"""
Region-Robust Biomarker Recognition under Biological Distribution Shift.

Task: multi-class classification of the IHC biomarker (marker column) from a
microscopy image.  Evaluation: 0.50 * Accuracy + 0.50 * Macro-F1 under a strict
biological-sample + tissue-region holdout.

v3 — MULTI-SCALE, multi-backbone diverse ensemble.
  The OOF confusion of v1/v2 is dominated by the CD3 / CD8 / CD45RO triad (the
  three T-cell markers share the same brown DAB chromogen and differ only by
  CELL DENSITY).  That signal is resolution-limited, so v3 adds a 512px ConvNeXt
  (sharper cells) alongside a 384px EfficientNetV2-S (different inductive bias):
    * cnx512   : ConvNeXt-Small  @512  (k folds)
    * effv2_384: EfficientNetV2-S @384  (k folds)
  Softmax probabilities are blended (weights validated on held-out OOF), the
  2-sample HE class is excluded on test (the holdout test has none), and every
  model's test + OOF probabilities are cached to working/*.npz for instant
  re-blending without retraining.

Design notes (from EDA):
  * 7 real classes + typo merges  CDRO45->CD45RO , LAG1->LAG3 ; HE (n=2) kept in
    training but never predicted on test.
  * COLOR is signal (separates super-groups) -> preserve it (no stain-norm, tiny
    hue jitter); heavy SPATIAL aug forces within-group density learning.
  * IMAGE-ONLY (region never fed) -> robust to the region<->marker shortcut
    (e.g. T is 100% IM in train).  Region-prior post-processing was tested on
    held-out OOF and gave only +0.0017 with real shift risk -> NOT used.

Output: ./working/submission.csv
"""

from __future__ import annotations

import gc
import math
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2  # noqa: E402

cv2.setNumThreads(0)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.metrics import f1_score, accuracy_score  # noqa: E402

import albumentations as A  # noqa: E402
from albumentations.pytorch import ToTensorV2  # noqa: E402
import timm  # noqa: E402


# ============================== config ====================================== #

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

LABEL_MAP = {"CDRO45": "CD45RO", "LAG1": "LAG3"}  # typo merges

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

if DEVICE.type == "cuda":
    MAXSIDE = 1536          # one cache, large enough for 512px crops
    EPOCHS = 14
    NUM_WORKERS = 4
    TIME_BUDGET_SEC = 100 * 60
    # (label, backbone, n_folds, img_size) — multi-scale + multi-arch.
    BACKBONES = [
        ("cnx512", "convnext_small.fb_in22k_ft_in1k", 4, 512),
        ("effv2_384", "tf_efficientnetv2_s.in21k_ft_in1k", 4, 384),
    ]
    BLEND_WEIGHTS = {"cnx512": 0.6, "effv2_384": 0.4}
else:
    MAXSIDE = 768
    EPOCHS = 4
    NUM_WORKERS = 0
    TIME_BUDGET_SEC = 60 * 60
    BACKBONES = [("cnx224", "convnext_nano.in12k_ft_in1k", 2, 224)]
    BLEND_WEIGHTS = {"cnx224": 1.0}

LR = 2e-4
WEIGHT_DECAY = 0.02
WARMUP_EPOCHS = 2
LABEL_SMOOTH = 0.05
EMA_DECAY = 0.998
DROP_RATE = 0.1
DROP_PATH = 0.1

CACHE_DIRNAME = f"cache_{MAXSIDE}"


def batch_for(size: int) -> int:
    return 16 if size <= 384 else (12 if size <= 512 else 8)


# ============================== paths ======================================= #


def find_root() -> Path:
    here = Path.cwd()
    for root in [here, here / "public", here / "dataset" / "public",
                 here / "dataset", here.parent / "public",
                 Path("/kaggle/input"), Path("./input")]:
        if (root / "train.csv").exists() and (root / "test.csv").exists():
            return root
    raise FileNotFoundError("Could not locate train.csv/test.csv.")


def resolve_image_path(root: Path, rel: str) -> Path:
    p = root / rel
    if p.exists():
        return p
    name = Path(rel).name
    for d in [root / "images", root, root / "public" / "images"]:
        if (d / name).exists():
            return d / name
    return p


def working_dir() -> Path:
    d = Path.cwd() / "working"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ============================== repro ======================================= #


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ============================== image cache ================================= #


def build_cache(df: pd.DataFrame, root: Path, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [(r["image_path"], cache_dir / f"{r['id']}.jpg")
            for _, r in df.iterrows() if not (cache_dir / f"{r['id']}.jpg").exists()]
    if not todo:
        return
    print(f"caching {len(todo)} images -> {cache_dir} ...", flush=True)
    t0 = time.time()
    for rel, out in todo:
        im = cv2.imread(str(resolve_image_path(root, rel)), cv2.IMREAD_COLOR)
        if im is None:
            im = np.full((256, 256, 3), 128, np.uint8)
        h, w = im.shape[:2]
        s = MAXSIDE / max(h, w)
        if s < 1.0:
            im = cv2.resize(im, (int(round(w * s)), int(round(h * s))),
                            interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  cache built in {time.time() - t0:.1f}s", flush=True)


def load_cached(cache_dir: Path, img_id: str) -> np.ndarray:
    im = cv2.imread(str(cache_dir / f"{img_id}.jpg"), cv2.IMREAD_COLOR)
    if im is None:
        im = np.full((256, 256, 3), 128, np.uint8)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


# ============================== transforms ================================== #
# COLOR IS SIGNAL -> heavy spatial aug, only mild photometric aug.


def train_tf(size: int) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(size=(size, size), scale=(0.35, 1.0),
                            ratio=(0.8, 1.25), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(rotate=(-25, 25), translate_percent=(0.0, 0.05),
                 scale=(0.9, 1.1), shear=(-6, 6), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.5),
        A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=10,
                             val_shift_limit=8, p=0.3),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                 A.MotionBlur(blur_limit=5, p=1.0)], p=0.15),
        A.GaussNoise(p=0.12),
        A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(12, 40),
                        hole_width_range=(12, 40), fill=0, p=0.2),
        A.Normalize(mean=tuple(IMAGENET_MEAN), std=tuple(IMAGENET_STD)),
        ToTensorV2(),
    ])


def eval_tf(size: int) -> A.Compose:
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=tuple(IMAGENET_MEAN), std=tuple(IMAGENET_STD)),
        ToTensorV2(),
    ])


def eval_resize(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def normalize_to_tensor(img):
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).contiguous()


def tta_views(img, size):
    """full + center + 4 corners (75%), each + h-flip -> 12 views."""
    h, w = img.shape[:2]
    ch, cw = int(h * 0.75), int(w * 0.75)
    boxes = [
        (0, 0, w, h),
        ((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch),
        (0, 0, cw, ch), (w - cw, 0, w, ch), (0, h - ch, cw, h), (w - cw, h - ch, w, h),
    ]
    views = []
    for x1, y1, x2, y2 in boxes:
        crop = eval_resize(img[y1:y2, x1:x2], size)
        views.append(normalize_to_tensor(crop))
        views.append(normalize_to_tensor(np.ascontiguousarray(crop[:, ::-1])))
    return torch.stack(views, 0)


# ============================== datasets ==================================== #


class TrainDS(Dataset):
    def __init__(self, df, cache_dir, tf):
        self.df = df.reset_index(drop=True); self.cache_dir = cache_dir; self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = self.tf(image=load_cached(self.cache_dir, r["id"]))["image"]
        return x, int(r["y"])


class TTADS(Dataset):
    def __init__(self, df, cache_dir, size):
        self.df = df.reset_index(drop=True); self.cache_dir = cache_dir; self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return tta_views(load_cached(self.cache_dir, r["id"]), self.size), r["id"]


# ============================== model / ema ================================= #


def build_model(name, n_classes):
    return timm.create_model(name, pretrained=True, num_classes=n_classes,
                             drop_rate=DROP_RATE, drop_path_rate=DROP_PATH)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}


def cosine_lr(epoch, total, warmup, base):
    if epoch < warmup:
        return base * (epoch + 1) / max(1, warmup)
    prog = (epoch - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * prog))


def metric_score(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return 0.5 * acc + 0.5 * f1, acc, f1


# ============================== eval helpers ================================ #


@torch.no_grad()
def predict_val(model, loader):
    model.eval()
    out, ys = [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            logits = model(x)
        out.append(F.softmax(logits.float(), 1).cpu()); ys.append(y)
    return torch.cat(out).numpy(), torch.cat(ys).numpy()


@torch.no_grad()
def predict_tta_probs(model, df, cache_dir, size):
    loader = DataLoader(TTADS(df, cache_dir, size), batch_size=max(4, batch_for(size) // 2),
                        shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))
    model.eval()
    probs, ids = [], []
    for views, batch_ids in loader:
        b, v, c, h, w = views.shape
        x = views.view(b * v, c, h, w).to(DEVICE, non_blocking=True)
        with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            logits = model(x)
        probs.append(F.softmax(logits.float(), 1).view(b, v, -1).mean(1).cpu().numpy())
        ids.extend(list(batch_ids))
    return np.concatenate(probs, 0), ids


# ============================== train one fold ============================== #


def train_fold(label, name, fold, df_tr, df_va, cache_dir, size, class_weights, n_classes):
    bs = batch_for(size)
    tr_loader = DataLoader(TrainDS(df_tr, cache_dir, train_tf(size)), batch_size=bs,
                           shuffle=True, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
                           drop_last=True, persistent_workers=NUM_WORKERS > 0)
    va_loader = DataLoader(TrainDS(df_va, cache_dir, eval_tf(size)), batch_size=bs * 2,
                           shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"),
                           persistent_workers=NUM_WORKERS > 0)

    model = build_model(name, n_classes).to(DEVICE)
    ema = EMA(model, EMA_DECAY)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)
    w = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTH)

    best_score, best_state, best_probs, best_tag = -1.0, None, None, "raw"
    for ep in range(EPOCHS):
        lr = cosine_lr(ep, EPOCHS, WARMUP_EPOCHS, LR)
        for pg in opt.param_groups:
            pg["lr"] = lr
        model.train()
        t0, run = time.time(), 0.0
        for x, y in tr_loader:
            x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            ema.update(model)
            run += float(loss.item())
        probs, yv = predict_val(model, va_loader)
        score, acc, f1 = metric_score(yv, probs.argmax(1))
        print(f"  [{label}] f{fold} ep{ep+1:02d} loss{run/max(1,len(tr_loader)):.3f}"
              f" val{score:.4f} acc{acc:.4f} f1{f1:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if score > best_score:
            best_score, best_probs, best_tag = score, probs, "raw"
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    ema_model = build_model(name, n_classes).to(DEVICE)
    ema_model.load_state_dict(ema.state_dict())
    probs, yv = predict_val(ema_model, va_loader)
    ema_score, acc, f1 = metric_score(yv, probs.argmax(1))
    if ema_score > best_score:
        best_score, best_probs, best_tag = ema_score, probs, "ema"
        best_state = {k: v.detach().cpu().clone() for k, v in ema_model.state_dict().items()}
    print(f"  [{label}] fold {fold} best={best_score:.4f} ({best_tag})", flush=True)

    del model, ema_model
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return best_state, best_score, best_probs, yv


# ============================== submission helper ========================== #


def write_submission(test_ids, prob, idx2cls, majority, out_csv, exclude=None):
    p = prob.copy()
    if exclude is not None:
        p[:, exclude] = -1.0
    preds = p.argmax(1)
    sub = pd.DataFrame({"id": test_ids, "prediction": [idx2cls[int(i)] for i in preds]})
    sub["prediction"] = sub["prediction"].fillna(majority)
    sub.to_csv(out_csv, index=False)
    return sub


# ============================== main ======================================== #


def main():
    t_start = time.time()
    seed_everything(SEED)
    root = find_root()
    wd = working_dir()
    out_csv = wd / "submission.csv"
    cache_dir = wd / CACHE_DIRNAME
    print("root:", root, "device:", DEVICE, flush=True)

    train_df = pd.read_csv(root / "train.csv")
    test_df = pd.read_csv(root / "test.csv")
    train_df["marker"] = train_df["marker"].replace(LABEL_MAP)
    classes = sorted(train_df["marker"].unique())
    cls2idx = {c: i for i, c in enumerate(classes)}
    idx2cls = {i: c for c, i in cls2idx.items()}
    train_df["y"] = train_df["marker"].map(cls2idx)
    n_classes = len(classes)
    counts = train_df["marker"].value_counts()
    HE_idx = cls2idx.get("HE", None)
    print("classes:", classes, "| counts:", counts.to_dict(), flush=True)

    med = float(np.median(counts.values))
    cw = np.array([min(4.0, max(0.5, math.sqrt(med / counts[c]))) for c in classes], np.float32)

    build_cache(pd.concat([train_df[["id", "image_path"]], test_df[["id", "image_path"]]]),
                root, cache_dir)

    majority = counts.idxmax()
    test_ids = test_df["id"].tolist()
    pd.DataFrame({"id": test_ids, "prediction": majority}).to_csv(out_csv, index=False)

    N = len(train_df)
    running_sum = None          # running ensemble (weighted) for incremental writes
    per_label_test, per_label_oof = {}, {}

    for label, name, n_folds, size in BACKBONES:
        if time.time() - t_start > TIME_BUDGET_SEC and per_label_test:
            print("time budget hit before", label, flush=True)
            break
        try:
            del_m = build_model(name, n_classes); del del_m
        except Exception as e:
            print(f"!! skipping {label} ({name}): {e}", flush=True)
            continue
        wgt = BLEND_WEIGHTS.get(label, 1.0)
        print(f"\n##### {label}: {name} @ {size}px, {n_folds} folds, weight {wgt} #####", flush=True)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        bb_test, bb_oof, done = None, np.zeros((N, n_classes), np.float32), 0
        for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, train_df["y"])):
            if time.time() - t_start > TIME_BUDGET_SEC and per_label_test:
                print("time budget hit mid-", label, flush=True); break
            df_tr = train_df.iloc[tr_idx].reset_index(drop=True)
            df_va = train_df.iloc[va_idx].reset_index(drop=True)
            state, sc, vprobs, _ = train_fold(label, name, fold, df_tr, df_va,
                                              cache_dir, size, cw, n_classes)
            bb_oof[va_idx] = vprobs
            m = build_model(name, n_classes).to(DEVICE)
            m.load_state_dict(state)
            tprobs, ids = predict_tta_probs(m, test_df, cache_dir, size)
            assert ids == test_ids
            del m, state
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            bb_test = tprobs if bb_test is None else bb_test + tprobs
            done += 1
            # incremental write: rebuild ensemble from finished labels + current partial
            ens = np.zeros((len(test_ids), n_classes), np.float32); wsum = 0.0
            for lb, v in per_label_test.items():
                ens += BLEND_WEIGHTS.get(lb, 1.0) * v; wsum += BLEND_WEIGHTS.get(lb, 1.0)
            ens += wgt * (bb_test / done); wsum += wgt
            sub = write_submission(test_ids, ens / wsum, idx2cls, majority, out_csv, exclude=HE_idx)
            print(f"   -> {label} {done}/{n_folds} folds; ens dist {sub['prediction'].value_counts().to_dict()}",
                  flush=True)
        if bb_test is not None and done > 0:
            per_label_test[label] = bb_test / done
            per_label_oof[label] = bb_oof
            cov = bb_oof.sum(1) > 0
            if cov.any():
                s, a, f = metric_score(train_df["y"].values[cov], bb_oof[cov].argmax(1))
                print(f"== {label} OOF: score{s:.4f} acc{a:.4f} f1{f:.4f} ==", flush=True)

    # ---- save artifacts ----
    if per_label_test:
        np.savez(wd / "test_probs_v3.npz", ids=np.array(test_ids), classes=np.array(classes),
                 y=train_df["y"].values,
                 **{f"test__{k}": v for k, v in per_label_test.items()},
                 **{f"oof__{k}": v for k, v in per_label_oof.items()})
        print("saved working/test_probs_v3.npz", flush=True)

        # blended OOF CV (labels covering all rows)
        full = {k: v for k, v in per_label_oof.items() if (v.sum(1) > 0).all()}
        if full:
            num = sum(BLEND_WEIGHTS.get(k, 1.0) * v for k, v in full.items())
            den = sum(BLEND_WEIGHTS.get(k, 1.0) for k in full)
            blend = num / den
            pr = blend.copy()
            if HE_idx is not None:
                pr[:, HE_idx] = -1.0
            yv = train_df["y"].values
            mask = yv != (HE_idx if HE_idx is not None else -1)
            s, a, f = metric_score(yv[mask], pr.argmax(1)[mask])
            print(f"\n=== BLENDED OOF CV (main classes): score{s:.4f} acc{a:.4f} f1{f:.4f} ===", flush=True)

    # ---- final weighted blend + HE exclusion ----
    if per_label_test:
        num = sum(BLEND_WEIGHTS.get(k, 1.0) * v for k, v in per_label_test.items())
        den = sum(BLEND_WEIGHTS.get(k, 1.0) for k in per_label_test)
        sub = write_submission(test_ids, num / den, idx2cls, majority, out_csv, exclude=HE_idx)
        print("\nFINAL pred distribution:", sub["prediction"].value_counts().to_dict(), flush=True)

    print(f"\nwrote {out_csv} rows={len(test_df)} total {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
