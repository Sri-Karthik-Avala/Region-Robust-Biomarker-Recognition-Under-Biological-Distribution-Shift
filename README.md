# Region-Robust Biomarker Recognition Under Biological Distribution Shift

| | |
| --- | --- |
| Final rank | #4 |
| Domain | Computer Vision |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | A10G |
| Challenge status | Accepted / closed |
| Solutions submitted | 3 |
| Last submission | 2026-06-15 |

## Problem statement

### Overview

Multiplex immunohistochemistry (IHC) is widely used to study the tumor microenvironment by measuring the spatial distribution of immune biomarkers within tissue samples.

In this challenge, participants must develop a computer vision model that identifies the biomarker represented in a microscopy image.

Unlike conventional image classification benchmarks, evaluation is performed under a strict biological sample holdout protocol. Images originating from the same biological sample never appear in both training and test sets.

This prevents sample memorization and encourages robust biomarker recognition under biological distribution shift.

### Domain Generalization Challenge

Conventional pathology classifiers frequently exploit correlations between tissue regions and biomarker labels.

To discourage shortcut learning, this challenge evaluates biomarker recognition under two simultaneous sources of distribution shift:

- Biological sample shift
- Tissue region shift

The training and evaluation sets may contain different region distributions and different biomarker-region combinations.

As a result, successful solutions must learn biomarker-specific visual representations that remain stable across tissue regions and previously unseen biological samples.

### Dataset

The dataset contains high-resolution microscopy images collected from multiple biological samples.

Each biological sample contains multiple images corresponding to different biomarkers and tissue regions.

All original filenames and identifiers are removed during challenge preparation and replaced with obfuscated identifiers.

### Image Data

The images are stored in JPG format.

Characteristics:

- RGB microscopy images
- High-resolution tissue images
- Multiple biomarker staining types
- Multiple tissue regions
- Variable image dimensions depending on the source image

The images directory contains all image files referenced by train.csv and test.csv.

### Files

Public files:

- train.csv
- test.csv
- sample_submission.csv
- images/

Private files:

- answers.csv

### File Descriptions

train.csv contains the labeled training data.

Columns:

- id (string): unique image identifier
- image_path (string): relative path to image file
- marker (string): biomarker class label
- region (string): tissue region label

test.csv contains the evaluation data.

Columns:

- id (string): unique image identifier
- image_path (string): relative path to image file
- region (string): tissue region label

sample_submission.csv provides an example submission format.

answers.csv contains the hidden ground-truth biomarker labels used exclusively for evaluation and is never accessible to participants.

### Region Definitions

CT = Tumor Core

IM = Invasive Margin

N = Normal Tissue

Region information is provided as contextual metadata but is not the prediction target.

### Target Classes

The prediction target is the biomarker label contained in the marker column.

The exact set of valid biomarker classes is determined by the training data provided to participants.

Example biomarker classes present in the dataset include:

- CD3
- TIM3
- PDL1
- HE

Additional biomarker classes may also be present depending on the dataset split.

Participants should treat this as a multi-class classification problem where each image belongs to exactly one biomarker class.

### Task

Given a microscopy image, predict the corresponding biomarker class.

### Submission Format

Submit a CSV file with a header row.

Required columns:

id,prediction

Example submission:

id,prediction

a1b2c3d4e5,CD3

f6g7h8i9j0,TIM3

z9y8x7w6v5,PDL1

Requirements:

- Every test ID must appear exactly once
- No duplicate IDs are allowed
- Prediction must be a valid biomarker class present in the training data
- Exactly one class prediction must be provided for each image
- Probabilities are not accepted

### Evaluation

Final Score = 0.50 × Accuracy + 0.50 × Macro F1

Accuracy measures overall classification correctness.

Macro F1 computes the F1 score independently for each biomarker class and averages the results.

This metric rewards both strong predictive performance and balanced behavior across biomarker classes.

### Scientific Motivation

Many pathology models achieve strong benchmark performance by relying on spurious correlations between tissue context and biomarker expression.

However, these shortcuts often fail when deployed on images collected from new patients or different tissue regions.

This challenge explicitly evaluates robustness under biological and regional distribution shift, encouraging the development of models that learn transferable biomarker representations rather than dataset-specific patterns.

### Goal

Develop a robust biomarker recognition model capable of generalizing to unseen biological samples under realistic biological distribution shift conditions.
