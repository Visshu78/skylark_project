# Skylar Drones – Aerial GCP Pose Estimation

## Overview

This repository contains a complete machine learning pipeline for automated **Ground Control Point (GCP) localization** and **marker shape classification** from aerial drone imagery.

The system performs two tasks simultaneously:

1. **Keypoint Localization** – Predict the center coordinates `(x, y)` of a GCP marker.
2. **Shape Classification** – Predict the physical marker shape:

   * Cross
   * Square
   * L-Shape

**Note:** Links to the trained model checkpoint and final predictions.json are provided in the Resources section below.


---

# Problem Statement

Given an aerial image containing a Ground Control Point (GCP) marker, the objective is to:

* Predict the exact center coordinates of the marker.
* Classify the marker shape into one of three classes.

The challenge arises because the marker occupies only a tiny fraction of a high-resolution aerial image and appears under varying illumination, terrain, and viewing conditions.

---

# Dataset Overview

The dataset contains aerial images collected from real-world drone surveying operations.

Training labels are provided through:

```text
train_dataset/gcp_marks.json
```

Example annotation:

```json
{
  "mark": {
    "x": 1024.5,
    "y": 850.2
  },
  "verified_shape": "L-Shape"
}
```

The test dataset contains only images and no labels.

---

# Exploratory Data Analysis (EDA)

Before designing the training pipeline, a detailed analysis of the dataset was performed to understand its structure, distribution, and potential challenges.

## Dataset Statistics

| Metric              | Value |
| ------------------- | ----- |
| Training Images     | 1000  |
| Test Images         | 300   |
| Projects            | 11    |
| Surveys             | 14    |
| Physical GCP Groups | 159   |

The dataset follows a production-style hierarchical organization rather than a simplified academic structure.

Multiple images often belong to the same physical GCP marker.

---

## Shape Distribution

| Shape          | Count |
| -------------- | ----- |
| L-Shape        | 491   |
| Square         | 328   |
| Cross          | 177   |
| Missing Labels | 4     |

### Observations

* Moderate class imbalance exists.
* L-Shape markers account for nearly half of the dataset.
* Cross markers are the least represented class.
* Four images contain valid coordinate annotations but no shape labels.

### Impact on Design

* Macro F1 Score was selected for evaluation.
* Missing labels are ignored during classification training using:

```python
ignore_index = -100
```

* Coordinate regression continues to learn from those samples.

---

## Resolution Analysis

The dataset contains two image resolutions:

| Resolution  |
| ----------- |
| 4096 × 3068 |
| 4096 × 2730 |

### Observations

* Images are extremely high resolution.
* The GCP marker occupies only a tiny region of the image.
* Direct full-image coordinate regression is difficult.

### Impact on Design

A crop-based training strategy was adopted to convert the problem into a local localization task.

---

## Coordinate Distribution

The marker center coordinates span nearly the entire image plane.

### Observations

* No strong center bias exists.
* Markers appear close to borders as well as near image centers.

### Impact on Design

* No positional prior was introduced.
* Marker placement inside crops is randomized.
* The model cannot rely on the marker always appearing near the center.

---

## Dataset Correlation Analysis

Images belonging to the same physical GCP share:

* Similar viewpoints
* Similar backgrounds
* The same marker

A random image-level split would therefore introduce severe data leakage.

### Impact on Design

A leakage-safe split was implemented using:

```text
GroupKFold
```

Grouping key:

```text
project/survey/gcp_id
```

This guarantees that images from the same physical marker never appear in both training and validation sets.

---

## Label Quality Analysis

Visual inspection revealed:

* Minor annotation inconsistencies in some Cross markers.
* Significant variation in:

  * Illumination
  * Shadows
  * Terrain
  * Viewing angle

### Impact on Design

The following augmentations were used:

* Horizontal Flip
* Vertical Flip
* Random Rotate 90
* ShiftScaleRotate
* Random BrightnessContrast

---

# Assumptions

The assignment intentionally contains ambiguities.

The following assumptions were made:

1. Every image contains exactly one GCP marker.
2. Marker center coordinates are accurate.
3. Shape labels are correct whenever provided.
4. Missing shape labels should not prevent coordinate learning.
5. Test images follow a similar distribution to the training dataset.
6. During inference, the tile with the highest classification confidence is assumed to contain the marker.

---

## Executive Summary

This project presents a multitask deep learning solution for automated Ground Control Point (GCP) localization and shape classification in aerial survey imagery.

The final solution uses an EfficientNet-B3 backbone with separate regression and classification heads. A crop-based training strategy and tile-based inference pipeline were adopted to handle extremely high-resolution drone imagery efficiently.

Final validation performance:

| Metric | Result |
|----------|----------|
| Mean Localization Error | 20.32 px |
| Median Localization Error | 17.20 px |
| Shape Accuracy | ~100% |
| Macro F1 Score | ~1.0 |

---

---

# Model Architecture

## Backbone

EfficientNet-B3

### Why EfficientNet-B3?

* Excellent accuracy-to-parameter ratio.
* Strong transfer learning performance.
* Lower memory requirements than larger CNNs.
* Suitable for training on a 6GB GPU.

---

## Multi-Task Architecture

```text
EfficientNet-B3 Backbone
            |
     Shared Features
            |
    -----------------
    |               |
Regression      Classification
   Head             Head
```

### Regression Head

```text
Linear
→ ReLU
→ Dropout(0.2)
→ Linear
→ Sigmoid
```

Output:

```text
(x, y)
```

### Classification Head

```text
Linear
→ ReLU
→ Dropout(0.2)
→ Linear
```

Output:

```text
Cross / Square / L-Shape
```

---

# Training Pipeline

## Crop-Based Training

Instead of training on entire 4096×3000 images:

* Marker-containing crops are generated dynamically.
* Crop size: 512×512
* Marker position is randomized inside the crop.
* Coordinates are converted into crop-relative normalized coordinates.

This significantly simplifies the localization task.

---

## Data Augmentation

Albumentations was used with keypoint-aware transformations.

Applied augmentations:

* Horizontal Flip
* Vertical Flip
* Random Rotate 90
* ShiftScaleRotate
* Random BrightnessContrast
* Resize
* Normalize

---

## Optimization

| Parameter       | Value             |
| --------------- | ----------------- |
| Optimizer       | AdamW             |
| Learning Rate   | 3e-4              |
| Weight Decay    | 1e-4              |
| Scheduler       | CosineAnnealingLR |
| Mixed Precision | Enabled           |
| Early Stopping  | Enabled           |

---

## Two-Stage Training

### Stage 1

Freeze EfficientNet backbone and train task-specific heads.

### Stage 2

Unfreeze the full network and fine-tune end-to-end.

This stabilizes learning and improves transfer learning effectiveness.

---

# Validation Strategy

A GroupKFold-based validation strategy was used.

Current experiments were performed on:

```text
Fold 0
```

Approximate split:

| Dataset    | Samples |
| ---------- | ------- |
| Training   | ~800    |
| Validation | ~200    |

No validation image appears in training.

---

# Results

## Classification Performance

| Metric   | Result |
| -------- | ------ |
| Accuracy | ≈ 100% |
| Macro F1 | ≈ 1.0  |

The model successfully distinguishes all marker shapes.

---

## Localization Performance

| Metric        | Value     |
| ------------- | --------- |
| Mean Error    | 20.32 px  |
| Median Error  | 17.20 px  |
| Minimum Error | 2.21 px   |
| Maximum Error | 144.00 px |

The majority of predictions fall very close to the true marker center.

---

### Final Training Configuration

| Parameter | Value |
|------------|------------|
| Image Size | 512 |
| Crop Size | 512 |
| Batch Size | 8 |
| Learning Rate | 3e-4 |
| Weight Decay | 1e-4 |
| Stage 1 Epochs | 3 |
| Stage 2 Epochs | 27 |

---

# Test Dataset Evaluation

Inference was successfully executed on all 300 test images, generating a submission-ready predictions.json file.

Ground-truth annotations are not available for the test dataset.

Therefore:

* Numerical evaluation is impossible.
* Qualitative evaluation was performed.

Random test images were manually inspected.

The predicted marker locations consistently aligned with visible GCP markers, indicating successful generalization to unseen projects.

---

# Inference Pipeline

Large aerial images are processed using overlapping tiles.

Pipeline:

1. Generate overlapping 512×512 tiles.
2. Run the model on every tile.
3. Compute classification confidence.
4. Select the tile with highest confidence.
5. Convert coordinates back to original image space.
6. Save predictions.

Output format:

```json
{
  "image_path": {
    "mark": {
      "x": 1234.5,
      "y": 987.6
    },
    "verified_shape": "Cross"
  }
}
```

---

# Resources

## Dataset

Dataset provided by Skylar Drones:

[https://drive.google.com/drive/folders/1RDNiAO1EynKrRDomcVNXQW1-ct5zzvE5](https://drive.google.com/drive/folders/1RDNiAO1EynKrRDomcVNXQW1-ct5zzvE5?usp=sharing)

---

## Trained Model

Best trained checkpoint:

[https://drive.google.com/file/d/1LWS9-Qd5VgPvSaHIHcksJaiG8BTquYwD/view?usp=drive_link](https://drive.google.com/file/d/1LWS9-Qd5VgPvSaHIHcksJaiG8BTquYwD/view?usp=drive_link)

Download:

best_model.pth

Place inside:

outputs/

---

## Final Predictions

Generated predictions on the complete test dataset:

[https://drive.google.com/file/d/1-mLi8JXy-7HsehZk-fhYo1nF_fAYpby9/view?usp=drive_link](https://drive.google.com/file/d/1-mLi8JXy-7HsehZk-fhYo1nF_fAYpby9/view?usp=drive_link)

File:

predictions.json
---

---

# How to Train

```bash
python train.py --config configs/fold0_30epochs.json
```

---

# How to Run Inference

```bash
python inference.py
```

Output:

```text
outputs/predictions.json
---

---

# Repository Structure

```text
src/
configs/

train.py
validate.py
inference.py
visualize_predictions.py

create_dataset_csv.py
create_folds.py

README.md
REPORT.md
requirements.txt
```

# Deliverables Checklist

| Requirement | Status |
|-------------|---------|
| EDA Analysis | ✅ |
| Data Processing Pipeline | ✅ |
| Training Pipeline | ✅ |
| Validation Pipeline | ✅ |
| Inference Pipeline | ✅ |
| Trained Model Weights | ✅ |
| predictions.json | ✅ |
| Documentation | ✅ |

---

# Future Improvements

Potential improvements include:

* Heatmap-based localization
* Hard negative mining
* Confidence calibration
* Higher-resolution crops
* Cross-fold ensembling
* Full 5-fold training

---

# Author

**Vishal Dhawal**

B.Tech Computer Science and Engineering
IIIT Kottayam
