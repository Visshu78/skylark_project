# Skylar Drones Aerial GCP Pose Estimation

Production-quality pipeline for detecting Ground Control Point marker center coordinates and marker shape from aerial imagery.

## Objective

Given an aerial image containing a GCP marker, predict:

- Marker center coordinates: `x`, `y`
- Marker shape class:
  - `Cross`
  - `Square`
  - `L-Shape`

This is implemented as a multitask learning problem with one shared visual backbone and two heads.

## Verified Dataset Facts

Training labels are stored in `train_dataset/gcp_marks.json`.

| Item | Value |
| --- | ---: |
| Labelled training images | 1000 |
| Test images | 300 |
| Projects | 11 |
| Surveys | 14 |
| Physical GCP groups | 159 |

Class distribution:

| Shape | Count |
| --- | ---: |
| `L-Shape` | 491 |
| `Square` | 328 |
| `Cross` | 177 |
| Missing shape labels | 4 |

Image resolutions:

- `4096 x 3068`
- `4096 x 2730`

The marker coordinate distribution spans the full image area, so the pipeline does not assume a center bias. Some label noise was observed in the `Cross` class during visual inspection.

## Leakage-Safe Splitting

Random image-level splitting is not valid for this dataset because multiple highly correlated images belong to the same physical GCP.

The split key is:

```text
project/survey/gcp_id
```

Folds are created using `GroupKFold`, ensuring images from the same physical GCP never appear in both training and validation.

Generate metadata and folds:

```powershell
.\.venv\Scripts\python.exe create_dataset_csv.py
.\.venv\Scripts\python.exe create_folds.py
```

## Repository Layout

```text
project/
  data/
  notebooks/
    eda.ipynb
  src/
    dataset.py
    augmentations.py
    model.py
    losses.py
    metrics.py
    train.py
    validate.py
    inference.py
  configs/
    default.json
  create_dataset_csv.py
  create_folds.py
  train.py
  validate.py
  inference.py
  dataset.csv
  dataset_folds.csv
  outputs/
    best_model.pth
    last_model.pth
    predictions.json
    training_curves.png
```

## Model Architecture

```text
EfficientNet-B3 backbone
        |
   shared features
        |
  -----------------
  |               |
Regression     Shape
Head           Head
```

Regression head:

```text
Linear -> ReLU -> Dropout(0.2) -> Linear(2) -> Sigmoid
```

Shape head:

```text
Linear -> ReLU -> Dropout(0.2) -> Linear(3)
```

The model is trained on marker-containing crops where the marker is deliberately
not centered, so the regression head returns coordinates normalized inside the
current crop/tile:

```text
x_crop_norm = x_crop / crop_width
y_crop_norm = y_crop / crop_height
```

`dataset.csv` also stores full-image normalized coordinates as metadata:
`x_norm = x / width` and `y_norm = y / height`.

No softmax is used inside the model. Classification uses `CrossEntropyLoss`.

## Crop-Based Image Pipeline

Full `4096 x ~3000` images are too large for efficient training on a 6GB GPU, so training uses marker-containing crops.

Default behavior:

- Crop size: `512 x 512`
- Final model input size: `512 x 512`
- Training crop placement: random marker position inside the crop
- Validation crop placement: deterministic pseudo-random marker position per image
- Crop target margin: keeps the marker away from crop borders when possible

Albumentations is used with keypoint-aware transforms so coordinate targets remain aligned after augmentation.

Required augmentations are included:

- `HorizontalFlip`
- `VerticalFlip`
- `RandomRotate90`
- `RandomBrightnessContrast`
- `ShiftScaleRotate`
- `Normalize`

## Missing Labels

Four rows have valid coordinate annotations but no `verified_shape`.

For these samples:

- Coordinate regression loss is still used
- Classification loss is ignored using `ignore_index=-100`

## Loss

Total loss:

```text
loss = SmoothL1Loss(coords) + 0.5 * CrossEntropyLoss(shape)
```

The classification weight is configurable in `configs/default.json`.

## Training Strategy

Training is split into two stages:

1. Freeze EfficientNet-B3 backbone and train heads only
2. Unfreeze the full model and fine-tune end-to-end

Default config:

- Optimizer: `AdamW`
- Learning rate: `3e-4`
- Scheduler: `CosineAnnealingLR`
- Mixed precision: `torch.cuda.amp`
- Best checkpoint metric: `PCK@10`
- Early stopping: enabled

CUDA is required for training. The code intentionally fails if no CUDA device is available.

Start training:

```powershell
.\.venv\Scripts\python.exe train.py --config configs/default.json
```

Outputs:

```text
outputs/best_model.pth
outputs/last_model.pth
outputs/training_curves.png
outputs/train.log
```

## Validation Metrics

Localization:

- Mean pixel error
- `PCK@10`
- `PCK@25`
- `PCK@50`

Classification:

- Accuracy
- Macro F1

Run validation:

```powershell
.\.venv\Scripts\python.exe validate.py --model outputs/best_model.pth
```

## Inference

Generate assignment-format predictions:

```powershell
.\.venv\Scripts\python.exe inference.py --model outputs/best_model.pth --data test_dataset --output outputs/predictions.json
```

Output format:

```json
{
  "path/to/image.JPG": {
    "mark": {
      "x": 1234.56,
      "y": 789.12
    },
    "verified_shape": "Cross"
  }
}
```

The default inference mode is tiled inference because test images do not provide annotations for crop centering. Since the requested architecture has no objectness head, tile selection uses the maximum shape-class probability as a practical confidence heuristic.

To run resized whole-image inference instead:

```powershell
.\.venv\Scripts\python.exe inference.py --mode whole
```

## Configuration

Main settings live in:

```text
configs/default.json
```

Important fields:

- `fold`
- `image_size`
- `crop_size`
- `crop_target_margin`
- `batch_size`
- `learning_rate`
- `stage1_epochs`
- `stage2_epochs`
- `inference_mode`
- `tile_stride`

## Environment

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For PyTorch with CUDA, install the wheel that matches the local CUDA setup if the default PyPI package is not GPU-enabled.
