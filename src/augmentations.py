from __future__ import annotations

from typing import Any


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _require_albumentations() -> Any:
    try:
        import albumentations as A
    except ImportError as exc:
        raise ImportError(
            "Albumentations is required. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return A


def get_train_transforms(image_size: int):
    """Build keypoint-aware training transforms."""
    A = _require_albumentations()
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.08,
                scale_limit=0.12,
                rotate_limit=20,
                border_mode=0,
                value=0,
                p=0.65,
            ),
            A.RandomBrightnessContrast(p=0.35),
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def get_valid_transforms(image_size: int):
    """Build deterministic validation/inference transforms."""
    A = _require_albumentations()
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
