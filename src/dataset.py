from __future__ import annotations

import random
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


SHAPE_TO_IDX = {"Cross": 0, "Square": 1, "L-Shape": 2}
IDX_TO_SHAPE = {v: k for k, v in SHAPE_TO_IDX.items()}
IGNORE_LABEL = -100


@dataclass(frozen=True)
class Sample:
    path: str
    full_path: str
    width: int
    height: int
    x: float
    y: float
    shape: str | float | None


def crop_containing_point(
    image: np.ndarray,
    x: float,
    y: float,
    crop_size: int,
    target_margin: int = 64,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, float, float, tuple[int, int]]:
    """Crop so the point is inside the crop at a non-centered target position."""
    h, w = image.shape[:2]
    del h, w
    rng = rng or random
    margin = min(max(0, target_margin), max(0, crop_size // 2 - 1))
    target_x = rng.uniform(margin, crop_size - margin)
    target_y = rng.uniform(margin, crop_size - margin)

    x1 = int(round(x - target_x))
    y1 = int(round(y - target_y))
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    h, w = image.shape[:2]
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if any((pad_left, pad_top, pad_right, pad_bottom)):
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    x1_pad = x1 + pad_left
    y1_pad = y1 + pad_top
    crop = image[y1_pad : y1_pad + crop_size, x1_pad : x1_pad + crop_size]
    point_x = x - x1
    point_y = y - y1
    return crop, point_x, point_y, (x1, y1)


class GCPDataset(Dataset):
    """Dataset for crop-based GCP coordinate regression and shape classification."""

    def __init__(
        self,
        csv_path: str | Path,
        fold: int,
        mode: str,
        transforms: Any,
        crop_size: int = 512,
        crop_target_margin: int = 64,
        train_crop_jitter: int | None = None,
    ) -> None:
        del train_crop_jitter
        self.df = pd.read_csv(csv_path)
        if mode == "train":
            self.df = self.df[self.df["fold"] != fold].reset_index(drop=True)
        elif mode in {"valid", "val"}:
            self.df = self.df[self.df["fold"] == fold].reset_index(drop=True)
        else:
            raise ValueError("mode must be 'train' or 'valid'")

        self.mode = mode
        self.transforms = transforms
        self.crop_size = crop_size
        self.crop_target_margin = crop_target_margin

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        sample = Sample(
            path=str(row["path"]),
            full_path=str(row["full_path"]),
            width=int(row["width"]),
            height=int(row["height"]),
            x=float(row["x"]),
            y=float(row["y"]),
            shape=row.get("shape"),
        )

        image_bgr = cv2.imread(sample.full_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {sample.full_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        rng = None
        if self.mode != "train":
            digest = hashlib.sha1(sample.path.encode("utf-8")).hexdigest()
            rng = random.Random(int(digest[:8], 16))

        crop, crop_x, crop_y, origin = crop_containing_point(
            image=image,
            x=sample.x,
            y=sample.y,
            crop_size=self.crop_size,
            target_margin=self.crop_target_margin,
            rng=rng,
        )

        transformed = self.transforms(image=crop, keypoints=[(crop_x, crop_y)])
        image_out = transformed["image"]
        keypoint = transformed["keypoints"][0]

        if image_out.ndim == 3:
            image_out = np.transpose(image_out, (2, 0, 1))

        _, out_h, out_w = image_out.shape
        coord = torch.tensor(
            [
                float(np.clip(keypoint[0] / out_w, 0.0, 1.0)),
                float(np.clip(keypoint[1] / out_h, 0.0, 1.0)),
            ],
            dtype=torch.float32,
        )

        shape_value = sample.shape
        shape_target = IGNORE_LABEL
        if isinstance(shape_value, str) and shape_value in SHAPE_TO_IDX:
            shape_target = SHAPE_TO_IDX[shape_value]

        return {
            "image": torch.tensor(image_out, dtype=torch.float32),
            "coord": coord,
            "shape": torch.tensor(shape_target, dtype=torch.long),
            "path": sample.path,
            "origin": torch.tensor(origin, dtype=torch.float32),
            "crop_size": torch.tensor(self.crop_size, dtype=torch.float32),
            "width": torch.tensor(sample.width, dtype=torch.float32),
            "height": torch.tensor(sample.height, dtype=torch.float32),
        }
