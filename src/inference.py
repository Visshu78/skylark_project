from __future__ import annotations
from tqdm import tqdm
import argparse
import json
import logging
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch

from src.augmentations import get_valid_transforms
from src.dataset import IDX_TO_SHAPE
from src.model import GCPMultiTaskModel
from src.utils import load_config


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def iter_images(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_checkpoint(model_path: str | Path, device: torch.device) -> tuple[dict, dict]:
    checkpoint = torch.load(model_path, map_location=device)
    return checkpoint["model"], checkpoint.get("config", {})


def preprocess_image(image: np.ndarray, image_size: int) -> torch.Tensor:
    transforms = get_valid_transforms(image_size)
    transformed = transforms(image=image, keypoints=[(image.shape[1] / 2, image.shape[0] / 2)])
    arr = transformed["image"]
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr, dtype=torch.float32)


@torch.no_grad()
def predict_whole_image(
    model: torch.nn.Module,
    image: np.ndarray,
    image_size: int,
    device: torch.device,
) -> tuple[float, float, str, float]:
    h, w = image.shape[:2]
    tensor = preprocess_image(image, image_size).unsqueeze(0).to(device)
    outputs = model(tensor)
    coord = outputs["coords"][0].detach().cpu().numpy()
    probs = torch.softmax(outputs["logits"], dim=1)[0].detach().cpu().numpy()
    shape_idx = int(np.argmax(probs))
    return (
        float(np.clip(coord[0] * w, 0, w - 1)),
        float(np.clip(coord[1] * h, 0, h - 1)),
        IDX_TO_SHAPE[shape_idx],
        float(probs[shape_idx]),
    )


def make_tiles(image: np.ndarray, tile_size: int, stride: int) -> Iterator[tuple[np.ndarray, int, int, int, int]]:
    h, w = image.shape[:2]
    xs = list(range(0, max(1, w - tile_size + 1), stride))
    ys = list(range(0, max(1, h - tile_size + 1), stride))
    if not xs or xs[-1] != max(0, w - tile_size):
        xs.append(max(0, w - tile_size))
    if not ys or ys[-1] != max(0, h - tile_size):
        ys.append(max(0, h - tile_size))

    for y0 in sorted(set(ys)):
        for x0 in sorted(set(xs)):
            x1 = min(w, x0 + tile_size)
            y1 = min(h, y0 + tile_size)
            tile = image[y0:y1, x0:x1]
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=image.dtype)
                padded[: tile.shape[0], : tile.shape[1]] = tile
                tile = padded
            yield tile, x0, y0, x1 - x0, y1 - y0


@torch.no_grad()
def predict_tiled(
    model: torch.nn.Module,
    image: np.ndarray,
    image_size: int,
    tile_size: int,
    stride: int,
    device: torch.device,
) -> tuple[float, float, str, float]:
    best: tuple[float, float, str, float] | None = None
    for tile, x0, y0, valid_w, valid_h in make_tiles(image, tile_size, stride):
        tensor = preprocess_image(tile, image_size).unsqueeze(0).to(device)
        outputs = model(tensor)
        coord = outputs["coords"][0].detach().cpu().numpy()
        probs = torch.softmax(outputs["logits"], dim=1)[0].detach().cpu().numpy()
        shape_idx = int(np.argmax(probs))
        score = float(probs[shape_idx])
        x = float(x0 + np.clip(coord[0], 0.0, 1.0) * valid_w)
        y = float(y0 + np.clip(coord[1], 0.0, 1.0) * valid_h)
        candidate = (x, y, IDX_TO_SHAPE[shape_idx], score)
        if best is None or score > best[3]:
            best = candidate
    if best is None:
        raise RuntimeError("No tiles generated for image")
    h, w = image.shape[:2]
    return (
        float(np.clip(best[0], 0, w - 1)),
        float(np.clip(best[1], 0, h - 1)),
        best[2],
        best[3],
    )


def run_inference(
    model_path: str | Path,
    data_root: str | Path,
    output_path: str | Path,
    config_path: str | Path = "configs/default.json",
    mode: str | None = None,
) -> None:
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        logging.warning("CUDA not detected; inference is running on CPU.")

    state_dict, checkpoint_config = load_checkpoint(model_path, device)

    merged = {**config, **checkpoint_config}

    model = GCPMultiTaskModel(
        pretrained=False,
        dropout=float(merged.get("dropout", 0.2)),
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    data_root = Path(data_root)

    predictions: dict[str, dict] = {}

    inference_mode = mode or str(
        merged.get("inference_mode", "tiles")
    )

    image_paths = list(iter_images(data_root))

    print(f"\nFound {len(image_paths)} images")
    print("Starting inference...\n")

    for image_path in tqdm(
        image_paths,
        desc="Running inference",
        unit="image",
    ):

        image_bgr = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image_bgr is None:
            logging.warning(
                "Skipping unreadable image: %s",
                image_path,
            )
            continue

        image = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        if inference_mode == "whole":

            x, y, shape, score = predict_whole_image(
                model,
                image,
                int(merged["image_size"]),
                device,
            )

        elif inference_mode == "tiles":

            x, y, shape, score = predict_tiled(
                model,
                image,
                int(merged["image_size"]),
                int(merged["crop_size"]),
                int(merged["tile_stride"]),
                device,
            )

        else:
            raise ValueError(
                "inference mode must be 'whole' or 'tiles'"
            )

        rel_path = image_path.relative_to(
            data_root
        ).as_posix()

        predictions[rel_path] = {
            "mark": {
                "x": float(x),
                "y": float(y),
            },
            "verified_shape": shape,
            "confidence": float(score),
        }

    print(f"\nFinished processing {len(predictions)} images")

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Saving predictions to: {output_path}")

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            predictions,
            f,
            indent=2,
        )

    logging.info(
        "Saved %d predictions to %s",
        len(predictions),
        output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/best_model.pth")
    parser.add_argument("--data", default="test_dataset")
    parser.add_argument("--output", default="outputs/predictions.json")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--mode", choices=["whole", "tiles"], default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_inference(args.model, args.data, args.output, args.config, args.mode)


if __name__ == "__main__":
    main()
