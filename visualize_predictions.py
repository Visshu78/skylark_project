from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.augmentations import get_valid_transforms
from src.dataset import GCPDataset, IDX_TO_SHAPE
from src.model import GCPMultiTaskModel


CONFIG = {
    "csv_path": "dataset_folds.csv",
    "fold": 0,
    "image_size": 512,
    "crop_size": 512,
    "crop_target_margin": 64,
    "checkpoint": "outputs/best_model.pth",
    "num_samples": 200,
    "output_dir": "prediction_visualizations",
}


def unnormalize(image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = image.transpose(1, 2, 0)
    image = image * std + mean
    image = np.clip(image, 0, 1)

    return image


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = GCPDataset(
        csv_path=CONFIG["csv_path"],
        fold=CONFIG["fold"],
        mode="valid",
        transforms=get_valid_transforms(CONFIG["image_size"]),
        crop_size=CONFIG["crop_size"],
        crop_target_margin=CONFIG["crop_target_margin"],
    )

    checkpoint = torch.load(
        CONFIG["checkpoint"],
        map_location=device,
    )

    model = GCPMultiTaskModel(
        pretrained=False,
        dropout=0.2,
    )

    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        all_errors = []
        for idx in range(min(CONFIG["num_samples"], len(dataset))):

            sample = dataset[idx]

            image_tensor = sample["image"]
            gt_coord = sample["coord"]
            gt_shape = sample["shape"]

            image_batch = image_tensor.unsqueeze(0).to(device)

            outputs = model(image_batch)

            pred_coord = outputs["coords"][0].cpu()

            pred_shape_idx = (
                outputs["logits"][0]
                .argmax()
                .cpu()
                .item()
            )

            pred_shape = IDX_TO_SHAPE[pred_shape_idx]

            image = unnormalize(image_tensor.numpy())

            h, w = image.shape[:2]

            gt_x = gt_coord[0].item() * w
            gt_y = gt_coord[1].item() * h

            pred_x = pred_coord[0].item() * w
            pred_y = pred_coord[1].item() * h

            plt.figure(figsize=(8, 8))
            plt.imshow(image)

            # GT = GREEN
            plt.scatter(
                gt_x,
                gt_y,
                c="lime",
                s=120,
                label="Ground Truth",
            )

            # Prediction = RED
            plt.scatter(
                pred_x,
                pred_y,
                c="red",
                s=120,
                label="Prediction",
            )
            error_px = np.sqrt(
             (pred_x - gt_x) ** 2 +
            (pred_y - gt_y) ** 2
            )
            error_pct = (error_px / 512) * 100

            all_errors.append(error_px)
            print(
                f"Sample {idx}: Error = {error_px:.2f}px "
                f"({error_pct:.2f}% of crop)"
                )
            
            plt.title(
                f"Sample {idx}\n"
                f"Err={error_px:.1f}px ({error_pct:.1f}%) | "
                f"GT: {IDX_TO_SHAPE.get(gt_shape.item(), 'Unknown')} | "
                f"Pred: {pred_shape}"
            )

            plt.legend()

            save_path = output_dir / f"sample_{idx:03d}.png"

            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

            print(f"Saved: {save_path}")

    print("\n===== ERROR STATS =====")
    print(f"Mean   : {np.mean(all_errors):.2f}px")
    print(f"Median : {np.median(all_errors):.2f}px")
    print(f"Max    : {np.max(all_errors):.2f}px")
    print(f"Min    : {np.min(all_errors):.2f}px")

    print("\nDone.")


if __name__ == "__main__":
    main()