from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch

from src.inference import predict_tiled
from src.model import GCPMultiTaskModel


MODEL_PATH = "outputs/best_model.pth"
TEST_ROOT = "test_dataset"
OUTPUT_DIR = "test_predictions"
NUM_IMAGES = 20


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = GCPMultiTaskModel(
        pretrained=False,
        dropout=0.2,
    )

    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    image_paths = list(Path(TEST_ROOT).rglob("*.JPG"))
    image_paths += list(Path(TEST_ROOT).rglob("*.jpg"))

    for idx, image_path in enumerate(image_paths[:NUM_IMAGES]):

        image_bgr = cv2.imread(str(image_path))
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        pred_x, pred_y, pred_shape, score = predict_tiled(
            model=model,
            image=image,
            image_size=512,
            tile_size=512,
            stride=384,
            device=device,
        )

        plt.figure(figsize=(10, 8))
        plt.imshow(image)

        plt.scatter(
            pred_x,
            pred_y,
            c="red",
            s=15,
            label="Prediction",
        )

        plt.title(
            f"{image_path.name}\n"
            f"Shape={pred_shape} | Score={score:.3f}"
        )

        plt.legend()

        save_path = output_dir / f"test_{idx:03d}.png"

        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        print(
            f"[{idx+1}] {image_path.name} "
            f"Shape={pred_shape} "
            f"Score={score:.3f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()