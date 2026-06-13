from __future__ import annotations

import argparse
import logging

import torch
from torch.utils.data import DataLoader

from src.augmentations import get_valid_transforms
from src.dataset import GCPDataset
from src.losses import MultiTaskLoss
from src.model import GCPMultiTaskModel
from src.utils import load_config
from src.validate import validate_one_epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/best_model.pth")
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = GCPDataset(
        csv_path=config["train_csv"],
        fold=config["fold"],
        mode="valid",
        transforms=get_valid_transforms(config["image_size"]),
        crop_size=config["crop_size"],
        crop_target_margin=config["crop_target_margin"],
    )
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    checkpoint = torch.load(args.model, map_location=device)
    model = GCPMultiTaskModel(pretrained=False, dropout=config["dropout"]).to(device)
    model.load_state_dict(checkpoint["model"])
    metrics = validate_one_epoch(
        model,
        loader,
        MultiTaskLoss(config["classification_loss_weight"]),
        device,
        config["image_size"],
    )
    logging.info("Validation metrics: %s", metrics)


if __name__ == "__main__":
    main()
