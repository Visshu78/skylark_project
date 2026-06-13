from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.augmentations import get_train_transforms, get_valid_transforms
from src.dataset import GCPDataset
from src.losses import MultiTaskLoss
from src.model import GCPMultiTaskModel
from src.utils import load_config, require_cuda, seed_everything, setup_logging
from src.validate import validate_one_epoch


def build_loader(
    config: dict,
    mode: str,
) -> DataLoader:
    transforms = (
        get_train_transforms(config["image_size"])
        if mode == "train"
        else get_valid_transforms(config["image_size"])
    )
    dataset = GCPDataset(
        csv_path=config["train_csv"],
        fold=config["fold"],
        mode=mode,
        transforms=transforms,
        crop_size=config["crop_size"],
        crop_target_margin=config["crop_target_margin"],
    )
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=mode == "train",
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=mode == "train",
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    reg_losses: list[float] = []
    cls_losses: list[float] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        coords = batch["coord"].to(device, non_blocking=True)
        shapes = batch["shape"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss, parts = criterion(outputs["coords"], coords, outputs["logits"], shapes)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(parts["loss"])
        reg_losses.append(parts["regression_loss"])
        cls_losses.append(parts["classification_loss"])

    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "regression_loss": float(sum(reg_losses) / max(1, len(reg_losses))),
        "classification_loss": float(sum(cls_losses) / max(1, len(cls_losses))),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_pck10: float,
    config: dict,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_pck10": best_pck10,
            "config": config,
        },
        path,
    )


def plot_curves(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return
    epochs = [item["epoch"] for item in history]
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [item["train_loss"] for item in history], label="train loss")
    plt.plot(epochs, [item["val_loss"] for item in history], label="val loss")
    plt.plot(epochs, [item["val_pck10"] for item in history], label="val PCK@10")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def run_training(config: dict) -> None:
    output_dir = Path(config["output_dir"])
    setup_logging(output_dir)
    seed_everything(config["seed"])
    device = require_cuda()
    logging.info("Using CUDA device: %s", torch.cuda.get_device_name(0))

    train_loader = build_loader(config, "train")
    valid_loader = build_loader(config, "valid")

    model = GCPMultiTaskModel(
        pretrained=config["pretrained"],
        dropout=config["dropout"],
    ).to(device)
    criterion = MultiTaskLoss(config["classification_loss_weight"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    total_epochs = int(config["stage1_epochs"]) + int(config["stage2_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config["amp"]))

    best_pck10 = -1.0
    bad_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(total_epochs):
        if epoch == 0:
            logging.info("Stage 1: freezing EfficientNet backbone")
            model.freeze_backbone()
        if epoch == int(config["stage1_epochs"]):
            logging.info("Stage 2: unfreezing full model")
            model.unfreeze_backbone()

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            bool(config["amp"]),
        )
        val_metrics = validate_one_epoch(
            model,
            valid_loader,
            criterion,
            device,
            image_size=int(config["image_size"]),
        )
        # DEBUG: inspect predictions
        model.eval()
        debug_batch = next(iter(valid_loader))
        debug_images = debug_batch["image"].to(device)
        debug_coords = debug_batch["coord"]
        with torch.no_grad():
            debug_outputs = model(debug_images)
        debug_preds = debug_outputs["coords"].cpu()
        print("\n===== DEBUG PREDICTIONS =====")
        for i in range(min(10, len(debug_preds))):
            print(
                f"GT={debug_coords[i].tolist()} "
                f"PR={debug_preds[i].tolist()}"
            )
        print("============================\n")
        scheduler.step()

        row = {
            "epoch": float(epoch + 1),
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_pck10": val_metrics["pck10"],
            "val_mean_pixel_error": val_metrics["mean_pixel_error"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        logging.info(
            "Epoch %03d | "
            "train_loss %.4f | "
            "train_reg %.4f | "
            "train_cls %.4f | "
            "val_loss %.4f | "
            "shape_accuracy %.4f | "
            "macro_f1 %.4f | "
            "mean_coord_error %.4f | "
            "PCK@10 %.4f | "
            "PCK@25 %.4f | "
            "PCK@50 %.4f",
            epoch + 1,
            train_metrics["loss"],
            train_metrics["regression_loss"],
            train_metrics["classification_loss"],
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
            val_metrics["mean_pixel_error"],
            val_metrics["pck10"],
            val_metrics["pck25"],
            val_metrics["pck50"],
        )

        save_checkpoint(
            output_dir / "last_model.pth",
            model,
            optimizer,
            scheduler,
            epoch + 1,
            best_pck10,
            config,
        )

        if val_metrics["pck10"] > best_pck10:
            best_pck10 = val_metrics["pck10"]
            bad_epochs = 0
            save_checkpoint(
                output_dir / "best_model.pth",
                model,
                optimizer,
                scheduler,
                epoch + 1,
                best_pck10,
                config,
            )
            logging.info("Saved new best model with PCK@10 %.4f", best_pck10)
        else:
            bad_epochs += 1

        plot_curves(history, output_dir / "training_curves.png")
        if bad_epochs >= int(config["early_stopping_patience"]):
            logging.info("Early stopping after %d stale epochs", bad_epochs)
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    run_training(load_config(args.config))


if __name__ == "__main__":
    main()
