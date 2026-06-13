from __future__ import annotations

from collections.abc import Iterable

import torch

from src.losses import MultiTaskLoss
from src.metrics import MetricAccumulator


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    criterion: MultiTaskLoss,
    device: torch.device,
    image_size: int,
) -> dict[str, float]:
    model.eval()
    metrics = MetricAccumulator()
    losses: list[float] = []
    reg_losses: list[float] = []
    cls_losses: list[float] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        coords = batch["coord"].to(device, non_blocking=True)
        shapes = batch["shape"].to(device, non_blocking=True)

        outputs = model(images)
        loss, loss_parts = criterion(
            outputs["coords"], coords, outputs["logits"], shapes
        )
        losses.append(loss_parts["loss"])
        reg_losses.append(loss_parts["regression_loss"])
        cls_losses.append(loss_parts["classification_loss"])
        metrics.update(outputs["coords"], coords, outputs["logits"], shapes)

    output = metrics.compute(image_size=image_size)
    output["loss"] = float(sum(losses) / max(1, len(losses)))
    output["regression_loss"] = float(sum(reg_losses) / max(1, len(reg_losses)))
    output["classification_loss"] = float(sum(cls_losses) / max(1, len(cls_losses)))
    return output
