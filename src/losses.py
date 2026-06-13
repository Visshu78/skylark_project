from __future__ import annotations

import torch
from torch import nn

from src.dataset import IGNORE_LABEL


class MultiTaskLoss(nn.Module):
    """SmoothL1 coordinate loss plus weighted CrossEntropy shape loss."""

    def __init__(self, classification_weight: float = 0.5) -> None:
        super().__init__()
        self.classification_weight = classification_weight
        self.regression_loss = nn.SmoothL1Loss()
        self.classification_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

    def forward(
        self,
        pred_coords: torch.Tensor,
        true_coords: torch.Tensor,
        pred_logits: torch.Tensor,
        true_shapes: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        reg_loss = self.regression_loss(pred_coords, true_coords)
        valid_shapes = true_shapes != IGNORE_LABEL
        if valid_shapes.any():
            cls_loss = self.classification_loss(pred_logits, true_shapes)
        else:
            cls_loss = pred_logits.sum() * 0.0

        total = reg_loss + self.classification_weight * cls_loss
        return total, {
            "loss": float(total.detach().cpu()),
            "regression_loss": float(reg_loss.detach().cpu()),
            "classification_loss": float(cls_loss.detach().cpu()),
        }
