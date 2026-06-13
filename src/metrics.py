from __future__ import annotations
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from src.dataset import IGNORE_LABEL


def localization_metrics(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    image_size: int,
) -> dict[str, float]:

    print("\n===== DEBUG =====")
    print("pred first 5:")
    print(pred_coords[:5])

    print("\ngt first 5:")
    print(true_coords[:5])

    pred_px = pred_coords.detach().cpu().numpy() * image_size
    true_px = true_coords.detach().cpu().numpy() * image_size

    errors = np.linalg.norm(pred_px - true_px, axis=1)

    print("\nerror first 5:")
    print(errors[:5])

    return {
        "mean_pixel_error": float(np.mean(errors)),
        "pck10": float(np.mean(errors <= 10.0)),
        "pck25": float(np.mean(errors <= 25.0)),
        "pck50": float(np.mean(errors <= 50.0)),
    }


def classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    valid = targets != IGNORE_LABEL
    if not valid.any():
        return {"accuracy": 0.0, "macro_f1": 0.0}

    y_true = targets[valid].detach().cpu().numpy()
    y_pred = logits[valid].argmax(dim=1).detach().cpu().numpy()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


class MetricAccumulator:
    """Accumulates predictions over a validation epoch."""

    def __init__(self) -> None:
        self.coords: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        self.logits: list[torch.Tensor] = []
        self.shapes: list[torch.Tensor] = []

    def update(
        self,
        pred_coords: torch.Tensor,
        true_coords: torch.Tensor,
        logits: torch.Tensor,
        shapes: torch.Tensor,
    ) -> None:
        self.coords.append(pred_coords.detach().cpu())
        self.targets.append(true_coords.detach().cpu())
        self.logits.append(logits.detach().cpu())
        self.shapes.append(shapes.detach().cpu())

    def compute(self, image_size: int) -> dict[str, float]:
        pred_coords = torch.cat(self.coords)
        true_coords = torch.cat(self.targets)
        logits = torch.cat(self.logits)
        shapes = torch.cat(self.shapes)
        output = localization_metrics(pred_coords, true_coords, image_size)
        output.update(classification_metrics(logits, shapes))
        return output
