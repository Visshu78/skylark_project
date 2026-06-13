from __future__ import annotations

import torch
from torch import nn


class GCPMultiTaskModel(nn.Module):
    """EfficientNet-B3 multitask model for marker coordinates and shape."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        try:
            from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for EfficientNet-B3. Install dependencies "
                "with `pip install -r requirements.txt`."
            ) from exc

        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b3(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        hidden = 512
        self.regression_head = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
            nn.Sigmoid(),
        )
        self.shape_head = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        return {
            "coords": self.regression_head(features),
            "logits": self.shape_head(features),
        }

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True
