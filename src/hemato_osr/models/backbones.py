"""Backbone factory exposing logits and embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    """Backbone configuration."""

    backbone: str = "resnet18"
    num_classes: int = 5
    pretrained: bool = True
    embedding_dim: int | None = None


class TinyCNN(nn.Module):
    """Small CNN used only for smoke tests and CI fixtures."""

    def __init__(self, num_classes: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embedding = nn.Linear(64, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.features(x).flatten(1)
        embedding = self.embedding(features)
        logits = self.classifier(torch.relu(embedding))
        return {"logits": logits, "embedding": embedding}


class TimmClassifier(nn.Module):
    """Wrap a timm feature extractor with a classifier head."""

    def __init__(self, backbone: str, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        import timm

        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        embedding_dim = int(cast(Any, self.encoder).num_features)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        return {"logits": logits, "embedding": embedding}


class TorchvisionResNet18(nn.Module):
    """Torchvision ResNet18 fallback."""

    def __init__(self, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights: Any = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        embedding_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
        self.encoder = model
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        return {"logits": logits, "embedding": embedding}


def create_classifier(config: ModelConfig) -> nn.Module:
    """Create an image classifier with ``logits`` and ``embedding`` outputs."""

    name = config.backbone.lower()
    if name == "tiny_cnn":
        return TinyCNN(
            num_classes=config.num_classes,
            embedding_dim=config.embedding_dim or 64,
        )
    try:
        return TimmClassifier(
            backbone=config.backbone,
            num_classes=config.num_classes,
            pretrained=config.pretrained,
        )
    except Exception:
        if name == "resnet18":
            return TorchvisionResNet18(
                num_classes=config.num_classes,
                pretrained=config.pretrained,
            )
        raise
