"""Representation-learning model wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.nn import functional as F

from hemato_osr.models.backbones import ModelConfig, create_classifier

AngularLoss = Literal["arcface", "cosface"]


def _encoder_from_classifier(
    backbone: str,
    num_classes: int,
    pretrained: bool,
) -> tuple[nn.Module, int]:
    base = create_classifier(
        ModelConfig(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
        )
    )
    if not hasattr(base, "encoder"):
        msg = f"Representation training requires an encoder-backed model, got {type(base)}"
        raise ValueError(msg)
    classifier = cast(Any, base).classifier
    embedding_dim = int(classifier.in_features)
    return cast(nn.Module, cast(Any, base).encoder), embedding_dim


class ProjectionHead(nn.Module):
    """512 -> 256 -> 128 projection head with output normalization."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, output_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(embedding), dim=1)


class SupConClassifier(nn.Module):
    """ResNet classifier with a supervised-contrastive projection head."""

    def __init__(
        self,
        *,
        backbone: str,
        num_classes: int,
        pretrained: bool,
        projection_hidden_dim: int = 256,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder, embedding_dim = _encoder_from_classifier(backbone, num_classes, pretrained)
        self.embedding_dim = embedding_dim
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.projection_head = ProjectionHead(embedding_dim, projection_hidden_dim, projection_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        projection = self.projection_head(embedding)
        return {"logits": logits, "embedding": embedding, "projection": projection}


class AngularMarginHead(nn.Module):
    """Normalized classifier supporting ArcFace and CosFace training logits."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        *,
        scale: float = 30.0,
        margin: float = 0.30,
        loss_type: AngularLoss = "arcface",
    ) -> None:
        super().__init__()
        if scale <= 0 or margin < 0:
            msg = "scale must be positive and margin non-negative"
            raise ValueError(msg)
        if loss_type not in {"arcface", "cosface"}:
            msg = f"Unsupported angular loss: {loss_type}"
            raise ValueError(msg)
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = float(scale)
        self.margin = float(margin)
        self.loss_type = loss_type

    def normalized_weight(self) -> torch.Tensor:
        return F.normalize(self.weight, dim=1)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        apply_margin: bool = True,
    ) -> torch.Tensor:
        normalized_embeddings = F.normalize(embeddings, dim=1)
        cosine = F.linear(normalized_embeddings, self.normalized_weight()).clamp(-1.0, 1.0)
        if labels is None or not apply_margin:
            return cosine * self.scale
        one_hot = F.one_hot(labels, num_classes=cosine.shape[1]).to(dtype=cosine.dtype)
        if self.loss_type == "arcface":
            theta = torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
            target = torch.cos(theta + self.margin)
        else:
            target = cosine - self.margin
        logits = cosine * (1.0 - one_hot) + target * one_hot
        return logits * self.scale


class AngularMarginModel(nn.Module):
    """ResNet encoder with ArcFace/CosFace-style normalized classifier."""

    def __init__(
        self,
        *,
        backbone: str,
        num_classes: int,
        pretrained: bool,
        scale: float = 30.0,
        margin: float = 0.30,
        loss_type: AngularLoss = "arcface",
    ) -> None:
        super().__init__()
        self.encoder, embedding_dim = _encoder_from_classifier(backbone, num_classes, pretrained)
        self.embedding_dim = embedding_dim
        self.classifier = AngularMarginHead(
            embedding_dim,
            num_classes,
            scale=scale,
            margin=margin,
            loss_type=loss_type,
        )

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        apply_margin: bool = True,
    ) -> dict[str, torch.Tensor]:
        embedding = self.encoder(x)
        logits = self.classifier(embedding, labels, apply_margin=apply_margin)
        return {"logits": logits, "embedding": F.normalize(embedding, dim=1)}


@dataclass(frozen=True)
class RepresentationModelSpec:
    """Serializable model spec saved into checkpoints."""

    representation: str
    backbone: str
    num_classes: int
    pretrained: bool
    scale: float = 30.0
    margin: float = 0.30
    projection_hidden_dim: int = 256
    projection_dim: int = 128


def create_representation_model(spec: RepresentationModelSpec) -> nn.Module:
    """Instantiate a Delivery 5 representation model from a checkpoint spec."""

    if spec.representation == "supcon":
        return SupConClassifier(
            backbone=spec.backbone,
            num_classes=spec.num_classes,
            pretrained=spec.pretrained,
            projection_hidden_dim=spec.projection_hidden_dim,
            projection_dim=spec.projection_dim,
        )
    if spec.representation in {"arcface", "cosface"}:
        return AngularMarginModel(
            backbone=spec.backbone,
            num_classes=spec.num_classes,
            pretrained=spec.pretrained,
            scale=spec.scale,
            margin=spec.margin,
            loss_type=cast(AngularLoss, spec.representation),
        )
    msg = f"Unknown representation: {spec.representation}"
    raise ValueError(msg)
