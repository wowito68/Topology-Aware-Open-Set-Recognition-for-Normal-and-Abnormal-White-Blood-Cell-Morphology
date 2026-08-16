"""Deep embedding plus precomputed topological feature fusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FusionConfig:
    """Fusion MLP dimensions."""

    deep_dim: int
    tda_dim: int
    num_classes: int
    hidden_dim: int = 128
    dropout: float = 0.1


class DeepTDAFusionMLP(nn.Module):
    """Concatenate deep embeddings and cached TDA features before classification."""

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.deep_dim + config.tda_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(
        self,
        deep_embedding: torch.Tensor,
        tda_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fused = torch.cat([deep_embedding, tda_features], dim=1)
        logits = self.network(fused)
        return {"logits": logits, "embedding": fused}
