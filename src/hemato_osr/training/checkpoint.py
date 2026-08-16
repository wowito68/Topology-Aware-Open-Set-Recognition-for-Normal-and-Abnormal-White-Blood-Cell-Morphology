"""Checkpoint IO helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    label_to_index: dict[str, int],
    config: dict[str, Any],
    metrics: dict[str, float],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int | None = None,
    seed: int | None = None,
) -> None:
    """Save model weights and reproducibility metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_to_index": label_to_index,
            "config": config,
            "metrics": metrics,
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "seed": seed,
        },
        path,
    )


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a training checkpoint."""

    return torch.load(path, map_location=map_location, weights_only=False)
