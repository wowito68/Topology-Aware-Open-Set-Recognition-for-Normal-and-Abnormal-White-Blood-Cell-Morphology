"""Hydra/OmegaConf helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(path: Path | None) -> dict[str, Any]:
    """Load a YAML config as plain containers."""

    if path is None:
        return {}
    cfg = OmegaConf.load(path)
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        msg = f"Expected mapping config at {path}"
        raise ValueError(msg)
    return {str(key): value for key, value in container.items()}


def save_config(path: Path, config: dict[str, Any] | DictConfig) -> None:
    """Save an exact experiment config."""

    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(config), f=path)
