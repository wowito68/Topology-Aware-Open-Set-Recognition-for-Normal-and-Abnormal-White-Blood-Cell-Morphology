"""Frozen ResNet activation-energy maps for feature-map topology."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from hemato_osr.data.dataset import ManifestImageDataset, collate_samples
from hemato_osr.data.transforms import TransformConfig, build_transforms
from hemato_osr.models.backbones import ModelConfig, create_classifier
from hemato_osr.training.checkpoint import load_checkpoint


@dataclass(frozen=True)
class ActivationConfig:
    """Configuration for activation-map extraction."""

    checkpoint_path: Path
    layer: str = "layer2"
    batch_size: int = 64
    image_size: int = 224
    num_workers: int = 0
    device: str = "auto"
    robust_lower_percentile: float = 1.0
    robust_upper_percentile: float = 99.0


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_frozen_classifier(checkpoint_path: Path, device: str = "auto") -> nn.Module:
    """Load the frozen classifier from a Delivery 1/2 checkpoint."""

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    label_to_index = {str(k): int(v) for k, v in dict(checkpoint["label_to_index"]).items()}
    cfg = dict(checkpoint.get("config", {}))
    model = create_classifier(
        ModelConfig(
            backbone=str(cfg.get("backbone", "resnet18")),
            num_classes=len(label_to_index),
            pretrained=False,
        )
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(_device(device))
    model.eval()
    return model


def resolve_layer(model: nn.Module, layer: str) -> nn.Module:
    """Resolve a ResNet layer name without assuming a specific backend."""

    candidates = [
        f"encoder.{layer}",
        layer,
    ]
    modules = dict(model.named_modules())
    for candidate in candidates:
        if candidate in modules:
            return modules[candidate]
    suffix_matches = [module for name, module in modules.items() if name.endswith(f".{layer}")]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    msg = f"Could not resolve activation layer {layer!r}; available examples: {list(modules)[:12]}"
    raise ValueError(msg)


def activation_energy_map(activation: torch.Tensor) -> np.ndarray:
    """Convert ``N x C x H x W`` activations to RMS energy maps."""

    if activation.ndim != 4:
        msg = "Activation tensor must have shape N x C x H x W"
        raise ValueError(msg)
    energy = torch.sqrt(torch.mean(torch.square(activation.float()), dim=1))
    return energy.detach().cpu().numpy()


def normalize_activation_map(
    values: np.ndarray,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    """Robustly normalize one scalar activation map to [0, 1]."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        msg = "Activation energy map must be 2D"
        raise ValueError(msg)
    if not np.isfinite(arr).all():
        msg = "Activation energy map contains NaN/Inf"
        raise ValueError(msg)
    lower = float(np.percentile(arr, lower_percentile))
    upper = float(np.percentile(arr, upper_percentile))
    if upper - lower <= 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - lower) / (upper - lower), 0.0, 1.0)


class ActivationCollector:
    """Context manager collecting one layer's forward activation."""

    def __init__(self, model: nn.Module, layer: str) -> None:
        self.model = model
        self.layer = resolve_layer(model, layer)
        self.output: torch.Tensor | None = None
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def __enter__(self) -> ActivationCollector:
        def hook(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            self.output = output.detach()

        self._handle = self.layer.register_forward_hook(hook)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._handle is not None:
            self._handle.remove()


def activation_maps_for_manifest(
    manifest: pd.DataFrame,
    config: ActivationConfig,
) -> tuple[list[str], list[np.ndarray], tuple[int, int, int]]:
    """Extract normalized activation-energy maps for a manifest frame."""

    checkpoint = load_checkpoint(config.checkpoint_path, map_location="cpu")
    label_to_index = {str(k): int(v) for k, v in dict(checkpoint["label_to_index"]).items()}
    model = load_frozen_classifier(config.checkpoint_path, device=config.device)
    device = _device(config.device)
    frame = manifest.sort_values("sample_id").reset_index(drop=True)
    dataset = ManifestImageDataset(
        frame,
        label_to_index=label_to_index,
        transform=build_transforms("test", TransformConfig(image_size=config.image_size)),
        include_unknown=True,
    )
    loader: DataLoader[Any] = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_samples,
    )
    sample_ids: list[str] = []
    maps: list[np.ndarray] = []
    activation_shape: tuple[int, int, int] | None = None
    with ActivationCollector(model, config.layer) as collector:
        with torch.inference_mode():
            for batch in loader:
                images = batch["image"].to(device)  # type: ignore[union-attr]
                _ = model(images)
                if collector.output is None:
                    msg = f"No activation captured for layer {config.layer}"
                    raise RuntimeError(msg)
                output = collector.output
                channels, height, width = output.shape[1:]
                activation_shape = (int(channels), int(height), int(width))
                energy = activation_energy_map(output)
                for item in energy:
                    maps.append(
                        normalize_activation_map(
                            item,
                            lower_percentile=config.robust_lower_percentile,
                            upper_percentile=config.robust_upper_percentile,
                        )
                    )
                sample_ids.extend(str(item) for item in batch["sample_id"])  # type: ignore[union-attr]
    if activation_shape is None:
        msg = "Activation extraction received an empty manifest"
        raise ValueError(msg)
    return sample_ids, maps, activation_shape


def activation_metadata(
    config: ActivationConfig,
    activation_shape: tuple[int, int, int],
) -> dict[str, object]:
    """Serializable activation-map metadata."""

    return {
        "checkpoint_path": str(config.checkpoint_path),
        "layer": config.layer,
        "activation_shape": list(activation_shape),
        "aggregation": "sqrt(mean_c activation^2)",
        "normalization": {
            "type": "per-sample robust percentile to [0,1]",
            "lower_percentile": config.robust_lower_percentile,
            "upper_percentile": config.robust_upper_percentile,
        },
    }


def checkpoint_label_mapping(checkpoint_path: Path) -> dict[str, int]:
    """Read the checkpoint class mapping."""

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    return {str(k): int(v) for k, v in dict(checkpoint["label_to_index"]).items()}


def checkpoint_config_json(checkpoint_path: Path) -> str:
    """Return a stable JSON view of checkpoint config for metadata."""

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    payload = {
        "config": checkpoint.get("config", {}),
        "label_to_index": checkpoint.get("label_to_index", {}),
    }
    return json.dumps(payload, sort_keys=True, default=str)
