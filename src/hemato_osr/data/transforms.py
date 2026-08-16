"""Image preprocessing and conservative hematology augmentations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from PIL import Image
from torchvision import transforms


@dataclass(frozen=True)
class TransformConfig:
    """Configurable image transform parameters."""

    image_size: int = 224
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.0
    rotation_degrees: float = 10.0
    brightness: float = 0.05
    contrast: float = 0.05


ImageTransform = Callable[[Image.Image], torch.Tensor]


def build_transforms(split: str, config: TransformConfig) -> ImageTransform:
    """Build transforms for train/validation/test.

    Validation and test transforms are deterministic. Normalization uses
    ImageNet statistics by default because the initial baseline uses ImageNet
    pretrained backbones.
    """

    common: list[Callable[[Image.Image], object]] = [
        transforms.Resize((config.image_size, config.image_size), antialias=True),
    ]
    if split == "train":
        common.extend(
            [
                transforms.RandomHorizontalFlip(p=config.horizontal_flip_p),
                transforms.RandomVerticalFlip(p=config.vertical_flip_p),
                transforms.RandomRotation(degrees=config.rotation_degrees),
                transforms.ColorJitter(brightness=config.brightness, contrast=config.contrast),
            ]
        )
    common.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std),
        ]
    )
    return transforms.Compose(common)  # type: ignore[return-value]
