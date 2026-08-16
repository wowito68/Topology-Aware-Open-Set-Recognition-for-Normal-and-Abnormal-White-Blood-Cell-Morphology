"""Synthetic data generator for tests and smoke runs only."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from hemato_osr.data.taxonomy import DEFAULT_KNOWN_CLASSES

SYNTHETIC_NOTICE = "SYNTHETIC TEST DATA - NOT SCIENTIFIC DATA"
SYNTHETIC_UNKNOWN_CLASSES = ("myeloblast", "plasma_cell")


@dataclass(frozen=True)
class SyntheticConfig:
    """Settings for deterministic synthetic fixture generation."""

    output_dir: Path
    samples_per_known_class: int = 6
    samples_per_unknown_class: int = 4
    image_size: int = 64
    seed: int = 13
    overwrite: bool = True


def _draw_cell(
    rng: np.random.Generator,
    label: str,
    index: int,
    image_size: int,
) -> Image.Image:
    base = np.full((image_size, image_size, 3), 235, dtype=np.uint8)
    stain = {
        "basophil": (92, 70, 150),
        "eosinophil": (210, 120, 92),
        "lymphocyte": (82, 105, 176),
        "monocyte": (115, 145, 165),
        "neutrophil_segmented": (150, 110, 175),
        "myeloblast": (65, 140, 115),
        "plasma_cell": (185, 85, 125),
    }.get(label, (120, 120, 120))
    noise = rng.normal(0, 5, size=base.shape)
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")

    cx = image_size // 2 + int(rng.integers(-4, 5))
    cy = image_size // 2 + int(rng.integers(-4, 5))
    radius = int(image_size * 0.30 + rng.integers(-2, 3))
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=(245, 205, 220, 175),
        outline=(160, 120, 155, 180),
        width=2,
    )

    if label == "neutrophil_segmented":
        for offset in (-10, 0, 10):
            draw.ellipse(
                (cx + offset - 7, cy - 8, cx + offset + 7, cy + 8),
                fill=(*stain, 210),
            )
    elif label in {"eosinophil", "basophil"}:
        for _ in range(18 if label == "basophil" else 10):
            x = int(rng.integers(cx - radius + 5, cx + radius - 5))
            y = int(rng.integers(cy - radius + 5, cy + radius - 5))
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(*stain, 220))
    elif label == "monocyte":
        draw.arc((cx - 15, cy - 12, cx + 15, cy + 14), 35, 325, fill=(*stain, 230), width=8)
    elif label == "plasma_cell":
        draw.ellipse((cx - 15, cy - 15, cx + 10, cy + 10), fill=(*stain, 220))
        draw.ellipse((cx + 7, cy + 7, cx + 13, cy + 13), fill=(240, 230, 235, 220))
    else:
        draw.ellipse((cx - 15, cy - 15, cx + 15, cy + 15), fill=(*stain, 220))

    draw.text((2, image_size - 10), f"SYN {index}", fill=(20, 20, 20, 255))
    return image


def generate_synthetic_dataset(config: SyntheticConfig) -> Path:
    """Create a deterministic tiny dataset for CI and smoke tests."""

    root = config.output_dir
    if config.overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    notice_path = root / "README_SYNTHETIC.txt"
    notice_path.write_text(SYNTHETIC_NOTICE + "\n", encoding="utf-8")

    rng = np.random.default_rng(config.seed)
    for label in DEFAULT_KNOWN_CLASSES:
        class_dir = root / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(config.samples_per_known_class):
            image = _draw_cell(rng, label, idx, config.image_size)
            image.save(class_dir / f"{label}_{idx:03d}.png")

    for label in SYNTHETIC_UNKNOWN_CLASSES:
        class_dir = root / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(config.samples_per_unknown_class):
            image = _draw_cell(rng, label, idx, config.image_size)
            image.save(class_dir / f"{label}_{idx:03d}.png")
    return root
