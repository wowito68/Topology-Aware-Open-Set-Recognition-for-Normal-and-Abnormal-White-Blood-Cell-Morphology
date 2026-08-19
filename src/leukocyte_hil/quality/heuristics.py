"""Deterministic quality indicators for model-inference suitability."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from PIL import Image

from leukocyte_hil.cropping.boxes import BoundingBox, touches_image_border


class QualityFlag(StrEnum):
    """Quality flags used by the human-review triage layer."""

    BLUR = "BLUR"
    UNDEREXPOSED = "UNDEREXPOSED"
    OVEREXPOSED = "OVEREXPOSED"
    CLIPPED_AT_IMAGE_BORDER = "CLIPPED_AT_IMAGE_BORDER"
    TOO_SMALL = "TOO_SMALL"
    INVALID_CROP = "INVALID_CROP"


@dataclass(frozen=True)
class QualityConfig:
    """Thresholds for deterministic crop quality heuristics."""

    min_crop_width: int = 32
    min_crop_height: int = 32
    blur_variance_threshold: float = 12.0
    underexposed_mean_threshold: float = 20.0
    overexposed_mean_threshold: float = 235.0
    saturation_fraction_threshold: float = 0.20


@dataclass(frozen=True)
class QualityResult:
    """Quality score and flags for one crop."""

    quality_score: float
    flags: tuple[QualityFlag, ...]
    focus_score: float
    saturation_fraction: float
    mean_intensity: float

    @property
    def failed(self) -> bool:
        return bool(self.flags)


def assess_crop_quality(
    crop: Image.Image | None,
    *,
    crop_box: BoundingBox | None = None,
    image_size: tuple[int, int] | None = None,
    config: QualityConfig | None = None,
) -> QualityResult:
    """Assess whether a crop has low image quality for model inference."""

    cfg = config or QualityConfig()
    flags: list[QualityFlag] = []
    if crop is None or crop_box is None or not crop_box.is_valid:
        return QualityResult(
            quality_score=0.0,
            flags=(QualityFlag.INVALID_CROP,),
            focus_score=0.0,
            saturation_fraction=0.0,
            mean_intensity=0.0,
        )

    width, height = crop.size
    if width < cfg.min_crop_width or height < cfg.min_crop_height:
        flags.append(QualityFlag.TOO_SMALL)
    if image_size is not None and touches_image_border(crop_box, image_size):
        flags.append(QualityFlag.CLIPPED_AT_IMAGE_BORDER)

    array = np.asarray(crop.convert("RGB"), dtype=np.float32)
    gray = array.mean(axis=2)
    focus_score = _laplacian_variance(gray)
    mean_intensity = float(gray.mean())
    saturation_fraction = float(np.mean((array <= 2.0) | (array >= 253.0)))

    if focus_score < cfg.blur_variance_threshold:
        flags.append(QualityFlag.BLUR)
    if mean_intensity < cfg.underexposed_mean_threshold:
        flags.append(QualityFlag.UNDEREXPOSED)
    if mean_intensity > cfg.overexposed_mean_threshold:
        flags.append(QualityFlag.OVEREXPOSED)
    if saturation_fraction > cfg.saturation_fraction_threshold:
        if mean_intensity <= 127.5 and QualityFlag.UNDEREXPOSED not in flags:
            flags.append(QualityFlag.UNDEREXPOSED)
        if mean_intensity > 127.5 and QualityFlag.OVEREXPOSED not in flags:
            flags.append(QualityFlag.OVEREXPOSED)

    penalties = {
        QualityFlag.INVALID_CROP: 1.0,
        QualityFlag.TOO_SMALL: 0.35,
        QualityFlag.CLIPPED_AT_IMAGE_BORDER: 0.25,
        QualityFlag.BLUR: 0.25,
        QualityFlag.UNDEREXPOSED: 0.30,
        QualityFlag.OVEREXPOSED: 0.30,
    }
    penalty = min(1.0, sum(penalties[flag] for flag in set(flags)))
    return QualityResult(
        quality_score=1.0 - penalty,
        flags=tuple(flags),
        focus_score=focus_score,
        saturation_fraction=saturation_fraction,
        mean_intensity=mean_intensity,
    )


def _laplacian_variance(gray: np.ndarray) -> float:
    center = -4.0 * gray[1:-1, 1:-1]
    neighbors = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    laplacian = center + neighbors
    return float(np.var(laplacian)) if laplacian.size else 0.0
