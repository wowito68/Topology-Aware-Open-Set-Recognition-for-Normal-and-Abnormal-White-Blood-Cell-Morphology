"""Deterministic candidate morphology masks for Delivery 3 TDA."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

MaskStatus = Literal["valid", "warning", "invalid"]


@dataclass(frozen=True)
class MorphologyConfig:
    """Configuration for non-learned candidate-region generation."""

    image_size: int = 96
    background_distance_min: float = 0.10
    saturation_min: float = 0.08
    value_max: float = 0.98
    min_cell_fraction: float = 0.03
    max_cell_fraction: float = 0.985
    min_nucleus_fraction: float = 0.005
    max_nucleus_to_cell_fraction: float = 0.85
    min_cytoplasm_fraction: float = 0.01
    max_components_warning: int = 8
    min_component_area: int = 12
    min_nucleus_darkness: float = 0.20
    nucleus_dark_quantile: float = 0.85
    nucleus_saturation_min: float = 0.08
    nucleus_blue_margin_min: float = -0.05
    cleanup_iterations: int = 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MorphologyResult:
    """Candidate masks and QC metadata for one cell crop."""

    whole_cell: np.ndarray
    nucleus: np.ndarray
    cytoplasm: np.ndarray
    status: MaskStatus
    warnings: tuple[str, ...]
    foreground_fraction: float
    nucleus_fraction: float
    cytoplasm_fraction: float
    nucleus_to_cell_fraction: float
    number_of_components: int


def load_rgb(path: Path, image_size: int) -> np.ndarray:
    """Load an RGB image resized for morphology/TDA processing."""

    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((image_size, image_size))
    return np.asarray(rgb, dtype=np.float32) / 255.0


def _saturation_and_value(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    saturation = np.divide(maxc - minc, np.maximum(maxc, 1e-6))
    return saturation, maxc


def _cleanup(mask: np.ndarray, config: MorphologyConfig) -> np.ndarray:
    cleaned = np.asarray(mask, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    if config.cleanup_iterations > 0:
        cleaned = ndi.binary_opening(
            cleaned,
            structure=structure,
            iterations=config.cleanup_iterations,
        )
        cleaned = ndi.binary_closing(
            cleaned,
            structure=structure,
            iterations=config.cleanup_iterations,
        )
    cleaned = ndi.binary_fill_holes(cleaned)
    labels, n_labels = ndi.label(cleaned)
    if n_labels == 0:
        return np.zeros_like(cleaned, dtype=bool)
    keep = np.zeros_like(cleaned, dtype=bool)
    for idx in range(1, n_labels + 1):
        component = labels == idx
        if int(component.sum()) >= config.min_component_area:
            keep |= component
    return keep


def _central_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels, n_labels = ndi.label(mask)
    if n_labels == 0:
        return np.zeros_like(mask, dtype=bool), 0
    h, w = mask.shape
    yy, xx = np.mgrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    sigma = max(h, w) / 4.0
    central_weight = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2)))
    best_label = 1
    best_score = -1.0
    for idx in range(1, n_labels + 1):
        component = labels == idx
        area = float(component.mean())
        score = float(central_weight[component].mean()) + 0.15 * area
        if score > best_score:
            best_label = idx
            best_score = score
    return labels == best_label, int(n_labels)


def whole_cell_candidate(rgb: np.ndarray, config: MorphologyConfig) -> tuple[np.ndarray, int]:
    """Estimate the central whole-cell foreground candidate."""

    border = np.concatenate(
        [
            rgb[0, :, :],
            rgb[-1, :, :],
            rgb[:, 0, :],
            rgb[:, -1, :],
        ],
        axis=0,
    )
    background = np.median(border, axis=0)
    color_distance = np.linalg.norm(rgb - background[None, None, :], axis=2)
    saturation, value = _saturation_and_value(rgb)
    mask = (
        (color_distance >= config.background_distance_min) | (saturation >= config.saturation_min)
    ) & (value <= config.value_max)
    cleaned = _cleanup(mask, config)
    return _central_component(cleaned)


def nucleus_candidate(
    rgb: np.ndarray,
    whole_cell: np.ndarray,
    config: MorphologyConfig,
) -> np.ndarray:
    """Estimate darker blue/purple chromatin-like nucleus candidate pixels."""

    if not np.any(whole_cell):
        return np.zeros(whole_cell.shape, dtype=bool)
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    darkness = 1.0 - luminance
    saturation, _ = _saturation_and_value(rgb)
    cell_darkness = darkness[whole_cell]
    darkness_threshold = max(
        config.min_nucleus_darkness,
        float(np.quantile(cell_darkness, config.nucleus_dark_quantile)),
    )
    blue_margin = blue - 0.5 * (red + green)
    mask = (
        whole_cell
        & (darkness >= darkness_threshold)
        & (saturation >= config.nucleus_saturation_min)
        & (blue_margin >= config.nucleus_blue_margin_min)
    )
    cleaned = _cleanup(mask, config)
    labels, n_labels = ndi.label(cleaned)
    if n_labels <= 1:
        return cleaned.astype(bool)
    selected = np.zeros_like(cleaned, dtype=bool)
    for idx in range(1, n_labels + 1):
        component = labels == idx
        if int(component.sum()) >= config.min_component_area:
            selected |= component
    return selected


def segment_morphology(path: Path, config: MorphologyConfig) -> MorphologyResult:
    """Generate candidate whole-cell, nucleus, and cytoplasm masks."""

    rgb = load_rgb(path, config.image_size)
    whole_cell, n_components = whole_cell_candidate(rgb, config)
    nucleus = nucleus_candidate(rgb, whole_cell, config)
    cytoplasm = whole_cell & ~nucleus
    cytoplasm = (_cleanup(cytoplasm, config) & whole_cell) & ~nucleus

    foreground_fraction = float(whole_cell.mean())
    nucleus_fraction = float(nucleus.mean())
    cytoplasm_fraction = float(cytoplasm.mean())
    nucleus_to_cell = float(nucleus.sum() / max(1, whole_cell.sum()))

    warnings: list[str] = []
    status: MaskStatus = "valid"
    if foreground_fraction < config.min_cell_fraction:
        warnings.append("empty_or_tiny_cell")
        status = "invalid"
    if foreground_fraction > config.max_cell_fraction:
        warnings.append("cell_covers_nearly_entire_image")
        status = "invalid"
    if nucleus_fraction < config.min_nucleus_fraction:
        warnings.append("empty_or_tiny_nucleus_candidate")
        status = "invalid"
    if nucleus_to_cell > config.max_nucleus_to_cell_fraction:
        warnings.append("nucleus_candidate_larger_than_expected_cell_fraction")
        status = "invalid"
    if cytoplasm_fraction < config.min_cytoplasm_fraction:
        warnings.append("cytoplasm_candidate_tiny_or_impossible")
        if status == "valid":
            status = "warning"
    if n_components > config.max_components_warning:
        warnings.append("extreme_component_fragmentation")
        if status == "valid":
            status = "warning"

    return MorphologyResult(
        whole_cell=whole_cell.astype(bool),
        nucleus=nucleus.astype(bool),
        cytoplasm=cytoplasm.astype(bool),
        status=status,
        warnings=tuple(warnings),
        foreground_fraction=foreground_fraction,
        nucleus_fraction=nucleus_fraction,
        cytoplasm_fraction=cytoplasm_fraction,
        nucleus_to_cell_fraction=nucleus_to_cell,
        number_of_components=n_components,
    )


def mask_distance_map(mask: np.ndarray) -> np.ndarray:
    """Return normalized inside-mask Euclidean distance-to-background map."""

    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return np.zeros(mask_bool.shape, dtype=np.float64)
    distances = ndi.distance_transform_edt(mask_bool).astype(np.float64)
    max_value = float(distances.max())
    if max_value <= 1e-12:
        return np.zeros(mask_bool.shape, dtype=np.float64)
    return distances / max_value


def morphology_metadata_row(
    sample_id: str,
    label: str,
    split: str,
    result: MorphologyResult,
) -> dict[str, object]:
    """Create a flat QC metadata row."""

    return {
        "sample_id": sample_id,
        "label": label,
        "split": split,
        "segmentation_status": result.status,
        "warnings": ";".join(result.warnings),
        "foreground_fraction": result.foreground_fraction,
        "nucleus_fraction": result.nucleus_fraction,
        "cytoplasm_fraction": result.cytoplasm_fraction,
        "nucleus_to_cell_fraction": result.nucleus_to_cell_fraction,
        "number_of_components": result.number_of_components,
    }


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    """Create a non-destructive colored mask overlay for QC panels."""

    base = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    overlay = base.copy()
    mask_bool = np.asarray(mask, dtype=bool)
    color_arr = np.asarray(color, dtype=np.uint8)
    overlay[mask_bool] = (0.55 * overlay[mask_bool] + 0.45 * color_arr).astype(np.uint8)
    return Image.fromarray(overlay)


def write_morphology_qc_sheet(
    frame: pd.DataFrame,
    output_path: Path,
    config: MorphologyConfig,
    *,
    max_rows: int = 100,
) -> pd.DataFrame:
    """Write QC contact sheet and metadata for candidate masks."""

    rows = []
    panel_w = config.image_size
    label_h = 34
    cols = 4
    selected = frame.head(max_rows).reset_index(drop=True)
    sheet = Image.new(
        "RGB",
        (cols * panel_w, len(selected) * (config.image_size + label_h)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for row_idx, row in enumerate(selected.itertuples(index=False)):
        path = Path(str(row.path))
        rgb = load_rgb(path, config.image_size)
        result = segment_morphology(path, config)
        rows.append(
            morphology_metadata_row(
                str(row.sample_id),
                str(row.canonical_label),
                str(row.split),
                result,
            )
        )
        panels = [
            Image.fromarray((rgb * 255).astype(np.uint8)),
            overlay_mask(rgb, result.whole_cell, (0, 180, 0)),
            overlay_mask(rgb, result.nucleus, (80, 80, 220)),
            overlay_mask(rgb, result.cytoplasm, (220, 140, 0)),
        ]
        y = row_idx * (config.image_size + label_h)
        for col_idx, panel in enumerate(panels):
            sheet.paste(panel, (col_idx * panel_w, y))
        text = (
            f"{row.sample_id} {row.canonical_label} {result.status} "
            f"cell={result.foreground_fraction:.2f} nuc={result.nucleus_fraction:.2f}"
        )
        draw.text((4, y + config.image_size + 2), text[:120], fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    metadata = pd.DataFrame(rows)
    metadata.to_csv(output_path.with_suffix(".csv"), index=False)
    return metadata
