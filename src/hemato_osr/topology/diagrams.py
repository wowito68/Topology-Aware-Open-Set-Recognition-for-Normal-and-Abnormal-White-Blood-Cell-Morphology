"""Persistent homology over 2D cell images."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TDAConfig:
    """Configuration for cubical-complex persistent homology."""

    filtration: str = "sublevel"
    homology_dimensions: tuple[int, ...] = (0, 1)
    image_size: int = 64
    infinite_policy: str = "keep"

    def to_dict(self) -> dict[str, object]:
        """Return a serializable representation."""

        return asdict(self)


PersistenceDiagram = dict[int, np.ndarray]


def _load_grayscale(path: Path, image_size: int) -> np.ndarray:
    """Load the TDA baseline grayscale image in [0, 1]."""

    with Image.open(path) as image:
        gray = image.convert("L").resize((image_size, image_size))
        arr = np.asarray(gray, dtype=np.float64) / 255.0
    return arr


def _fallback_diagram(values: np.ndarray, dimensions: tuple[int, ...]) -> PersistenceDiagram:
    """Deterministic fallback when GUDHI is not installed.

    This is not a scientific substitute for GUDHI. It exists so cache/version
    logic can still be tested in minimal environments.
    """

    q25, q50, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    diagrams: PersistenceDiagram = {}
    if 0 in dimensions:
        diagrams[0] = np.asarray([[float(values.min()), float(q50)], [float(q25), float(q75)]])
    if 1 in dimensions:
        diagrams[1] = np.asarray([[float(q50), float(values.max())]])
    for dim in dimensions:
        diagrams.setdefault(dim, np.empty((0, 2), dtype=float))
    return diagrams


def compute_diagram(path: Path, config: TDAConfig) -> PersistenceDiagram:
    """Compute H0/H1 persistence diagrams using cubical complexes."""

    values = _load_grayscale(path, config.image_size)
    return compute_diagram_from_array(values, config)


def compute_diagram_from_array(values: np.ndarray, config: TDAConfig) -> PersistenceDiagram:
    """Compute H0/H1 persistence diagrams from a normalized 2D scalar field."""

    if any(dim not in {0, 1} for dim in config.homology_dimensions):
        msg = "The 2D cubical-complex baseline only supports H0/H1"
        raise ValueError(msg)
    if config.infinite_policy != "keep":
        msg = "Only infinite_policy='keep' is supported; vectorizers drop essential bars explicitly"
        raise ValueError(msg)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        msg = "Persistent homology scalar field must be 2D"
        raise ValueError(msg)
    if not np.isfinite(values).all():
        msg = "Persistent homology scalar field contains NaN/Inf"
        raise ValueError(msg)
    lower = float(values.min())
    upper = float(values.max())
    if lower < -1e-6 or upper > 1.0 + 1e-6:
        msg = "Persistent homology scalar field must be normalized to [0, 1]"
        raise ValueError(msg)
    values = np.clip(values, 0.0, 1.0)
    if config.filtration == "superlevel":
        values = 1.0 - values
    elif config.filtration != "sublevel":
        msg = f"Unsupported filtration: {config.filtration}"
        raise ValueError(msg)

    try:
        import gudhi as gd
    except ImportError:
        return _fallback_diagram(values, config.homology_dimensions)

    complex_ = gd.CubicalComplex(
        dimensions=values.shape,
        top_dimensional_cells=values.flatten(),
    )
    complex_.compute_persistence()
    diagrams: PersistenceDiagram = {}
    for dim in config.homology_dimensions:
        intervals = complex_.persistence_intervals_in_dimension(dim)
        arr = np.asarray(intervals, dtype=float).reshape(-1, 2)
        diagrams[int(dim)] = arr
    return diagrams


def diagram_to_jsonable(diagram: PersistenceDiagram) -> dict[str, list[list[float]]]:
    """Convert a diagram to JSON-serializable lists."""

    return {str(dim): values.astype(float).tolist() for dim, values in diagram.items()}
