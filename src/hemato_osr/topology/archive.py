"""Compact persistence-diagram archive utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from hemato_osr.topology.diagrams import PersistenceDiagram


@dataclass(frozen=True)
class DiagramArchive:
    """Loaded diagram archive."""

    sample_ids: list[str]
    labels: np.ndarray
    known_status: np.ndarray
    splits: np.ndarray
    filtrations: tuple[str, ...]
    dimensions: tuple[int, ...]
    diagrams: dict[tuple[str, int], list[np.ndarray]]
    essential_counts: dict[tuple[str, int], np.ndarray]
    metadata: dict[str, Any]


def _pack_diagrams(diagrams: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite_parts = []
    offsets = [0]
    essential_counts = []
    for diagram in diagrams:
        arr = np.asarray(diagram, dtype=np.float64).reshape(-1, 2)
        finite = arr[np.isfinite(arr).all(axis=1)]
        finite_parts.append(finite)
        offsets.append(offsets[-1] + len(finite))
        essential_counts.append(int(np.isinf(arr).any(axis=1).sum()) if len(arr) else 0)
    points = (
        np.vstack(finite_parts).astype(np.float32)
        if finite_parts and any(len(part) for part in finite_parts)
        else np.empty((0, 2), dtype=np.float32)
    )
    return (
        np.asarray(offsets, dtype=np.int64),
        points,
        np.asarray(essential_counts, dtype=np.int16),
    )


def _unpack_diagrams(offsets: np.ndarray, points: np.ndarray) -> list[np.ndarray]:
    rows = []
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        rows.append(np.asarray(points[int(start) : int(end)], dtype=np.float64).reshape(-1, 2))
    return rows


def save_diagram_archive(
    path: Path,
    *,
    sample_ids: list[str],
    labels: np.ndarray,
    known_status: np.ndarray,
    splits: np.ndarray,
    diagrams_by_key: dict[tuple[str, int], list[np.ndarray]],
    metadata: dict[str, Any],
) -> Path:
    """Save finite diagram intervals and essential bar counts in one archive."""

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "sample_id": np.asarray(sample_ids, dtype=object),
        "true_label": labels.astype(object),
        "known_status": known_status.astype(object),
        "split": splits.astype(object),
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True, default=str)),
    }
    for (filtration, dim), diagrams in diagrams_by_key.items():
        offsets, points, essential_counts = _pack_diagrams(diagrams)
        prefix = f"{filtration}_h{dim}"
        arrays[f"{prefix}_offsets"] = offsets
        arrays[f"{prefix}_points"] = points
        arrays[f"{prefix}_essential_counts"] = essential_counts
    np.savez_compressed(path, **cast(Any, arrays))
    return path


def load_diagram_archive(path: Path) -> DiagramArchive:
    """Load a compact diagram archive."""

    with np.load(path, allow_pickle=True) as data:
        metadata = json.loads(str(data["metadata"]))
        sample_ids = [str(item) for item in data["sample_id"].tolist()]
        labels = data["true_label"].astype(str)
        known_status = data["known_status"].astype(str)
        splits = data["split"].astype(str)
        filtrations: set[str] = set()
        dimensions: set[int] = set()
        diagrams: dict[tuple[str, int], list[np.ndarray]] = {}
        essential_counts: dict[tuple[str, int], np.ndarray] = {}
        for key in data.files:
            if not key.endswith("_offsets"):
                continue
            prefix = key.removesuffix("_offsets")
            filtration, dim_text = prefix.rsplit("_h", maxsplit=1)
            dim = int(dim_text)
            filtrations.add(filtration)
            dimensions.add(dim)
            diagrams[(filtration, dim)] = _unpack_diagrams(
                np.asarray(data[key]),
                np.asarray(data[f"{prefix}_points"]),
            )
            essential_counts[(filtration, dim)] = np.asarray(data[f"{prefix}_essential_counts"])
    return DiagramArchive(
        sample_ids=sample_ids,
        labels=labels,
        known_status=known_status,
        splits=splits,
        filtrations=tuple(sorted(filtrations)),
        dimensions=tuple(sorted(dimensions)),
        diagrams=diagrams,
        essential_counts=essential_counts,
        metadata=metadata,
    )


def diagrams_for_filtration(archive: DiagramArchive, filtration: str) -> list[PersistenceDiagram]:
    """Return list-of-dicts diagrams for a single filtration."""

    rows: list[PersistenceDiagram] = []
    n = len(archive.sample_ids)
    for idx in range(n):
        diagram: PersistenceDiagram = {}
        for dim in archive.dimensions:
            diagram[dim] = archive.diagrams[(filtration, dim)][idx]
        rows.append(diagram)
    return rows
