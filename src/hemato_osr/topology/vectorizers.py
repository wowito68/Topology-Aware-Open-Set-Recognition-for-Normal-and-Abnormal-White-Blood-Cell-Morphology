"""Vectorizers for persistence diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from hemato_osr.topology.diagrams import PersistenceDiagram


class DiagramVectorizer(Protocol):
    """Fit/transform interface for topological vectorizers."""

    def fit(self, diagrams: list[PersistenceDiagram]) -> DiagramVectorizer:
        """Fit vectorizer parameters on training diagrams only."""

    def transform(self, diagrams: list[PersistenceDiagram]) -> np.ndarray:
        """Vectorize diagrams."""

    def fit_transform(self, diagrams: list[PersistenceDiagram]) -> np.ndarray:
        """Fit then transform diagrams."""


def _finite_points(diagram: PersistenceDiagram, dim: int) -> np.ndarray:
    points = diagram.get(dim, np.empty((0, 2), dtype=float))
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    points = points[np.isfinite(points).all(axis=1)]
    points = points[points[:, 1] >= points[:, 0]]
    return points


@dataclass
class PersistenceEntropy:
    """Persistence entropy for H0/H1."""

    dimensions: tuple[int, ...] = (0, 1)
    eps: float = 1e-12

    def fit(self, diagrams: list[PersistenceDiagram]) -> PersistenceEntropy:
        return self

    def transform(self, diagrams: list[PersistenceDiagram]) -> np.ndarray:
        rows: list[list[float]] = []
        for diagram in diagrams:
            row: list[float] = []
            for dim in self.dimensions:
                points = _finite_points(diagram, dim)
                persistence = points[:, 1] - points[:, 0] if len(points) else np.asarray([])
                total = float(persistence.sum())
                if total <= self.eps:
                    row.append(0.0)
                else:
                    probs = persistence / total
                    row.append(float(-np.sum(probs * np.log(probs + self.eps))))
            rows.append(row)
        return np.asarray(rows, dtype=float)

    def fit_transform(self, diagrams: list[PersistenceDiagram]) -> np.ndarray:
        return self.fit(diagrams).transform(diagrams)


@dataclass
class BettiCurve:
    """Betti curve sampled over a fitted filtration grid."""

    dimensions: tuple[int, ...] = (0, 1)
    n_bins: int = 16
    grid_: np.ndarray | None = None

    def fit(self, diagrams: list[PersistenceDiagram]) -> BettiCurve:
        values: list[float] = []
        for diagram in diagrams:
            for dim in self.dimensions:
                points = _finite_points(diagram, dim)
                values.extend(points.flatten().tolist())
        if values:
            lower = float(np.min(values))
            upper = float(np.max(values))
        else:
            lower, upper = 0.0, 1.0
        if abs(upper - lower) < 1e-12:
            upper = lower + 1.0
        self.grid_ = np.linspace(lower, upper, self.n_bins)
        return self

    def transform(self, diagrams: list[PersistenceDiagram]) -> np.ndarray:
        if self.grid_ is None:
            msg = "BettiCurve must be fit before transform"
            raise ValueError(msg)
        rows: list[list[float]] = []
        for diagram in diagrams:
            row: list[float] = []
            for dim in self.dimensions:
                points = _finite_points(diagram, dim)
                births = points[:, 0] if len(points) else np.asarray([])
                deaths = points[:, 1] if len(points) else np.asarray([])
                curve = [
                    float(np.sum((births <= value) & (deaths > value))) for value in self.grid_
                ]
                row.extend(curve)
            rows.append(row)
        return np.asarray(rows, dtype=float)

    def fit_transform(self, diagrams: list[PersistenceDiagram]) -> np.ndarray:
        return self.fit(diagrams).transform(diagrams)


@dataclass
class PersistenceImage:
    """Simple persistence image over birth-persistence coordinates."""

    dimensions: tuple[int, ...] = (0, 1)
    resolution: int = 8
    sigma: float = 0.1
    bounds_: tuple[float, float, float, float] | None = None

    def fit(self, diagrams: list[PersistenceDiagram]) -> PersistenceImage:
        coords: list[tuple[float, float]] = []
        for diagram in diagrams:
            for dim in self.dimensions:
                points = _finite_points(diagram, dim)
                coords.extend((float(b), float(d - b)) for b, d in points)
        if coords:
            arr = np.asarray(coords, dtype=float)
            b_min, p_min = arr.min(axis=0)
            b_max, p_max = arr.max(axis=0)
        else:
            b_min, b_max, p_min, p_max = 0.0, 1.0, 0.0, 1.0
        if abs(b_max - b_min) < 1e-12:
            b_max = b_min + 1.0
        if abs(p_max - p_min) < 1e-12:
            p_max = p_min + 1.0
        self.bounds_ = (float(b_min), float(b_max), float(p_min), float(p_max))
        return self

    def transform(self, diagrams: list[PersistenceDiagram]) -> np.ndarray:
        if self.bounds_ is None:
            msg = "PersistenceImage must be fit before transform"
            raise ValueError(msg)
        b_min, b_max, p_min, p_max = self.bounds_
        x_edges = np.linspace(b_min, b_max, self.resolution + 1)
        y_edges = np.linspace(p_min, p_max, self.resolution + 1)
        rows: list[np.ndarray] = []
        for diagram in diagrams:
            channels: list[np.ndarray] = []
            for dim in self.dimensions:
                points = _finite_points(diagram, dim)
                if len(points):
                    births = points[:, 0]
                    persistence = points[:, 1] - points[:, 0]
                    image, _, _ = np.histogram2d(
                        persistence,
                        births,
                        bins=[y_edges, x_edges],
                        weights=persistence,
                    )
                else:
                    image = np.zeros((self.resolution, self.resolution), dtype=float)
                channels.append(image.flatten())
            rows.append(np.concatenate(channels))
        return np.vstack(rows) if rows else np.empty((0, len(self.dimensions) * self.resolution**2))

    def fit_transform(self, diagrams: list[PersistenceDiagram]) -> np.ndarray:
        return self.fit(diagrams).transform(diagrams)


@dataclass
class PersistenceLandscape:
    """Persistence landscape summary sampled over a fitted grid."""

    dimensions: tuple[int, ...] = (0, 1)
    n_bins: int = 16
    n_layers: int = 3
    grid_: np.ndarray | None = None

    def fit(self, diagrams: list[PersistenceDiagram]) -> PersistenceLandscape:
        values: list[float] = []
        for diagram in diagrams:
            for dim in self.dimensions:
                points = _finite_points(diagram, dim)
                values.extend(points.flatten().tolist())
        lower, upper = (float(np.min(values)), float(np.max(values))) if values else (0.0, 1.0)
        if abs(upper - lower) < 1e-12:
            upper = lower + 1.0
        self.grid_ = np.linspace(lower, upper, self.n_bins)
        return self

    def transform(self, diagrams: list[PersistenceDiagram]) -> np.ndarray:
        if self.grid_ is None:
            msg = "PersistenceLandscape must be fit before transform"
            raise ValueError(msg)
        rows: list[list[float]] = []
        for diagram in diagrams:
            row: list[float] = []
            for dim in self.dimensions:
                points = _finite_points(diagram, dim)
                for value in self.grid_:
                    tents = (
                        np.maximum(
                            0.0,
                            np.minimum(value - points[:, 0], points[:, 1] - value),
                        )
                        if len(points)
                        else np.asarray([])
                    )
                    sorted_tents = np.sort(tents)[::-1]
                    padded = np.pad(
                        sorted_tents[: self.n_layers],
                        (0, max(0, self.n_layers - len(sorted_tents))),
                    )
                    row.extend(float(x) for x in padded[: self.n_layers])
            rows.append(row)
        return np.asarray(rows, dtype=float)

    def fit_transform(self, diagrams: list[PersistenceDiagram]) -> np.ndarray:
        return self.fit(diagrams).transform(diagrams)


def create_vectorizer(name: str) -> DiagramVectorizer:
    """Create a vectorizer by name."""

    normalized = name.lower().replace("_", "-")
    if normalized == "persistence-entropy":
        return PersistenceEntropy()
    if normalized == "betti-curve":
        return BettiCurve()
    if normalized == "persistence-image":
        return PersistenceImage()
    if normalized == "persistence-landscape":
        return PersistenceLandscape()
    msg = f"Unsupported TDA vectorizer: {name}"
    raise ValueError(msg)
