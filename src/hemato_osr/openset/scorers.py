"""Open-set scoring baselines.

All scorers return higher values for samples that look more likely unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.special import logsumexp, softmax


class OpenSetScorer(Protocol):
    """Protocol for open-set scoring methods."""

    def fit(
        self,
        logits: np.ndarray,
        embeddings: np.ndarray | None = None,
        labels: np.ndarray | None = None,
    ) -> OpenSetScorer:
        """Fit scorer parameters using training/development data only."""

    def score(
        self,
        logits: np.ndarray,
        embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return unknownness scores; larger means more likely unknown."""


@dataclass
class MaximumSoftmaxProbability:
    """Unknown score based on one minus maximum softmax probability."""

    def fit(
        self,
        logits: np.ndarray,
        embeddings: np.ndarray | None = None,
        labels: np.ndarray | None = None,
    ) -> MaximumSoftmaxProbability:
        return self

    def score(
        self,
        logits: np.ndarray,
        embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        probabilities = softmax(logits, axis=1)
        return 1.0 - probabilities.max(axis=1)


@dataclass
class PredictiveEntropy:
    """Unknown score based on softmax entropy."""

    eps: float = 1e-12

    def fit(
        self,
        logits: np.ndarray,
        embeddings: np.ndarray | None = None,
        labels: np.ndarray | None = None,
    ) -> PredictiveEntropy:
        return self

    def score(
        self,
        logits: np.ndarray,
        embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        probabilities = softmax(logits, axis=1)
        return -np.sum(probabilities * np.log(probabilities + self.eps), axis=1)


@dataclass
class EnergyScore:
    """Energy unknown score from logits."""

    temperature: float = 1.0

    def fit(
        self,
        logits: np.ndarray,
        embeddings: np.ndarray | None = None,
        labels: np.ndarray | None = None,
    ) -> EnergyScore:
        return self

    def score(
        self,
        logits: np.ndarray,
        embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        energy = -self.temperature * logsumexp(logits / self.temperature, axis=1)
        return energy


@dataclass
class MahalanobisScorer:
    """Minimum class-conditional Mahalanobis distance over embeddings."""

    regularization: float = 1e-4
    means_: dict[int, np.ndarray] | None = None
    precision_: np.ndarray | None = None

    def fit(
        self,
        logits: np.ndarray,
        embeddings: np.ndarray | None = None,
        labels: np.ndarray | None = None,
    ) -> MahalanobisScorer:
        if embeddings is None or labels is None:
            msg = "MahalanobisScorer requires embeddings and labels for fit"
            raise ValueError(msg)
        if embeddings.ndim != 2:
            msg = "Embeddings must be a 2D array"
            raise ValueError(msg)
        means: dict[int, np.ndarray] = {}
        centered: list[np.ndarray] = []
        for label in sorted(set(labels.astype(int).tolist())):
            class_embeddings = embeddings[labels == label]
            mean = class_embeddings.mean(axis=0)
            means[int(label)] = mean
            centered.append(class_embeddings - mean)
        residuals = np.concatenate(centered, axis=0)
        covariance = np.cov(residuals, rowvar=False)
        covariance = np.atleast_2d(covariance)
        covariance += np.eye(covariance.shape[0]) * self.regularization
        self.means_ = means
        self.precision_ = np.linalg.pinv(covariance)
        return self

    def score(
        self,
        logits: np.ndarray,
        embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        if embeddings is None:
            msg = "MahalanobisScorer requires embeddings for score"
            raise ValueError(msg)
        if self.means_ is None or self.precision_ is None:
            msg = "MahalanobisScorer must be fit before score"
            raise ValueError(msg)
        distances = []
        for mean in self.means_.values():
            delta = embeddings - mean
            distances.append(np.sum(delta @ self.precision_ * delta, axis=1))
        return np.min(np.vstack(distances), axis=0)


def create_scorer(name: str) -> OpenSetScorer:
    """Create a scorer from a config name."""

    normalized = name.lower().replace("_", "-")
    if normalized in {"msp", "maximum-softmax-probability"}:
        return MaximumSoftmaxProbability()
    if normalized in {"entropy", "predictive-entropy"}:
        return PredictiveEntropy()
    if normalized == "energy":
        return EnergyScore()
    if normalized == "mahalanobis":
        return MahalanobisScorer()
    msg = f"Unsupported open-set scorer: {name}"
    raise ValueError(msg)
