"""Threshold calibration protocols for open-set recognition."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdResult:
    """Calibrated threshold and protocol metadata."""

    threshold: float
    protocol: str
    target_known_recall: float
    used_unknown_classes: tuple[str, ...] = ()


def calibrate_strict_open_set(
    known_validation_scores: np.ndarray,
    *,
    target_known_recall: float = 0.95,
) -> ThresholdResult:
    """Calibrate without unknown examples.

    Since larger scores mean more unknown-like, the threshold is the
    ``target_known_recall`` quantile of known validation scores.
    """

    if known_validation_scores.size == 0:
        msg = "Strict calibration requires known validation scores"
        raise ValueError(msg)
    quantile = float(np.quantile(known_validation_scores, target_known_recall))
    return ThresholdResult(
        threshold=quantile,
        protocol="strict_open_set",
        target_known_recall=target_known_recall,
    )


def calibrate_development_open_set(
    known_validation_scores: np.ndarray,
    unknown_development_scores: np.ndarray,
    *,
    unknown_development_classes: Iterable[str],
    unknown_test_classes: Iterable[str],
    target_known_recall: float = 0.95,
) -> ThresholdResult:
    """Calibrate with reserved unknown development classes only."""

    development = tuple(sorted(set(unknown_development_classes)))
    test = tuple(sorted(set(unknown_test_classes)))
    overlap = set(development).intersection(test)
    if overlap:
        msg = "Unknown development and test classes must be disjoint: " + ", ".join(sorted(overlap))
        raise ValueError(msg)
    if known_validation_scores.size == 0 or unknown_development_scores.size == 0:
        msg = "Development calibration requires known validation and unknown development scores"
        raise ValueError(msg)

    known_threshold = float(np.quantile(known_validation_scores, target_known_recall))
    candidates = np.unique(np.concatenate([known_validation_scores, unknown_development_scores]))
    best_threshold = known_threshold
    best_balanced = -1.0
    for candidate in candidates:
        known_recall = float(np.mean(known_validation_scores <= candidate))
        if known_recall + 1e-12 < target_known_recall:
            continue
        unknown_recall = float(np.mean(unknown_development_scores > candidate))
        balanced = 0.5 * (known_recall + unknown_recall)
        if balanced > best_balanced:
            best_balanced = balanced
            best_threshold = float(candidate)
    return ThresholdResult(
        threshold=best_threshold,
        protocol="development_open_set",
        target_known_recall=target_known_recall,
        used_unknown_classes=development,
    )


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Return 1 for predicted unknown and 0 for predicted known."""

    return (scores > threshold).astype(int)
