"""Closed-set morphology predictions and MSP-oriented unknown scores."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import softmax


@dataclass(frozen=True)
class MorphologyPrediction:
    """Classifier output for one cell crop."""

    predicted_class: str
    confidence: float
    second_class: str
    second_confidence: float
    margin: float
    unknown_score: float
    probabilities: dict[str, float]


def classification_signals(
    logits: np.ndarray,
    class_names: Sequence[str],
) -> list[MorphologyPrediction]:
    """Convert logits to triage-ready softmax, margin, and MSP unknown signals.

    Larger ``unknown_score`` means more unknown-like and is defined as
    ``1 - max softmax probability``.
    """

    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2:
        msg = "logits must be a 2D array"
        raise ValueError(msg)
    if logits.shape[1] != len(class_names):
        msg = "number of class names must match logits.shape[1]"
        raise ValueError(msg)
    if len(class_names) < 2:
        msg = "at least two classes are required to compute a top1-top2 margin"
        raise ValueError(msg)

    probabilities = softmax(logits, axis=1)
    results: list[MorphologyPrediction] = []
    for row in probabilities:
        order = np.argsort(row)[::-1]
        top1 = int(order[0])
        top2 = int(order[1])
        confidence = float(row[top1])
        second_confidence = float(row[top2])
        results.append(
            MorphologyPrediction(
                predicted_class=class_names[top1],
                confidence=confidence,
                second_class=class_names[top2],
                second_confidence=second_confidence,
                margin=confidence - second_confidence,
                unknown_score=1.0 - confidence,
                probabilities={
                    class_name: float(row[idx]) for idx, class_name in enumerate(class_names)
                },
            )
        )
    return results
