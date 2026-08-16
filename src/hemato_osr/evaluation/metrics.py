"""Closed-set, open-set, and calibration metrics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def closed_set_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, object]:
    """Compute standard closed-set classification metrics."""

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    per_class = {
        class_name: {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx, class_name in enumerate(class_names)
    }
    observed = support > 0
    balanced_accuracy = float(np.mean(recall[observed])) if np.any(observed) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
        ).tolist(),
    }


def fpr_at_tpr(y_unknown: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95) -> float:
    """Compute FPR at target TPR where positive means unknown."""

    positives = scores[y_unknown == 1]
    negatives = scores[y_unknown == 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    threshold = float(np.quantile(positives, 1.0 - target_tpr))
    return float(np.mean(negatives > threshold))


def open_set_metrics(
    y_unknown: Sequence[int],
    unknown_scores: Sequence[float],
    y_true_with_unknown: Sequence[str],
    y_pred_with_unknown: Sequence[str],
) -> dict[str, float]:
    """Compute open-set metrics with UNKNOWN as a prediction label."""

    y_unknown_arr = np.asarray(y_unknown, dtype=int)
    score_arr = np.asarray(unknown_scores, dtype=float)
    metrics: dict[str, float] = {
        "auroc_known_unknown": float(roc_auc_score(y_unknown_arr, score_arr)),
        "aupr_known_unknown": float(average_precision_score(y_unknown_arr, score_arr)),
        "fpr_at_95_tpr": fpr_at_tpr(y_unknown_arr, score_arr, target_tpr=0.95),
        "unknown_recall": float(
            np.mean(np.asarray(y_pred_with_unknown, dtype=object)[y_unknown_arr == 1] == "UNKNOWN")
        ),
        "known_recall": float(
            np.mean(np.asarray(y_pred_with_unknown, dtype=object)[y_unknown_arr == 0] != "UNKNOWN")
        ),
        "macro_f1_with_unknown": float(
            f1_score(y_true_with_unknown, y_pred_with_unknown, average="macro", zero_division=0)
        ),
    }
    return metrics


def expected_calibration_error(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Compute multiclass expected calibration error."""

    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == y_true
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        mask = (confidences > lower) & (confidences <= upper)
        if not np.any(mask):
            continue
        accuracy = float(np.mean(correct[mask]))
        confidence = float(np.mean(confidences[mask]))
        ece += float(np.mean(mask)) * abs(accuracy - confidence)
    return float(ece)


def calibration_metrics(logits: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """Compute ECE, NLL, and multiclass Brier score."""

    probabilities = softmax(logits, axis=1)
    one_hot = np.eye(probabilities.shape[1])[y_true]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    return {
        "ece": expected_calibration_error(probabilities, y_true),
        "nll": float(log_loss(y_true, probabilities, labels=list(range(probabilities.shape[1])))),
        "brier_score": brier,
    }


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap confidence interval."""

    metric: str
    estimate: float
    lower: float
    upper: float
    n_bootstraps: int
    seed: int


def bootstrap_ci(
    values: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    *,
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
    seed: int = 13,
) -> BootstrapResult:
    """Seeded bootstrap confidence interval for a metric over rows."""

    rng = np.random.default_rng(seed)
    n = values.shape[0]
    estimates = []
    for _ in range(n_bootstraps):
        indices = rng.integers(0, n, size=n)
        estimates.append(float(metric_fn(values[indices])))
    alpha = 1.0 - confidence
    estimate = float(metric_fn(values))
    return BootstrapResult(
        metric=getattr(metric_fn, "__name__", "metric"),
        estimate=estimate,
        lower=float(np.quantile(estimates, alpha / 2.0)),
        upper=float(np.quantile(estimates, 1.0 - alpha / 2.0)),
        n_bootstraps=n_bootstraps,
        seed=seed,
    )


def oscr(
    y_true_known: np.ndarray,
    predicted_known: np.ndarray,
    known_confidence: np.ndarray,
    is_unknown: np.ndarray,
) -> float:
    """Compute a compact OSCR approximation from thresholded CCR/FPR curve."""

    thresholds = np.unique(known_confidence)[::-1]
    ccr_values: list[float] = []
    fpr_values: list[float] = []
    known_mask = is_unknown == 0
    unknown_mask = is_unknown == 1
    for threshold in thresholds:
        accepted_known = known_mask & (known_confidence >= threshold)
        correct_known = accepted_known & (predicted_known == y_true_known)
        ccr = float(np.sum(correct_known) / max(1, np.sum(known_mask)))
        fpr = float(
            np.sum(unknown_mask & (known_confidence >= threshold)) / max(1, np.sum(unknown_mask))
        )
        ccr_values.append(ccr)
        fpr_values.append(fpr)
    order = np.argsort(fpr_values)
    return float(np.trapezoid(np.asarray(ccr_values)[order], np.asarray(fpr_values)[order]))
