from __future__ import annotations

import numpy as np

from hemato_osr.evaluation.metrics import (
    calibration_metrics,
    closed_set_metrics,
    fpr_at_tpr,
    open_set_metrics,
)


def test_closed_set_metrics() -> None:
    metrics = closed_set_metrics([0, 1, 1], [0, 0, 1], ["a", "b"])

    assert metrics["accuracy"] == 2 / 3
    assert "confusion_matrix" in metrics


def test_open_set_and_calibration_metrics() -> None:
    y_unknown = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    metrics = open_set_metrics(
        y_unknown,
        scores,
        ["a", "b", "UNKNOWN", "UNKNOWN"],
        ["a", "UNKNOWN", "UNKNOWN", "UNKNOWN"],
    )
    calib = calibration_metrics(np.asarray([[3.0, 0.1], [0.2, 2.0]]), np.asarray([0, 1]))

    assert metrics["auroc_known_unknown"] == 1.0
    assert fpr_at_tpr(np.asarray(y_unknown), np.asarray(scores)) == 0.0
    assert 0.0 <= calib["ece"] <= 1.0
