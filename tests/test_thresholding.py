from __future__ import annotations

import numpy as np
import pytest

from hemato_osr.openset.thresholds import (
    apply_threshold,
    calibrate_development_open_set,
    calibrate_strict_open_set,
)


def test_strict_threshold_and_application() -> None:
    scores = np.asarray([0.1, 0.2, 0.3, 0.4])
    result = calibrate_strict_open_set(scores, target_known_recall=0.75)

    assert result.protocol == "strict_open_set"
    assert apply_threshold(np.asarray([0.1, 0.9]), result.threshold).tolist() == [0, 1]


def test_development_unknown_classes_must_be_disjoint() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        calibrate_development_open_set(
            np.asarray([0.1, 0.2]),
            np.asarray([0.8, 0.9]),
            unknown_development_classes=("myeloblast",),
            unknown_test_classes=("myeloblast",),
        )
