from __future__ import annotations

from pathlib import Path

import pytest

from hemato_osr.data.taxonomy import Taxonomy
from leukocyte_hil.triage.calibration import (
    CalibrationRecord,
    TriageStrategy,
    assign_strategy_status,
    calibrate_operating_points,
    known_validation_records,
    operating_points_to_yaml,
)
from leukocyte_hil.triage.rules import TriageStatus
from leukocyte_hil.utils.checkpoints import sha256_file, verify_sha256


def test_checkpoint_sha256_exact_match_and_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny_checkpoint.pt"
    checkpoint.write_bytes(b"small synthetic checkpoint bytes")
    observed = sha256_file(checkpoint)

    matched = verify_sha256(checkpoint, observed)
    mismatched = verify_sha256(checkpoint, "0" * 64)
    missing = verify_sha256(tmp_path / "missing.pt", observed)

    assert matched.exact_match
    assert matched.exists
    assert not mismatched.exact_match
    assert missing.observed_sha256 is None
    assert not missing.exists


def test_known_validation_records_exclude_test_and_unknown_rows() -> None:
    records = [
        CalibrationRecord("kv1", "known_validation", "known", 0.95, 0.30, 0.05),
        CalibrationRecord("kt1", "known_test", "known", 0.10, 0.01, 0.90),
        CalibrationRecord("ut1", "unknown_test", "unknown", 0.99, 0.80, 0.01),
    ]

    selected = known_validation_records(records)

    assert [record.sample_id for record in selected] == ["kv1"]


def test_calibrate_operating_points_and_assign_statuses() -> None:
    records = [
        CalibrationRecord("kv1", "known_validation", "known", 0.99, 0.50, 0.01),
        CalibrationRecord("kv2", "known_validation", "known", 0.90, 0.30, 0.10),
        CalibrationRecord("kv3", "known_validation", "known", 0.70, 0.12, 0.30),
        CalibrationRecord("kv4", "known_validation", "known", 0.40, 0.02, 0.60),
        CalibrationRecord("ut1", "unknown_test", "unknown", 0.99, 0.50, 0.01),
    ]

    points = calibrate_operating_points(records, targets=[0.5])
    point_by_strategy = {point.strategy: point for point in points}

    assert set(point_by_strategy) == set(TriageStrategy)
    for point in points:
        assert point.calibration_sample_count == 4
        assert len(point.config_hash) == 64

    status = assign_strategy_status(
        CalibrationRecord("bad", "known_test", "known", 0.99, 0.50, 0.01, quality_pass=False),
        point_by_strategy[TriageStrategy.FULL_TRIAGE],
    )
    assert status == TriageStatus.LOW_QUALITY


def test_operating_points_yaml_records_test_isolation() -> None:
    points = calibrate_operating_points(
        [
            CalibrationRecord("kv1", "known_validation", "known", 0.99, 0.50, 0.01),
            CalibrationRecord("kv2", "known_validation", "known", 0.70, 0.20, 0.30),
        ],
        targets=[1.0],
    )

    rendered = operating_points_to_yaml(points)

    assert "known_test_used_for_threshold_calibration: false" in rendered
    assert "unknown_test_used_for_threshold_calibration: false" in rendered
    assert "confidence_margin_msp_qc" in rendered


def test_calibration_requires_known_validation_data() -> None:
    with pytest.raises(ValueError, match="No known-validation records"):
        calibrate_operating_points(
            [CalibrationRecord("ut1", "unknown_test", "unknown", 0.5, 0.2, 0.5)]
        )


def test_delivery11_taxonomy_hard_invariants() -> None:
    taxonomy = Taxonomy()

    assert taxonomy.canonicalize("neutrophil_band") != "neutrophil_segmented"
    assert taxonomy.known_status("neutrophil_band") == "unknown"
    assert "diagnosis" != "UNKNOWN"
