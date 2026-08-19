"""Prospective operating-point calibration for selective triage."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256

import numpy as np

from leukocyte_hil.triage.rules import TriageStatus


class TriageStrategy(StrEnum):
    """The three Delivery 1.1 triage strategies."""

    CONFIDENCE_ONLY = "confidence_only"
    CONFIDENCE_MARGIN = "confidence_margin"
    FULL_TRIAGE = "confidence_margin_msp_qc"


@dataclass(frozen=True)
class CalibrationRecord:
    """One validation/evaluation signal row."""

    sample_id: str
    split: str
    known_or_unknown: str
    confidence: float
    margin: float
    unknown_score: float
    quality_pass: bool = True
    correct_if_known: bool | None = None


@dataclass(frozen=True)
class OperatingPoint:
    """Frozen threshold set for one strategy and target coverage."""

    strategy: TriageStrategy
    target_coverage: float
    actual_known_validation_coverage: float
    calibration_sample_count: int
    confidence_threshold: float | None
    margin_threshold: float | None
    unknown_threshold: float | None
    qc_required: bool
    config_hash: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "strategy": self.strategy.value,
            "target_coverage": self.target_coverage,
            "actual_known_validation_coverage": self.actual_known_validation_coverage,
            "calibration_sample_count": self.calibration_sample_count,
            "confidence_threshold": self.confidence_threshold,
            "margin_threshold": self.margin_threshold,
            "unknown_threshold": self.unknown_threshold,
            "qc_required": self.qc_required,
            "config_hash": self.config_hash,
        }


def known_validation_records(records: Iterable[CalibrationRecord]) -> list[CalibrationRecord]:
    """Return allowed calibration rows and reject test/unknown leakage."""

    selected = [
        record
        for record in records
        if record.split == "known_validation" and record.known_or_unknown == "known"
    ]
    if not selected:
        msg = "No known-validation records available for triage calibration"
        raise ValueError(msg)
    return selected


def calibrate_operating_points(
    records: Iterable[CalibrationRecord],
    *,
    targets: Iterable[float] = (0.95, 0.90, 0.80, 0.70, 0.60),
) -> list[OperatingPoint]:
    """Calibrate strategies A/B/C using known-validation coverage only."""

    calibration = known_validation_records(records)
    points: list[OperatingPoint] = []
    for strategy in TriageStrategy:
        for target in targets:
            if not 0.0 < target <= 1.0:
                msg = f"Invalid coverage target: {target}"
                raise ValueError(msg)
            point = _choose_point_for_strategy(strategy, calibration, float(target))
            points.append(point)
    return points


def assign_strategy_status(
    record: CalibrationRecord,
    point: OperatingPoint,
) -> TriageStatus:
    """Assign a status under one frozen operating point."""

    if point.strategy == TriageStrategy.CONFIDENCE_ONLY:
        if record.confidence >= _required(point.confidence_threshold, "confidence"):
            return TriageStatus.AUTO_ACCEPT
        return TriageStatus.HUMAN_REVIEW
    if point.strategy == TriageStrategy.CONFIDENCE_MARGIN:
        if record.confidence >= _required(
            point.confidence_threshold, "confidence"
        ) and record.margin >= _required(point.margin_threshold, "margin"):
            return TriageStatus.AUTO_ACCEPT
        return TriageStatus.HUMAN_REVIEW

    if point.qc_required and not record.quality_pass:
        return TriageStatus.LOW_QUALITY
    if record.unknown_score >= _required(point.unknown_threshold, "unknown"):
        return TriageStatus.POSSIBLE_UNKNOWN
    if record.confidence < _required(point.confidence_threshold, "confidence"):
        return TriageStatus.HUMAN_REVIEW
    if record.margin < _required(point.margin_threshold, "margin"):
        return TriageStatus.HUMAN_REVIEW
    return TriageStatus.AUTO_ACCEPT


def operating_points_to_yaml(points: Iterable[OperatingPoint]) -> str:
    """Serialize operating points as simple YAML without custom tags."""

    payload = {
        "status": "frozen_after_known_validation_calibration",
        "known_test_used_for_threshold_calibration": False,
        "unknown_test_used_for_threshold_calibration": False,
        "operating_points": [point.to_mapping() for point in points],
    }
    return _to_simple_yaml(payload)


def _choose_point_for_strategy(
    strategy: TriageStrategy,
    calibration: list[CalibrationRecord],
    target: float,
) -> OperatingPoint:
    if strategy == TriageStrategy.CONFIDENCE_ONLY:
        candidates = [
            _candidate_point(strategy, calibration, target, confidence_threshold=threshold)
            for threshold in _candidate_thresholds([record.confidence for record in calibration])
        ]
    elif strategy == TriageStrategy.CONFIDENCE_MARGIN:
        candidates = [
            _candidate_point(
                strategy,
                calibration,
                target,
                confidence_threshold=confidence_threshold,
                margin_threshold=margin_threshold,
            )
            for confidence_threshold in _candidate_thresholds(
                [record.confidence for record in calibration]
            )
            for margin_threshold in _candidate_thresholds([record.margin for record in calibration])
        ]
    else:
        candidates = [
            _candidate_point(
                strategy,
                calibration,
                target,
                confidence_threshold=confidence_threshold,
                margin_threshold=margin_threshold,
                unknown_threshold=unknown_threshold,
                qc_required=True,
            )
            for confidence_threshold in _candidate_thresholds(
                [record.confidence for record in calibration]
            )
            for margin_threshold in _candidate_thresholds([record.margin for record in calibration])
            for unknown_threshold in _candidate_thresholds(
                [record.unknown_score for record in calibration]
            )
        ]
    return min(
        candidates,
        key=lambda point: (
            abs(point.actual_known_validation_coverage - target),
            -_none_low(point.confidence_threshold),
            -_none_low(point.margin_threshold),
        ),
    )


def _candidate_point(
    strategy: TriageStrategy,
    calibration: list[CalibrationRecord],
    target: float,
    *,
    confidence_threshold: float | None = None,
    margin_threshold: float | None = None,
    unknown_threshold: float | None = None,
    qc_required: bool = False,
) -> OperatingPoint:
    provisional = OperatingPoint(
        strategy=strategy,
        target_coverage=target,
        actual_known_validation_coverage=0.0,
        calibration_sample_count=len(calibration),
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        unknown_threshold=unknown_threshold,
        qc_required=qc_required,
        config_hash="pending",
    )
    coverage = float(
        np.mean(
            [
                assign_strategy_status(record, provisional) == TriageStatus.AUTO_ACCEPT
                for record in calibration
            ]
        )
    )
    payload = {
        "strategy": strategy.value,
        "target_coverage": target,
        "actual_known_validation_coverage": coverage,
        "calibration_sample_count": len(calibration),
        "confidence_threshold": confidence_threshold,
        "margin_threshold": margin_threshold,
        "unknown_threshold": unknown_threshold,
        "qc_required": qc_required,
    }
    config_hash = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return OperatingPoint(
        strategy=strategy,
        target_coverage=target,
        actual_known_validation_coverage=coverage,
        calibration_sample_count=len(calibration),
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        unknown_threshold=unknown_threshold,
        qc_required=qc_required,
        config_hash=config_hash,
    )


def _candidate_thresholds(values: Iterable[float]) -> list[float]:
    array = np.asarray(list(values), dtype=float)
    quantiles = np.linspace(0.0, 1.0, 21)
    thresholds = np.quantile(array, quantiles)
    return sorted({float(value) for value in thresholds})


def _required(value: float | None, name: str) -> float:
    if value is None:
        msg = f"{name} threshold is required for this strategy"
        raise ValueError(msg)
    return value


def _none_low(value: float | None) -> float:
    return -1.0 if value is None else value


def _to_simple_yaml(value: object, *, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, list | dict):
                lines.append(f"{prefix}{key}:")
                lines.append(_to_simple_yaml(item, indent=indent + 2).rstrip())
            else:
                lines.append(f"{prefix}{key}: {_format_scalar(item)}")
        return "\n".join(lines) + "\n"
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{prefix}-")
                lines.append(_to_simple_yaml(item, indent=indent + 2).rstrip())
            else:
                lines.append(f"{prefix}- {_format_scalar(item)}")
        return "\n".join(lines) + "\n"
    return f"{prefix}{_format_scalar(value)}\n"


def _format_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return str(value)
