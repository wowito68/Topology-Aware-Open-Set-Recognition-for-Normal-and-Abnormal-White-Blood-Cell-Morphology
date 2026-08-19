"""Priority-ordered triage rules for leukocyte morphology review."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from hashlib import sha256

from leukocyte_hil.quality.heuristics import QualityFlag, QualityResult


class TriageStatus(StrEnum):
    """Primary status assigned to each candidate crop."""

    AUTO_ACCEPT = "AUTO_ACCEPT"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    POSSIBLE_UNKNOWN = "POSSIBLE_UNKNOWN"
    LOW_QUALITY = "LOW_QUALITY"
    INVALID_DETECTION = "INVALID_DETECTION"


@dataclass(frozen=True)
class TriageThresholds:
    """Frozen thresholds for rule-based selective prediction."""

    confidence: float
    margin: float
    unknown_score: float
    min_quality_score: float

    def stable_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TriageDecision:
    """Triage result and compact reason code."""

    status: TriageStatus
    reason: str


def triage_cell(
    *,
    valid_detection: bool,
    quality: QualityResult,
    confidence: float,
    margin: float,
    unknown_score: float,
    thresholds: TriageThresholds,
) -> TriageDecision:
    """Assign exactly one primary triage status with deterministic priority."""

    if not valid_detection or QualityFlag.INVALID_CROP in quality.flags:
        return TriageDecision(TriageStatus.INVALID_DETECTION, "invalid_detection")
    if quality.failed or quality.quality_score < thresholds.min_quality_score:
        return TriageDecision(TriageStatus.LOW_QUALITY, "quality_failed")
    if unknown_score >= thresholds.unknown_score:
        return TriageDecision(TriageStatus.POSSIBLE_UNKNOWN, "unknown_score")
    if confidence < thresholds.confidence:
        return TriageDecision(TriageStatus.HUMAN_REVIEW, "low_confidence")
    if margin < thresholds.margin:
        return TriageDecision(TriageStatus.HUMAN_REVIEW, "low_margin")
    return TriageDecision(TriageStatus.AUTO_ACCEPT, "all_acceptance_conditions_passed")
