"""JSON-safe records for research review output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from leukocyte_hil.cropping.boxes import BoundingBox
from leukocyte_hil.quality.heuristics import QualityFlag
from leukocyte_hil.triage.rules import TriageStatus

FORBIDDEN_OUTPUT_KEYS = {
    "diagnosis",
    "cancer_probability",
    "leukemia_probability",
    "malignancy_probability",
}


@dataclass(frozen=True)
class CellDetail:
    """One detected-cell record for human verification."""

    detection_id: str
    bbox: BoundingBox
    expanded_bbox: BoundingBox
    detection_confidence: float
    predicted_class: str
    classification_confidence: float
    second_class: str
    margin: float
    unknown_score: float
    quality_score: float
    quality_flags: tuple[QualityFlag, ...]
    triage_status: TriageStatus
    crop_checksum: str | None = None

    def to_json_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "detection_id": self.detection_id,
            "bbox": self.bbox.to_xyxy(),
            "expanded_bbox": self.expanded_bbox.to_xyxy(),
            "detection_confidence": self.detection_confidence,
            "predicted_class": self.predicted_class,
            "classification_confidence": self.classification_confidence,
            "second_class": self.second_class,
            "margin": self.margin,
            "unknown_score": self.unknown_score,
            "quality_score": self.quality_score,
            "quality_flags": [flag.value for flag in self.quality_flags],
            "triage_status": self.triage_status.value,
            "crop_checksum": self.crop_checksum,
        }
        forbidden = FORBIDDEN_OUTPUT_KEYS.intersection(record)
        if forbidden:
            msg = f"Forbidden diagnostic output keys present: {sorted(forbidden)}"
            raise ValueError(msg)
        return record


@dataclass(frozen=True)
class ImageAnalysisResult:
    """Machine-readable output for one full-smear image."""

    image_id: str
    detections: tuple[CellDetail, ...]

    def to_json_record(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "detections": [detection.to_json_record() for detection in self.detections],
        }
