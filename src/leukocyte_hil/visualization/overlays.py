"""Neutral static overlays for detected leukocyte candidates."""

from __future__ import annotations

from PIL import Image, ImageDraw

from leukocyte_hil.pipeline.records import CellDetail
from leukocyte_hil.triage.rules import TriageStatus

STATUS_COLORS: dict[TriageStatus, tuple[int, int, int]] = {
    TriageStatus.AUTO_ACCEPT: (70, 120, 180),
    TriageStatus.HUMAN_REVIEW: (145, 95, 180),
    TriageStatus.POSSIBLE_UNKNOWN: (200, 140, 40),
    TriageStatus.LOW_QUALITY: (120, 120, 120),
    TriageStatus.INVALID_DETECTION: (40, 40, 40),
}


def draw_review_overlay(image: Image.Image, detections: list[CellDetail]) -> Image.Image:
    """Draw boxes and neutral labels for research review."""

    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    for detection in detections:
        color = STATUS_COLORS[detection.triage_status]
        box = detection.bbox.to_int_tuple()
        draw.rectangle(box, outline=color, width=3)
        label = _label_for_detection(detection)
        text_bbox = draw.textbbox((box[0], box[1]), label)
        background = (text_bbox[0], text_bbox[1], text_bbox[2] + 4, text_bbox[3] + 4)
        draw.rectangle(background, fill=(255, 255, 255))
        draw.text((box[0] + 2, box[1] + 2), label, fill=color)
    return overlay


def _label_for_detection(detection: CellDetail) -> str:
    if detection.triage_status == TriageStatus.POSSIBLE_UNKNOWN:
        return "UNKNOWN? REVIEW"
    if detection.triage_status == TriageStatus.HUMAN_REVIEW:
        return f"{detection.predicted_class.upper()}? REVIEW"
    if detection.triage_status == TriageStatus.LOW_QUALITY:
        return "LOW QUALITY"
    if detection.triage_status == TriageStatus.INVALID_DETECTION:
        return "INVALID"
    return f"{detection.predicted_class.upper()} AUTO"
