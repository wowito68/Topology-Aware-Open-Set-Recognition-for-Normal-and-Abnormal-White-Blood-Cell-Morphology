"""IoU matching for full-smear WBC candidate detections."""

from __future__ import annotations

from dataclasses import dataclass

from leukocyte_hil.cropping.boxes import BoundingBox


@dataclass(frozen=True)
class DetectionPrediction:
    """One detector output box."""

    detection_id: str
    bbox: BoundingBox
    confidence: float


@dataclass(frozen=True)
class GroundTruthBox:
    """One annotated leukocyte box."""

    gt_cell_id: str
    bbox: BoundingBox
    morphology: str | None = None
    known_or_unknown: str | None = None


@dataclass(frozen=True)
class DetectionMatch:
    """Greedy one-to-one match result."""

    detection_id: str | None
    gt_cell_id: str | None
    iou: float
    status: str


def bbox_iou(first: BoundingBox, second: BoundingBox) -> float:
    """Compute intersection over union for two pixel-space boxes."""

    x_min = max(first.x_min, second.x_min)
    y_min = max(first.y_min, second.y_min)
    x_max = min(first.x_max, second.x_max)
    y_max = min(first.y_max, second.y_max)
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    union = first.area + second.area - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def match_detections(
    detections: list[DetectionPrediction],
    ground_truth: list[GroundTruthBox],
    *,
    iou_threshold: float,
) -> list[DetectionMatch]:
    """Greedily match detections to annotations by confidence then IoU."""

    if not 0.0 <= iou_threshold <= 1.0:
        msg = "iou_threshold must be in [0, 1]"
        raise ValueError(msg)

    matched_gt: set[str] = set()
    matches: list[DetectionMatch] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        candidates = [
            (bbox_iou(detection.bbox, gt.bbox), gt)
            for gt in ground_truth
            if gt.gt_cell_id not in matched_gt
        ]
        best_iou, best_gt = max(candidates, key=lambda item: item[0], default=(0.0, None))
        if best_gt is not None and best_iou >= iou_threshold:
            matched_gt.add(best_gt.gt_cell_id)
            matches.append(
                DetectionMatch(detection.detection_id, best_gt.gt_cell_id, best_iou, "TP_DETECTION")
            )
        else:
            matches.append(DetectionMatch(detection.detection_id, None, 0.0, "FP_DETECTION"))

    for gt in ground_truth:
        if gt.gt_cell_id not in matched_gt:
            matches.append(DetectionMatch(None, gt.gt_cell_id, 0.0, "MISSED_GT_WBC"))
    return matches
