from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from hemato_osr.data.taxonomy import DEFAULT_KNOWN_CLASSES
from leukocyte_hil.cropping.boxes import BoundingBox, crop_image, crop_sha256, expand_bbox
from leukocyte_hil.detection.matching import DetectionPrediction, GroundTruthBox, match_detections
from leukocyte_hil.evaluation.selective import SelectiveRecord, selective_metrics
from leukocyte_hil.morphology.signals import classification_signals
from leukocyte_hil.pipeline.records import FORBIDDEN_OUTPUT_KEYS, CellDetail, ImageAnalysisResult
from leukocyte_hil.quality.heuristics import QualityConfig, QualityFlag, assess_crop_quality
from leukocyte_hil.triage.rules import TriageStatus, TriageThresholds, triage_cell
from leukocyte_hil.visualization.overlays import draw_review_overlay


def test_bbox_expansion_clipping_and_crop_checksum() -> None:
    image = Image.new("RGB", (100, 80), color=(128, 128, 128))
    bbox = BoundingBox(10, 10, 30, 30)

    expanded = expand_bbox(bbox, image.size, margin_fraction=0.25)
    crop = crop_image(image, expanded)

    assert expanded.to_xyxy() == [5.0, 5.0, 35.0, 35.0]
    assert crop.size == (30, 30)
    assert len(crop_sha256(crop)) == 64


def test_invalid_crop_raises() -> None:
    image = Image.new("RGB", (20, 20))

    with pytest.raises(ValueError, match="invalid bounding box"):
        crop_image(image, BoundingBox(5, 5, 5, 10))


def test_classification_signals_msp_orientation_and_margin() -> None:
    logits = np.asarray([[8.0, 0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0, 0.0]])

    confident, ambiguous = classification_signals(logits, DEFAULT_KNOWN_CLASSES)

    assert confident.predicted_class == "basophil"
    assert confident.confidence > ambiguous.confidence
    assert ambiguous.unknown_score > confident.unknown_score
    assert confident.margin > ambiguous.margin


def test_quality_flags_for_invalid_border_small_and_exposure() -> None:
    config = QualityConfig(min_crop_width=16, min_crop_height=16, underexposed_mean_threshold=5.0)
    dark_small_crop = Image.new("RGB", (10, 10), color=(0, 0, 0))

    result = assess_crop_quality(
        dark_small_crop,
        crop_box=BoundingBox(0, 0, 10, 10),
        image_size=(100, 100),
        config=config,
    )
    invalid = assess_crop_quality(None, crop_box=None)

    assert QualityFlag.TOO_SMALL in result.flags
    assert QualityFlag.CLIPPED_AT_IMAGE_BORDER in result.flags
    assert QualityFlag.UNDEREXPOSED in result.flags
    assert invalid.flags == (QualityFlag.INVALID_CROP,)


def test_triage_priority_order() -> None:
    thresholds = TriageThresholds(
        confidence=0.80,
        margin=0.10,
        unknown_score=0.40,
        min_quality_score=0.75,
    )
    good_quality = assess_crop_quality(
        _checkerboard_image(40),
        crop_box=BoundingBox(10, 10, 50, 50),
        image_size=(100, 100),
        config=QualityConfig(blur_variance_threshold=0.0),
    )
    low_quality = assess_crop_quality(None, crop_box=None)

    assert (
        triage_cell(
            valid_detection=False,
            quality=good_quality,
            confidence=1.0,
            margin=1.0,
            unknown_score=0.0,
            thresholds=thresholds,
        ).status
        == TriageStatus.INVALID_DETECTION
    )
    assert (
        triage_cell(
            valid_detection=True,
            quality=low_quality,
            confidence=1.0,
            margin=1.0,
            unknown_score=1.0,
            thresholds=thresholds,
        ).status
        == TriageStatus.INVALID_DETECTION
    )
    assert (
        triage_cell(
            valid_detection=True,
            quality=good_quality,
            confidence=0.99,
            margin=0.99,
            unknown_score=0.80,
            thresholds=thresholds,
        ).status
        == TriageStatus.POSSIBLE_UNKNOWN
    )
    assert (
        triage_cell(
            valid_detection=True,
            quality=good_quality,
            confidence=0.60,
            margin=0.99,
            unknown_score=0.10,
            thresholds=thresholds,
        ).status
        == TriageStatus.HUMAN_REVIEW
    )
    assert (
        triage_cell(
            valid_detection=True,
            quality=good_quality,
            confidence=0.95,
            margin=0.02,
            unknown_score=0.10,
            thresholds=thresholds,
        ).status
        == TriageStatus.HUMAN_REVIEW
    )
    assert (
        triage_cell(
            valid_detection=True,
            quality=good_quality,
            confidence=0.95,
            margin=0.30,
            unknown_score=0.10,
            thresholds=thresholds,
        ).status
        == TriageStatus.AUTO_ACCEPT
    )
    assert len(thresholds.stable_hash()) == 64


def test_detection_matching_tracks_tp_fp_and_missed() -> None:
    detections = [
        DetectionPrediction("d1", BoundingBox(0, 0, 10, 10), 0.9),
        DetectionPrediction("d2", BoundingBox(80, 80, 90, 90), 0.8),
    ]
    ground_truth = [
        GroundTruthBox("g1", BoundingBox(1, 1, 11, 11)),
        GroundTruthBox("g2", BoundingBox(40, 40, 50, 50)),
    ]

    matches = match_detections(detections, ground_truth, iou_threshold=0.5)
    statuses = [match.status for match in matches]

    assert statuses == ["TP_DETECTION", "FP_DETECTION", "MISSED_GT_WBC"]


def test_selective_metrics_known_unknown_tradeoff() -> None:
    records = [
        SelectiveRecord("known", TriageStatus.AUTO_ACCEPT, True),
        SelectiveRecord("known", TriageStatus.AUTO_ACCEPT, False),
        SelectiveRecord("known", TriageStatus.HUMAN_REVIEW, None),
        SelectiveRecord("unknown", TriageStatus.AUTO_ACCEPT, None),
        SelectiveRecord("unknown", TriageStatus.POSSIBLE_UNKNOWN, None),
    ]

    metrics = selective_metrics(records)

    assert metrics["coverage"] == 3 / 5
    assert metrics["selective_risk"] == 1 / 2
    assert metrics["known_review_rate"] == 1 / 3
    assert metrics["unknown_auto_accept_rate"] == 1 / 2
    assert metrics["unknown_possible_unknown_rate"] == 1 / 2


def test_json_output_schema_excludes_diagnostic_fields() -> None:
    detail = CellDetail(
        detection_id="d1",
        bbox=BoundingBox(1, 2, 20, 30),
        expanded_bbox=BoundingBox(0, 0, 22, 32),
        detection_confidence=0.91,
        predicted_class="lymphocyte",
        classification_confidence=0.88,
        second_class="monocyte",
        margin=0.40,
        unknown_score=0.12,
        quality_score=1.0,
        quality_flags=(),
        triage_status=TriageStatus.AUTO_ACCEPT,
        crop_checksum="abc",
    )
    record = ImageAnalysisResult("img1", (detail,)).to_json_record()

    serialized = json.dumps(record)
    assert not FORBIDDEN_OUTPUT_KEYS.intersection(record["detections"][0])
    for forbidden in FORBIDDEN_OUTPUT_KEYS:
        assert forbidden not in serialized


def test_review_overlay_changes_pixels() -> None:
    image = Image.new("RGB", (64, 64), color=(240, 240, 240))
    detail = CellDetail(
        detection_id="d1",
        bbox=BoundingBox(8, 8, 40, 40),
        expanded_bbox=BoundingBox(4, 4, 44, 44),
        detection_confidence=0.9,
        predicted_class="monocyte",
        classification_confidence=0.7,
        second_class="lymphocyte",
        margin=0.1,
        unknown_score=0.3,
        quality_score=1.0,
        quality_flags=(),
        triage_status=TriageStatus.HUMAN_REVIEW,
    )

    overlay = draw_review_overlay(image, [detail])

    assert overlay.size == image.size
    assert np.any(np.asarray(overlay) != np.asarray(image))


def _checkerboard_image(size: int) -> Image.Image:
    values = np.indices((size, size)).sum(axis=0) % 2
    array = (100 + values * 60).astype(np.uint8)
    rgb = np.stack([array, array, array], axis=2)
    return Image.fromarray(rgb, mode="RGB")
