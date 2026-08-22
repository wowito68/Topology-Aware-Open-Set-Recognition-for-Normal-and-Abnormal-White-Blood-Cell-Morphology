from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from leukocyte_hil.cropping.boxes import BoundingBox
from leukocyte_hil.detection.matching import DetectionPrediction, GroundTruthBox, match_detections


def _load_delivery11_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts/experiments/run_delivery11.py"
    spec = importlib.util.spec_from_file_location("run_delivery11", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _target() -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.tensor([[1.0, 2.0, 8.0, 9.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
        "image_id": torch.tensor([37], dtype=torch.int64),
    }


def test_nonfinite_loss_writes_first_event_and_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_delivery11_module()
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "detector_run"

    with pytest.raises(FloatingPointError, match="loss"):
        module._fail_if_nonfinite_loss(
            loss_dict={
                "loss_classifier": torch.tensor(float("nan")),
                "loss_box_reg": torch.tensor(0.1),
            },
            total_loss=torch.tensor(float("nan")),
            output_dir=output_dir,
            epoch=1,
            batch_index=4,
            targets=[_target()],
        )

    event = (output_dir / "detector_first_nonfinite_event.json").read_text()
    mirrored_event = (
        tmp_path / "artifacts/metrics/delivery11/detector_first_nonfinite_event.json"
    ).read_text()

    assert '"phase": "FORWARD_LOSS"' in event
    assert '"epoch": 1' in event
    assert '"batch_index": 4' in event
    assert '"loss_classifier": false' in event
    assert event == mirrored_event


def test_gradient_finite_stats_counts_nonfinite_gradients() -> None:
    module = _load_delivery11_module()
    model = torch.nn.Linear(2, 1)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    model.weight.grad[0, 1] = float("inf")

    stats = module._gradient_finite_stats(model)

    assert stats["nonfinite_gradient_values"] == 1
    assert stats["examples"][0]["name"] == "weight"
    assert stats["global_grad_norm"] > 0.0


def test_parameter_finite_stats_counts_nonfinite_parameters() -> None:
    module = _load_delivery11_module()
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.bias[0] = float("nan")

    stats = module._parameter_finite_stats(model)

    assert stats["parameter_tensors"] == 2
    assert stats["nonfinite_tensors"] == 1
    assert stats["nonfinite_values"] == 1
    assert stats["examples"][0]["name"] == "bias"


def test_raabin_detection_dataset_uses_finite_xyxy_targets(
    tmp_path: Path,
) -> None:
    module = _load_delivery11_module()
    image_path = tmp_path / "field.jpg"
    Image.new("RGB", (16, 12), color=(32, 64, 96)).save(image_path)
    frame = pd.DataFrame(
        [
            {
                "split": "train",
                "detector_eligible": True,
                "image_id": "film_a/image_001",
                "image_path": str(image_path),
                "x_min": 2.0,
                "y_min": 3.0,
                "x_max": 11.0,
                "y_max": 10.0,
            }
        ]
    )

    dataset = module.RaabinDetectionDataset(frame, split="train")
    image, target = dataset[0]

    assert torch.isfinite(image).all()
    assert image.shape == (3, 12, 16)
    assert image.dtype == torch.float32
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0
    assert torch.isfinite(target["boxes"]).all()
    assert target["boxes"].tolist() == [[2.0, 3.0, 11.0, 10.0]]
    assert target["area"].tolist() == [63.0]
    assert target["labels"].tolist() == [1]


def test_raabin_box_clipping_rejects_nonpositive_area() -> None:
    module = _load_delivery11_module()

    assert module._clip_xyxy((-5.0, 1.0, 8.0, 9.0), width=16, height=12) == (
        0.0,
        1.0,
        8.0,
        9.0,
    )
    assert module._clip_xyxy((3.0, 4.0, 3.0, 9.0), width=16, height=12) is None
    assert module._clip_xyxy((9.0, 4.0, 3.0, 9.0), width=16, height=12) is None


def test_detection_matching_is_one_to_one_for_duplicate_predictions() -> None:
    detections = [
        DetectionPrediction("high", BoundingBox(0.0, 0.0, 10.0, 10.0), 0.95),
        DetectionPrediction("duplicate", BoundingBox(0.0, 0.0, 10.0, 10.0), 0.90),
    ]
    ground_truth = [GroundTruthBox("gt", BoundingBox(0.0, 0.0, 10.0, 10.0))]

    matches = match_detections(detections, ground_truth, iou_threshold=0.5)

    assert [match.status for match in matches] == ["TP_DETECTION", "FP_DETECTION"]
    assert [match.gt_cell_id for match in matches] == ["gt", None]


def test_detector_average_precision_uses_one_to_one_matching() -> None:
    module = _load_delivery11_module()
    predictions = [
        ("image_a", 0.95, BoundingBox(0.0, 0.0, 10.0, 10.0)),
        ("image_a", 0.90, BoundingBox(0.0, 0.0, 10.0, 10.0)),
    ]
    ground_truth = {"image_a": [BoundingBox(0.0, 0.0, 10.0, 10.0)]}

    ap = module._average_precision_for_iou(
        predictions=predictions,
        ground_truth=ground_truth,
        iou_threshold=0.5,
    )

    assert ap == pytest.approx(1.0)
