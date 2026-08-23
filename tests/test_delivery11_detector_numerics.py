from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
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


def test_detector_guard_policy_uses_tiered_cadence() -> None:
    module = _load_delivery11_module()
    policy = module.DetectorGuardPolicy()

    assert policy.should_check_gradient(1, is_epoch_last_step=False)
    assert policy.should_check_gradient(100, is_epoch_last_step=False)
    assert not policy.should_check_gradient(101, is_epoch_last_step=False)
    assert policy.should_check_gradient(125, is_epoch_last_step=False)
    assert policy.should_check_gradient(101, is_epoch_last_step=True)
    assert policy.should_scan_parameters_after_step(1)
    assert policy.should_scan_parameters_after_step(10)
    assert policy.should_scan_parameters_after_step(100)
    assert not policy.should_scan_parameters_after_step(101)


def test_numerical_diagnostics_do_not_mutate_gradients_or_parameters() -> None:
    module = _load_delivery11_module()
    model = torch.nn.Linear(3, 2)
    inputs = torch.randn(4, 3)
    targets = torch.randn(4, 2)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    gradients_before = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    parameters_before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

    module._gradient_finite_stats(model)
    module._parameter_finite_stats(model)

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.detach(), parameters_before[name])
        assert parameter.grad is not None
        assert torch.equal(parameter.grad.detach(), gradients_before[name])


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


def test_checkpoint_finite_validation_reads_model_state_dict(tmp_path: Path) -> None:
    module = _load_delivery11_module()
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": {
                "finite": torch.tensor([1.0, 2.0]),
                "bad": torch.tensor([float("nan")]),
                "integer_buffer": torch.tensor([1], dtype=torch.int64),
            }
        },
        checkpoint_path,
    )

    stats = module._verify_checkpoint_finite_cpu(checkpoint_path)

    assert stats["parameter_tensors"] == 2
    assert stats["nonfinite_tensors"] == 1
    assert stats["nonfinite_values"] == 1


def test_rng_state_round_trips_python_numpy_and_torch() -> None:
    module = _load_delivery11_module()
    module._seed_everything(37)
    state = module._rng_state()
    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = torch.rand(3)

    random.random()
    np.random.rand()
    torch.rand(3)
    module._restore_rng_state(state)

    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)


def test_phase_timing_schema_contains_required_fields() -> None:
    module = _load_delivery11_module()
    timer = module.DetectorPhaseTimer()

    assert set(timer.__dict__) == {
        "train_seconds",
        "validation_inference_seconds",
        "validation_metric_seconds",
        "checkpoint_seconds",
        "finite_guard_seconds",
    }


def test_dataloader_worker_kwargs_do_not_change_sample_contents(
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
    baseline_image, baseline_target = dataset[0]
    kwargs = module._dataloader_kwargs(
        num_workers=4,
        device=torch.device("cuda"),
        collate_fn=module._detection_collate,
    )

    repeated_image, repeated_target = dataset[0]

    assert "persistent_workers" not in kwargs
    assert "prefetch_factor" not in kwargs
    assert torch.equal(repeated_image, baseline_image)
    assert torch.equal(repeated_target["boxes"], baseline_target["boxes"])
    assert torch.equal(repeated_target["labels"], baseline_target["labels"])


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
