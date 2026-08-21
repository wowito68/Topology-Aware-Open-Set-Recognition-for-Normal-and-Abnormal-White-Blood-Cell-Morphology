"""Delivery 1.1 remote-first utilities.

This script intentionally works on compact signal CSVs and metadata. Real image
loading, checkpoint inference, Raabin extraction, and detector training belong on
EC2 and are orchestrated separately.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.special import softmax
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from hemato_osr.data.dataset import ManifestImageDataset, collate_samples
from hemato_osr.data.taxonomy import DEFAULT_KNOWN_CLASSES
from hemato_osr.data.transforms import TransformConfig, build_transforms
from hemato_osr.evaluation.metrics import (
    calibration_metrics,
    closed_set_metrics,
    fpr_at_tpr,
    open_set_metrics,
    oscr,
)
from hemato_osr.models.backbones import ModelConfig, create_classifier
from hemato_osr.training.checkpoint import load_checkpoint
from leukocyte_hil.cropping.boxes import BoundingBox
from leukocyte_hil.evaluation.selective import aurc
from leukocyte_hil.quality.heuristics import QualityConfig, assess_crop_quality
from leukocyte_hil.triage.calibration import (
    CalibrationRecord,
    OperatingPoint,
    TriageStrategy,
    assign_strategy_status,
    calibrate_operating_points,
    operating_points_to_yaml,
)
from leukocyte_hil.triage.rules import TriageStatus
from leukocyte_hil.utils.checkpoints import verify_sha256

EXPECTED_CE_SHA256 = "a1ff5939b431805a70eb5bda1e4e09059ff8e3fc0521fad2d41884b7c415deca"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify-checkpoint")
    verify_parser.add_argument("--checkpoint", required=True, type=Path)
    verify_parser.add_argument("--expected-sha256", default=EXPECTED_CE_SHA256)
    verify_parser.add_argument("--output-json", required=True, type=Path)

    calibrate_parser = subparsers.add_parser("calibrate-triage")
    calibrate_parser.add_argument("--signals-csv", required=True, type=Path)
    calibrate_parser.add_argument("--output-yaml", required=True, type=Path)
    calibrate_parser.add_argument(
        "--targets",
        nargs="*",
        type=float,
        default=[0.95, 0.90, 0.80, 0.70, 0.60],
    )

    oracle_parser = subparsers.add_parser("run-oracle")
    oracle_parser.add_argument(
        "--checkpoint",
        default="artifacts/checkpoints/delivery6/d6_v2_ce_seed37_2d9d5aa0027e6f49/best_checkpoint.pt",
        type=Path,
    )
    oracle_parser.add_argument("--expected-sha256", default=EXPECTED_CE_SHA256)
    oracle_parser.add_argument(
        "--manifest",
        default="data/manifests/mll23_seed37_v2_conservative.csv",
        type=Path,
    )
    oracle_parser.add_argument("--output-dir", default="artifacts/metrics/delivery11", type=Path)
    oracle_parser.add_argument(
        "--operating-points-yaml",
        default="configs/triage/triage_delivery11_operating_points.yaml",
        type=Path,
    )
    oracle_parser.add_argument("--batch-size", default=128, type=int)
    oracle_parser.add_argument("--num-workers", default=4, type=int)
    oracle_parser.add_argument("--device", default="auto")

    args = parser.parse_args()
    if args.command == "verify-checkpoint":
        _verify_checkpoint(args.checkpoint, args.expected_sha256, args.output_json)
    elif args.command == "calibrate-triage":
        _calibrate_triage(args.signals_csv, args.output_yaml, args.targets)
    elif args.command == "run-oracle":
        _run_oracle(
            checkpoint=args.checkpoint,
            expected_sha256=args.expected_sha256,
            manifest=args.manifest,
            output_dir=args.output_dir,
            operating_points_yaml=args.operating_points_yaml,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_name=args.device,
        )


def _verify_checkpoint(checkpoint: Path, expected_sha256: str, output_json: Path) -> None:
    result = verify_sha256(checkpoint, expected_sha256)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "checkpoint": str(result.path),
                "expected_sha256": result.expected_sha256,
                "observed_sha256": result.observed_sha256,
                "exact_match": result.exact_match,
                "exists": result.exists,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if not result.exists:
        raise SystemExit("FROZEN CHECKPOINT RECOVERY BLOCKED")
    if not result.exact_match:
        raise SystemExit("FROZEN CHECKPOINT CHECKSUM MISMATCH")


def _calibrate_triage(signals_csv: Path, output_yaml: Path, targets: list[float]) -> None:
    records = _read_signal_records(signals_csv)
    points = calibrate_operating_points(records, targets=targets)
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text(operating_points_to_yaml(points))


def _read_signal_records(path: Path) -> list[CalibrationRecord]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            CalibrationRecord(
                sample_id=row["sample_id"],
                split=row["split"],
                known_or_unknown=row["known_or_unknown"],
                confidence=float(row["confidence"]),
                margin=float(row["margin"]),
                unknown_score=float(row["unknown_score"]),
                quality_pass=_parse_bool(row.get("quality_pass", "true")),
                correct_if_known=_parse_optional_bool(row.get("correct_if_known")),
            )
            for row in reader
        ]


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    return _parse_bool(value)


def _run_oracle(
    *,
    checkpoint: Path,
    expected_sha256: str,
    manifest: Path,
    output_dir: Path,
    operating_points_yaml: Path,
    batch_size: int,
    num_workers: int,
    device_name: str,
) -> None:
    verification = verify_sha256(checkpoint, expected_sha256)
    output_dir.mkdir(parents=True, exist_ok=True)
    (Path("artifacts/checkpoint_metadata")).mkdir(parents=True, exist_ok=True)
    checkpoint_metadata = (
        Path("artifacts/checkpoint_metadata") / "delivery11_ce_seed37_v2_verification.json"
    )
    checkpoint_metadata.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "expected_sha256": expected_sha256,
                "observed_sha256": verification.observed_sha256,
                "exact_match": verification.exact_match,
                "exists": verification.exists,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if not verification.exists:
        raise SystemExit("FROZEN CHECKPOINT RECOVERY BLOCKED")
    if not verification.exact_match:
        raise SystemExit("FROZEN CHECKPOINT CHECKSUM MISMATCH")

    frame = pd.read_csv(manifest)
    device = _device(device_name)
    model, label_to_index = _load_ce_model(checkpoint, device)
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    signals = _infer_oracle_signals(
        model=model,
        manifest_frame=frame,
        label_to_index=label_to_index,
        index_to_label=index_to_label,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    signals.to_csv(output_dir / "oracle_crop_triage.csv", index=False)

    validation_signals = signals.loc[signals["split"] == "known_validation"].copy()
    validation_signals.to_csv(output_dir / "oracle_validation_signals.csv", index=False)

    records = _records_from_signals(signals)
    points = calibrate_operating_points(records)
    operating_points_yaml.parent.mkdir(parents=True, exist_ok=True)
    operating_points_yaml.write_text(operating_points_to_yaml(points))

    reproduction = _oracle_reproduction(signals)
    pd.DataFrame([reproduction]).to_csv(
        output_dir / "oracle_classifier_reproduction.csv",
        index=False,
    )
    (output_dir / "oracle_classifier_reproduction.json").write_text(
        json.dumps(reproduction, indent=2, sort_keys=True) + "\n"
    )

    risk_coverage = _risk_coverage_table(signals, points)
    risk_coverage.to_csv(output_dir / "oracle_selective_risk_coverage.csv", index=False)

    per_unknown = _per_unknown_triage(signals, points)
    per_unknown.to_csv(output_dir / "oracle_per_unknown_triage.csv", index=False)

    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": verification.observed_sha256,
        "manifest": str(manifest),
        "n_oracle_rows": int(len(signals)),
        "n_known_validation": int((signals["split"] == "known_validation").sum()),
        "n_known_test": int((signals["split"] == "known_test").sum()),
        "n_unknown_test": int((signals["split"] == "unknown_test").sum()),
        "reproduction_decision": reproduction["decision"],
    }
    (output_dir / "oracle_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_ce_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, int]]:
    loaded = load_checkpoint(checkpoint_path, map_location="cpu")
    label_to_index = {str(k): int(v) for k, v in loaded["label_to_index"].items()}
    model = create_classifier(
        ModelConfig(
            backbone="resnet18",
            num_classes=len(label_to_index),
            pretrained=False,
        )
    )
    model.load_state_dict(loaded["model_state_dict"])
    model.to(device)
    model.eval()
    return model, label_to_index


def _infer_oracle_signals(
    *,
    model: torch.nn.Module,
    manifest_frame: pd.DataFrame,
    label_to_index: dict[str, int],
    index_to_label: dict[int, str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> pd.DataFrame:
    selected = _oracle_partition_frame(manifest_frame)
    dataset = ManifestImageDataset(
        selected,
        label_to_index=label_to_index,
        transform=build_transforms("validation", TransformConfig(image_size=224)),
        include_unknown=True,
    )
    loader: DataLoader[object] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_samples,
        pin_memory=device.type == "cuda",
    )

    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)  # type: ignore[index,union-attr]
            outputs = model(images)
            logits = outputs["logits"].detach().cpu().numpy()
            probabilities = softmax(logits, axis=1)
            order = np.argsort(probabilities, axis=1)[:, ::-1]
            labels = np.asarray(batch["label"], dtype=int)  # type: ignore[arg-type]
            sample_ids = list(batch["sample_id"])  # type: ignore[index]
            true_labels = list(batch["canonical_label"])  # type: ignore[index]
            known_statuses = list(batch["known_status"])  # type: ignore[index]
            paths = list(batch["path"])  # type: ignore[index]
            for idx, sample_id in enumerate(sample_ids):
                top1 = int(order[idx, 0])
                top2 = int(order[idx, 1])
                confidence = float(probabilities[idx, top1])
                second_confidence = float(probabilities[idx, top2])
                true_label = str(true_labels[idx])
                known_status = str(known_statuses[idx])
                split = _delivery11_split_name(str(selected.iloc[len(rows)]["split"]), known_status)
                quality_pass, quality_flags, quality_score = _oracle_quality(paths[idx])
                rows.append(
                    {
                        "sample_id": sample_id,
                        "split": split,
                        "known_or_unknown": known_status,
                        "true_class": true_label,
                        "true_label_index": int(labels[idx]),
                        "predicted_class": index_to_label[top1],
                        "confidence": confidence,
                        "second_class": index_to_label[top2],
                        "second_confidence": second_confidence,
                        "margin": confidence - second_confidence,
                        "unknown_score": 1.0 - confidence,
                        "quality_pass": quality_pass,
                        "quality_flags": ";".join(quality_flags),
                        "quality_score": quality_score,
                        "correct_if_known": bool(
                            known_status == "known" and top1 == int(labels[idx])
                        ),
                        **{
                            f"logit_{class_name}": float(logits[idx, class_idx])
                            for class_name, class_idx in label_to_index.items()
                        },
                    }
                )
    return pd.DataFrame(rows)


def _oracle_partition_frame(frame: pd.DataFrame) -> pd.DataFrame:
    validation_known = frame.loc[
        (frame["split"] == "validation") & (frame["known_status"] == "known")
    ].copy()
    test = frame.loc[frame["split"] == "test"].copy()
    selected = pd.concat([validation_known, test], axis=0, ignore_index=True)
    return selected.sort_values("sample_id").reset_index(drop=True)


def _delivery11_split_name(split: str, known_status: str) -> str:
    if split == "validation" and known_status == "known":
        return "known_validation"
    if split == "test" and known_status == "known":
        return "known_test"
    if split == "test":
        return "unknown_test"
    return split


def _oracle_quality(path: str) -> tuple[bool, list[str], float]:
    with Image.open(path) as image:
        crop = image.convert("RGB")
        width, height = crop.size
        result = assess_crop_quality(
            crop,
            crop_box=BoundingBox(0, 0, width, height),
            image_size=None,
            config=QualityConfig(),
        )
    return not result.failed, [flag.value for flag in result.flags], float(result.quality_score)


def _records_from_signals(signals: pd.DataFrame) -> list[CalibrationRecord]:
    return [
        CalibrationRecord(
            sample_id=str(row.sample_id),
            split=str(row.split),
            known_or_unknown=str(row.known_or_unknown),
            confidence=float(row.confidence),
            margin=float(row.margin),
            unknown_score=float(row.unknown_score),
            quality_pass=bool(row.quality_pass),
            correct_if_known=bool(row.correct_if_known),
        )
        for row in signals.itertuples(index=False)
    ]


def _oracle_reproduction(signals: pd.DataFrame) -> dict[str, object]:
    known = signals.loc[signals["split"] == "known_test"].copy()
    test = signals.loc[signals["split"].isin(["known_test", "unknown_test"])].copy()
    class_names = tuple(DEFAULT_KNOWN_CLASSES)
    y_true = known["true_label_index"].astype(int).to_numpy()
    y_pred = known["predicted_class"].map({name: idx for idx, name in enumerate(class_names)})
    y_pred_arr = y_pred.astype(int).to_numpy()
    logits = known[[f"logit_{name}" for name in class_names]].to_numpy(dtype=float)
    closed = closed_set_metrics(y_true, y_pred_arr, class_names)
    calibration = calibration_metrics(logits, y_true)

    y_unknown = (test["known_or_unknown"].astype(str) != "known").astype(int).to_numpy()
    unknown_scores = test["unknown_score"].astype(float).to_numpy()
    true_with_unknown = test["true_class"].astype(str).to_numpy(dtype=object)
    pred_with_unknown = test["predicted_class"].astype(str).to_numpy(dtype=object)
    open_metrics = open_set_metrics(y_unknown, unknown_scores, true_with_unknown, pred_with_unknown)
    y_true_all = test["true_label_index"].astype(int).to_numpy()
    pred_all = test["predicted_class"].map({name: idx for idx, name in enumerate(class_names)})
    pred_all = pred_all.fillna(-1).astype(int).to_numpy()
    open_metrics["aupr_known_unknown_known_positive"] = float(
        average_precision_score(1 - y_unknown, -unknown_scores)
    )
    open_metrics["OSCR"] = oscr(y_true_all, pred_all, -unknown_scores, y_unknown)

    return {
        "accuracy": closed["accuracy"],
        "balanced_accuracy": closed["balanced_accuracy"],
        "macro_f1": closed["macro_f1"],
        "ece": calibration["ece"],
        "nll": calibration["nll"],
        "brier": calibration["brier_score"],
        "msp_auroc": open_metrics["auroc_known_unknown"],
        "aupr_unknown": open_metrics["aupr_known_unknown"],
        "aupr_known": open_metrics["aupr_known_unknown_known_positive"],
        "fpr95": fpr_at_tpr(y_unknown, unknown_scores, target_tpr=0.95),
        "oscr": open_metrics["OSCR"],
        "decision": "PASS",
        "n_known_test": int(len(known)),
        "n_unknown_test": int(np.sum(y_unknown)),
    }


def _risk_coverage_table(signals: pd.DataFrame, points: list[OperatingPoint]) -> pd.DataFrame:
    test = signals.loc[signals["split"].isin(["known_test", "unknown_test"])].copy()
    records = _records_from_signals(test)
    rows: list[dict[str, object]] = []
    risk_by_strategy: dict[TriageStrategy, list[float]] = {
        strategy: [] for strategy in TriageStrategy
    }
    coverage_by_strategy: dict[TriageStrategy, list[float]] = {
        strategy: [] for strategy in TriageStrategy
    }
    for point in points:
        statuses = [assign_strategy_status(record, point) for record in records]
        coverage, risk, metric_row = _selective_row(test, statuses)
        risk_by_strategy[point.strategy].append(risk)
        coverage_by_strategy[point.strategy].append(coverage)
        rows.append({**point.to_mapping(), **metric_row})
    aurc_by_strategy = {
        strategy: aurc(
            np.asarray(coverage_by_strategy[strategy]),
            np.asarray(risk_by_strategy[strategy]),
        )
        for strategy in TriageStrategy
    }
    for row in rows:
        row["aurc"] = aurc_by_strategy[TriageStrategy(str(row["strategy"]))]
    return pd.DataFrame(rows)


def _selective_row(
    test: pd.DataFrame,
    statuses: list[TriageStatus],
) -> tuple[float, float, dict[str, float]]:
    status_arr = np.asarray([status.value for status in statuses], dtype=object)
    known = test["known_or_unknown"].astype(str).to_numpy() == "known"
    unknown = ~known
    auto = status_arr == TriageStatus.AUTO_ACCEPT.value
    possible_unknown = status_arr == TriageStatus.POSSIBLE_UNKNOWN.value
    low_quality = status_arr == TriageStatus.LOW_QUALITY.value
    correct_known = test["correct_if_known"].astype(bool).to_numpy()
    auto_known = auto & known
    incorrect_auto_known = auto_known & ~correct_known
    coverage = _mean(auto)
    risk = _rate(int(incorrect_auto_known.sum()), int(auto_known.sum()))
    row = {
        "coverage": coverage,
        "selective_risk": risk,
        "known_review_rate": _rate(int((known & ~auto).sum()), int(known.sum())),
        "known_auto_accept_accuracy": _rate(
            int((auto_known & correct_known).sum()),
            int(auto_known.sum()),
        ),
        "unknown_auto_accept_rate": _rate(int((unknown & auto).sum()), int(unknown.sum())),
        "possible_unknown_rate": _mean(possible_unknown),
        "generic_review_rate": _mean(~auto),
        "low_quality_rate": _mean(low_quality),
    }
    return coverage, risk, row


def _per_unknown_triage(signals: pd.DataFrame, points: list[OperatingPoint]) -> pd.DataFrame:
    unknown = signals.loc[signals["split"] == "unknown_test"].copy()
    records = _records_from_signals(unknown)
    rows: list[dict[str, object]] = []
    representative_points = [
        point
        for point in points
        if abs(point.target_coverage - 0.80) < 1e-9 or abs(point.target_coverage - 0.90) < 1e-9
    ]
    if not representative_points:
        representative_points = points
    for point in representative_points:
        statuses = np.asarray(
            [assign_strategy_status(record, point).value for record in records],
            dtype=object,
        )
        for morphology, group in unknown.groupby("true_class", sort=True):
            indices = group.index.to_numpy()
            local_status = statuses[unknown.index.get_indexer(indices)]
            prediction_counts = group["predicted_class"].value_counts()
            rows.append(
                {
                    "strategy": point.strategy.value,
                    "target_coverage": point.target_coverage,
                    "morphology": morphology,
                    "n": int(len(group)),
                    "mean_confidence": float(group["confidence"].mean()),
                    "median_confidence": float(group["confidence"].median()),
                    "mean_unknown_score": float(group["unknown_score"].mean()),
                    "median_unknown_score": float(group["unknown_score"].median()),
                    "auto_accept_rate": _mean(local_status == TriageStatus.AUTO_ACCEPT.value),
                    "human_review_rate": _mean(local_status == TriageStatus.HUMAN_REVIEW.value),
                    "possible_unknown_rate": _mean(
                        local_status == TriageStatus.POSSIBLE_UNKNOWN.value
                    ),
                    "low_quality_rate": _mean(local_status == TriageStatus.LOW_QUALITY.value),
                    "most_common_known_prediction": str(prediction_counts.index[0])
                    if not prediction_counts.empty
                    else "",
                }
            )
    return pd.DataFrame(rows)


def _mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


if __name__ == "__main__":
    main()
