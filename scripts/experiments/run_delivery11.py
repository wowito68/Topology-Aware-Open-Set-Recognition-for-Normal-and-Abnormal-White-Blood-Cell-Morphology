"""Delivery 1.1 remote-first utilities.

This script intentionally works on compact signal CSVs and metadata. Real image
loading, checkpoint inference, Raabin extraction, and detector training belong on
EC2 and are orchestrated separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.special import softmax
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

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

    materialize_parser = subparsers.add_parser("materialize-raabin-fullfield")
    materialize_parser.add_argument(
        "--archive",
        default="/opt/dlami/nvme/datasets/raabin_wbc/raw/WBCData.rar",
        type=Path,
    )
    materialize_parser.add_argument(
        "--output-root",
        default="/opt/dlami/nvme/datasets/raabin_wbc/fullfield",
        type=Path,
    )
    materialize_parser.add_argument(
        "--summary-json",
        default="artifacts/data_audit/delivery11/raabin_fullfield_materialization.json",
        type=Path,
    )

    manifest_parser = subparsers.add_parser("build-raabin-detection-manifest")
    manifest_parser.add_argument(
        "--materialized-root",
        default="/opt/dlami/nvme/datasets/raabin_wbc/fullfield",
        type=Path,
    )
    manifest_parser.add_argument(
        "--output-csv",
        default="data/manifests/delivery11_raabin_detection.csv",
        type=Path,
    )
    manifest_parser.add_argument(
        "--audit-json",
        default="artifacts/data_audit/delivery11/raabin_detection_manifest_audit.json",
        type=Path,
    )
    manifest_parser.add_argument("--seed", default=37, type=int)

    duplicate_parser = subparsers.add_parser("audit-raabin-duplicates")
    duplicate_parser.add_argument(
        "--manifest",
        default="data/manifests/delivery11_raabin_detection.csv",
        type=Path,
    )
    duplicate_parser.add_argument(
        "--output-json",
        default="artifacts/data_audit/delivery11/raabin_duplicate_audit.json",
        type=Path,
    )
    duplicate_parser.add_argument(
        "--output-csv",
        default="artifacts/data_audit/delivery11/raabin_duplicate_audit.csv",
        type=Path,
    )

    train_detector_parser = subparsers.add_parser("train-raabin-detector")
    train_detector_parser.add_argument(
        "--manifest",
        default="data/manifests/delivery11_raabin_detection.csv",
        type=Path,
    )
    train_detector_parser.add_argument(
        "--output-dir",
        default="artifacts/checkpoints/delivery11/faster_rcnn_wbc_candidate_v1",
        type=Path,
    )
    train_detector_parser.add_argument("--epochs", default=30, type=int)
    train_detector_parser.add_argument("--batch-size", default=2, type=int)
    train_detector_parser.add_argument("--num-workers", default=4, type=int)
    train_detector_parser.add_argument("--device", default="auto")
    train_detector_parser.add_argument("--seed", default=37, type=int)
    train_detector_parser.add_argument("--smoke-limit", default=0, type=int)

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
    elif args.command == "materialize-raabin-fullfield":
        _materialize_raabin_fullfield(args.archive, args.output_root, args.summary_json)
    elif args.command == "build-raabin-detection-manifest":
        _build_raabin_detection_manifest(
            materialized_root=args.materialized_root,
            output_csv=args.output_csv,
            audit_json=args.audit_json,
            seed=args.seed,
        )
    elif args.command == "audit-raabin-duplicates":
        _audit_raabin_duplicates(args.manifest, args.output_json, args.output_csv)
    elif args.command == "train-raabin-detector":
        _train_raabin_detector(
            manifest=args.manifest,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_name=args.device,
            seed=args.seed,
            smoke_limit=args.smoke_limit,
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


def _materialize_raabin_fullfield(archive: Path, output_root: Path, summary_json: Path) -> None:
    """Extract only Raabin full-field image/json ZIP payloads from the outer RAR."""

    if not archive.exists():
        raise SystemExit(f"Raabin archive missing: {archive}")
    output_root.mkdir(parents=True, exist_ok=True)
    listing = subprocess.check_output(["unrar", "lb", str(archive)], text=True)
    inner_archives = [
        line.strip()
        for line in listing.splitlines()
        if (
            line.strip().endswith(".zip")
            and (
                "/Index of WBC First_microscope/" in line
                or "/Index of WBC Second_microscope/" in line
            )
        )
    ]
    if not inner_archives:
        raise SystemExit("RAABIN FULL-FIELD MATERIALIZATION BLOCKED: no microscope ZIPs found")

    materialized: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="raabin-inner-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for inner in inner_archives:
            source = (
                "first_microscope"
                if "/Index of WBC First_microscope/" in inner
                else "second_microscope"
            )
            film_id = Path(inner).stem
            target_dir = output_root / source / film_id
            done_marker = target_dir / ".delivery11_extracted"
            if done_marker.exists():
                counts = _count_materialized_pairs(target_dir)
                materialized.append(
                    {
                        "inner_archive": inner,
                        "source": source,
                        "film_id": film_id,
                        "target_dir": str(target_dir),
                        "skipped_existing": True,
                        **counts,
                    }
                )
                continue

            target_dir.mkdir(parents=True, exist_ok=True)
            temp_zip = tmp_path / f"{source}_{film_id}.zip"
            with temp_zip.open("wb") as handle:
                subprocess.run(
                    ["unrar", "p", "-inul", str(archive), inner],
                    stdout=handle,
                    check=True,
                )
            _safe_extract_zip(temp_zip, target_dir)
            temp_zip.unlink(missing_ok=True)
            done_marker.write_text("delivery11 materialized\n")
            counts = _count_materialized_pairs(target_dir)
            materialized.append(
                {
                    "inner_archive": inner,
                    "source": source,
                    "film_id": film_id,
                    "target_dir": str(target_dir),
                    "skipped_existing": False,
                    **counts,
                }
            )

    summary = {
        "archive": str(archive),
        "output_root": str(output_root),
        "inner_archive_count": len(inner_archives),
        "materialized": materialized,
        "total_images": sum(int(str(row["image_count"])) for row in materialized),
        "total_jsons": sum(int(str(row["json_count"])) for row in materialized),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")


def _safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise SystemExit(f"Unsafe ZIP member path: {member.filename}")
            archive.extract(member, output_dir)


def _count_materialized_pairs(root: Path) -> dict[str, int]:
    return {
        "image_count": sum(
            1 for path in root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ),
        "json_count": sum(1 for path in root.rglob("*.json")),
    }


def _build_raabin_detection_manifest(
    *,
    materialized_root: Path,
    output_csv: Path,
    audit_json: Path,
    seed: int,
) -> None:
    rows: list[dict[str, object]] = []
    label_counts: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()
    invalid_boxes: list[dict[str, object]] = []
    image_groups: dict[str, str] = {}

    for json_path in sorted(materialized_root.rglob("jsons/*.json")):
        archive_root = json_path.parents[1]
        image_path = archive_root / "images" / f"{json_path.stem}.jpg"
        if not image_path.exists():
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        payload = json.loads(json_path.read_text(errors="replace"))
        source = archive_root.parent.name
        film_id = str(payload.get("Film ID") or archive_root.name)
        image_id = f"{source}/{film_id}/{json_path.stem}"
        image_groups[image_id] = film_id
        for key, value in sorted(payload.items()):
            if not key.startswith("Cell_") or not isinstance(value, dict):
                continue
            label = str(value.get("Label1") or value.get("Label2") or "").strip()
            label_counts[label] += 1
            bbox = _raabin_cell_bbox(value)
            if bbox is None:
                invalid_boxes.append({"image_id": image_id, "cell_key": key, "label": label})
                continue
            clipped = _clip_xyxy(bbox, width, height)
            if clipped is None:
                invalid_boxes.append({"image_id": image_id, "cell_key": key, "label": label})
                continue
            mapped, known_status, detector_eligible, exclusion_reason = _map_raabin_label(label)
            if not detector_eligible:
                excluded_counts[exclusion_reason] += 1
            rows.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "json_path": str(json_path),
                    "source": source,
                    "film_id": film_id,
                    "width": width,
                    "height": height,
                    "gt_cell_id": f"{image_id}/{key}",
                    "cell_key": key,
                    "raw_label1": label,
                    "raw_label2": str(value.get("Label2") or "").strip(),
                    "mapped_morphology": mapped,
                    "known_or_unknown": known_status,
                    "detector_eligible": detector_eligible,
                    "exclusion_reason": exclusion_reason,
                    "x_min": clipped[0],
                    "y_min": clipped[1],
                    "x_max": clipped[2],
                    "y_max": clipped[3],
                }
            )

    if not rows:
        raise SystemExit("RAABIN FULL-FIELD DETECTION GATE: FAIL")

    split_by_film = _deterministic_group_splits(sorted(set(image_groups.values())), seed=seed)
    for row in rows:
        row["split"] = split_by_film[str(row["film_id"])]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_csv, index=False)

    eligible = frame.loc[frame["detector_eligible"].astype(bool)].copy()
    image_split_counts = (
        eligible.drop_duplicates("image_id").groupby("split")["image_id"].count().to_dict()
    )
    box_split_counts = eligible.groupby("split")["gt_cell_id"].count().to_dict()
    audit = {
        "materialized_root": str(materialized_root),
        "output_csv": str(output_csv),
        "seed": seed,
        "row_count_all_annotations": int(len(frame)),
        "row_count_detector_eligible": int(len(eligible)),
        "image_count_detector_eligible": int(eligible["image_id"].nunique()),
        "film_id_count": int(frame["film_id"].nunique()),
        "split_unit": "film_id",
        "split_limitation": (
            "Raabin patient identifiers were not established; Film ID is used as the "
            "deterministic grouping unit."
        ),
        "image_split_counts": {str(k): int(v) for k, v in image_split_counts.items()},
        "box_split_counts": {str(k): int(v) for k, v in box_split_counts.items()},
        "raw_label_counts": dict(label_counts.most_common()),
        "excluded_counts": dict(excluded_counts.most_common()),
        "mapped_morphology_counts": eligible["mapped_morphology"].value_counts().to_dict(),
        "known_unknown_counts": eligible["known_or_unknown"].value_counts().to_dict(),
        "invalid_box_count": len(invalid_boxes),
        "invalid_box_samples": invalid_boxes[:25],
        "known_taxonomy": list(DEFAULT_KNOWN_CLASSES),
        "detector_class": "LEUKOCYTE_CANDIDATE",
    }
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    audit_json.write_text(json.dumps(audit, indent=2) + "\n")


def _raabin_cell_bbox(cell: dict[str, object]) -> tuple[float, float, float, float] | None:
    try:
        return (
            float(str(cell["x1"])),
            float(str(cell["y1"])),
            float(str(cell["x2"])),
            float(str(cell["y2"])),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _clip_xyxy(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = bbox
    clipped = BoundingBox(
        x_min=min(max(0.0, x1), float(width)),
        y_min=min(max(0.0, y1), float(height)),
        x_max=min(max(0.0, x2), float(width)),
        y_max=min(max(0.0, y2), float(height)),
    )
    if not clipped.is_valid:
        return None
    return (clipped.x_min, clipped.y_min, clipped.x_max, clipped.y_max)


def _map_raabin_label(label: str) -> tuple[str, str, bool, str]:
    normalized = label.strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    if normalized in {"artifact", "artefact", "burst", ""}:
        return "", "excluded", False, normalized or "empty_label"
    if normalized == "basophil":
        return "basophil", "known", True, ""
    if normalized == "eosinophil":
        return "eosinophil", "known", True, ""
    if normalized in {"small lymph", "lymphocyte", "lymph"}:
        return "lymphocyte", "known", True, ""
    if normalized == "monocyte":
        return "monocyte", "known", True, ""
    if normalized == "neutrophil":
        return "neutrophil_segmented", "known", True, ""
    return "unknown", "unknown", True, ""


def _deterministic_group_splits(groups: list[str], *, seed: int) -> dict[str, str]:
    keyed = sorted(
        (hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(), group) for group in groups
    )
    n = len(keyed)
    train_end = int(round(0.70 * n))
    val_end = train_end + int(round(0.15 * n))
    split_by_group: dict[str, str] = {}
    for idx, (_, group) in enumerate(keyed):
        if idx < train_end:
            split = "train"
        elif idx < val_end:
            split = "validation"
        else:
            split = "test"
        split_by_group[group] = split
    return split_by_group


def _audit_raabin_duplicates(manifest: Path, output_json: Path, output_csv: Path) -> None:
    frame = pd.read_csv(manifest)
    images = (
        frame.loc[frame["detector_eligible"].astype(bool), ["image_id", "image_path", "split"]]
        .drop_duplicates("image_id")
        .sort_values("image_id")
    )
    rows: list[dict[str, object]] = []
    for row in images.itertuples(index=False):
        path = Path(str(row.image_path))
        sha = _file_sha256(path)
        ahash = _average_hash(path)
        rows.append(
            {
                "image_id": str(row.image_id),
                "image_path": str(path),
                "split": str(row.split),
                "sha256": sha,
                "average_hash": ahash,
            }
        )

    audit_frame = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    audit_frame.to_csv(output_csv, index=False)

    exact_cross_split = _cross_split_duplicates(audit_frame, "sha256")
    phash_candidates = _average_hash_candidates(audit_frame, max_hamming=4)
    summary = {
        "manifest": str(manifest),
        "image_count": int(len(audit_frame)),
        "exact_cross_split_duplicate_count": len(exact_cross_split),
        "exact_cross_split_duplicates": exact_cross_split[:25],
        "average_hash_candidate_count_hamming_le_4": len(phash_candidates),
        "average_hash_candidates_hamming_le_4": phash_candidates[:50],
        "verdict": "PASS" if not exact_cross_split else "FAIL_EXACT_CROSS_SPLIT_DUPLICATES",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n")
    if exact_cross_split:
        raise SystemExit("RAABIN DUPLICATE AUDIT FAIL: exact cross-split duplicate")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _average_hash(path: Path) -> str:
    with Image.open(path) as image:
        image.draft("L", (64, 64))
        gray = image.convert("L")
        gray.thumbnail((8, 8), Image.Resampling.BILINEAR)
        if gray.size != (8, 8):
            gray = gray.resize((8, 8), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32)
    bits = arr >= float(arr.mean())
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _cross_split_duplicates(frame: pd.DataFrame, column: str) -> list[dict[str, object]]:
    duplicates: list[dict[str, object]] = []
    for value, group in frame.groupby(column):
        splits = sorted(set(str(item) for item in group["split"]))
        if len(splits) <= 1:
            continue
        duplicates.append(
            {
                column: str(value),
                "splits": splits,
                "image_ids": group["image_id"].astype(str).head(10).tolist(),
            }
        )
    return duplicates


def _average_hash_candidates(frame: pd.DataFrame, *, max_hamming: int) -> list[dict[str, object]]:
    split_hashes: dict[str, dict[int, list[str]]] = {}
    for split, group in frame.groupby("split"):
        hashes: dict[int, list[str]] = defaultdict(list)
        for row in group[["image_id", "average_hash"]].itertuples(index=False):
            hashes[int(str(row.average_hash), 16)].append(str(row.image_id))
        split_hashes[str(split)] = hashes

    candidates: list[dict[str, object]] = []
    split_pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    for left_split, right_split in split_pairs:
        left_hashes = split_hashes.get(left_split, {})
        right_hashes = split_hashes.get(right_split, {})
        tree = _BKTree(list(right_hashes))
        for left_hash, left_ids in left_hashes.items():
            for right_hash, distance in tree.query(left_hash, max_hamming):
                right_ids = right_hashes[right_hash]
                for left_id in left_ids[:3]:
                    for right_id in right_ids[:3]:
                        candidates.append(
                            {
                                "left_split": left_split,
                                "left_image_id": left_id,
                                "right_split": right_split,
                                "right_image_id": right_id,
                                "hamming": distance,
                            }
                        )
                        if len(candidates) >= 10000:
                            return candidates
    return candidates


class _BKTree:
    def __init__(self, values: list[int]) -> None:
        self.root: _BKNode | None = None
        for value in values:
            self.add(value)

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = _BKNode(value=value)
            return
        node = self.root
        while True:
            distance = _hamming64(value, node.value)
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(value=value)
                return
            node = child

    def query(self, value: int, max_distance: int) -> list[tuple[int, int]]:
        if self.root is None:
            return []
        matches: list[tuple[int, int]] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = _hamming64(value, node.value)
            if distance <= max_distance:
                matches.append((node.value, distance))
            lower = distance - max_distance
            upper = distance + max_distance
            stack.extend(child for edge, child in node.children.items() if lower <= edge <= upper)
        return matches


@dataclass
class _BKNode:
    value: int
    children: dict[int, _BKNode] = field(default_factory=dict)


def _hamming64(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class RaabinDetectionDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    def __init__(self, frame: pd.DataFrame, *, split: str, limit: int = 0) -> None:
        selected = frame.loc[
            (frame["split"] == split) & (frame["detector_eligible"].astype(bool))
        ].copy()
        grouped: list[tuple[str, pd.DataFrame]] = list(selected.groupby("image_id", sort=True))
        if limit > 0:
            grouped = grouped[:limit]
        self.grouped = grouped

    def __len__(self) -> int:
        return len(self.grouped)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, group = self.grouped[index]
        image_path = Path(str(group.iloc[0]["image_path"]))
        with Image.open(image_path) as image:
            image_tensor = _pil_to_tensor(image.convert("RGB"))
        boxes = torch.as_tensor(
            group[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=np.float32).copy(),
            dtype=torch.float32,
        )
        labels = torch.ones((len(group),), dtype=torch.int64)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": torch.zeros((len(group),), dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        return image_tensor, target


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _train_raabin_detector(
    *,
    manifest: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    device_name: str,
    seed: int,
    smoke_limit: int,
) -> None:
    _seed_everything(seed)
    frame = pd.read_csv(manifest)
    train_dataset = RaabinDetectionDataset(frame, split="train", limit=smoke_limit)
    val_limit = max(1, min(smoke_limit, 64)) if smoke_limit > 0 else 0
    val_dataset = RaabinDetectionDataset(frame, split="validation", limit=val_limit)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise SystemExit("RAABIN DETECTOR TRAINING BLOCKED: empty train/validation split")

    device = _device(device_name)
    model = _create_faster_rcnn_detector().to(device)
    optimizer = torch.optim.SGD(
        [param for param in model.parameters() if param.requires_grad],
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_detection_collate,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_detection_collate,
        pin_memory=device.type == "cuda",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = Path("artifacts/metrics/delivery11/detector_training_history_recovery.csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, object]] = []
    best_recall = -1.0
    best_path = output_dir / "best_checkpoint.pt"
    run_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        losses: list[float] = []
        loss_component_totals: dict[str, list[float]] = defaultdict(list)
        for batch_index, (images, targets) in enumerate(train_loader):
            images = [image.to(device) for image in images]
            targets = [
                {key: value.to(device) for key, value in target.items()} for target in targets
            ]
            loss_dict = model(images, targets)
            loss = sum(value for value in loss_dict.values())
            loss_values = {key: float(value.detach().cpu()) for key, value in loss_dict.items()}
            _fail_if_nonfinite_loss(
                loss_dict=loss_dict,
                total_loss=loss,
                output_dir=output_dir,
                epoch=epoch,
                batch_index=batch_index,
                targets=targets,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_stats = _gradient_finite_stats(model)
            if int(str(gradient_stats["nonfinite_gradient_values"])) > 0:
                _write_detection_nonfinite_event(
                    output_dir=output_dir,
                    phase="BACKWARD_GRADIENT",
                    epoch=epoch,
                    batch_index=batch_index,
                    targets=targets,
                    loss_values=loss_values,
                    total_loss=float(loss.detach().cpu()),
                    gradient_stats=gradient_stats,
                )
                raise FloatingPointError("Non-finite Faster R-CNN gradient detected")
            optimizer.step()
            parameter_stats = _parameter_finite_stats(model)
            if int(str(parameter_stats["nonfinite_values"])) > 0:
                _write_detection_nonfinite_event(
                    output_dir=output_dir,
                    phase="OPTIMIZER_STEP",
                    epoch=epoch,
                    batch_index=batch_index,
                    targets=targets,
                    loss_values=loss_values,
                    total_loss=float(loss.detach().cpu()),
                    gradient_stats=gradient_stats,
                    parameter_stats=parameter_stats,
                )
                raise FloatingPointError("Non-finite Faster R-CNN parameter detected")
            losses.append(float(loss.detach().cpu()))
            for key, value in loss_values.items():
                loss_component_totals[key].append(value)

        val_metrics = _evaluate_detector(model, val_loader, device=device, score_threshold=0.05)
        parameter_stats = _parameter_finite_stats(model)
        epoch_seconds = time.perf_counter() - epoch_start
        cumulative_seconds = time.perf_counter() - run_start
        row: dict[str, object] = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            **{
                f"train_{key}": float(np.mean(values)) if values else float("nan")
                for key, values in sorted(loss_component_totals.items())
            },
            "finite_state": "PASS",
            "finite_losses": True,
            "finite_gradients": True,
            "finite_parameters": int(str(parameter_stats["nonfinite_values"])) == 0,
            "checkpoint_nonfinite_tensors": parameter_stats["nonfinite_tensors"],
            "checkpoint_nonfinite_values": parameter_stats["nonfinite_values"],
            "epoch_seconds": epoch_seconds,
            "cumulative_seconds": cumulative_seconds,
            **val_metrics,
        }
        metrics_rows.append(row)
        pd.DataFrame(metrics_rows).to_csv(output_dir / "training_metrics.csv", index=False)
        pd.DataFrame(metrics_rows).to_csv(history_path, index=False)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "seed": seed,
            "detector": "torchvision_faster_rcnn_resnet50_fpn",
            "score_threshold": 0.05,
            "validation_metrics": val_metrics,
        }
        torch.save(checkpoint, output_dir / f"epoch_{epoch:03d}.pt")
        if float(val_metrics["recall"]) > best_recall:
            best_recall = float(val_metrics["recall"])
            torch.save(checkpoint, best_path)

    _freeze_detector_threshold(
        best_path,
        frame,
        output_dir,
        device=device,
        num_workers=num_workers,
        limit=val_limit,
    )


def _fail_if_nonfinite_loss(
    *,
    loss_dict: dict[str, torch.Tensor],
    total_loss: torch.Tensor,
    output_dir: Path,
    epoch: int,
    batch_index: int,
    targets: list[dict[str, torch.Tensor]],
) -> None:
    loss_values = {key: float(value.detach().cpu()) for key, value in loss_dict.items()}
    component_finite = {
        key: bool(torch.isfinite(value).all().item()) for key, value in loss_dict.items()
    }
    total_finite = bool(torch.isfinite(total_loss).all().item())
    if all(component_finite.values()) and total_finite:
        return
    _write_detection_nonfinite_event(
        output_dir=output_dir,
        phase="FORWARD_LOSS",
        epoch=epoch,
        batch_index=batch_index,
        targets=targets,
        loss_values=loss_values,
        total_loss=float(total_loss.detach().cpu()),
        component_finite=component_finite,
    )
    raise FloatingPointError("Non-finite Faster R-CNN loss detected")


def _write_detection_nonfinite_event(
    *,
    output_dir: Path,
    phase: str,
    epoch: int,
    batch_index: int,
    targets: list[dict[str, torch.Tensor]],
    loss_values: dict[str, float],
    total_loss: float,
    component_finite: dict[str, bool] | None = None,
    gradient_stats: dict[str, object] | None = None,
    parameter_stats: dict[str, object] | None = None,
) -> None:
    event = {
        "phase": phase,
        "epoch": epoch,
        "batch_index": batch_index,
        "target_summaries": [_target_summary(target) for target in targets],
        "loss_values": loss_values,
        "total_loss": total_loss,
        "component_finite": component_finite,
        "gradient_stats": gradient_stats,
        "parameter_stats": parameter_stats,
        "amp_used": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "detector_first_nonfinite_event.json").write_text(
        json.dumps(event, indent=2) + "\n"
    )
    metrics_dir = Path("artifacts/metrics/delivery11")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "detector_first_nonfinite_event.json").write_text(
        json.dumps(event, indent=2) + "\n"
    )


def _target_summary(target: dict[str, torch.Tensor]) -> dict[str, object]:
    boxes = target["boxes"].detach().cpu()
    labels = target["labels"].detach().cpu()
    image_id = target["image_id"].detach().cpu().tolist()
    return {
        "image_id": image_id,
        "box_count": int(len(boxes)),
        "boxes_finite": bool(torch.isfinite(boxes).all().item()),
        "labels": labels.tolist(),
        "boxes": boxes[:20].tolist(),
    }


def _gradient_finite_stats(model: torch.nn.Module) -> dict[str, object]:
    total_sq = 0.0
    max_norm = 0.0
    nonfinite_values = 0
    examples: list[dict[str, object]] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        finite = torch.isfinite(gradient)
        bad = int((~finite).sum().item())
        if bad:
            nonfinite_values += bad
            if len(examples) < 10:
                examples.append({"name": name, "nonfinite_values": bad, "numel": gradient.numel()})
        if finite.any():
            norm = float(torch.linalg.vector_norm(gradient[finite]).cpu())
            total_sq += norm * norm
            max_norm = max(max_norm, norm)
    return {
        "global_grad_norm": float(total_sq**0.5),
        "max_grad_norm": max_norm,
        "nonfinite_gradient_values": nonfinite_values,
        "examples": examples,
    }


def _parameter_finite_stats(model: torch.nn.Module) -> dict[str, object]:
    tensors = 0
    nonfinite_tensors = 0
    nonfinite_values = 0
    total_values = 0
    examples: list[dict[str, object]] = []
    for name, tensor in model.state_dict().items():
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            continue
        tensors += 1
        total_values += tensor.numel()
        bad = int((~torch.isfinite(tensor)).sum().item())
        if bad:
            nonfinite_tensors += 1
            nonfinite_values += bad
            if len(examples) < 10:
                examples.append({"name": name, "nonfinite_values": bad, "numel": tensor.numel()})
    return {
        "parameter_tensors": tensors,
        "nonfinite_tensors": nonfinite_tensors,
        "nonfinite_values": nonfinite_values,
        "total_values": total_values,
        "examples": examples,
    }


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _create_faster_rcnn_detector() -> torch.nn.Module:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    return fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=2)


def _detection_collate(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def _evaluate_detector(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    score_threshold: float,
) -> dict[str, float]:
    from leukocyte_hil.cropping.boxes import BoundingBox
    from leukocyte_hil.detection.matching import (
        DetectionPrediction,
        GroundTruthBox,
        match_detections,
    )

    model.eval()
    tp = fp = missed = annotated = matched = 0
    predictions = 0
    image_index = 0
    ap_predictions: list[tuple[str, float, BoundingBox]] = []
    ap_ground_truth: dict[str, list[BoundingBox]] = {}
    with torch.no_grad():
        for images, targets in loader:
            outputs = model([image.to(device) for image in images])
            for output, target in zip(outputs, targets, strict=True):
                image_key = f"image_{image_index}"
                image_index += 1
                output_boxes = [
                    BoundingBox(*[float(value) for value in box.tolist()])
                    for box in output["boxes"].cpu()
                ]
                output_scores = [float(score) for score in output["scores"].cpu()]
                detections = [
                    DetectionPrediction(
                        detection_id=f"det_{idx}",
                        bbox=box,
                        confidence=score,
                    )
                    for idx, (box, score) in enumerate(
                        zip(output_boxes, output_scores, strict=True)
                    )
                    if score >= score_threshold
                ]
                gt = [
                    GroundTruthBox(
                        gt_cell_id=f"gt_{idx}",
                        bbox=BoundingBox(*[float(value) for value in box.tolist()]),
                    )
                    for idx, box in enumerate(target["boxes"].cpu())
                ]
                ap_predictions.extend(
                    (image_key, score, box)
                    for box, score in zip(output_boxes, output_scores, strict=True)
                    if box.is_valid
                )
                ap_ground_truth[image_key] = [item.bbox for item in gt]
                matches = match_detections(detections, gt, iou_threshold=0.5)
                tp += sum(match.status == "TP_DETECTION" for match in matches)
                fp += sum(match.status == "FP_DETECTION" for match in matches)
                missed += sum(match.status == "MISSED_GT_WBC" for match in matches)
                annotated += len(gt)
                matched += sum(match.status == "TP_DETECTION" for match in matches)
                predictions += len(detections)
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, annotated)
    map50 = _average_precision_for_iou(
        predictions=ap_predictions,
        ground_truth=ap_ground_truth,
        iou_threshold=0.5,
    )
    map50_95 = float(
        np.mean(
            [
                _average_precision_for_iou(
                    predictions=ap_predictions,
                    ground_truth=ap_ground_truth,
                    iou_threshold=threshold,
                )
                for threshold in np.arange(0.5, 1.0, 0.05)
            ]
        )
    )
    return {
        "precision": precision,
        "recall": recall,
        "annotated_wbcs": float(annotated),
        "predictions": float(predictions),
        "matched": float(matched),
        "missed": float(missed),
        "false_positives": float(fp),
        "map50": map50,
        "map50_95": map50_95,
    }


def _average_precision_for_iou(
    *,
    predictions: list[tuple[str, float, BoundingBox]],
    ground_truth: dict[str, list[BoundingBox]],
    iou_threshold: float,
) -> float:
    from leukocyte_hil.detection.matching import bbox_iou

    total_gt = sum(len(items) for items in ground_truth.values())
    if total_gt == 0:
        return 0.0

    matched_gt: dict[str, set[int]] = defaultdict(set)
    tp_values: list[float] = []
    fp_values: list[float] = []
    for image_key, _, prediction_box in sorted(
        predictions,
        key=lambda item: item[1],
        reverse=True,
    ):
        gt_boxes = ground_truth.get(image_key, [])
        candidates = [
            (bbox_iou(prediction_box, gt_box), gt_index)
            for gt_index, gt_box in enumerate(gt_boxes)
            if gt_index not in matched_gt[image_key]
        ]
        best_iou, best_index = max(candidates, key=lambda item: item[0], default=(0.0, -1))
        if best_index >= 0 and best_iou >= iou_threshold:
            matched_gt[image_key].add(best_index)
            tp_values.append(1.0)
            fp_values.append(0.0)
        else:
            tp_values.append(0.0)
            fp_values.append(1.0)

    if not tp_values:
        return 0.0

    tp_cumulative = np.cumsum(np.asarray(tp_values, dtype=np.float64))
    fp_cumulative = np.cumsum(np.asarray(fp_values, dtype=np.float64))
    recall = tp_cumulative / float(total_gt)
    precision = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1e-12)
    recall_points = np.concatenate(([0.0], recall, [1.0]))
    precision_points = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(precision_points) - 1, 0, -1):
        precision_points[index - 1] = max(precision_points[index - 1], precision_points[index])
    changed = np.where(recall_points[1:] != recall_points[:-1])[0]
    return float(
        np.sum(
            (recall_points[changed + 1] - recall_points[changed]) * precision_points[changed + 1]
        )
    )


def _freeze_detector_threshold(
    checkpoint_path: Path,
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    device: torch.device,
    num_workers: int,
    limit: int,
) -> None:
    model = _create_faster_rcnn_detector().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_dataset = RaabinDetectionDataset(frame, split="validation", limit=limit)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_detection_collate,
        pin_memory=device.type == "cuda",
    )
    candidates = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    rows = []
    for threshold in candidates:
        metrics = _evaluate_detector(model, val_loader, device=device, score_threshold=threshold)
        precision = float(metrics["precision"])
        recall = float(metrics["recall"])
        f1 = (
            (2.0 * precision * recall) / (precision + recall)
            if precision > 0 and recall > 0
            else 0.0
        )
        rows.append({"score_threshold": threshold, "f1": f1, **metrics})
    threshold_frame = pd.DataFrame(rows)
    threshold_frame.to_csv(output_dir / "validation_threshold_sweep.csv", index=False)
    best = threshold_frame.sort_values(["f1", "recall"], ascending=False).iloc[0].to_dict()
    (output_dir / "detector_threshold.json").write_text(json.dumps(best, indent=2) + "\n")


if __name__ == "__main__":
    main()
