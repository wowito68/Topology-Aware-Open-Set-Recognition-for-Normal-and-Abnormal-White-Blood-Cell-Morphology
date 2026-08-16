"""Evaluation artifact writer for open-set experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix

from hemato_osr.evaluation.metrics import (
    calibration_metrics,
    closed_set_metrics,
    open_set_metrics,
    oscr,
)
from hemato_osr.openset.scorers import create_scorer
from hemato_osr.openset.thresholds import apply_threshold, calibrate_strict_open_set
from hemato_osr.utils.config import save_config
from hemato_osr.utils.tracking import environment_metadata, write_json


@dataclass(frozen=True)
class OpenSetEvaluationConfig:
    """Open-set evaluation configuration."""

    embeddings_path: Path
    output_dir: Path
    open_set_method: str = "msp"
    target_known_recall: float = 0.95


def _load_embeddings(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _save_figures(y_unknown: np.ndarray, scores: np.ndarray, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    RocCurveDisplay.from_predictions(y_unknown, scores)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_known_unknown.png", dpi=160)
    plt.close()
    PrecisionRecallDisplay.from_predictions(y_unknown, scores)
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_known_unknown.png", dpi=160)
    plt.close()


def _save_confusion_png(matrix: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(5, len(matrix.columns) * 0.6), max(4, len(matrix) * 0.6)))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(matrix.columns)), labels=matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), labels=matrix.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_per_class(metrics: dict[str, object], output_path: Path) -> None:
    rows = []
    per_class = metrics.get("per_class", {})
    if isinstance(per_class, dict):
        for class_name, values in per_class.items():
            if isinstance(values, dict):
                rows.append({"class": class_name, **values})
    pd.DataFrame(rows).to_csv(output_path, index=False)


def evaluate_open_set(config: OpenSetEvaluationConfig) -> Path:
    """Evaluate open-set scores and write standard artifacts."""

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = _load_embeddings(config.embeddings_path)
    logits = arrays["logits"].astype(float)
    embeddings = arrays["embedding"].astype(float)
    true_label = arrays["true_label"].astype(str)
    known_status = arrays["known_status"].astype(str)
    split = arrays["split"].astype(str)
    prediction = arrays["prediction"].astype(str)
    label_to_index = json.loads(str(arrays["label_to_index"].tolist()))
    index_to_label = {int(idx): label for label, idx in label_to_index.items()}
    numeric_true = np.asarray([label_to_index.get(label, -1) for label in true_label], dtype=int)

    scorer = create_scorer(config.open_set_method)
    train_mask = (split == "train") & (known_status == "known")
    scorer.fit(logits[train_mask], embeddings[train_mask], numeric_true[train_mask])
    scores = scorer.score(logits, embeddings)
    validation_known = (split == "validation") & (known_status == "known")
    threshold = calibrate_strict_open_set(
        scores[validation_known],
        target_known_recall=config.target_known_recall,
    )

    test_mask = split == "test"
    test_unknown = (known_status[test_mask] != "known").astype(int)
    test_scores = scores[test_mask]
    unknown_pred = apply_threshold(test_scores, threshold.threshold)
    pred_with_unknown = prediction[test_mask].astype(object)
    pred_with_unknown[unknown_pred == 1] = "UNKNOWN"
    true_with_unknown = true_label[test_mask].astype(object)
    true_with_unknown[test_unknown == 1] = "UNKNOWN"

    known_test_mask = test_mask & (known_status == "known")
    closed_metrics = {}
    calib_metrics = {}
    if np.any(known_test_mask):
        closed_true = numeric_true[known_test_mask]
        closed_pred = logits[known_test_mask].argmax(axis=1)
        class_names = [index_to_label[idx] for idx in sorted(index_to_label)]
        closed_metrics = closed_set_metrics(closed_true, closed_pred, class_names)
        calib_metrics = calibration_metrics(logits[known_test_mask], closed_true)

    open_metrics = open_set_metrics(test_unknown, test_scores, true_with_unknown, pred_with_unknown)
    probabilities = softmax(logits[test_mask], axis=1)
    open_metrics["oscr"] = oscr(
        np.asarray([label_to_index.get(label, -1) for label in true_label[test_mask]], dtype=int),
        probabilities.argmax(axis=1),
        probabilities.max(axis=1),
        test_unknown,
    )

    predictions_frame = pd.DataFrame(
        {
            "sample_id": arrays["sample_id"][test_mask].astype(str),
            "true_label": true_label[test_mask],
            "known_status": known_status[test_mask],
            "closed_set_prediction": prediction[test_mask],
            "open_set_prediction": pred_with_unknown,
            "unknown_score": test_scores,
            "threshold": threshold.threshold,
        }
    )
    predictions_frame.to_csv(output_dir / "predictions.csv", index=False)
    predictions_frame.to_csv(output_dir / "known_unknown_predictions.csv", index=False)

    cm_labels = sorted(set(true_with_unknown.tolist()) | set(pred_with_unknown.tolist()))
    cm_frame = pd.DataFrame(
        confusion_matrix(true_with_unknown, pred_with_unknown, labels=cm_labels),
        index=cm_labels,
        columns=cm_labels,
    )
    cm_frame.to_csv(output_dir / "confusion_matrix.csv")
    _save_confusion_png(cm_frame, output_dir / "confusion_matrix.png")
    if closed_metrics:
        _save_per_class(closed_metrics, output_dir / "per_class_metrics.csv")
    pd.DataFrame({"y_unknown": test_unknown, "unknown_score": test_scores}).to_csv(
        output_dir / "roc_known_unknown.csv",
        index=False,
    )
    pd.DataFrame({"y_unknown": test_unknown, "unknown_score": test_scores}).to_csv(
        output_dir / "precision_recall_known_unknown.csv",
        index=False,
    )
    _save_figures(test_unknown, test_scores, output_dir)

    metrics = {
        "threshold": threshold.__dict__,
        "closed_set": closed_metrics,
        "open_set": open_metrics,
        "calibration": calib_metrics,
    }
    write_json(output_dir / "metrics.json", metrics)
    if closed_metrics:
        write_json(output_dir / "closed_set_metrics.json", closed_metrics)
    write_json(output_dir / "open_set_metrics.json", open_metrics)
    manifest_hash_value = (
        str(arrays["manifest_hash"].tolist()) if "manifest_hash" in arrays else None
    )
    write_json(output_dir / "environment.json", environment_metadata(manifest_hash_value))
    save_config(
        output_dir / "experiment_config.yaml",
        {
            "embeddings_path": str(config.embeddings_path),
            "open_set_method": config.open_set_method,
            "target_known_recall": config.target_known_recall,
        },
    )
    return output_dir / "metrics.json"
