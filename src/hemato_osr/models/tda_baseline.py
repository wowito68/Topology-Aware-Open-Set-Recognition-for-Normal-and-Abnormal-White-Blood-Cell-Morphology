"""Classical baselines on cached topological features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hemato_osr.data.taxonomy import DEFAULT_KNOWN_CLASSES
from hemato_osr.evaluation.metrics import closed_set_metrics, open_set_metrics, oscr
from hemato_osr.openset.thresholds import apply_threshold, calibrate_strict_open_set
from hemato_osr.topology.pipeline import load_tda_feature_archive
from hemato_osr.utils.tracking import write_json


@dataclass(frozen=True)
class TDABaselineConfig:
    """TDA baseline classifier configuration."""

    classifier: str = "logistic_regression"
    seed: int = 13
    max_iter: int = 500


@dataclass(frozen=True)
class TDAOnlyExperimentConfig:
    """Configuration for TDA-only closed/open-set evaluation."""

    features_path: Path
    output_dir: Path
    known_classes: tuple[str, ...] = DEFAULT_KNOWN_CLASSES
    feature_components: tuple[str, ...] = ()
    classifier: str = "logistic_regression"
    osr_method: str = "centroid_distance"
    seed: int = 37
    max_iter: int = 1000
    target_known_recall: float = 0.95


def create_tda_baseline(config: TDABaselineConfig) -> Pipeline:
    """Create a standardization plus classifier pipeline."""

    if config.classifier == "logistic_regression":
        classifier = LogisticRegression(max_iter=config.max_iter, random_state=config.seed)
    elif config.classifier == "mlp":
        classifier = MLPClassifier(
            hidden_layer_sizes=(64,),
            max_iter=config.max_iter,
            random_state=config.seed,
        )
    else:
        msg = f"Unsupported TDA baseline classifier: {config.classifier}"
        raise ValueError(msg)
    return Pipeline([("standardize", StandardScaler()), ("classifier", classifier)])


def fit_tda_baseline(
    features: np.ndarray,
    labels: np.ndarray,
    config: TDABaselineConfig,
) -> Pipeline:
    """Fit a TDA-only baseline using training features only."""

    model = create_tda_baseline(config)
    model.fit(features, labels)
    return model


def _component_indices(metadata: dict[str, Any], components: tuple[str, ...]) -> np.ndarray:
    if not components:
        return np.asarray([], dtype=int)
    slices = metadata.get("component_slices", {})
    indices: list[int] = []
    for component in components:
        start, end = slices[component]
        indices.extend(range(int(start), int(end)))
    return np.asarray(indices, dtype=int)


def _label_mapping(known_classes: tuple[str, ...]) -> dict[str, int]:
    return {label: idx for idx, label in enumerate(known_classes)}


def _centroid_scores(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    distances = []
    for label in sorted(set(train_labels.astype(int).tolist())):
        centroid = train_features[train_labels == label].mean(axis=0)
        distances.append(np.linalg.norm(features - centroid, axis=1))
    return np.min(np.vstack(distances), axis=0)


def _mahalanobis_scores(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    features: np.ndarray,
    regularization: float = 1e-4,
) -> np.ndarray:
    distances = []
    residuals = []
    means = []
    for label in sorted(set(train_labels.astype(int).tolist())):
        class_features = train_features[train_labels == label]
        mean = class_features.mean(axis=0)
        means.append(mean)
        residuals.append(class_features - mean)
    covariance = np.cov(np.vstack(residuals), rowvar=False)
    covariance = np.atleast_2d(covariance)
    covariance += np.eye(covariance.shape[0]) * regularization
    precision = np.linalg.pinv(covariance)
    for mean in means:
        delta = features - mean
        distances.append(np.sum(delta @ precision * delta, axis=1))
    return np.min(np.vstack(distances), axis=0)


def evaluate_tda_only(config: TDAOnlyExperimentConfig) -> Path:
    """Train/evaluate the TDA-only baseline without using unknowns for tuning."""

    archive = load_tda_feature_archive(config.features_path)
    features = archive["features"]
    selected = _component_indices(archive["metadata"], config.feature_components)
    if selected.size:
        features = features[:, selected]
    if not np.isfinite(features).all():
        msg = "TDA-only evaluation received NaN/Inf features"
        raise ValueError(msg)

    labels = archive["true_label"].astype(str)
    known_status = archive["known_status"].astype(str)
    splits = archive["split"].astype(str)
    label_to_index = _label_mapping(config.known_classes)
    y = np.asarray([label_to_index.get(label, -1) for label in labels], dtype=int)
    known = y >= 0
    train_mask = known & (splits == "train")
    validation_known = known & (splits == "validation")
    known_test = known & (splits == "test")
    test_mask = splits == "test"
    if not np.any(train_mask) or not np.any(validation_known):
        msg = "TDA-only evaluation requires known train and validation rows"
        raise ValueError(msg)

    model = create_tda_baseline(
        TDABaselineConfig(
            classifier=config.classifier,
            seed=config.seed,
            max_iter=config.max_iter,
        )
    )
    model.fit(features[train_mask], y[train_mask])
    pred_numeric = model.predict(features)
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    pred_label = np.asarray([index_to_label[int(idx)] for idx in pred_numeric], dtype=object)

    scaler = StandardScaler().fit(features[train_mask])
    standardized = scaler.transform(features)
    train_standardized = standardized[train_mask]
    if config.osr_method == "centroid_distance":
        scores = _centroid_scores(train_standardized, y[train_mask], standardized)
    elif config.osr_method == "mahalanobis":
        scores = _mahalanobis_scores(train_standardized, y[train_mask], standardized)
    else:
        msg = f"Unsupported TDA OSR method: {config.osr_method}"
        raise ValueError(msg)

    threshold = calibrate_strict_open_set(
        scores[validation_known],
        target_known_recall=config.target_known_recall,
    )
    unknown_pred = apply_threshold(scores[test_mask], threshold.threshold)
    pred_with_unknown = pred_label[test_mask].astype(object)
    pred_with_unknown[unknown_pred == 1] = "UNKNOWN"
    test_unknown = (known_status[test_mask] != "known").astype(int)
    true_with_unknown = labels[test_mask].astype(object)
    true_with_unknown[test_unknown == 1] = "UNKNOWN"

    class_names = list(config.known_classes)
    closed_metrics = closed_set_metrics(
        y[known_test],
        pred_numeric[known_test],
        class_names,
    )
    open_metrics = open_set_metrics(
        test_unknown,
        scores[test_mask],
        true_with_unknown,
        pred_with_unknown,
    )
    known_confidence = -scores[test_mask]
    open_metrics["oscr"] = oscr(
        y[test_mask],
        pred_numeric[test_mask],
        known_confidence,
        test_unknown,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.DataFrame(
        {
            "sample_id": archive["sample_id"][test_mask],
            "true_label": labels[test_mask],
            "known_status": known_status[test_mask],
            "closed_set_prediction": pred_label[test_mask],
            "open_set_prediction": pred_with_unknown,
            "unknown_score": scores[test_mask],
            "threshold": threshold.threshold,
        }
    )
    prediction_frame.to_csv(config.output_dir / "predictions.csv", index=False)
    write_json(config.output_dir / "closed_set_metrics.json", closed_metrics)
    write_json(config.output_dir / "open_set_metrics.json", open_metrics)
    metrics = {
        "closed_set": closed_metrics,
        "open_set": open_metrics,
        "threshold": threshold.__dict__,
        "feature_components": list(config.feature_components),
        "feature_dimension": int(features.shape[1]),
        "feature_metadata": archive["metadata"],
    }
    write_json(config.output_dir / "metrics.json", metrics)
    (config.output_dir / "experiment_config.json").write_text(
        json.dumps(config.__dict__, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return config.output_dir / "metrics.json"
