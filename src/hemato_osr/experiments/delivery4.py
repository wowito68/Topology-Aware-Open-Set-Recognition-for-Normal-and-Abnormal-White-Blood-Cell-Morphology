"""Delivery 4 post-hoc OSR and embedding-geometry experiments."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors

from hemato_osr.data.taxonomy import DEFAULT_KNOWN_CLASSES
from hemato_osr.evaluation.metrics import fpr_at_tpr, open_set_metrics, oscr
from hemato_osr.openset.thresholds import apply_threshold, calibrate_strict_open_set
from hemato_osr.topology.cache import config_hash
from hemato_osr.training.checkpoint import load_checkpoint
from hemato_osr.utils.tracking import environment_metadata, write_json

UNKNOWN_GROUPS: dict[str, tuple[str, ...]] = {
    "immature_related_myeloid": (
        "myeloblast",
        "promyelocyte",
        "promyelocyte_atypical",
        "myelocyte",
        "metamyelocyte",
        "neutrophil_band",
    ),
    "lymphoid_related": (
        "lymphocyte_large_granular",
        "lymphocyte_neoplastic",
        "lymphocyte_reactive",
        "hairy_cell",
        "plasma_cell",
    ),
    "other": ("normoblast", "smudge_cell"),
}

HISTORICAL_BASELINES = {
    "msp": {"AUROC": 0.8602, "FPR95": 0.5500},
    "energy_T1": {"AUROC": 0.8586, "FPR95": 0.5511},
    "historical_mahalanobis": {"AUROC": 0.7693, "FPR95": 0.7651},
}


@dataclass(frozen=True)
class Delivery4Config:
    """Delivery 4 fixed input paths."""

    embeddings_path: Path
    checkpoint_path: Path
    matrix_path: Path
    output_dir: Path
    seed: int = 37
    target_known_recall: float = 0.95
    expected_checkpoint_sha256: str = (
        "7fb2e2a6741a4fcd1b7d9e3a985ab136fa0550741b69dcd62f0f21c28b6ae39d"
    )


@dataclass(frozen=True)
class EmbeddingArchive:
    """Loaded frozen-embedding archive."""

    sample_id: np.ndarray
    true_label: np.ndarray
    known_status: np.ndarray
    split: np.ndarray
    embedding: np.ndarray
    logits: np.ndarray
    prediction: np.ndarray
    label_to_index: dict[str, int]
    manifest_hash: str


@dataclass(frozen=True)
class FittedScores:
    """Scores plus fit/score efficiency metadata."""

    scores: np.ndarray
    fit_time_seconds: float
    score_time_ms_per_sample: float
    extra_memory_mb: float
    fitted_state: dict[str, Any]


def sha256_file(path: Path) -> str:
    """Compute SHA256 without reading secrets into logs."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_embeddings(path: Path) -> EmbeddingArchive:
    """Load the frozen Delivery 1/2 embedding archive."""

    with np.load(path, allow_pickle=True) as data:
        label_to_index = json.loads(str(data["label_to_index"].tolist()))
        manifest_hash = str(data["manifest_hash"].tolist())
        return EmbeddingArchive(
            sample_id=np.asarray(data["sample_id"]).astype(str),
            true_label=np.asarray(data["true_label"]).astype(str),
            known_status=np.asarray(data["known_status"]).astype(str),
            split=np.asarray(data["split"]).astype(str),
            embedding=np.asarray(data["embedding"], dtype=np.float64),
            logits=np.asarray(data["logits"], dtype=np.float64),
            prediction=np.asarray(data["prediction"]).astype(str),
            label_to_index={str(key): int(value) for key, value in label_to_index.items()},
            manifest_hash=manifest_hash,
        )


def label_indices(labels: np.ndarray, label_to_index: dict[str, int]) -> np.ndarray:
    """Map known labels to indices and unknown labels to -1."""

    return np.asarray([label_to_index.get(str(label), -1) for label in labels], dtype=int)


def known_train_mask(archive: EmbeddingArchive) -> np.ndarray:
    """Return the only deployable fit mask: known train."""

    return (archive.split == "train") & (archive.known_status == "known")


def validate_train_only_fit_mask(
    split: np.ndarray,
    known_status: np.ndarray,
    fit_mask: np.ndarray,
) -> None:
    """Fail loudly if a fitter receives anything other than known-train rows."""

    mask = np.asarray(fit_mask, dtype=bool)
    if mask.shape[0] != split.shape[0] or mask.shape[0] != known_status.shape[0]:
        msg = "Fit mask length must match split and known_status arrays"
        raise ValueError(msg)
    if np.any(mask & (np.asarray(split).astype(str) != "train")):
        msg = "Fit mask includes validation or test rows"
        raise ValueError(msg)
    if np.any(mask & (np.asarray(known_status).astype(str) != "known")):
        msg = "Fit mask includes unknown rows"
        raise ValueError(msg)
    if not np.any(mask):
        msg = "Fit mask is empty"
        raise ValueError(msg)


def l2_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-normalize vectors with an epsilon guard."""

    arr = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, eps)


def msp_anomaly_scores(logits: np.ndarray) -> np.ndarray:
    """MSP anomaly: 1 - max softmax probability."""

    probabilities = softmax(logits, axis=1)
    return 1.0 - probabilities.max(axis=1)


def predictive_entropy_scores(logits: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Predictive entropy; larger means less confident."""

    probabilities = softmax(logits, axis=1)
    return -np.sum(probabilities * np.log(probabilities + eps), axis=1)


def energy_scores(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Temperature-scaled energy oriented as unknownness."""

    if temperature <= 0:
        msg = "Energy temperature must be positive"
        raise ValueError(msg)
    return -temperature * logsumexp(np.asarray(logits, dtype=np.float64) / temperature, axis=1)


def class_means(
    embeddings: np.ndarray,
    labels: np.ndarray,
    known_classes: tuple[str, ...] = DEFAULT_KNOWN_CLASSES,
) -> dict[int, np.ndarray]:
    """Compute class means from already-filtered known-train embeddings."""

    means: dict[int, np.ndarray] = {}
    for idx, _class_name in enumerate(known_classes):
        class_embeddings = embeddings[labels == idx]
        if class_embeddings.size == 0:
            msg = f"No training embeddings for class index {idx}"
            raise ValueError(msg)
        means[idx] = class_embeddings.mean(axis=0)
    return means


def ncm_euclidean_scores(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    embeddings: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Nearest class mean using Euclidean distance."""

    means = class_means(train_embeddings, train_labels)
    distances = [np.linalg.norm(embeddings - mean, axis=1) for mean in means.values()]
    return np.min(np.vstack(distances), axis=0), means


def ncm_cosine_scores(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    embeddings: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Nearest normalized prototype using cosine distance."""

    train_norm = l2_normalize(train_embeddings)
    query_norm = l2_normalize(embeddings)
    raw_means = class_means(train_norm, train_labels)
    prototypes = {label: l2_normalize(mean.reshape(1, -1))[0] for label, mean in raw_means.items()}
    similarities = [query_norm @ prototype for prototype in prototypes.values()]
    return 1.0 - np.max(np.vstack(similarities), axis=0), prototypes


def knn_cosine_scores(
    train_embeddings: np.ndarray,
    embeddings: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, NearestNeighbors]:
    """Mean cosine distance to k nearest known-train embeddings."""

    if k <= 0:
        msg = "kNN k must be positive"
        raise ValueError(msg)
    bank = l2_normalize(train_embeddings)
    query = l2_normalize(embeddings)
    neighbors = NearestNeighbors(n_neighbors=min(k, len(bank)), metric="cosine")
    neighbors.fit(bank)
    distances, _indices = neighbors.kneighbors(query)
    return distances.mean(axis=1), neighbors


@dataclass
class PooledMahalanobis:
    """Class-conditional pooled Mahalanobis distance."""

    means: dict[int, np.ndarray]
    precision: np.ndarray
    covariance: np.ndarray
    estimator: str

    @classmethod
    def fit_historical(
        cls,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        *,
        regularization: float = 1e-4,
    ) -> PooledMahalanobis:
        """Match the historical pooled covariance implementation."""

        means = class_means(train_embeddings, train_labels)
        residuals = []
        for label, mean in means.items():
            residuals.append(train_embeddings[train_labels == label] - mean)
        covariance = np.cov(np.vstack(residuals), rowvar=False)
        covariance = np.atleast_2d(covariance)
        covariance += np.eye(covariance.shape[0]) * regularization
        precision = np.linalg.pinv(covariance)
        return cls(means=means, precision=precision, covariance=covariance, estimator="pinv")

    @classmethod
    def fit_ledoit_wolf(
        cls,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
    ) -> PooledMahalanobis:
        """Fit pooled residual covariance with Ledoit-Wolf shrinkage."""

        means = class_means(train_embeddings, train_labels)
        residuals = []
        for label, mean in means.items():
            residuals.append(train_embeddings[train_labels == label] - mean)
        estimator = LedoitWolf().fit(np.vstack(residuals))
        covariance = np.asarray(estimator.covariance_, dtype=np.float64)
        precision = np.asarray(estimator.precision_, dtype=np.float64)
        if not np.isfinite(precision).all():
            msg = "Ledoit-Wolf precision is non-finite"
            raise ValueError(msg)
        return cls(
            means=means,
            precision=precision,
            covariance=covariance,
            estimator="ledoit_wolf",
        )

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Minimum class-conditional squared Mahalanobis distance."""

        distances = []
        for mean in self.means.values():
            delta = embeddings - mean
            distances.append(np.sum(delta @ self.precision * delta, axis=1))
        return np.min(np.vstack(distances), axis=0)


@dataclass
class RelativeMahalanobis:
    """Relative Mahalanobis: class distance minus global background distance."""

    class_model: PooledMahalanobis
    global_mean: np.ndarray
    global_precision: np.ndarray
    global_covariance: np.ndarray

    @classmethod
    def fit(
        cls,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
    ) -> RelativeMahalanobis:
        """Fit class-conditional and global Ledoit-Wolf models on known train."""

        class_model = PooledMahalanobis.fit_ledoit_wolf(train_embeddings, train_labels)
        global_estimator = LedoitWolf().fit(train_embeddings)
        return cls(
            class_model=class_model,
            global_mean=train_embeddings.mean(axis=0),
            global_precision=np.asarray(global_estimator.precision_, dtype=np.float64),
            global_covariance=np.asarray(global_estimator.covariance_, dtype=np.float64),
        )

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Return class-conditional distance minus global distance."""

        class_distance = self.class_model.score(embeddings)
        delta = embeddings - self.global_mean
        global_distance = np.sum(delta @ self.global_precision * delta, axis=1)
        return class_distance - global_distance


def classifier_head_from_checkpoint(checkpoint_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the frozen linear classifier head."""

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    state = checkpoint["model_state_dict"]
    weight_key = next((key for key in state if key.endswith("classifier.weight")), None)
    bias_key = next((key for key in state if key.endswith("classifier.bias")), None)
    if weight_key is None:
        msg = "Checkpoint does not contain classifier.weight"
        raise ValueError(msg)
    weight = state[weight_key].detach().cpu().numpy().astype(np.float64)
    bias = (
        state[bias_key].detach().cpu().numpy().astype(np.float64)
        if bias_key is not None
        else np.zeros(weight.shape[0], dtype=np.float64)
    )
    return weight, bias


def react_clipped_logits(
    embeddings: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    clip_value: float,
) -> np.ndarray:
    """Apply ReAct-style upper activation clipping and recompute logits."""

    clipped = np.minimum(np.asarray(embeddings, dtype=np.float64), float(clip_value))
    return clipped @ weight.T + bias


def react_scores(
    train_embeddings: np.ndarray,
    embeddings: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    percentile: float,
    temperature: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Fit ReAct clipping threshold on known train and score with energy."""

    clip_value = float(np.percentile(train_embeddings, percentile))
    logits = react_clipped_logits(embeddings, weight, bias, clip_value)
    return energy_scores(logits, temperature=temperature), clip_value


@dataclass
class ViMModel:
    """ViM-style residual score fit on known-train embeddings."""

    mean: np.ndarray
    principal_axes: np.ndarray
    alpha: float
    explained_variance_ratio: float
    n_components: int

    @classmethod
    def fit(
        cls,
        train_embeddings: np.ndarray,
        train_logits: np.ndarray,
        *,
        explained_variance: float = 0.95,
    ) -> ViMModel:
        """Fit PCA subspace and residual scale using ID-only statistics."""

        if not 0.0 < explained_variance < 1.0:
            msg = "ViM explained_variance must be in (0, 1)"
            raise ValueError(msg)
        mean = train_embeddings.mean(axis=0)
        centered = train_embeddings - mean
        full_pca = PCA(svd_solver="full").fit(centered)
        cumulative = np.cumsum(full_pca.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumulative, explained_variance, side="left") + 1)
        n_components = min(n_components, train_embeddings.shape[1] - 1)
        principal_axes = np.asarray(full_pca.components_[:n_components], dtype=np.float64)
        residual = residual_norm(centered, principal_axes)
        logit_energy = logsumexp(train_logits, axis=1)
        alpha = float(np.mean(logit_energy) / max(float(np.mean(residual)), 1e-12))
        return cls(
            mean=mean,
            principal_axes=principal_axes,
            alpha=alpha,
            explained_variance_ratio=float(cumulative[n_components - 1]),
            n_components=n_components,
        )

    def score(self, embeddings: np.ndarray, logits: np.ndarray) -> np.ndarray:
        """ViM unknownness: alpha * residual_norm - logsumexp(logits)."""

        centered = embeddings - self.mean
        virtual_logit = self.alpha * residual_norm(centered, self.principal_axes)
        return virtual_logit - logsumexp(logits, axis=1)


def residual_norm(centered_embeddings: np.ndarray, principal_axes: np.ndarray) -> np.ndarray:
    """Norm outside the PCA principal subspace."""

    projection = centered_embeddings @ principal_axes.T @ principal_axes
    residual = centered_embeddings - projection
    return np.linalg.norm(residual, axis=1)


def write_predeclared_matrix(output_path: Path) -> Path:
    """Persist Delivery 4 methods and progression rules before final evaluation."""

    methods: list[dict[str, Any]] = [
        {"method": "msp", "variant": "primary", "primary": True, "status": "implemented"},
        {"method": "entropy", "variant": "baseline", "primary": False, "status": "implemented"},
        {"method": "energy", "variant": "T=1", "primary": True, "status": "implemented"},
        {"method": "energy", "variant": "T=0.5", "primary": False, "status": "implemented"},
        {"method": "energy", "variant": "T=2", "primary": False, "status": "implemented"},
        {
            "method": "ncm_euclidean",
            "variant": "primary",
            "primary": True,
            "status": "implemented",
        },
        {
            "method": "ncm_cosine",
            "variant": "primary",
            "primary": True,
            "status": "implemented",
        },
        {"method": "knn_cosine", "variant": "k=1", "primary": False, "status": "implemented"},
        {"method": "knn_cosine", "variant": "k=5", "primary": True, "status": "implemented"},
        {"method": "knn_cosine", "variant": "k=10", "primary": False, "status": "implemented"},
        {
            "method": "historical_mahalanobis",
            "variant": "pinv_regularization_1e-4",
            "primary": False,
            "status": "implemented",
        },
        {
            "method": "regularized_mahalanobis",
            "variant": "ledoit_wolf",
            "primary": True,
            "status": "implemented",
        },
        {
            "method": "relative_mahalanobis",
            "variant": "ledoit_wolf_class_minus_global",
            "primary": True,
            "status": "implemented",
        },
        {
            "method": "react_energy",
            "variant": "p85",
            "primary": False,
            "status": "implemented",
        },
        {
            "method": "react_energy",
            "variant": "p90",
            "primary": True,
            "status": "implemented",
        },
        {
            "method": "react_energy",
            "variant": "p95",
            "primary": False,
            "status": "implemented",
        },
        {
            "method": "vim",
            "variant": "pca_95pct_variance",
            "primary": True,
            "status": "implemented",
        },
        {
            "method": "dice",
            "variant": "not_applicable",
            "primary": False,
            "status": "not implemented",
            "reason": (
                "Skipped because a rigorous DICE contribution-pruning implementation "
                "for signed pooled ResNet embeddings is not established in this codebase."
            ),
        },
    ]
    payload = {
        "status": "predeclared_before_delivery4_final_evaluation",
        "seed": 37,
        "split": "V1",
        "primary_baseline": "msp",
        "score_orientation": "larger means more UNKNOWN-like",
        "decision_criteria": {
            "criterion_a": "Delta AUROC vs MSP >= +0.015 with favorable paired bootstrap CI",
            "criterion_b": "Delta FPR95 vs MSP <= -0.075 while Delta AUROC >= -0.005",
            "criterion_c": (
                "Delta lymphoid-related subgroup AUROC >= +0.03 without overall AUROC "
                "degradation > 0.01"
            ),
        },
        "ranking_priority_if_multiple_pass": [
            "overall AUROC",
            "FPR95",
            "OSCR",
            "simplicity / no additional forward pass",
            "runtime",
        ],
        "unknown_groups": UNKNOWN_GROUPS,
        "methods": methods,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _method_id(method: str, variant: str) -> str:
    clean = (
        f"{method}_{variant}".replace("=", "").replace(".", "_").replace("-", "_").replace(" ", "_")
    )
    return clean.lower()


def _score_no_fit(
    name: str,
    variant: str,
    archive: EmbeddingArchive,
) -> FittedScores:
    start = time.perf_counter()
    fit_time = time.perf_counter() - start
    score_start = time.perf_counter()
    if name == "msp":
        scores = msp_anomaly_scores(archive.logits)
    elif name == "entropy":
        scores = predictive_entropy_scores(archive.logits)
    elif name == "energy":
        temperature = float(variant.removeprefix("T="))
        scores = energy_scores(archive.logits, temperature=temperature)
    else:
        msg = f"Unsupported no-fit method: {name}/{variant}"
        raise ValueError(msg)
    score_time = (time.perf_counter() - score_start) * 1000.0 / len(scores)
    return FittedScores(scores, fit_time, score_time, 0.0, {"fit": "none"})


def fit_and_score_method(
    method: dict[str, Any],
    archive: EmbeddingArchive,
    *,
    checkpoint_path: Path,
) -> FittedScores:
    """Fit method on known train only and score all rows."""

    name = str(method["method"])
    variant = str(method["variant"])
    if name in {"msp", "entropy", "energy"}:
        return _score_no_fit(name, variant, archive)

    fit_mask = known_train_mask(archive)
    validate_train_only_fit_mask(archive.split, archive.known_status, fit_mask)
    y = label_indices(archive.true_label, archive.label_to_index)
    train_embeddings = archive.embedding[fit_mask]
    train_logits = archive.logits[fit_mask]
    train_labels = y[fit_mask]

    fit_start = time.perf_counter()
    fitted_state: dict[str, Any] = {"fit_split": "known train only"}
    extra_memory_mb = 0.0
    scorer: Any
    score_fn: Any
    if name == "ncm_euclidean":
        _train_scores, means = ncm_euclidean_scores(
            train_embeddings,
            train_labels,
            train_embeddings,
        )
        fitted_state["prototype_count"] = len(means)
        extra_memory_mb = sum(mean.nbytes for mean in means.values()) / 1024**2

        def score_fn(embeddings: np.ndarray) -> np.ndarray:
            distances = [np.linalg.norm(embeddings - mean, axis=1) for mean in means.values()]
            return np.min(np.vstack(distances), axis=0)

    elif name == "ncm_cosine":
        _train_scores, prototypes = ncm_cosine_scores(
            train_embeddings,
            train_labels,
            train_embeddings,
        )
        fitted_state["prototype_count"] = len(prototypes)
        extra_memory_mb = sum(proto.nbytes for proto in prototypes.values()) / 1024**2

        def score_fn(embeddings: np.ndarray) -> np.ndarray:
            query = l2_normalize(embeddings)
            similarities = [query @ prototype for prototype in prototypes.values()]
            return 1.0 - np.max(np.vstack(similarities), axis=0)

    elif name == "knn_cosine":
        k = int(variant.removeprefix("k="))
        _scores, scorer = knn_cosine_scores(train_embeddings, train_embeddings, k=k)
        fitted_state["k"] = k
        extra_memory_mb = train_embeddings.nbytes / 1024**2

        def score_fn(embeddings: np.ndarray) -> np.ndarray:
            distances, _indices = scorer.kneighbors(l2_normalize(embeddings))
            return distances.mean(axis=1)

    elif name == "historical_mahalanobis":
        scorer = PooledMahalanobis.fit_historical(train_embeddings, train_labels)
        fitted_state["estimator"] = scorer.estimator
        extra_memory_mb = scorer.covariance.nbytes / 1024**2
        score_fn = scorer.score
    elif name == "regularized_mahalanobis":
        scorer = PooledMahalanobis.fit_ledoit_wolf(train_embeddings, train_labels)
        fitted_state["estimator"] = scorer.estimator
        extra_memory_mb = scorer.covariance.nbytes / 1024**2
        score_fn = scorer.score
    elif name == "relative_mahalanobis":
        scorer = RelativeMahalanobis.fit(train_embeddings, train_labels)
        fitted_state["equation"] = "min class Mahalanobis distance - global Mahalanobis distance"
        extra_memory_mb = (
            scorer.class_model.covariance.nbytes + scorer.global_covariance.nbytes
        ) / 1024**2
        score_fn = scorer.score
    elif name == "react_energy":
        percentile = float(variant.removeprefix("p"))
        weight, bias = classifier_head_from_checkpoint(checkpoint_path)
        clip_value = float(np.percentile(train_embeddings, percentile))
        fitted_state["clip_percentile"] = percentile
        fitted_state["clip_value"] = clip_value
        extra_memory_mb = (weight.nbytes + bias.nbytes) / 1024**2

        def score_fn(embeddings: np.ndarray) -> np.ndarray:
            logits = react_clipped_logits(embeddings, weight, bias, clip_value)
            return energy_scores(logits, temperature=1.0)

    elif name == "vim":
        scorer = ViMModel.fit(train_embeddings, train_logits)
        fitted_state.update(
            {
                "n_components": scorer.n_components,
                "explained_variance_ratio": scorer.explained_variance_ratio,
                "alpha": scorer.alpha,
                "score": "alpha * residual_norm - logsumexp(logits)",
            }
        )
        extra_memory_mb = (scorer.mean.nbytes + scorer.principal_axes.nbytes) / 1024**2

        def score_fn(embeddings: np.ndarray) -> np.ndarray:
            return scorer.score(embeddings, archive.logits)

    else:
        msg = f"Method {name}/{variant} is not implemented"
        raise ValueError(msg)

    fit_time = time.perf_counter() - fit_start
    score_start = time.perf_counter()
    scores = np.asarray(score_fn(archive.embedding), dtype=np.float64)
    score_time = (time.perf_counter() - score_start) * 1000.0 / len(scores)
    if not np.isfinite(scores).all():
        msg = f"Non-finite scores from {name}/{variant}"
        raise ValueError(msg)
    return FittedScores(scores, fit_time, score_time, extra_memory_mb, fitted_state)


def _score_metrics(
    archive: EmbeddingArchive,
    scores: np.ndarray,
    *,
    method_id: str,
    output_dir: Path,
    target_known_recall: float = 0.95,
) -> dict[str, float]:
    test_mask = archive.split == "test"
    validation_known = (archive.split == "validation") & (archive.known_status == "known")
    threshold = calibrate_strict_open_set(
        scores[validation_known],
        target_known_recall=target_known_recall,
    )
    y_unknown = (archive.known_status[test_mask] != "known").astype(int)
    test_scores = scores[test_mask]
    unknown_pred = apply_threshold(test_scores, threshold.threshold)
    pred_with_unknown = archive.prediction[test_mask].astype(object)
    pred_with_unknown[unknown_pred == 1] = "UNKNOWN"
    true_with_unknown = archive.true_label[test_mask].astype(object)
    true_with_unknown[y_unknown == 1] = "UNKNOWN"
    metrics = open_set_metrics(y_unknown, test_scores, true_with_unknown, pred_with_unknown)
    metrics["aupr_known_positive"] = float(average_precision_score(1 - y_unknown, -test_scores))
    y_true = label_indices(archive.true_label[test_mask], archive.label_to_index)
    metrics["oscr"] = oscr(
        y_true,
        archive.logits[test_mask].argmax(axis=1),
        -test_scores,
        y_unknown,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sample_id": archive.sample_id[test_mask],
            "true_label": archive.true_label[test_mask],
            "known_status": archive.known_status[test_mask],
            "closed_set_prediction": archive.prediction[test_mask],
            "open_set_prediction": pred_with_unknown,
            "unknown_score": test_scores,
            "threshold": threshold.threshold,
            "method": method_id,
        }
    ).to_csv(output_dir / f"{method_id}_predictions.csv", index=False)
    return metrics


def paired_bootstrap_deltas(
    y_unknown: np.ndarray,
    baseline_scores: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    n_bootstraps: int = 1000,
    seed: int = 37,
) -> dict[str, dict[str, float | int]]:
    """Paired bootstrap deltas for AUROC and FPR@95TPR."""

    y = np.asarray(y_unknown, dtype=int)
    baseline = np.asarray(baseline_scores, dtype=np.float64)
    candidate = np.asarray(candidate_scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    auroc_delta = []
    fpr_delta = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, len(y), size=len(y))
        if len(set(y[idx].tolist())) < 2:
            continue
        auroc_delta.append(
            roc_auc_score(y[idx], candidate[idx]) - roc_auc_score(y[idx], baseline[idx])
        )
        fpr_delta.append(fpr_at_tpr(y[idx], candidate[idx]) - fpr_at_tpr(y[idx], baseline[idx]))
    return {
        "delta_auroc": {
            "estimate": float(roc_auc_score(y, candidate) - roc_auc_score(y, baseline)),
            "lower": float(np.quantile(auroc_delta, 0.025)),
            "upper": float(np.quantile(auroc_delta, 0.975)),
            "n_bootstraps": n_bootstraps,
            "seed": seed,
        },
        "delta_fpr95": {
            "estimate": float(fpr_at_tpr(y, candidate) - fpr_at_tpr(y, baseline)),
            "lower": float(np.quantile(fpr_delta, 0.025)),
            "upper": float(np.quantile(fpr_delta, 0.975)),
            "n_bootstraps": n_bootstraps,
            "seed": seed,
        },
    }


def reproduce_delivery4_baselines(
    embeddings_path: Path,
    checkpoint_path: Path,
    output_path: Path,
) -> Path:
    """Recompute historical post-hoc baselines before new-method evaluation."""

    archive = load_embeddings(embeddings_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    methods = [
        {"method": "msp", "variant": "primary", "status": "implemented"},
        {"method": "energy", "variant": "T=1", "status": "implemented"},
        {
            "method": "historical_mahalanobis",
            "variant": "pinv_regularization_1e-4",
            "status": "implemented",
        },
    ]
    rows = {}
    for method in methods:
        result = fit_and_score_method(method, archive, checkpoint_path=checkpoint_path)
        metrics = _score_metrics(
            archive,
            result.scores,
            method_id=_method_id(str(method["method"]), str(method["variant"])),
            output_dir=output_path.parent / "baseline_predictions",
        )
        method_key = str(method["method"])
        if method_key == "energy":
            method_key = "energy_T1"
        expected = HISTORICAL_BASELINES[method_key]
        rows[method_key] = {
            "AUROC": metrics["auroc_known_unknown"],
            "FPR95": metrics["fpr_at_95_tpr"],
            "expected_AUROC": expected["AUROC"],
            "expected_FPR95": expected["FPR95"],
            "delta_AUROC": metrics["auroc_known_unknown"] - expected["AUROC"],
            "delta_FPR95": metrics["fpr_at_95_tpr"] - expected["FPR95"],
            "passed": (
                abs(metrics["auroc_known_unknown"] - expected["AUROC"]) <= 0.002
                and abs(metrics["fpr_at_95_tpr"] - expected["FPR95"]) <= 0.002
            ),
        }
    payload = {
        "checkpoint_sha256": checkpoint_hash,
        "manifest_hash": archive.manifest_hash,
        "baselines": rows,
        "passed": all(bool(row["passed"]) for row in rows.values()),
    }
    write_json(output_path, payload)
    if not payload["passed"]:
        msg = "Baseline reproduction failed; inspect manifest/checkpoint/score orientation"
        raise RuntimeError(msg)
    return output_path


def evaluate_delivery4(config: Delivery4Config) -> Path:
    """Evaluate the predeclared Delivery 4 matrix on V1 test data."""

    checkpoint_hash = sha256_file(config.checkpoint_path)
    if checkpoint_hash != config.expected_checkpoint_sha256:
        msg = f"Checkpoint hash mismatch: {checkpoint_hash}"
        raise RuntimeError(msg)
    matrix = json.loads(config.matrix_path.read_text(encoding="utf-8"))
    archive = load_embeddings(config.embeddings_path)
    methods = [method for method in matrix["methods"] if method["status"] == "implemented"]
    config.output_dir.mkdir(parents=True, exist_ok=True)

    score_by_id: dict[str, np.ndarray] = {}
    fitted_by_id: dict[str, FittedScores] = {}
    rows = []
    for method in methods:
        method_id = _method_id(str(method["method"]), str(method["variant"]))
        result = fit_and_score_method(method, archive, checkpoint_path=config.checkpoint_path)
        score_by_id[method_id] = result.scores
        fitted_by_id[method_id] = result
        metrics = _score_metrics(
            archive,
            result.scores,
            method_id=method_id,
            output_dir=config.output_dir / "predictions",
            target_known_recall=config.target_known_recall,
        )
        method_config = {
            "method": method["method"],
            "variant": method["variant"],
            "fitted_state": result.fitted_state,
        }
        rows.append(
            {
                "method": method["method"],
                "variant": method["variant"],
                "method_id": method_id,
                "primary": bool(method.get("primary", False)),
                "fit_data": (
                    "known_train_only"
                    if method["method"] not in {"msp", "entropy", "energy"}
                    else "none"
                ),
                "AUROC": metrics["auroc_known_unknown"],
                "AUPR_unknown": metrics["aupr_known_unknown"],
                "AUPR_known": metrics["aupr_known_positive"],
                "FPR95": metrics["fpr_at_95_tpr"],
                "OSCR": metrics["oscr"],
                "known_recall": metrics["known_recall"],
                "unknown_recall": metrics["unknown_recall"],
                "macro_f1_with_unknown": metrics["macro_f1_with_unknown"],
                "fit_time_seconds": result.fit_time_seconds,
                "score_time_ms_per_sample": result.score_time_ms_per_sample,
                "config_hash": config_hash(method_config),
                "checkpoint_hash": checkpoint_hash,
                "manifest_hash": archive.manifest_hash,
            }
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(config.output_dir / "method_comparison.csv", index=False)
    _write_data_usage_audit(methods, config.output_dir / "data_usage_audit.csv")
    _write_efficiency_table(fitted_by_id, methods, config.output_dir / "efficiency.csv")

    geometry = compute_embedding_geometry(archive)
    geometry["class_geometry"].to_csv(config.output_dir / "embedding_geometry.csv", index=False)
    geometry["compactness"].to_csv(config.output_dir / "embedding_compactness.csv", index=False)
    geometry["attractor"].to_csv(config.output_dir / "unknown_to_known_attractor.csv")
    geometry["class_distance"].to_csv(config.output_dir / "class_centroid_distance_matrix.csv")

    per_class = per_unknown_class_table(
        archive,
        score_by_id,
        comparison,
        geometry["nearest_lookup"],
    )
    per_class.to_csv(config.output_dir / "per_unknown_class.csv", index=False)
    subgroup = subgroup_results(per_class)
    subgroup.to_csv(config.output_dir / "subgroup_results.csv", index=False)
    bootstrap = bootstrap_primary_methods(archive, score_by_id, comparison)
    write_json(config.output_dir / "paired_bootstrap.json", bootstrap)
    decision = apply_progression_rule(comparison, subgroup, bootstrap)
    write_json(config.output_dir / "progression_decision.json", decision)
    _write_delivery4_figures(
        archive,
        comparison,
        per_class,
        geometry,
        score_by_id,
        config.output_dir,
    )
    write_json(config.output_dir / "environment.json", environment_metadata(archive.manifest_hash))
    write_json(
        config.output_dir / "method_state.json",
        {method_id: result.fitted_state for method_id, result in fitted_by_id.items()},
    )
    return config.output_dir / "method_comparison.csv"


def _write_data_usage_audit(methods: list[dict[str, Any]], output_path: Path) -> None:
    rows = []
    for method in methods:
        needs_fit = method["method"] not in {"msp", "entropy", "energy"}
        rows.append(
            {
                "method": method["method"],
                "variant": method["variant"],
                "fit_split": "known_train" if needs_fit else "none",
                "calibration_split": "known_validation",
                "test_split": "test",
                "unknown_data_used_during_fit": "NO",
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _write_efficiency_table(
    fitted_by_id: dict[str, FittedScores],
    methods: list[dict[str, Any]],
    output_path: Path,
) -> None:
    rows = []
    for method in methods:
        method_id = _method_id(str(method["method"]), str(method["variant"]))
        result = fitted_by_id[method_id]
        rows.append(
            {
                "method": method["method"],
                "variant": method["variant"],
                "method_id": method_id,
                "fit_time_seconds": result.fit_time_seconds,
                "score_ms_per_sample": result.score_time_ms_per_sample,
                "extra_memory_mb": result.extra_memory_mb,
                "extra_forward_pass": "no",
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def nearest_known_centroids(
    embeddings: np.ndarray,
    prototypes: dict[int, np.ndarray],
    index_to_label: dict[int, str],
) -> pd.DataFrame:
    """Nearest known centroid and prototype margin for rows."""

    labels = list(prototypes)
    distances = np.vstack(
        [np.linalg.norm(embeddings - prototypes[label], axis=1) for label in labels]
    ).T
    order = np.argsort(distances, axis=1)
    nearest = [index_to_label[labels[idx]] for idx in order[:, 0]]
    margin = (
        distances[np.arange(len(embeddings)), order[:, 1]]
        - distances[np.arange(len(embeddings)), order[:, 0]]
    )
    return pd.DataFrame(
        {
            "nearest_known_class": nearest,
            "nearest_centroid_distance": distances[np.arange(len(embeddings)), order[:, 0]],
            "prototype_margin": margin,
        }
    )


def compute_embedding_geometry(archive: EmbeddingArchive) -> dict[str, Any]:
    """Compute ID-only geometry plus analysis-only unknown summaries."""

    y = label_indices(archive.true_label, archive.label_to_index)
    train_mask = known_train_mask(archive)
    train_embeddings = archive.embedding[train_mask]
    train_labels = y[train_mask]
    prototypes = class_means(train_embeddings, train_labels)
    index_to_label = {idx: label for label, idx in archive.label_to_index.items()}
    test_mask = archive.split == "test"
    test_geometry = nearest_known_centroids(
        archive.embedding[test_mask],
        prototypes,
        index_to_label,
    )
    test_frame = pd.DataFrame(
        {
            "sample_id": archive.sample_id[test_mask],
            "true_label": archive.true_label[test_mask],
            "known_status": archive.known_status[test_mask],
        }
    ).join(test_geometry)
    bank = l2_normalize(train_embeddings)
    neighbors = NearestNeighbors(n_neighbors=5, metric="cosine").fit(bank)
    knn_distances, _indices = neighbors.kneighbors(l2_normalize(archive.embedding[test_mask]))
    test_frame["knn_distance"] = knn_distances.mean(axis=1)

    rows = []
    for label, group in test_frame.groupby("true_label", sort=True):
        class_embeddings = archive.embedding[test_mask][group.index.to_numpy()]
        class_centroid = class_embeddings.mean(axis=0)
        centroid_geo = nearest_known_centroids(
            class_centroid.reshape(1, -1),
            prototypes,
            index_to_label,
        )
        rows.append(
            {
                "class": str(label),
                "known_status": str(group["known_status"].iloc[0]),
                "n": int(len(group)),
                "nearest_known_centroid": str(centroid_geo["nearest_known_class"].iloc[0]),
                "centroid_distance": float(centroid_geo["nearest_centroid_distance"].iloc[0]),
                "mean_knn_distance": float(group["knn_distance"].mean()),
                "median_knn_distance": float(group["knn_distance"].median()),
                "prototype_margin": float(group["prototype_margin"].mean()),
                "analysis_only_unknown_centroid": str(group["known_status"].iloc[0]) != "known",
            }
        )
    class_geometry = pd.DataFrame(rows)

    unknown = test_frame.loc[test_frame["known_status"] != "known"]
    attractor = (
        pd.crosstab(unknown["true_label"], unknown["nearest_known_class"], normalize="index")
        .reindex(columns=list(DEFAULT_KNOWN_CLASSES), fill_value=0.0)
        .sort_index()
    )

    class_centroids = []
    class_names = []
    for label in sorted(set(archive.true_label[test_mask].astype(str).tolist())):
        mask = test_mask & (archive.true_label == label)
        class_centroids.append(archive.embedding[mask].mean(axis=0))
        class_names.append(label)
    centroid_array = np.vstack(class_centroids)
    distance_matrix = pd.DataFrame(
        np.linalg.norm(centroid_array[:, None, :] - centroid_array[None, :, :], axis=2),
        index=class_names,
        columns=class_names,
    )

    known_train_rows = []
    for label_idx, class_name in enumerate(DEFAULT_KNOWN_CLASSES):
        class_embeddings = train_embeddings[train_labels == label_idx]
        dispersion = float(
            np.mean(np.linalg.norm(class_embeddings - prototypes[label_idx], axis=1))
        )
        other_distances = [
            np.linalg.norm(prototypes[label_idx] - prototypes[other])
            for other in prototypes
            if other != label_idx
        ]
        known_train_rows.append(
            {
                "class": class_name,
                "within_class_dispersion": dispersion,
                "nearest_other_centroid_distance": float(np.min(other_distances)),
                "fisher_style_ratio": float(np.min(other_distances) / max(dispersion, 1e-12)),
            }
        )
    compactness = pd.DataFrame(known_train_rows)
    return {
        "sample_geometry": test_frame,
        "class_geometry": class_geometry,
        "attractor": attractor,
        "class_distance": distance_matrix,
        "compactness": compactness,
        "nearest_lookup": class_geometry.set_index("class")["nearest_known_centroid"].to_dict(),
    }


def _unknown_group(label: str) -> str:
    for group, labels in UNKNOWN_GROUPS.items():
        if label in labels:
            return group
    return "unassigned"


def per_unknown_class_table(
    archive: EmbeddingArchive,
    score_by_id: dict[str, np.ndarray],
    comparison: pd.DataFrame,
    nearest_lookup: dict[str, str],
) -> pd.DataFrame:
    """Per-unknown-class metrics for each scored method."""

    deep_scores = score_by_id["msp_primary"]
    deep = _test_prediction_frame(archive, deep_scores, "msp_primary")
    deep_metrics = _per_unknown_for_predictions(deep).set_index("unknown_class")
    frames = []
    for method_id in comparison["method_id"].astype(str):
        pred = _test_prediction_frame(archive, score_by_id[method_id], method_id)
        metrics = _per_unknown_for_predictions(pred)
        metrics["method"] = method_id
        metrics["group"] = metrics["unknown_class"].map(_unknown_group)
        metrics["nearest_known_class"] = metrics["unknown_class"].map(nearest_lookup)
        metrics["delta_AUROC_vs_MSP"] = metrics.apply(
            lambda row: float(row["AUROC"] - deep_metrics.loc[row["unknown_class"], "AUROC"]),
            axis=1,
        )
        metrics["delta_FPR95_vs_MSP"] = metrics.apply(
            lambda row: float(row["FPR95"] - deep_metrics.loc[row["unknown_class"], "FPR95"]),
            axis=1,
        )
        frames.append(metrics)
    return pd.concat(frames, ignore_index=True)


def _test_prediction_frame(
    archive: EmbeddingArchive,
    scores: np.ndarray,
    method_id: str,
) -> pd.DataFrame:
    test_mask = archive.split == "test"
    return pd.DataFrame(
        {
            "sample_id": archive.sample_id[test_mask],
            "true_label": archive.true_label[test_mask],
            "known_status": archive.known_status[test_mask],
            "unknown_score": scores[test_mask],
            "method": method_id,
        }
    )


def _per_unknown_for_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    known = predictions.loc[predictions["known_status"].astype(str) == "known"]
    unknown = predictions.loc[predictions["known_status"].astype(str) != "known"]
    rows = []
    for label, group in unknown.groupby("true_label", sort=True):
        frame = pd.concat([known, group], ignore_index=True)
        y_unknown = (frame["known_status"].astype(str) != "known").astype(int).to_numpy()
        scores = frame["unknown_score"].astype(float).to_numpy()
        rows.append(
            {
                "unknown_class": str(label),
                "n": int(len(group)),
                "AUROC": float(roc_auc_score(y_unknown, scores)),
                "FPR95": fpr_at_tpr(y_unknown, scores),
                "mean_score": float(group["unknown_score"].astype(float).mean()),
                "median_score": float(group["unknown_score"].astype(float).median()),
            }
        )
    return pd.DataFrame(rows)


def subgroup_results(per_class: pd.DataFrame) -> pd.DataFrame:
    """Aggregate predefined unknown groups."""

    rows = []
    deep = per_class.loc[per_class["method"] == "msp_primary"].set_index("unknown_class")
    for method, method_frame in per_class.groupby("method"):
        for group, labels in UNKNOWN_GROUPS.items():
            group_frame = method_frame.loc[method_frame["unknown_class"].isin(labels)]
            if group_frame.empty:
                continue
            deep_group = deep.loc[list(group_frame["unknown_class"])]
            rows.append(
                {
                    "group": group,
                    "classes": ",".join(labels),
                    "method": method,
                    "n": int(group_frame["n"].sum()),
                    "AUROC": float(np.average(group_frame["AUROC"], weights=group_frame["n"])),
                    "FPR95": float(np.average(group_frame["FPR95"], weights=group_frame["n"])),
                    "delta_AUROC": float(
                        np.average(
                            group_frame["AUROC"].to_numpy() - deep_group["AUROC"].to_numpy(),
                            weights=group_frame["n"],
                        )
                    ),
                    "delta_FPR95": float(
                        np.average(
                            group_frame["FPR95"].to_numpy() - deep_group["FPR95"].to_numpy(),
                            weights=group_frame["n"],
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_primary_methods(
    archive: EmbeddingArchive,
    score_by_id: dict[str, np.ndarray],
    comparison: pd.DataFrame,
) -> dict[str, Any]:
    """Paired bootstrap for every primary method against MSP."""

    test_mask = archive.split == "test"
    y_unknown = (archive.known_status[test_mask] != "known").astype(int)
    baseline = score_by_id["msp_primary"][test_mask]
    report = {}
    for row in comparison.loc[comparison["primary"]].itertuples(index=False):
        method_id = str(row.method_id)
        if method_id == "msp_primary":
            continue
        report[method_id] = paired_bootstrap_deltas(
            y_unknown,
            baseline,
            score_by_id[method_id][test_mask],
        )
    return report


def apply_progression_rule(
    comparison: pd.DataFrame,
    subgroup: pd.DataFrame,
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    """Apply predeclared Delivery 4 progression criteria."""

    msp = comparison.loc[comparison["method_id"] == "msp_primary"].iloc[0]
    primary = comparison.loc[comparison["primary"] & (comparison["method_id"] != "msp_primary")]
    criterion_a = False
    criterion_b = False
    for row in primary.itertuples(index=False):
        boot = bootstrap.get(str(row.method_id), {})
        auroc_ci = boot.get("delta_auroc", {})
        delta_auroc = float(row.AUROC - msp.AUROC)
        delta_fpr = float(row.FPR95 - msp.FPR95)
        criterion_a = criterion_a or (
            delta_auroc >= 0.015 and float(auroc_ci.get("lower", -1.0)) > 0.0
        )
        criterion_b = criterion_b or (delta_fpr <= -0.075 and delta_auroc >= -0.005)
    lymphoid = subgroup.loc[subgroup["group"] == "lymphoid_related"]
    primary_ids = set(primary["method_id"].astype(str))
    lymphoid_primary = lymphoid.loc[lymphoid["method"].isin(primary_ids)]
    criterion_c = bool(
        np.any(
            (lymphoid_primary["delta_AUROC"] >= 0.03)
            & (
                lymphoid_primary["method"].map(comparison.set_index("method_id")["AUROC"].to_dict())
                >= float(msp.AUROC) - 0.01
            )
        )
    )
    if criterion_a or criterion_b or criterion_c:
        decision = "Proceed to Delivery 5/confirmatory validation for passing post-hoc method"
    else:
        decision = "Proceed to representation learning"
    return {
        "criterion_a": "PASS" if criterion_a else "FAIL",
        "criterion_b": "PASS" if criterion_b else "FAIL",
        "criterion_c": "PASS" if criterion_c else "FAIL",
        "scientific_decision": decision,
    }


def _write_delivery4_figures(
    archive: EmbeddingArchive,
    comparison: pd.DataFrame,
    per_class: pd.DataFrame,
    geometry: dict[str, Any],
    score_by_id: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    figures = output_dir.parent.parent / "figures" / "delivery4"
    figures.mkdir(parents=True, exist_ok=True)
    primary = comparison.loc[comparison["primary"]].copy()
    test_mask = archive.split == "test"
    y = (archive.known_status[test_mask] != "known").astype(int)

    fig, ax = plt.subplots(figsize=(7, 5))
    for row in primary.itertuples(index=False):
        scores = score_by_id[str(row.method_id)][test_mask]
        fpr, tpr, _thresholds = roc_curve(y, scores)
        ax.plot(fpr, tpr, label=str(row.method_id))
    ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "open_set_roc_delivery4.png", dpi=160)
    plt.close(fig)

    for metric, filename, ylabel in [
        ("FPR95", "open_set_fpr95_delivery4.png", "FPR@95TPR"),
        ("OSCR", "open_set_oscr_delivery4.png", "OSCR"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(primary["method_id"], primary[metric])
        ax.tick_params(axis="x", rotation=60)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=160)
        plt.close(fig)

    msp_auroc = float(comparison.loc[comparison["method_id"] == "msp_primary", "AUROC"].iloc[0])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(primary["method_id"], primary["AUROC"] - msp_auroc)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.tick_params(axis="x", rotation=60)
    ax.set_ylabel("Delta AUROC vs MSP")
    fig.tight_layout()
    fig.savefig(figures / "delta_auroc_vs_msp.png", dpi=160)
    plt.close(fig)

    msp_per = per_class.loc[per_class["method"] == "msp_primary"].sort_values("AUROC")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(msp_per["unknown_class"], msp_per["AUROC"])
    ax.set_xlabel("MSP AUROC")
    fig.tight_layout()
    fig.savefig(figures / "per_unknown_auroc_delivery4.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(geometry["attractor"].to_numpy(), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(geometry["attractor"].columns)), labels=geometry["attractor"].columns)
    ax.set_yticks(range(len(geometry["attractor"].index)), labels=geometry["attractor"].index)
    ax.tick_params(axis="x", rotation=45)
    fig.colorbar(image, ax=ax, label="Fraction")
    fig.tight_layout()
    fig.savefig(figures / "unknown_to_known_attractor_heatmap.png", dpi=160)
    plt.close(fig)

    sample_geo = geometry["sample_geometry"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for status, group in sample_geo.groupby("known_status"):
        ax.hist(group["nearest_centroid_distance"], bins=40, alpha=0.45, density=True, label=status)
    ax.set_xlabel("Distance to nearest known centroid")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "embedding_distance_distributions.png", dpi=160)
    plt.close(fig)

    lymphoid_labels = ["lymphocyte", *UNKNOWN_GROUPS["lymphoid_related"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    for label in lymphoid_labels:
        values = sample_geo.loc[sample_geo["true_label"] == label, "knn_distance"]
        if not values.empty:
            ax.hist(values, bins=35, alpha=0.4, density=True, label=label)
    ax.set_xlabel("Mean cosine distance to 5 known-train neighbors")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "lymphoid_knn_distance_distributions.png", dpi=160)
    plt.close(fig)

    neut = per_class.loc[
        (per_class["unknown_class"] == "neutrophil_band")
        & (per_class["method"].isin(primary["method_id"].astype(str)))
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(neut["method"], neut["AUROC"])
    ax.tick_params(axis="x", rotation=60)
    ax.set_ylabel("AUROC for neutrophil_band")
    fig.tight_layout()
    fig.savefig(figures / "neutrophil_band_method_comparison.png", dpi=160)
    plt.close(fig)

    best = comparison.sort_values("AUROC", ascending=False).iloc[0]["method_id"]
    best_scores = score_by_id[str(best)][test_mask]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(best_scores[y == 0], bins=50, alpha=0.5, density=True, label="known")
    ax.hist(best_scores[y == 1], bins=50, alpha=0.5, density=True, label="unknown")
    ax.set_xlabel(f"Unknownness score: {best}")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "known_unknown_score_distributions_best_method.png", dpi=160)
    plt.close(fig)
