"""Delivery 5 representation-learning evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from hemato_osr.data.taxonomy import DEFAULT_KNOWN_CLASSES
from hemato_osr.evaluation.metrics import (
    calibration_metrics,
    closed_set_metrics,
    open_set_metrics,
    oscr,
)
from hemato_osr.openset.thresholds import apply_threshold, calibrate_strict_open_set
from hemato_osr.topology.cache import config_hash
from hemato_osr.training.representation.train import _sha256_file
from hemato_osr.utils.tracking import environment_metadata, write_json

from .delivery4 import (
    UNKNOWN_GROUPS,
    EmbeddingArchive,
    ViMModel,
    energy_scores,
    knn_cosine_scores,
    l2_normalize,
    label_indices,
    load_embeddings,
    msp_anomaly_scores,
    paired_bootstrap_deltas,
)

OSR_METHODS = ("msp", "energy", "knn_cosine_k5", "vim")
CE_VIM_AUROC = 0.872604
CE_VIM_FPR95 = 0.514223
CLOSED_MACRO_F1_FLOOR = 0.9616


@dataclass(frozen=True)
class Delivery5Config:
    """Inputs for Delivery 5 evaluation."""

    ce_embeddings_path: Path
    supcon_embeddings_path: Path
    arcface_embeddings_path: Path
    output_dir: Path
    ce_checkpoint_path: Path | None = None
    supcon_checkpoint_path: Path | None = None
    arcface_checkpoint_path: Path | None = None
    seed: int = 37
    target_known_recall: float = 0.95


@dataclass(frozen=True)
class Delivery5V2Config:
    """Minimal V2 sensitivity inputs for the passing Delivery 5 candidate."""

    ce_embeddings_path: Path
    candidate_embeddings_path: Path
    output_dir: Path
    candidate_representation: str = "arcface"
    osr_method: str = "msp"
    seed: int = 37
    target_known_recall: float = 0.95


def write_predeclared_matrix(output_path: Path) -> Path:
    """Persist representation matrix and progression criteria before unknown evaluation."""

    payload: dict[str, Any] = {
        "status": "predeclared_before_delivery5_unknown_evaluation",
        "seed": 37,
        "split": "V1",
        "manifest_hash": "4b8da8df49655b6f04b53aa75a912f0efceac776d81761b2b45b68c440c27bae",
        "delivery_status": "exploratory_representation_development",
        "tda_policy": "STOP TDA METHOD DEVELOPMENT; historical negative ablation only",
        "representations": [
            {"name": "ce", "status": "historical", "checkpoint_selection": "historical"},
            {
                "name": "supcon",
                "loss": "CE + lambda * SupCon",
                "lambda": 0.50,
                "temperature": 0.10,
                "projection_head": "512 -> 256 -> 128 with L2 normalization",
                "primary_embedding": "512-dimensional backbone embedding h",
                "checkpoint_selection": "known-validation macro-F1",
                "tie_breakers": ["known-validation Fisher ratio", "within-class dispersion"],
            },
            {
                "name": "arcface",
                "loss": "ArcFace-style additive angular margin",
                "scale": 30.0,
                "margin": 0.30,
                "primary_embedding": "512-dimensional normalized backbone embedding",
                "checkpoint_selection": "known-validation macro-F1",
                "tie_breakers": ["known-validation angular/Fisher separation"],
            },
        ],
        "osr_methods": list(OSR_METHODS),
        "score_orientation": "larger means more UNKNOWN-like",
        "closed_set_safeguard": {
            "metric": "known-test macro-F1",
            "minimum": CLOSED_MACRO_F1_FLOOR,
        },
        "progression_baseline": {
            "representation": "ce",
            "osr_method": "vim",
            "AUROC": CE_VIM_AUROC,
            "FPR95": CE_VIM_FPR95,
        },
        "progression_criteria": {
            "criterion_a": "Delta AUROC vs CE+ViM >= +0.015 with favorable paired bootstrap CI",
            "criterion_b": (
                "Delta FPR95 vs CE+ViM <= -0.075, Delta AUROC >= -0.005, "
                "and closed macro-F1 safeguard passes"
            ),
            "criterion_c": (
                "Delta lymphoid-related AUROC >= +0.03 vs best CE scorer, "
                "overall AUROC degradation <= 0.01, and closed macro-F1 safeguard passes"
            ),
        },
        "unknown_groups": UNKNOWN_GROUPS,
        "unknown_usage": "analysis only; never training, checkpoint selection, or OSR selection",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _representation_archives(config: Delivery5Config) -> dict[str, EmbeddingArchive]:
    return {
        "ce": load_embeddings(config.ce_embeddings_path),
        "supcon": load_embeddings(config.supcon_embeddings_path),
        "arcface": load_embeddings(config.arcface_embeddings_path),
    }


def _checkpoint_hash(path: Path | None) -> str:
    return _sha256_file(path) if path is not None and path.exists() else ""


def _score_representation(
    archive: EmbeddingArchive,
    method: str,
) -> tuple[np.ndarray, dict[str, Any], float]:
    fit_mask = (archive.split == "train") & (archive.known_status == "known")
    train_embeddings = archive.embedding[fit_mask]
    train_logits = archive.logits[fit_mask]
    if method == "msp":
        return msp_anomaly_scores(archive.logits), {"fit": "none"}, 0.0
    if method == "energy":
        return energy_scores(archive.logits, temperature=1.0), {"fit": "none", "T": 1.0}, 0.0
    if method == "knn_cosine_k5":
        import time

        start = time.perf_counter()
        scores, _neighbors = knn_cosine_scores(train_embeddings, archive.embedding, k=5)
        elapsed = (time.perf_counter() - start) * 1000.0 / len(scores)
        return scores, {"fit_split": "known_train", "k": 5, "metric": "cosine"}, elapsed
    if method == "vim":
        import time

        start = time.perf_counter()
        model = ViMModel.fit(train_embeddings, train_logits)
        scores = model.score(archive.embedding, archive.logits)
        elapsed = (time.perf_counter() - start) * 1000.0 / len(scores)
        return (
            scores,
            {
                "fit_split": "known_train",
                "n_components": model.n_components,
                "explained_variance_ratio": model.explained_variance_ratio,
                "alpha": model.alpha,
            },
            elapsed,
        )
    msg = f"Unknown OSR method: {method}"
    raise ValueError(msg)


def _closed_metrics(archive: EmbeddingArchive) -> tuple[dict[str, Any], pd.DataFrame]:
    test = (archive.split == "test") & (archive.known_status == "known")
    y_true = label_indices(archive.true_label[test], archive.label_to_index)
    logits = archive.logits[test]
    y_pred = logits.argmax(axis=1)
    closed = closed_set_metrics(
        y_true.astype(int).tolist(),
        y_pred.astype(int).tolist(),
        DEFAULT_KNOWN_CLASSES,
    )
    cal = calibration_metrics(logits, y_true)
    rows = []
    per_class = cast(dict[str, dict[str, Any]], closed["per_class"])
    for class_name, values in per_class.items():
        rows.append({"class": class_name, **values})
    return {**{k: v for k, v in closed.items() if k not in {"per_class"}}, **cal}, pd.DataFrame(
        rows
    )


def _open_metrics(
    archive: EmbeddingArchive,
    scores: np.ndarray,
    *,
    target_known_recall: float,
) -> dict[str, float]:
    validation_known = (archive.split == "validation") & (archive.known_status == "known")
    test = archive.split == "test"
    threshold = calibrate_strict_open_set(
        scores[validation_known],
        target_known_recall=target_known_recall,
    )
    y_unknown = (archive.known_status[test] != "known").astype(int)
    test_scores = scores[test]
    unknown_pred = apply_threshold(test_scores, threshold.threshold)
    pred_with_unknown = archive.prediction[test].astype(object)
    pred_with_unknown[unknown_pred == 1] = "UNKNOWN"
    true_with_unknown = archive.true_label[test].astype(object)
    true_with_unknown[y_unknown == 1] = "UNKNOWN"
    metrics = open_set_metrics(y_unknown, test_scores, true_with_unknown, pred_with_unknown)
    metrics["AUPR_known"] = float(average_precision_score(1 - y_unknown, -test_scores))
    y_true = label_indices(archive.true_label[test], archive.label_to_index)
    metrics["OSCR"] = oscr(y_true, archive.logits[test].argmax(axis=1), -test_scores, y_unknown)
    metrics["threshold"] = threshold.threshold
    return metrics


def _prediction_frame(
    archive: EmbeddingArchive, scores: np.ndarray, representation: str, method: str
) -> pd.DataFrame:
    test = archive.split == "test"
    return pd.DataFrame(
        {
            "sample_id": archive.sample_id[test],
            "true_label": archive.true_label[test],
            "known_status": archive.known_status[test],
            "unknown_score": scores[test],
            "representation": representation,
            "osr_method": method,
        }
    )


def _unknown_group(label: str) -> str:
    for group, labels in UNKNOWN_GROUPS.items():
        if label in labels:
            return group
    return "unassigned"


def _per_unknown_rows(
    predictions: pd.DataFrame,
    *,
    nearest_lookup: dict[str, str],
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    known = predictions.loc[predictions["known_status"].astype(str) == "known"]
    unknown = predictions.loc[predictions["known_status"].astype(str) != "known"]
    baseline_index = (
        baseline.set_index(["unknown_class", "osr_method"]) if not baseline.empty else None
    )
    rows = []
    for label, group in unknown.groupby("true_label", sort=True):
        frame = pd.concat([known, group], ignore_index=True)
        y_unknown = (frame["known_status"].astype(str) != "known").astype(int).to_numpy()
        scores = frame["unknown_score"].astype(float).to_numpy()
        method = str(group["osr_method"].iloc[0])
        base_auroc = np.nan
        base_fpr = np.nan
        if baseline_index is not None and (str(label), method) in baseline_index.index:
            base_row = baseline_index.loc[(str(label), method)]
            base_auroc = float(base_row["AUROC"])
            base_fpr = float(base_row["FPR95"])
        auroc = float(roc_auc_score(y_unknown, scores))
        fpr = float(_fpr_at_95(y_unknown, scores))
        rows.append(
            {
                "unknown_class": str(label),
                "group": _unknown_group(str(label)),
                "representation": str(group["representation"].iloc[0]),
                "osr_method": method,
                "n": int(len(group)),
                "AUROC": auroc,
                "FPR95": fpr,
                "mean_score": float(group["unknown_score"].astype(float).mean()),
                "median_score": float(group["unknown_score"].astype(float).median()),
                "nearest_known_class": nearest_lookup.get(str(label), ""),
                "delta_AUROC_vs_CE": float(auroc - base_auroc) if np.isfinite(base_auroc) else 0.0,
                "delta_FPR95_vs_CE": float(fpr - base_fpr) if np.isfinite(base_fpr) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _fpr_at_95(y_unknown: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[y_unknown == 1]
    negatives = scores[y_unknown == 0]
    threshold = float(np.quantile(positives, 0.05))
    return float(np.mean(negatives > threshold))


def _known_geometry(
    archive: EmbeddingArchive, representation: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = label_indices(archive.true_label, archive.label_to_index)
    rows = []
    summary_rows = []
    for split in ["train", "validation", "test"]:
        mask = (archive.split == split) & (archive.known_status == "known")
        embeddings = archive.embedding[mask]
        labels = y[mask]
        centroids = {}
        for class_idx, _class_name in enumerate(DEFAULT_KNOWN_CLASSES):
            class_embeddings = embeddings[labels == class_idx]
            centroid = class_embeddings.mean(axis=0)
            centroids[class_idx] = centroid
        centroid_array = np.vstack([centroids[idx] for idx in range(len(DEFAULT_KNOWN_CLASSES))])
        centroid_dist = np.linalg.norm(
            centroid_array[:, None, :] - centroid_array[None, :, :],
            axis=2,
        )
        nearest = np.partition(centroid_dist + np.eye(len(DEFAULT_KNOWN_CLASSES)) * 1e9, 1, axis=1)[
            :, 0
        ]
        between = centroid_dist[np.triu_indices_from(centroid_dist, k=1)]
        class_stats = []
        for class_idx, class_name in enumerate(DEFAULT_KNOWN_CLASSES):
            class_embeddings = embeddings[labels == class_idx]
            centroid = centroids[class_idx]
            within = float(np.mean(np.linalg.norm(class_embeddings - centroid, axis=1)))
            normed = l2_normalize(class_embeddings)
            proto = l2_normalize(centroid.reshape(1, -1))[0]
            cosine = float(np.mean(normed @ proto))
            class_stats.append((class_idx, class_name, within, cosine, centroid))
        split_within = [item[2] for item in class_stats]
        split_cosine = [item[3] for item in class_stats]
        split_fisher = float(np.mean(between) / max(float(np.mean(split_within)), 1e-12))
        for class_idx, class_name, within, cosine, centroid in class_stats:
            rows.append(
                {
                    "representation": representation,
                    "split": split,
                    "class": class_name,
                    "within_class_dispersion": within,
                    "centroid_norm": float(np.linalg.norm(centroid)),
                    "nearest_centroid_distance": float(nearest[class_idx]),
                    "mean_cosine_to_prototype": cosine,
                    "fisher_ratio": split_fisher,
                }
            )
        summary_rows.append(
            {
                "representation": representation,
                "split": split,
                "within_class_dispersion": float(np.mean(split_within)),
                "between_class_separation": float(np.mean(between)),
                "fisher_ratio": float(np.mean(between) / max(float(np.mean(split_within)), 1e-12)),
                "mean_cosine_compactness": float(np.mean(split_cosine)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def _unknown_geometry(
    archive: EmbeddingArchive,
    representation: str,
    vim_scores: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    y = label_indices(archive.true_label, archive.label_to_index)
    train = (archive.split == "train") & (archive.known_status == "known")
    train_embeddings = archive.embedding[train]
    train_labels = y[train]
    prototypes = {
        idx: train_embeddings[train_labels == idx].mean(axis=0)
        for idx in range(len(DEFAULT_KNOWN_CLASSES))
    }
    test = archive.split == "test"
    labels = list(prototypes)
    distances = np.vstack(
        [np.linalg.norm(archive.embedding[test] - prototypes[idx], axis=1) for idx in labels]
    ).T
    order = np.argsort(distances, axis=1)
    nearest_idx = order[:, 0]
    nearest_class = [DEFAULT_KNOWN_CLASSES[labels[idx]] for idx in nearest_idx]
    margin = (
        distances[np.arange(len(distances)), order[:, 1]]
        - distances[np.arange(len(distances)), order[:, 0]]
    )
    neighbors = NearestNeighbors(n_neighbors=5, metric="cosine").fit(l2_normalize(train_embeddings))
    knn_dist, _indices = neighbors.kneighbors(l2_normalize(archive.embedding[test]))
    sample = pd.DataFrame(
        {
            "representation": representation,
            "sample_id": archive.sample_id[test],
            "true_label": archive.true_label[test],
            "known_status": archive.known_status[test],
            "nearest_known_class": nearest_class,
            "nearest_known_distance": distances[np.arange(len(distances)), nearest_idx],
            "knn_distance": knn_dist.mean(axis=1),
            "prototype_margin": margin,
            "vim_score": vim_scores[test],
        }
    )
    unknown = sample.loc[sample["known_status"] != "known"].copy()
    attractor = (
        pd.crosstab(unknown["true_label"], unknown["nearest_known_class"], normalize="index")
        .reindex(columns=list(DEFAULT_KNOWN_CLASSES), fill_value=0.0)
        .sort_index()
    )
    nearest_lookup = (
        unknown.groupby("true_label")["nearest_known_class"]
        .agg(lambda values: values.value_counts().index[0])
        .to_dict()
    )
    return sample, attractor, {str(k): str(v) for k, v in nearest_lookup.items()}


def _subgroup_results(per_class: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (representation, method), frame in per_class.groupby(["representation", "osr_method"]):
        for group_name, labels in UNKNOWN_GROUPS.items():
            group = frame.loc[frame["unknown_class"].isin(labels)]
            if group.empty:
                continue
            rows.append(
                {
                    "representation": representation,
                    "osr_method": method,
                    "group": group_name,
                    "n": int(group["n"].sum()),
                    "AUROC": float(np.average(group["AUROC"], weights=group["n"])),
                    "FPR95": float(np.average(group["FPR95"], weights=group["n"])),
                }
            )
    return pd.DataFrame(rows)


def _progression(
    comparison: pd.DataFrame,
    subgroup: pd.DataFrame,
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    ce_vim = comparison.loc[
        (comparison["representation"] == "ce") & (comparison["osr_method"] == "vim")
    ].iloc[0]
    candidates = comparison.loc[comparison["representation"].isin(["supcon", "arcface"])].copy()
    best = candidates.sort_values(["AUROC", "FPR95"], ascending=[False, True]).iloc[0]
    best_key = f"{best.representation}__{best.osr_method}"
    boot = bootstrap.get(best_key, {})
    macro_pass_by_rep = {
        rep: bool(frame["closed_macro_f1"].iloc[0] >= CLOSED_MACRO_F1_FLOOR)
        for rep, frame in candidates.groupby("representation")
    }
    best_macro_pass = bool(best.closed_macro_f1 >= CLOSED_MACRO_F1_FLOOR)
    delta_auroc = float(best.AUROC - ce_vim.AUROC)
    delta_fpr = float(best.FPR95 - ce_vim.FPR95)
    criterion_a = (
        best_macro_pass
        and delta_auroc >= 0.015
        and float(boot.get("delta_auroc", {}).get("lower", -1)) > 0
    )
    criterion_b = best_macro_pass and delta_fpr <= -0.075 and delta_auroc >= -0.005
    ce_lymphoid = subgroup.loc[
        (subgroup["representation"] == "ce") & (subgroup["group"] == "lymphoid_related")
    ]["AUROC"].max()
    lymphoid_candidates = subgroup.loc[
        subgroup["representation"].isin(["supcon", "arcface"])
        & (subgroup["group"] == "lymphoid_related")
    ].copy()
    lymphoid_candidates["overall_auroc"] = lymphoid_candidates.apply(
        lambda row: float(
            comparison.loc[
                (comparison["representation"] == row["representation"])
                & (comparison["osr_method"] == row["osr_method"]),
                "AUROC",
            ].iloc[0]
        ),
        axis=1,
    )
    lymphoid_candidates["closed_pass"] = lymphoid_candidates["representation"].map(
        macro_pass_by_rep
    )
    criterion_c = bool(
        np.any(
            (lymphoid_candidates["AUROC"] - float(ce_lymphoid) >= 0.03)
            & (lymphoid_candidates["overall_auroc"] >= float(ce_vim.AUROC) - 0.01)
            & lymphoid_candidates["closed_pass"]
        )
    )
    if criterion_a or criterion_b or criterion_c:
        passing = candidates.loc[candidates["closed_macro_f1"] >= CLOSED_MACRO_F1_FLOOR]
        winner = passing.sort_values(
            ["AUROC", "FPR95", "closed_macro_f1"],
            ascending=[False, True, False],
        ).iloc[0]
        decision = f"{winner.representation} progresses to Delivery 6 confirmatory validation"
    else:
        decision = "Neither progresses; move to external validation with strongest existing system"
    return {
        "best_candidate": {
            "representation": str(best.representation),
            "osr_method": str(best.osr_method),
            "AUROC": float(best.AUROC),
            "FPR95": float(best.FPR95),
            "delta_AUROC_vs_CE_ViM": delta_auroc,
            "delta_FPR95_vs_CE_ViM": delta_fpr,
            "closed_macro_f1": float(best.closed_macro_f1),
        },
        "closed_set_safeguard": "PASS" if best_macro_pass else "FAIL",
        "criterion_a": "PASS" if criterion_a else "FAIL",
        "criterion_b": "PASS" if criterion_b else "FAIL",
        "criterion_c": "PASS" if criterion_c else "FAIL",
        "scientific_decision": decision,
    }


def evaluate_delivery5(config: Delivery5Config) -> Path:
    """Evaluate CE/SupCon/ArcFace representation matrix."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = config.output_dir.parent.parent / "figures" / "delivery5"
    figures_dir.mkdir(parents=True, exist_ok=True)
    archives = _representation_archives(config)
    checkpoint_hash = {
        "ce": _checkpoint_hash(config.ce_checkpoint_path),
        "supcon": _checkpoint_hash(config.supcon_checkpoint_path),
        "arcface": _checkpoint_hash(config.arcface_checkpoint_path),
    }
    score_by_key: dict[str, np.ndarray] = {}
    method_state: dict[str, Any] = {}
    comparison_rows = []
    per_class_frames = []
    closed_rows = []
    per_class_closed = []
    geometry_frames = []
    geometry_summary_frames = []
    unknown_geometry_frames = []
    attractors: dict[str, pd.DataFrame] = {}
    nearest_lookup: dict[str, dict[str, str]] = {}
    scoring_ms: dict[str, float] = {}
    ce_baseline_per_class = pd.DataFrame()
    (config.output_dir / "predictions").mkdir(parents=True, exist_ok=True)

    for representation, archive in archives.items():
        closed, closed_per_class = _closed_metrics(archive)
        closed_rows.append({"representation": representation, **closed})
        closed_per_class["representation"] = representation
        per_class_closed.append(closed_per_class)
        rep_scores: dict[str, np.ndarray] = {}
        for method in OSR_METHODS:
            scores, state, score_ms = _score_representation(archive, method)
            key = f"{representation}__{method}"
            score_by_key[key] = scores
            rep_scores[method] = scores
            method_state[key] = state
            scoring_ms[key] = score_ms
        sample_geo, attractor, lookup = _unknown_geometry(
            archive, representation, rep_scores["vim"]
        )
        unknown_geometry_frames.append(sample_geo)
        attractors[representation] = attractor
        nearest_lookup[representation] = lookup
        known_geo, known_summary = _known_geometry(archive, representation)
        geometry_frames.append(known_geo)
        geometry_summary_frames.append(known_summary)

        for method in OSR_METHODS:
            metrics = _open_metrics(
                archive,
                rep_scores[method],
                target_known_recall=config.target_known_recall,
            )
            comparison_rows.append(
                {
                    "representation": representation,
                    "training_loss": "historical_ce"
                    if representation == "ce"
                    else ("CE+SupCon" if representation == "supcon" else "ArcFace"),
                    "osr_method": method,
                    "seed": config.seed,
                    "closed_accuracy": closed["accuracy"],
                    "closed_balanced_accuracy": closed["balanced_accuracy"],
                    "closed_macro_f1": closed["macro_f1"],
                    "ECE": closed["ece"],
                    "NLL": closed["nll"],
                    "Brier": closed["brier_score"],
                    "AUROC": metrics["auroc_known_unknown"],
                    "AUPR_unknown": metrics["aupr_known_unknown"],
                    "AUPR_known": metrics["AUPR_known"],
                    "FPR95": metrics["fpr_at_95_tpr"],
                    "OSCR": metrics["OSCR"],
                    "known_recall": metrics["known_recall"],
                    "unknown_recall": metrics["unknown_recall"],
                    "config_hash": config_hash(
                        {"representation": representation, "osr_method": method}
                    ),
                    "checkpoint_hash": checkpoint_hash[representation],
                    "scoring_ms_per_image": scoring_ms[f"{representation}__{method}"],
                }
            )
            predictions = _prediction_frame(archive, rep_scores[method], representation, method)
            predictions.to_csv(
                config.output_dir / "predictions" / f"{representation}_{method}_predictions.csv",
                index=False,
            )
            per_rows = _per_unknown_rows(
                predictions,
                nearest_lookup=nearest_lookup[representation],
                baseline=ce_baseline_per_class,
            )
            if representation == "ce":
                ce_baseline_per_class = pd.concat(
                    [ce_baseline_per_class, per_rows], ignore_index=True
                )
            per_class_frames.append(per_rows)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(config.output_dir / "method_comparison.csv", index=False)
    pd.DataFrame(closed_rows).to_csv(config.output_dir / "closed_set_results.csv", index=False)
    pd.concat(per_class_closed, ignore_index=True).to_csv(
        config.output_dir / "closed_set_per_class.csv",
        index=False,
    )
    representation_geometry = pd.concat(geometry_frames, ignore_index=True)
    representation_geometry.to_csv(config.output_dir / "representation_geometry.csv", index=False)
    geometry_summary = pd.concat(geometry_summary_frames, ignore_index=True)
    geometry_summary.to_csv(config.output_dir / "representation_geometry_summary.csv", index=False)
    unknown_geometry = pd.concat(unknown_geometry_frames, ignore_index=True)
    unknown_geometry.to_csv(config.output_dir / "unknown_geometry.csv", index=False)
    for representation, attractor in attractors.items():
        attractor.to_csv(config.output_dir / f"unknown_to_known_attractor_{representation}.csv")
    per_class = pd.concat(per_class_frames, ignore_index=True)
    per_class.to_csv(config.output_dir / "per_unknown_class.csv", index=False)
    subgroup = _subgroup_results(per_class)
    subgroup.to_csv(config.output_dir / "subgroup_results.csv", index=False)
    bootstrap = _bootstrap_against_ce_vim(archives, score_by_key, comparison, config.seed)
    write_json(config.output_dir / "paired_bootstrap_vs_ce_vim.json", bootstrap)
    decision = _progression(comparison, subgroup, bootstrap)
    write_json(config.output_dir / "progression_decision.json", decision)
    _write_analysis_tables(config.output_dir, per_class, unknown_geometry, attractors)
    _write_figures(
        figures_dir,
        comparison,
        geometry_summary,
        per_class,
        unknown_geometry,
        attractors,
        archives,
        score_by_key,
    )
    write_json(config.output_dir / "method_state.json", method_state)
    manifest_hash = next(iter(archives.values())).manifest_hash
    write_json(config.output_dir / "environment.json", environment_metadata(manifest_hash))
    return config.output_dir / "method_comparison.csv"


def evaluate_delivery5_v2(config: Delivery5V2Config) -> Path:
    """Evaluate only the winning candidate/scorer against CE on Split V2."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    archives = {
        "ce": load_embeddings(config.ce_embeddings_path),
        config.candidate_representation: load_embeddings(config.candidate_embeddings_path),
    }
    rows = []
    score_by_rep = {}
    for representation, archive in archives.items():
        closed, _closed_per_class = _closed_metrics(archive)
        scores, state, score_ms = _score_representation(archive, config.osr_method)
        metrics = _open_metrics(
            archive,
            scores,
            target_known_recall=config.target_known_recall,
        )
        score_by_rep[representation] = scores
        rows.append(
            {
                "split": "V2",
                "representation": representation,
                "osr_method": config.osr_method,
                "closed_accuracy": closed["accuracy"],
                "closed_balanced_accuracy": closed["balanced_accuracy"],
                "closed_macro_f1": closed["macro_f1"],
                "ECE": closed["ece"],
                "NLL": closed["nll"],
                "Brier": closed["brier_score"],
                "AUROC": metrics["auroc_known_unknown"],
                "AUPR_unknown": metrics["aupr_known_unknown"],
                "AUPR_known": metrics["AUPR_known"],
                "FPR95": metrics["fpr_at_95_tpr"],
                "OSCR": metrics["OSCR"],
                "known_recall": metrics["known_recall"],
                "unknown_recall": metrics["unknown_recall"],
                "score_ms_per_image": score_ms,
                "state": json.dumps(state, sort_keys=True),
            }
        )
    result = pd.DataFrame(rows)
    ce = result.loc[result["representation"] == "ce"].iloc[0]
    candidate = result.loc[result["representation"] == config.candidate_representation].iloc[0]
    test = archives["ce"].split == "test"
    y_unknown = (archives["ce"].known_status[test] != "known").astype(int)
    bootstrap = paired_bootstrap_deltas(
        y_unknown,
        score_by_rep["ce"][test],
        score_by_rep[config.candidate_representation][test],
        seed=config.seed,
    )
    summary = {
        "candidate": config.candidate_representation,
        "osr_method": config.osr_method,
        "delta_AUROC": float(candidate.AUROC - ce.AUROC),
        "delta_FPR95": float(candidate.FPR95 - ce.FPR95),
        "delta_closed_macro_f1": float(candidate.closed_macro_f1 - ce.closed_macro_f1),
        "paired_bootstrap": bootstrap,
    }
    result.to_csv(config.output_dir / "v2_sensitivity.csv", index=False)
    write_json(config.output_dir / "v2_sensitivity_summary.json", summary)
    return config.output_dir / "v2_sensitivity.csv"


def _bootstrap_against_ce_vim(
    archives: dict[str, EmbeddingArchive],
    score_by_key: dict[str, np.ndarray],
    comparison: pd.DataFrame,
    seed: int,
) -> dict[str, Any]:
    test = archives["ce"].split == "test"
    y_unknown = (archives["ce"].known_status[test] != "known").astype(int)
    baseline = score_by_key["ce__vim"][test]
    report = {}
    for row in comparison.itertuples(index=False):
        key = f"{row.representation}__{row.osr_method}"
        if key == "ce__vim":
            continue
        report[key] = paired_bootstrap_deltas(
            y_unknown,
            baseline,
            score_by_key[key][test],
            seed=seed,
        )
    return report


def _write_analysis_tables(
    output_dir: Path,
    per_class: pd.DataFrame,
    unknown_geometry: pd.DataFrame,
    attractors: dict[str, pd.DataFrame],
) -> None:
    lymphoid = [
        "lymphocyte_large_granular",
        "lymphocyte_neoplastic",
        "lymphocyte_reactive",
        "hairy_cell",
        "plasma_cell",
    ]
    rows = []
    for representation, frame in unknown_geometry.groupby("representation"):
        attractor = attractors[str(representation)]
        for label in lymphoid:
            group = frame.loc[frame["true_label"] == label]
            rows.append(
                {
                    "representation": representation,
                    "unknown_class": label,
                    "nearest_known_distance": float(group["nearest_known_distance"].mean()),
                    "knn_distance": float(group["knn_distance"].mean()),
                    "prototype_margin": float(group["prototype_margin"].mean()),
                    "vim_residual_score": float(group["vim_score"].mean()),
                    "attractor_proportion_to_lymphocyte": float(
                        attractor.loc[label, "lymphocyte"] if label in attractor.index else 0.0
                    ),
                    "vim_AUROC": float(
                        per_class.loc[
                            (per_class["representation"] == representation)
                            & (per_class["osr_method"] == "vim")
                            & (per_class["unknown_class"] == label),
                            "AUROC",
                        ].iloc[0]
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "lymphoid_failure_analysis.csv", index=False)

    neut_rows = []
    for representation, frame in unknown_geometry.groupby("representation"):
        group = frame.loc[frame["true_label"] == "neutrophil_band"]
        for method in ["msp", "knn_cosine_k5", "vim"]:
            per = per_class.loc[
                (per_class["representation"] == representation)
                & (per_class["osr_method"] == method)
                & (per_class["unknown_class"] == "neutrophil_band")
            ].iloc[0]
            neut_rows.append(
                {
                    "representation": representation,
                    "osr_method": method,
                    "AUROC": float(per["AUROC"]),
                    "FPR95": float(per["FPR95"]),
                    "distance_to_neutrophil_segmented": float(
                        group["nearest_known_distance"].mean()
                    ),
                    "knn_distance": float(group["knn_distance"].mean()),
                    "vim_residual_score": float(group["vim_score"].mean()),
                }
            )
    pd.DataFrame(neut_rows).to_csv(output_dir / "neutrophil_band_analysis.csv", index=False)


def _write_figures(
    figures_dir: Path,
    comparison: pd.DataFrame,
    geometry_summary: pd.DataFrame,
    per_class: pd.DataFrame,
    unknown_geometry: pd.DataFrame,
    attractors: dict[str, pd.DataFrame],
    archives: dict[str, EmbeddingArchive],
    score_by_key: dict[str, np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    closed = comparison.drop_duplicates("representation")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(closed["representation"], closed["closed_macro_f1"])
    ax.axhline(CLOSED_MACRO_F1_FLOOR, color="black", linestyle=":", linewidth=0.8)
    ax.set_ylabel("Known-test macro-F1")
    fig.tight_layout()
    fig.savefig(figures_dir / "closed_macro_f1_by_representation.png", dpi=160)
    plt.close(fig)

    for metric, filename in [
        ("AUROC", "open_set_auroc_representation_score_matrix.png"),
        ("FPR95", "open_set_fpr95_representation_score_matrix.png"),
        ("OSCR", "oscr_representation_score_matrix.png"),
    ]:
        pivot = comparison.pivot(index="representation", columns="osr_method", values=metric).loc[
            ["ce", "supcon", "arcface"], list(OSR_METHODS)
        ]
        fig, ax = plt.subplots(figsize=(7, 4))
        image = ax.imshow(pivot.to_numpy(), cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=30)
        ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
        fig.colorbar(image, ax=ax, label=metric)
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=160)
        plt.close(fig)

    train_summary = geometry_summary.loc[geometry_summary["split"] == "train"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        train_summary["representation"],
        train_summary["within_class_dispersion"],
        marker="o",
        label="within",
    )
    ax.plot(
        train_summary["representation"],
        train_summary["between_class_separation"],
        marker="o",
        label="between",
    )
    ax.set_ylabel("Euclidean distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "geometry_within_between_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    unknown = unknown_geometry.loc[unknown_geometry["known_status"] != "known"]
    for representation, group in unknown.groupby("representation"):
        ax.hist(
            group["nearest_known_distance"], bins=40, alpha=0.4, density=True, label=representation
        )
    ax.set_xlabel("Unknown distance to nearest known prototype")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "unknown_distance_by_representation.png", dpi=160)
    plt.close(fig)

    lymphoid = unknown.loc[unknown["true_label"].isin(UNKNOWN_GROUPS["lymphoid_related"])]
    fig, ax = plt.subplots(figsize=(7, 4))
    for representation, group in lymphoid.groupby("representation"):
        ax.hist(
            group["nearest_known_distance"], bins=35, alpha=0.4, density=True, label=representation
        )
    ax.set_xlabel("Lymphoid unknown distance to nearest known prototype")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "lymphoid_distance_comparison.png", dpi=160)
    plt.close(fig)

    lymphoid_auc = per_class.loc[
        (per_class["group"] == "lymphoid_related") & (per_class["osr_method"] == "vim")
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    for representation, group in lymphoid_auc.groupby("representation"):
        ax.plot(group["unknown_class"], group["AUROC"], marker="o", label=representation)
    ax.tick_params(axis="x", rotation=60)
    ax.set_ylabel("ViM AUROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "lymphoid_auroc_comparison.png", dpi=160)
    plt.close(fig)

    neut = per_class.loc[
        (per_class["unknown_class"] == "neutrophil_band")
        & (per_class["osr_method"].isin(["msp", "knn_cosine_k5", "vim"]))
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    for method, group in neut.groupby("osr_method"):
        ax.plot(group["representation"], group["AUROC"], marker="o", label=method)
    ax.set_ylabel("neutrophil_band AUROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "neutrophil_band_comparison.png", dpi=160)
    plt.close(fig)

    for representation, attractor in attractors.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        image = ax.imshow(attractor.to_numpy(), vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(attractor.columns)), labels=attractor.columns, rotation=45)
        ax.set_yticks(range(len(attractor.index)), labels=attractor.index)
        fig.colorbar(image, ax=ax, label="Fraction")
        fig.tight_layout()
        fig.savefig(figures_dir / f"attractor_heatmap_{representation}.png", dpi=160)
        plt.close(fig)

    for representation, archive in archives.items():
        _write_embedding_visualization(figures_dir, representation, archive)


def _write_embedding_visualization(
    figures_dir: Path, representation: str, archive: EmbeddingArchive
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(37)
    test = archive.split == "test"
    embeddings = archive.embedding[test]
    labels = archive.true_label[test].astype(str)
    max_points = min(4000, len(labels))
    idx = rng.choice(np.arange(len(labels)), size=max_points, replace=False)
    selected = embeddings[idx]
    selected_labels = labels[idx]
    pca = PCA(n_components=2, random_state=37).fit_transform(selected)
    fig, ax = plt.subplots(figsize=(7, 5))
    for label in sorted(set(selected_labels.tolist())):
        mask = selected_labels == label
        ax.scatter(pca[mask, 0], pca[mask, 1], s=4, alpha=0.5, label=label)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=5, ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / f"pca_{representation}.png", dpi=160)
    plt.close(fig)

    try:
        import umap  # type: ignore[import-not-found]

        coords = umap.UMAP(
            n_neighbors=30,
            min_dist=0.10,
            metric="cosine",
            random_state=37,
        ).fit_transform(selected)
    except Exception:
        coords = pca
    fig, ax = plt.subplots(figsize=(7, 5))
    for label in sorted(set(selected_labels.tolist())):
        mask = selected_labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=4, alpha=0.5, label=label)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=5, ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / f"umap_{representation}.png", dpi=160)
    plt.close(fig)
