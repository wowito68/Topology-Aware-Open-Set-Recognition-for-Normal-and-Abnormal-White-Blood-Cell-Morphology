"""Delivery 2 comparison tables, per-class analysis, bootstrap, and figures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from hemato_osr.evaluation.metrics import fpr_at_tpr


@dataclass(frozen=True)
class MethodArtifact:
    """Paths and labels for one evaluated method."""

    split_protocol: str
    representation: str
    classifier: str
    osr_method: str
    metrics_path: Path
    predictions_path: Path
    config_hash: str = ""
    seed: int = 37


def _method_label(method: MethodArtifact) -> str:
    return f"{method.split_protocol} {method.representation}/{method.osr_method}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_model_comparison(methods: list[MethodArtifact], output_path: Path) -> Path:
    """Write the required model comparison CSV."""

    rows = []
    for method in methods:
        metrics = _read_json(method.metrics_path)
        closed = metrics.get("closed_set", {})
        open_set = metrics.get("open_set", {})
        rows.append(
            {
                "split_protocol": method.split_protocol,
                "representation": method.representation,
                "classifier": method.classifier,
                "osr_method": method.osr_method,
                "closed_macro_f1": closed.get("macro_f1", np.nan),
                "closed_balanced_accuracy": closed.get("balanced_accuracy", np.nan),
                "open_auroc": open_set.get("auroc_known_unknown", np.nan),
                "open_aupr_unknown_positive": open_set.get("aupr_known_unknown", np.nan),
                "fpr95": open_set.get("fpr_at_95_tpr", np.nan),
                "oscr": open_set.get("oscr", np.nan),
                "seed": method.seed,
                "config_hash": method.config_hash,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def per_unknown_class_metrics(
    predictions: pd.DataFrame,
    *,
    method: str,
) -> pd.DataFrame:
    """Compute one-vs-known metrics for each unknown morphology."""

    known = predictions.loc[predictions["known_status"].astype(str) == "known"]
    unknown = predictions.loc[predictions["known_status"].astype(str) != "known"]
    rows = []
    for label, group in unknown.groupby("true_label", sort=True):
        frame = pd.concat([known, group], ignore_index=True)
        y_unknown = (frame["known_status"].astype(str) != "known").astype(int).to_numpy()
        scores = frame["unknown_score"].astype(float).to_numpy()
        auroc = (
            float(roc_auc_score(y_unknown, scores)) if len(set(y_unknown.tolist())) == 2 else np.nan
        )
        rows.append(
            {
                "unknown_class": str(label),
                "n_samples": int(len(group)),
                "method": method,
                "AUROC": auroc,
                "FPR95": fpr_at_tpr(y_unknown, scores, target_tpr=0.95),
                "mean_score": float(np.mean(group["unknown_score"].astype(float))),
                "median_score": float(np.median(group["unknown_score"].astype(float))),
            }
        )
    return pd.DataFrame(rows)


def write_per_unknown_class_table(
    methods: list[MethodArtifact],
    output_path: Path,
) -> Path:
    """Write per-unknown-class metrics for all methods."""

    rows = []
    for method in methods:
        predictions = pd.read_csv(method.predictions_path)
        rows.append(
            per_unknown_class_metrics(
                predictions,
                method=_method_label(method),
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(output_path, index=False)
    return output_path


def _bootstrap_indices(n: int, n_bootstraps: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, size=n) for _ in range(n_bootstraps)]


def bootstrap_open_metric(
    predictions: pd.DataFrame,
    *,
    metric: str,
    n_bootstraps: int = 1000,
    seed: int = 37,
) -> dict[str, float | int | str]:
    """Bootstrap AUROC or FPR95 over open-set test rows."""

    y_unknown = (predictions["known_status"].astype(str) != "known").astype(int).to_numpy()
    scores = predictions["unknown_score"].astype(float).to_numpy()

    def compute(indices: np.ndarray) -> float:
        y = y_unknown[indices]
        s = scores[indices]
        if len(set(y.tolist())) < 2:
            return float("nan")
        if metric == "auroc":
            return float(roc_auc_score(y, s))
        if metric == "fpr95":
            return fpr_at_tpr(y, s, target_tpr=0.95)
        msg = f"Unsupported metric: {metric}"
        raise ValueError(msg)

    estimates = np.asarray(
        [compute(idx) for idx in _bootstrap_indices(len(scores), n_bootstraps, seed)]
    )
    estimates = estimates[np.isfinite(estimates)]
    point = compute(np.arange(len(scores)))
    return {
        "metric": metric,
        "estimate": float(point),
        "lower": float(np.quantile(estimates, 0.025)),
        "upper": float(np.quantile(estimates, 0.975)),
        "n_bootstraps": n_bootstraps,
        "seed": seed,
    }


def paired_bootstrap_delta(
    baseline_predictions: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
    *,
    metric: str,
    n_bootstraps: int = 1000,
    seed: int = 37,
) -> dict[str, float | int | str]:
    """Paired bootstrap delta for matching sample IDs."""

    base = baseline_predictions[["sample_id", "known_status", "unknown_score"]].merge(
        candidate_predictions[["sample_id", "known_status", "unknown_score"]],
        on="sample_id",
        suffixes=("_base", "_candidate"),
        validate="one_to_one",
    )
    y_unknown = (base["known_status_base"].astype(str) != "known").astype(int).to_numpy()
    base_scores = base["unknown_score_base"].astype(float).to_numpy()
    cand_scores = base["unknown_score_candidate"].astype(float).to_numpy()

    def compute(scores: np.ndarray, indices: np.ndarray) -> float:
        y = y_unknown[indices]
        s = scores[indices]
        if len(set(y.tolist())) < 2:
            return float("nan")
        if metric == "auroc":
            return float(roc_auc_score(y, s))
        if metric == "fpr95":
            return fpr_at_tpr(y, s, target_tpr=0.95)
        msg = f"Unsupported paired metric: {metric}"
        raise ValueError(msg)

    deltas = []
    for idx in _bootstrap_indices(len(base), n_bootstraps, seed):
        deltas.append(compute(cand_scores, idx) - compute(base_scores, idx))
    deltas_arr = np.asarray(deltas, dtype=float)
    deltas_arr = deltas_arr[np.isfinite(deltas_arr)]
    all_idx = np.arange(len(base))
    point = compute(cand_scores, all_idx) - compute(base_scores, all_idx)
    return {
        "metric": f"delta_{metric}",
        "estimate": float(point),
        "lower": float(np.quantile(deltas_arr, 0.025)),
        "upper": float(np.quantile(deltas_arr, 0.975)),
        "n_bootstraps": n_bootstraps,
        "seed": seed,
    }


def bootstrap_closed_macro_f1(
    predictions: pd.DataFrame,
    *,
    n_bootstraps: int = 1000,
    seed: int = 37,
) -> dict[str, float | int | str]:
    """Bootstrap closed-set macro-F1 over known test predictions."""

    known = predictions.loc[predictions["known_status"].astype(str) == "known"].reset_index(
        drop=True
    )
    true = known["true_label"].astype(str).to_numpy()
    pred = known["closed_set_prediction"].astype(str).to_numpy()
    labels = sorted(set(true.tolist()))

    def compute(indices: np.ndarray) -> float:
        return float(
            f1_score(
                true[indices],
                pred[indices],
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )

    estimates = np.asarray(
        [compute(idx) for idx in _bootstrap_indices(len(known), n_bootstraps, seed)]
    )
    point = compute(np.arange(len(known)))
    return {
        "metric": "closed_macro_f1",
        "estimate": float(point),
        "lower": float(np.quantile(estimates, 0.025)),
        "upper": float(np.quantile(estimates, 0.975)),
        "n_bootstraps": n_bootstraps,
        "seed": seed,
    }


def write_bootstrap_report(
    baseline_predictions: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
    output_path: Path,
    *,
    n_bootstraps: int = 1000,
    seed: int = 37,
) -> Path:
    """Write bootstrap CIs and paired deltas."""

    report = {
        "baseline": {
            "closed_macro_f1": bootstrap_closed_macro_f1(
                baseline_predictions,
                n_bootstraps=n_bootstraps,
                seed=seed,
            ),
            "open_auroc": bootstrap_open_metric(
                baseline_predictions,
                metric="auroc",
                n_bootstraps=n_bootstraps,
                seed=seed,
            ),
            "fpr95": bootstrap_open_metric(
                baseline_predictions,
                metric="fpr95",
                n_bootstraps=n_bootstraps,
                seed=seed,
            ),
        },
        "candidate": {
            "closed_macro_f1": bootstrap_closed_macro_f1(
                candidate_predictions,
                n_bootstraps=n_bootstraps,
                seed=seed,
            ),
            "open_auroc": bootstrap_open_metric(
                candidate_predictions,
                metric="auroc",
                n_bootstraps=n_bootstraps,
                seed=seed,
            ),
            "fpr95": bootstrap_open_metric(
                candidate_predictions,
                metric="fpr95",
                n_bootstraps=n_bootstraps,
                seed=seed,
            ),
        },
        "paired_delta": {
            "delta_auroc": paired_bootstrap_delta(
                baseline_predictions,
                candidate_predictions,
                metric="auroc",
                n_bootstraps=n_bootstraps,
                seed=seed,
            ),
            "delta_fpr95": paired_bootstrap_delta(
                baseline_predictions,
                candidate_predictions,
                metric="fpr95",
                n_bootstraps=n_bootstraps,
                seed=seed,
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def write_delivery2_figures(
    methods: list[MethodArtifact],
    output_dir: Path,
    *,
    per_unknown_path: Path,
) -> None:
    """Write aggregate figures required for Delivery 2."""

    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    for method in methods:
        pred = pd.read_csv(method.predictions_path)
        y_unknown = (pred["known_status"].astype(str) != "known").astype(int)
        RocCurveDisplay.from_predictions(
            y_unknown,
            pred["unknown_score"].astype(float),
            name=_method_label(method),
            ax=ax,
        )
    fig.tight_layout()
    fig.savefig(output_dir / "open_set_roc_comparison.png", dpi=160)
    plt.close(fig)

    comparison = []
    for method in methods:
        metrics = _read_json(method.metrics_path)
        comparison.append(
            {
                "method": _method_label(method),
                "fpr95": metrics["open_set"]["fpr_at_95_tpr"],
            }
        )
    comp_frame = pd.DataFrame(comparison)
    fig, ax = plt.subplots(figsize=(max(6, len(comp_frame) * 0.6), 4))
    ax.bar(comp_frame["method"], comp_frame["fpr95"])
    ax.set_ylabel("FPR@95TPR")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "open_set_fpr95_comparison.png", dpi=160)
    plt.close(fig)

    selected = methods[: min(4, len(methods))]
    fig, axes = plt.subplots(len(selected), 1, figsize=(7, max(3, 2.2 * len(selected))))
    axes_arr = np.atleast_1d(axes)
    for ax, method in zip(axes_arr, selected, strict=True):
        pred = pd.read_csv(method.predictions_path)
        known = pred.loc[pred["known_status"].astype(str) == "known", "unknown_score"].astype(float)
        unknown = pred.loc[
            pred["known_status"].astype(str) != "known",
            "unknown_score",
        ].astype(float)
        ax.hist(known, bins=40, alpha=0.55, label="known", density=True)
        ax.hist(unknown, bins=40, alpha=0.55, label="unknown", density=True)
        ax.set_title(_method_label(method))
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "open_set_score_distributions.png", dpi=160)
    plt.close(fig)

    per_unknown = pd.read_csv(per_unknown_path)
    best_method = per_unknown.groupby("method")["AUROC"].mean().idxmax()
    best = per_unknown.loc[per_unknown["method"] == best_method].sort_values("AUROC")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(best["unknown_class"], best["AUROC"])
    ax.set_xlabel("AUROC vs known")
    ax.set_title(best_method)
    fig.tight_layout()
    fig.savefig(output_dir / "per_unknown_class_auroc.png", dpi=160)
    plt.close(fig)

    selected_method = next(
        (method for method in methods if _method_label(method) == best_method),
        methods[-1],
    )
    pred = pd.read_csv(selected_method.predictions_path)
    unknown = pred.loc[pred["known_status"].astype(str) != "known"].copy()
    if not unknown.empty:
        order = (
            unknown.groupby("true_label")["unknown_score"]
            .median()
            .sort_values()
            .index.astype(str)
            .tolist()
        )
        data = [
            unknown.loc[unknown["true_label"].astype(str) == label, "unknown_score"]
            .astype(float)
            .to_numpy()
            for label in order
        ]
        fig, ax = plt.subplots(figsize=(9, max(4, 0.33 * len(order))))
        ax.boxplot(data, vert=False, tick_labels=order, showfliers=False)
        ax.set_xlabel("Unknown score")
        ax.set_title(f"Unknown-class score distributions: {best_method}")
        fig.tight_layout()
        fig.savefig(output_dir / "open_set_unknown_class_score_distributions.png", dpi=160)
        plt.close(fig)


def write_confusion_matrix_figure(predictions_path: Path, output_path: Path) -> None:
    """Write a confusion matrix PNG from prediction rows."""

    import matplotlib.pyplot as plt

    pred = pd.read_csv(predictions_path)
    known = pred.loc[pred["known_status"].astype(str) == "known"]
    labels = sorted(
        set(known["true_label"].astype(str)) | set(known["closed_set_prediction"].astype(str))
    )
    matrix = confusion_matrix(
        known["true_label"].astype(str),
        known["closed_set_prediction"].astype(str),
        labels=labels,
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
