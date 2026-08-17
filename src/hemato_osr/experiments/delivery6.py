"""Delivery 6 multiseed ArcFace stability analysis."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from hemato_osr.data.manifest import manifest_hash
from hemato_osr.embeddings.extract import EmbeddingExtractConfig, extract_embeddings
from hemato_osr.experiments.delivery4 import (
    UNKNOWN_GROUPS,
    load_embeddings,
)
from hemato_osr.experiments.delivery5 import (
    CLOSED_MACRO_F1_FLOOR,
    _closed_metrics,
    _known_geometry,
    _open_metrics,
    _per_unknown_rows,
    _score_representation,
    _subgroup_results,
    _unknown_geometry,
)
from hemato_osr.topology.cache import config_hash
from hemato_osr.training.checkpoint import load_checkpoint
from hemato_osr.training.representation import (
    RepresentationTrainConfig,
    export_representation_embeddings,
    train_representation_model,
)
from hemato_osr.training.representation.train import _sha256_file
from hemato_osr.training.train import TrainConfig, train_closed_set
from hemato_osr.utils.tracking import write_json

DELIVERY6_SEEDS = (13, 37, 73, 101, 137)
DELIVERY6_SPLITS = ("v1", "v2")
DELIVERY6_REPRESENTATIONS = ("ce", "arcface")
DELIVERY6_OSR_METHODS = ("msp", "vim")

V1_MANIFEST = Path("data/manifests/mll23_split_seed37.csv")
V2_MANIFEST = Path("data/manifests/mll23_seed37_v2_conservative.csv")
V1_MANIFEST_HASH = "4b8da8df49655b6f04b53aa75a912f0efceac776d81761b2b45b68c440c27bae"
V2_MANIFEST_HASH = "81ca3db900f788a0446f80ea5d36e8766d8ca050404b755111b3ca898f809026"

LOCKED_CE_CONFIG: dict[str, Any] = {
    "representation": "ce",
    "backbone": "resnet18",
    "pretrained": True,
    "image_size": 224,
    "batch_size": 64,
    "epochs": 30,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "imbalance_strategy": "weighted-cross-entropy",
    "precision": "amp",
    "checkpoint_selection": "known-validation macro-F1",
    "early_stopping_patience": 7,
    "osr_methods": ["msp", "vim"],
}
LOCKED_ARCFACE_CONFIG: dict[str, Any] = {
    **LOCKED_CE_CONFIG,
    "representation": "arcface",
    "loss": "ArcFace-style additive angular margin",
    "embedding_dimension": 512,
    "margin": 0.30,
    "scale": 30.0,
}
CE_CONFIG_HASH = config_hash(LOCKED_CE_CONFIG)
ARCFACE_CONFIG_HASH = config_hash(LOCKED_ARCFACE_CONFIG)


@dataclass(frozen=True)
class Delivery6RunSpec:
    """One locked Delivery 6 training run."""

    split: str
    manifest_path: Path
    manifest_hash: str
    seed: int
    representation: str
    config_hash: str

    @property
    def run_id(self) -> str:
        return f"d6_{self.split}_{self.representation}_seed{self.seed}_{self.config_hash}"


@dataclass(frozen=True)
class Delivery6Paths:
    """Artifact roots for Delivery 6."""

    checkpoint_dir: Path = Path("artifacts/checkpoints/delivery6")
    embedding_dir: Path = Path("artifacts/embeddings/delivery6")
    metrics_dir: Path = Path("artifacts/metrics/delivery6")
    figures_dir: Path = Path("artifacts/figures/delivery6")
    logs_dir: Path = Path("artifacts/logs/delivery6")


def build_run_matrix() -> list[Delivery6RunSpec]:
    """Return the exact predeclared 20-run matrix."""

    split_specs = {
        "v1": (V1_MANIFEST, V1_MANIFEST_HASH),
        "v2": (V2_MANIFEST, V2_MANIFEST_HASH),
    }
    reps = {"ce": CE_CONFIG_HASH, "arcface": ARCFACE_CONFIG_HASH}
    return [
        Delivery6RunSpec(
            split=split,
            manifest_path=manifest,
            manifest_hash=expected_hash,
            seed=seed,
            representation=representation,
            config_hash=rep_hash,
        )
        for split in DELIVERY6_SPLITS
        for seed in DELIVERY6_SEEDS
        for representation, rep_hash in reps.items()
        for manifest, expected_hash in [split_specs[split]]
    ]


def assert_locked_configs() -> None:
    """Fail if predeclared Delivery 6 constants drift."""

    if DELIVERY6_SEEDS != (13, 37, 73, 101, 137):
        msg = f"Unexpected Delivery 6 seeds: {DELIVERY6_SEEDS}"
        raise ValueError(msg)
    if LOCKED_ARCFACE_CONFIG["margin"] != 0.30 or LOCKED_ARCFACE_CONFIG["scale"] != 30.0:
        msg = "Delivery 6 ArcFace margin/scale are not locked"
        raise ValueError(msg)
    if CE_CONFIG_HASH != "2d9d5aa0027e6f49":
        msg = f"CE config hash drifted: {CE_CONFIG_HASH}"
        raise ValueError(msg)
    if ARCFACE_CONFIG_HASH != "bda304d847e6cd2c":
        msg = f"ArcFace config hash drifted: {ARCFACE_CONFIG_HASH}"
        raise ValueError(msg)


def assert_manifest_hashes(repo_root: Path = Path(".")) -> None:
    """Validate V1/V2 manifest hashes before any training."""

    for spec in build_run_matrix()[:: len(DELIVERY6_REPRESENTATIONS)]:
        frame = pd.read_csv(repo_root / spec.manifest_path)
        found = manifest_hash(frame)
        if found != spec.manifest_hash:
            msg = (
                f"Manifest hash mismatch for {spec.manifest_path}: "
                f"expected {spec.manifest_hash}, found {found}"
            )
            raise ValueError(msg)


def write_run_manifest(path: Path, specs: list[Delivery6RunSpec] | None = None) -> Path:
    """Create the pre-execution Delivery 6 run manifest."""

    specs = specs or build_run_matrix()
    rows = [
        {
            "run_id": spec.run_id,
            "split": spec.split,
            "manifest_hash": spec.manifest_hash,
            "seed": spec.seed,
            "representation": spec.representation,
            "status": "pending",
            "checkpoint_path": "",
            "config_hash": spec.config_hash,
        }
        for spec in specs
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def update_run_status(
    manifest_path: Path,
    spec: Delivery6RunSpec,
    status: str,
    *,
    checkpoint_path: Path | None = None,
) -> None:
    """Update one run's status in the run manifest."""

    if not manifest_path.exists():
        write_run_manifest(manifest_path)
    frame = pd.read_csv(manifest_path)
    frame["checkpoint_path"] = frame["checkpoint_path"].fillna("").astype(str)
    mask = frame["run_id"].astype(str) == spec.run_id
    if not mask.any():
        msg = f"Run {spec.run_id} not present in run manifest"
        raise ValueError(msg)
    frame.loc[mask, "status"] = status
    if checkpoint_path is not None:
        frame.loc[mask, "checkpoint_path"] = str(checkpoint_path)
    frame.to_csv(manifest_path, index=False)


def run_metadata(spec: Delivery6RunSpec) -> dict[str, Any]:
    """Compatibility metadata for resume/restart safety."""

    return {
        "run_id": spec.run_id,
        "split": spec.split,
        "manifest_path": str(spec.manifest_path),
        "manifest_hash": spec.manifest_hash,
        "seed": spec.seed,
        "representation": spec.representation,
        "config_hash": spec.config_hash,
        "delivery": 6,
    }


def assert_resume_compatible(metadata_path: Path, spec: Delivery6RunSpec) -> None:
    """Ensure an existing run directory belongs to exactly this run."""

    if not metadata_path.exists():
        return
    found = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = run_metadata(spec)
    keys = ("run_id", "split", "manifest_hash", "seed", "representation", "config_hash")
    mismatches = {}
    for key in keys:
        if found.get(key) != expected[key]:
            mismatches[key] = (found.get(key), expected[key])
    if mismatches:
        msg = f"Resume metadata mismatch for {metadata_path}: {mismatches}"
        raise ValueError(msg)


def _train_ce(
    spec: Delivery6RunSpec,
    output_dir: Path,
    log_path: Path,
    *,
    device: str,
    num_workers: int,
    epochs: int,
    smoke_max_train_per_class: int | None,
) -> Path:
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    checkpoint = train_closed_set(
        TrainConfig(
            manifest_path=spec.manifest_path,
            output_dir=output_dir,
            backbone="resnet18",
            pretrained=True,
            image_size=224,
            batch_size=64,
            num_workers=num_workers,
            epochs=epochs,
            learning_rate=3e-4,
            weight_decay=1e-4,
            imbalance_strategy="weighted-cross-entropy",
            early_stopping_patience=7,
            seed=spec.seed,
            device=device,
            precision="amp",
            log_path=log_path,
            smoke_max_train_per_class=smoke_max_train_per_class,
        )
    )
    elapsed = time.perf_counter() - start
    loaded = load_checkpoint(checkpoint, map_location="cpu")
    peak = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    write_json(
        output_dir / "training_summary.json",
        {
            "representation": "ce",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "epochs_requested": epochs,
            "epochs_completed": _epochs_completed(log_path),
            "best_epoch": float(loaded["metrics"]["epoch"]),
            "wall_time_seconds": elapsed,
            "gpu_hours": elapsed / 3600.0 if torch.cuda.is_available() else 0.0,
            "peak_vram_mb": peak,
            "selection_rule": "known-validation macro-F1",
            "unknown_used_during_training": False,
            "ce_weighting": "weighted_cross_entropy",
            **run_metadata(spec),
        },
    )
    return checkpoint


def _train_arcface(
    spec: Delivery6RunSpec,
    output_dir: Path,
    log_path: Path,
    *,
    device: str,
    num_workers: int,
    epochs: int,
    smoke_max_train_per_class: int | None,
) -> Path:
    checkpoint = train_representation_model(
        RepresentationTrainConfig(
            manifest_path=spec.manifest_path,
            output_dir=output_dir,
            representation="arcface",
            image_size=224,
            batch_size=64,
            num_workers=num_workers,
            epochs=epochs,
            learning_rate=3e-4,
            weight_decay=1e-4,
            early_stopping_patience=7,
            seed=spec.seed,
            device=device,
            precision="amp",
            angular_scale=30.0,
            angular_margin=0.30,
            ce_weighting="weighted",
            smoke_max_train_per_class=smoke_max_train_per_class,
            log_path=log_path,
        )
    )
    summary_path = output_dir / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(run_metadata(spec))
    write_json(summary_path, summary)
    return checkpoint


def _epochs_completed(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    return int(len(pd.read_csv(log_path)))


def train_delivery6_run(
    spec: Delivery6RunSpec,
    paths: Delivery6Paths,
    *,
    device: str = "auto",
    num_workers: int = 4,
    smoke: bool = False,
    skip_completed: bool = True,
) -> Path:
    """Train one Delivery 6 run and return the best checkpoint."""

    output_dir = paths.checkpoint_dir / spec.run_id
    metadata_path = output_dir / "run_metadata.json"
    assert_resume_compatible(metadata_path, spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(metadata_path, run_metadata(spec))
    checkpoint = output_dir / "best_checkpoint.pt"
    if skip_completed and checkpoint.exists() and (output_dir / "training_summary.json").exists():
        return checkpoint
    log_path = paths.logs_dir / f"{spec.run_id}.log"
    epochs = 1 if smoke else 30
    smoke_per_class = 8 if smoke else None
    if spec.representation == "ce":
        return _train_ce(
            spec,
            output_dir,
            log_path,
            device=device,
            num_workers=num_workers,
            epochs=epochs,
            smoke_max_train_per_class=smoke_per_class,
        )
    if spec.representation == "arcface":
        return _train_arcface(
            spec,
            output_dir,
            log_path,
            device=device,
            num_workers=num_workers,
            epochs=epochs,
            smoke_max_train_per_class=smoke_per_class,
        )
    msg = f"Unknown Delivery 6 representation: {spec.representation}"
    raise ValueError(msg)


def export_delivery6_run(
    spec: Delivery6RunSpec,
    checkpoint_path: Path,
    paths: Delivery6Paths,
    *,
    device: str = "auto",
    num_workers: int = 4,
    skip_existing: bool = True,
) -> Path:
    """Export logits and embeddings for one completed run."""

    output_path = paths.embedding_dir / f"{spec.run_id}_embeddings.npz"
    if skip_existing and output_path.exists():
        return output_path
    if spec.representation == "ce":
        return extract_embeddings(
            EmbeddingExtractConfig(
                manifest_path=spec.manifest_path,
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                batch_size=64,
                num_workers=num_workers,
                image_size=224,
                device=device,
            )
        )
    return export_representation_embeddings(
        spec.manifest_path,
        checkpoint_path,
        output_path,
        batch_size=64,
        num_workers=num_workers,
        image_size=224,
        device=device,
    )


def evaluate_delivery6_run(
    spec: Delivery6RunSpec,
    embedding_path: Path,
    checkpoint_path: Path,
    training_summary: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate MSP/ViM, geometry, per-class unknowns, and attractors for one run."""

    archive = load_embeddings(embedding_path)
    if archive.manifest_hash != spec.manifest_hash:
        msg = f"Embedding manifest hash mismatch for {spec.run_id}"
        raise ValueError(msg)
    closed, _closed_per_class = _closed_metrics(archive)
    scores_by_method: dict[str, np.ndarray] = {}
    method_metrics: dict[str, dict[str, float]] = {}
    method_state: dict[str, Any] = {}
    score_ms: dict[str, float] = {}
    predictions = {}
    for method in DELIVERY6_OSR_METHODS:
        scores, state, scoring_ms = _score_representation(archive, method)
        scores_by_method[method] = scores
        method_state[method] = state
        score_ms[method] = scoring_ms
        method_metrics[method] = _open_metrics(archive, scores, target_known_recall=0.95)
        predictions[method] = _prediction_frame_for_run(
            archive,
            scores,
            spec.representation,
            method,
        )
    known_geo, known_summary = _known_geometry(archive, spec.representation)
    test_geometry = known_summary.loc[known_summary["split"] == "test"].iloc[0]
    sample_geo, attractor, nearest_lookup = _unknown_geometry(
        archive,
        spec.representation,
        scores_by_method["vim"],
    )
    unknown_distance = float(
        sample_geo.loc[sample_geo["known_status"] != "known", "nearest_known_distance"].mean()
    )
    per_frames = []
    for method in DELIVERY6_OSR_METHODS:
        per = _per_unknown_rows(
            predictions[method],
            nearest_lookup=nearest_lookup,
            baseline=pd.DataFrame(),
        )
        per["split"] = spec.split
        per["seed"] = spec.seed
        per_frames.append(per)
    per_unknown = pd.concat(per_frames, ignore_index=True)
    subgroup = _subgroup_results(per_unknown)
    subgroup["split"] = spec.split
    subgroup["seed"] = spec.seed
    attractor_rows = _attractor_rows(spec, attractor)
    row = {
        "split": spec.split,
        "manifest_hash": spec.manifest_hash,
        "seed": spec.seed,
        "representation": spec.representation,
        "checkpoint_hash": _sha256_file(checkpoint_path),
        "best_epoch": float(training_summary.get("best_epoch", np.nan)),
        "closed_macro_f1": float(closed["macro_f1"]),
        "closed_balanced_accuracy": float(closed["balanced_accuracy"]),
        "msp_auroc": float(method_metrics["msp"]["auroc_known_unknown"]),
        "msp_fpr95": float(method_metrics["msp"]["fpr_at_95_tpr"]),
        "msp_oscr": float(method_metrics["msp"]["OSCR"]),
        "vim_auroc": float(method_metrics["vim"]["auroc_known_unknown"]),
        "vim_fpr95": float(method_metrics["vim"]["fpr_at_95_tpr"]),
        "vim_oscr": float(method_metrics["vim"]["OSCR"]),
        "fisher_ratio": float(test_geometry["fisher_ratio"]),
        "within_dispersion": float(test_geometry["within_class_dispersion"]),
        "between_separation": float(test_geometry["between_class_separation"]),
        "cosine_compactness": float(test_geometry["mean_cosine_compactness"]),
        "mean_unknown_nearest_distance": unknown_distance,
        "training_seconds": float(training_summary.get("wall_time_seconds", np.nan)),
        "peak_vram_mb": float(training_summary.get("peak_vram_mb", np.nan)),
        "msp_scoring_ms_per_image": score_ms["msp"],
        "vim_scoring_ms_per_image": score_ms["vim"],
        "vim_state": json.dumps(method_state["vim"], sort_keys=True),
    }
    geometry = pd.DataFrame(
        [
            {
                "split": spec.split,
                "seed": spec.seed,
                "representation": spec.representation,
                "within_dispersion": row["within_dispersion"],
                "between_separation": row["between_separation"],
                "fisher_ratio": row["fisher_ratio"],
                "cosine_compactness": row["cosine_compactness"],
                "mean_unknown_nearest_distance": row["mean_unknown_nearest_distance"],
            }
        ]
    )
    return row, per_unknown, geometry, pd.DataFrame(attractor_rows)


def _prediction_frame_for_run(
    archive: Any,
    scores: np.ndarray,
    representation: str,
    method: str,
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


def _attractor_rows(spec: Delivery6RunSpec, attractor: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for unknown_class, values in attractor.iterrows():
        for known_class, fraction in values.items():
            rows.append(
                {
                    "split": spec.split,
                    "seed": spec.seed,
                    "representation": spec.representation,
                    "unknown_class": str(unknown_class),
                    "known_class": str(known_class),
                    "fraction": float(fraction),
                }
            )
    return rows


def evaluate_all_completed(paths: Delivery6Paths) -> Path:
    """Evaluate all completed runs and write run-level/intermediate tables."""

    rows: list[dict[str, Any]] = []
    per_frames: list[pd.DataFrame] = []
    geometry_frames: list[pd.DataFrame] = []
    attractor_frames: list[pd.DataFrame] = []
    for spec in build_run_matrix():
        checkpoint = paths.checkpoint_dir / spec.run_id / "best_checkpoint.pt"
        embedding = paths.embedding_dir / f"{spec.run_id}_embeddings.npz"
        summary_path = paths.checkpoint_dir / spec.run_id / "training_summary.json"
        if not checkpoint.exists() or not embedding.exists() or not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row, per_unknown, geometry, attractor = evaluate_delivery6_run(
            spec,
            embedding,
            checkpoint,
            summary,
        )
        rows.append(row)
        per_frames.append(per_unknown)
        geometry_frames.append(geometry)
        attractor_frames.append(attractor)
    paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    run_level = pd.DataFrame(rows)
    run_level.to_csv(paths.metrics_dir / "run_level_results.csv", index=False)
    if per_frames:
        pd.concat(per_frames, ignore_index=True).to_csv(
            paths.metrics_dir / "per_unknown_seed_results.csv",
            index=False,
        )
    if geometry_frames:
        pd.concat(geometry_frames, ignore_index=True).to_csv(
            paths.metrics_dir / "geometry_stability.csv",
            index=False,
        )
    if attractor_frames:
        pd.concat(attractor_frames, ignore_index=True).to_csv(
            paths.metrics_dir / "attractor_seed_results.csv",
            index=False,
        )
    return paths.metrics_dir / "run_level_results.csv"


def analyze_delivery6(paths: Delivery6Paths) -> Path:
    """Create final Delivery 6 summaries, criteria, and figures."""

    run_level = pd.read_csv(paths.metrics_dir / "run_level_results.csv")
    per_unknown = pd.read_csv(paths.metrics_dir / "per_unknown_seed_results.csv")
    geometry = pd.read_csv(paths.metrics_dir / "geometry_stability.csv")
    attractor = pd.read_csv(paths.metrics_dir / "attractor_seed_results.csv")
    if len(run_level) != 20:
        msg = f"Delivery 6 analysis requires 20 completed rows, found {len(run_level)}"
        raise ValueError(msg)
    matched = matched_seed_deltas(run_level, per_unknown)
    matched.to_csv(paths.metrics_dir / "matched_seed_deltas.csv", index=False)
    summary = multiseed_summary(run_level)
    summary.to_csv(paths.metrics_dir / "multiseed_summary.csv", index=False)
    subgroup = subgroup_seed_results(per_unknown)
    subgroup.to_csv(paths.metrics_dir / "subgroup_seed_results.csv", index=False)
    correlations = geometry_osr_correlations(run_level)
    correlations.to_csv(paths.metrics_dir / "geometry_osr_correlations.csv", index=False)
    class_stability = per_unknown_class_stability(per_unknown)
    class_stability.to_csv(paths.metrics_dir / "per_unknown_class_stability.csv", index=False)
    attractor_summary = attractor_stability(attractor)
    attractor_summary.to_csv(paths.metrics_dir / "attractor_stability.csv", index=False)
    decision = decision_from_deltas(run_level, matched)
    write_json(paths.metrics_dir / "confirmation_decision.json", decision)
    write_figures(paths, run_level, matched, geometry, subgroup, attractor_summary)
    return paths.metrics_dir / "confirmation_decision.json"


def t_ci(values: pd.Series | np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """t-based CI over independent seed-level values."""

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))
    mean = float(np.mean(arr))
    sem = float(stats.sem(arr))
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, df=len(arr) - 1))
    return mean - critical * sem, mean + critical * sem


def matched_seed_deltas(run_level: pd.DataFrame, per_unknown: pd.DataFrame) -> pd.DataFrame:
    """Compute ArcFace minus CE matched seed deltas."""

    metrics = [
        "closed_macro_f1",
        "msp_auroc",
        "msp_fpr95",
        "msp_oscr",
        "vim_auroc",
        "vim_fpr95",
        "vim_oscr",
        "fisher_ratio",
        "within_dispersion",
        "between_separation",
        "mean_unknown_nearest_distance",
    ]
    rows = []
    for (split, seed), frame in run_level.groupby(["split", "seed"], sort=True):
        ce = frame.loc[frame["representation"] == "ce"].iloc[0]
        arc = frame.loc[frame["representation"] == "arcface"].iloc[0]
        for metric in metrics:
            rows.append(
                {
                    "split": split,
                    "seed": int(seed),
                    "metric": metric,
                    "ce_value": float(ce[metric]),
                    "arcface_value": float(arc[metric]),
                    "delta": float(arc[metric] - ce[metric]),
                }
            )
    focus = per_unknown.loc[per_unknown["osr_method"].isin(["msp", "vim"])].copy()
    for (split, seed, method, unknown_class), frame in focus.groupby(
        ["split", "seed", "osr_method", "unknown_class"],
        sort=True,
    ):
        if set(frame["representation"]) != {"ce", "arcface"}:
            continue
        ce = frame.loc[frame["representation"] == "ce"].iloc[0]
        arc = frame.loc[frame["representation"] == "arcface"].iloc[0]
        for metric in ["AUROC", "FPR95"]:
            rows.append(
                {
                    "split": split,
                    "seed": int(seed),
                    "metric": f"{unknown_class}_{method}_{metric.lower()}",
                    "ce_value": float(ce[metric]),
                    "arcface_value": float(arc[metric]),
                    "delta": float(arc[metric] - ce[metric]),
                }
            )
    subgroup = subgroup_seed_results(per_unknown)
    for (split, seed, method, group), frame in subgroup.groupby(
        ["split", "seed", "osr_method", "group"],
        sort=True,
    ):
        if set(frame["representation"]) != {"ce", "arcface"}:
            continue
        ce = frame.loc[frame["representation"] == "ce"].iloc[0]
        arc = frame.loc[frame["representation"] == "arcface"].iloc[0]
        for metric in ["AUROC", "FPR95"]:
            rows.append(
                {
                    "split": split,
                    "seed": int(seed),
                    "metric": f"{group}_{method}_{metric.lower()}",
                    "ce_value": float(ce[metric]),
                    "arcface_value": float(arc[metric]),
                    "delta": float(arc[metric] - ce[metric]),
                }
            )
    return pd.DataFrame(rows)


def multiseed_summary(run_level: pd.DataFrame) -> pd.DataFrame:
    """Summarize run-level metrics by split and representation."""

    metrics = [
        "closed_macro_f1",
        "msp_auroc",
        "msp_fpr95",
        "msp_oscr",
        "vim_auroc",
        "vim_fpr95",
        "vim_oscr",
        "fisher_ratio",
        "within_dispersion",
        "between_separation",
        "mean_unknown_nearest_distance",
    ]
    rows = []
    for (split, representation), frame in run_level.groupby(
        ["split", "representation"],
        sort=True,
    ):
        for metric in metrics:
            values = frame[metric].astype(float)
            ci_low, ci_high = t_ci(values)
            rows.append(
                {
                    "split": split,
                    "representation": representation,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "median": float(values.median()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "n": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def subgroup_seed_results(per_unknown: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-class results into predeclared unknown groups per seed."""

    rows = []
    for (split, seed, representation, method), frame in per_unknown.groupby(
        ["split", "seed", "representation", "osr_method"],
        sort=True,
    ):
        for group, labels in UNKNOWN_GROUPS.items():
            group_frame = frame.loc[frame["unknown_class"].isin(labels)]
            if group_frame.empty:
                continue
            rows.append(
                {
                    "split": split,
                    "seed": int(seed),
                    "representation": representation,
                    "osr_method": method,
                    "group": group,
                    "n": int(group_frame["n"].sum()),
                    "AUROC": float(np.average(group_frame["AUROC"], weights=group_frame["n"])),
                    "FPR95": float(np.average(group_frame["FPR95"], weights=group_frame["n"])),
                }
            )
    return pd.DataFrame(rows)


def decision_from_deltas(run_level: pd.DataFrame, matched: pd.DataFrame) -> dict[str, Any]:
    """Apply the exact Delivery 6 criteria A-E."""

    v1_auroc = _metric_deltas(matched, "v1", "msp_auroc")
    v1_fpr = _metric_deltas(matched, "v1", "msp_fpr95")
    v2_auroc = _metric_deltas(matched, "v2", "msp_auroc")
    v1_auroc_ci = t_ci(v1_auroc)
    v1_fpr_ci = t_ci(v1_fpr)
    criterion_a = float(v1_auroc.mean()) >= 0.015 and v1_auroc_ci[0] > 0
    criterion_b = float(v1_fpr.mean()) <= -0.075 and v1_fpr_ci[1] < 0
    criterion_c = int((v1_auroc > 0).sum()) >= 4
    criterion_d = float(v2_auroc.mean()) >= -0.01 and int((v2_auroc >= 0).sum()) >= 3
    arcface = run_level.loc[run_level["representation"] == "arcface"].copy()
    direct_macro = bool((arcface["closed_macro_f1"] >= CLOSED_MACRO_F1_FLOOR).all())
    macro_delta = _metric_deltas(matched, "all", "closed_macro_f1")
    matched_macro = bool((macro_delta >= -0.01).all())
    criterion_e = direct_macro or matched_macro
    if criterion_a and criterion_c and criterion_d and criterion_e:
        decision = "ArcFace confirmed as reproducible within MLL23"
    elif criterion_a and criterion_c and criterion_e and not criterion_d:
        decision = "ArcFace improves V1 but is split-sensitive"
    elif not criterion_c:
        decision = "ArcFace is seed-unstable"
    else:
        decision = "ArcFace fails confirmation"
    return {
        "criterion_a": "PASS" if criterion_a else "FAIL",
        "criterion_b": "PASS" if criterion_b else "FAIL",
        "criterion_c": "PASS" if criterion_c else "FAIL",
        "criterion_d": "PASS" if criterion_d else "FAIL",
        "criterion_e": "PASS" if criterion_e else "FAIL",
        "decision": decision,
        "evidence": {
            "v1_mean_delta_auroc": float(v1_auroc.mean()),
            "v1_delta_auroc_ci95": list(v1_auroc_ci),
            "v1_positive_auroc_seeds": int((v1_auroc > 0).sum()),
            "v1_mean_delta_fpr95": float(v1_fpr.mean()),
            "v1_delta_fpr95_ci95": list(v1_fpr_ci),
            "v2_mean_delta_auroc": float(v2_auroc.mean()),
            "v2_nonnegative_auroc_seeds": int((v2_auroc >= 0).sum()),
            "arcface_all_macro_f1_above_floor": direct_macro,
            "arcface_all_macro_f1_within_0_01_of_ce": matched_macro,
        },
    }


def _metric_deltas(matched: pd.DataFrame, split: str, metric: str) -> pd.Series:
    frame = matched.loc[matched["metric"] == metric]
    if split != "all":
        frame = frame.loc[frame["split"] == split]
    return frame.sort_values(["split", "seed"])["delta"].astype(float)


def geometry_osr_correlations(run_level: pd.DataFrame) -> pd.DataFrame:
    """Exploratory Pearson/Spearman correlations with MSP AUROC."""

    rows = []
    for metric in [
        "fisher_ratio",
        "within_dispersion",
        "between_separation",
        "mean_unknown_nearest_distance",
    ]:
        x = run_level[metric].astype(float)
        y = run_level["msp_auroc"].astype(float)
        pearson = stats.pearsonr(x, y)
        spearman = stats.spearmanr(x, y)
        rows.append(
            {
                "x_metric": metric,
                "y_metric": "msp_auroc",
                "n": int(len(run_level)),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_r": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "interpretation": "exploratory_n20",
            }
        )
    return pd.DataFrame(rows)


def per_unknown_class_stability(per_unknown: pd.DataFrame) -> pd.DataFrame:
    """Classify per-unknown ArcFace effects by sign consistency."""

    msp = per_unknown.loc[per_unknown["osr_method"] == "msp"].copy()
    rows = []
    for (split, unknown_class), frame in msp.groupby(["split", "unknown_class"], sort=True):
        deltas = []
        for _seed, seed_frame in frame.groupby("seed", sort=True):
            if set(seed_frame["representation"]) != {"ce", "arcface"}:
                continue
            ce = seed_frame.loc[seed_frame["representation"] == "ce"].iloc[0]
            arc = seed_frame.loc[seed_frame["representation"] == "arcface"].iloc[0]
            deltas.append(float(arc["AUROC"] - ce["AUROC"]))
        arr = np.asarray(deltas, dtype=float)
        favorable = int((arr > 0).sum())
        degraded = int((arr < 0).sum())
        if favorable >= 4:
            label = "consistently_improved"
        elif degraded >= 4:
            label = "consistently_degraded"
        else:
            label = "inconsistent"
        rows.append(
            {
                "split": split,
                "unknown_class": unknown_class,
                "n_seeds": int(len(arr)),
                "mean_delta_AUROC": float(arr.mean()) if len(arr) else np.nan,
                "sd_delta_AUROC": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "favorable_seeds": favorable,
                "degraded_seeds": degraded,
                "stability_label": label,
            }
        )
    return pd.DataFrame(rows)


def attractor_stability(attractor: pd.DataFrame) -> pd.DataFrame:
    """Aggregate unknown-to-known attraction probabilities over seeds."""

    rows = []
    for (split, representation, unknown_class, known_class), frame in attractor.groupby(
        ["split", "representation", "unknown_class", "known_class"],
        sort=True,
    ):
        values = frame["fraction"].astype(float)
        rows.append(
            {
                "split": split,
                "representation": representation,
                "unknown_class": unknown_class,
                "known_class": known_class,
                "mean_fraction": float(values.mean()),
                "sd_fraction": float(values.std(ddof=1)),
                "min_fraction": float(values.min()),
                "max_fraction": float(values.max()),
                "n": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def write_figures(
    paths: Delivery6Paths,
    run_level: pd.DataFrame,
    matched: pd.DataFrame,
    geometry: pd.DataFrame,
    subgroup: pd.DataFrame,
    attractor_summary: pd.DataFrame,
) -> None:
    """Generate required Delivery 6 raw-seed figures."""

    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    for split in DELIVERY6_SPLITS:
        _line_by_seed(
            run_level.loc[run_level["split"] == split],
            "msp_auroc",
            paths.figures_dir / f"auroc_by_seed_{split}.png",
            ylabel="MSP AUROC",
        )
        _line_by_seed(
            run_level.loc[run_level["split"] == split],
            "msp_fpr95",
            paths.figures_dir / f"fpr95_by_seed_{split}.png",
            ylabel="MSP FPR95",
        )
    _delta_plot(matched, "msp_auroc", paths.figures_dir / "matched_delta_auroc.png")
    _delta_plot(matched, "msp_fpr95", paths.figures_dir / "matched_delta_fpr95.png")
    _distribution_plot(
        run_level,
        paths.figures_dir / "auroc_distribution_by_representation_split.png",
    )
    _scatter(
        run_level,
        "fisher_ratio",
        "msp_auroc",
        paths.figures_dir / "geometry_fisher_vs_auroc.png",
    )
    _scatter(
        run_level,
        "within_dispersion",
        "msp_auroc",
        paths.figures_dir / "within_dispersion_vs_auroc.png",
    )
    lymphoid_mask = (subgroup["group"] == "lymphoid_related") & (subgroup["osr_method"] == "msp")
    _subgroup_plot(
        subgroup.loc[lymphoid_mask],
        paths.figures_dir / "lymphoid_subgroup_by_seed.png",
    )
    neutrophil_metrics = ["neutrophil_band_msp_auroc", "neutrophil_band_vim_auroc"]
    neut_metric = matched.loc[matched["metric"].isin(neutrophil_metrics)]
    _delta_lines(neut_metric, paths.figures_dir / "neutrophil_band_by_seed.png")
    _attractor_focus_plot(attractor_summary, paths.figures_dir / "unknown_attractor_stability.png")


def _line_by_seed(frame: pd.DataFrame, metric: str, path: Path, *, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for representation, group in frame.groupby("representation", sort=True):
        group = group.sort_values("seed")
        ax.plot(group["seed"], group[metric], marker="o", label=representation)
    ax.set_xlabel("seed")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _delta_plot(matched: pd.DataFrame, metric: str, path: Path) -> None:
    frame = matched.loc[matched["metric"] == metric].copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    for split, group in frame.groupby("split", sort=True):
        ax.plot(group["seed"], group["delta"], marker="o", label=split)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("seed")
    ax.set_ylabel(f"ArcFace - CE {metric}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _distribution_plot(run_level: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = []
    values = []
    for (split, representation), group in run_level.groupby(["split", "representation"], sort=True):
        labels.append(f"{split}-{representation}")
        values.append(group["msp_auroc"].astype(float).to_numpy())
    ax.boxplot(values, showmeans=True)
    ax.set_xticks(range(1, len(labels) + 1), labels=labels)
    for idx, vals in enumerate(values, start=1):
        ax.scatter(np.full_like(vals, idx, dtype=float), vals, color="black", s=18, zorder=3)
    ax.set_ylabel("MSP AUROC")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _scatter(run_level: pd.DataFrame, x_metric: str, y_metric: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for (split, representation), group in run_level.groupby(["split", "representation"], sort=True):
        ax.scatter(group[x_metric], group[y_metric], label=f"{split}-{representation}", s=36)
    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _subgroup_plot(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for (split, representation), group in frame.groupby(["split", "representation"], sort=True):
        group = group.sort_values("seed")
        ax.plot(group["seed"], group["AUROC"], marker="o", label=f"{split}-{representation}")
    ax.set_xlabel("seed")
    ax.set_ylabel("lymphoid MSP AUROC")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _delta_lines(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for (split, metric), group in frame.groupby(["split", "metric"], sort=True):
        group = group.sort_values("seed")
        ax.plot(group["seed"], group["delta"], marker="o", label=f"{split}-{metric}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("seed")
    ax.set_ylabel("ArcFace - CE AUROC")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _attractor_focus_plot(attractor_summary: pd.DataFrame, path: Path) -> None:
    focus = {
        "lymphocyte_large_granular": "lymphocyte",
        "lymphocyte_neoplastic": "lymphocyte",
        "lymphocyte_reactive": "lymphocyte",
        "hairy_cell": "lymphocyte",
        "plasma_cell": "lymphocyte",
        "neutrophil_band": "neutrophil_segmented",
    }
    rows = []
    for unknown, known in focus.items():
        frame = attractor_summary.loc[
            (attractor_summary["unknown_class"] == unknown)
            & (attractor_summary["known_class"] == known)
        ]
        for _, row in frame.iterrows():
            rows.append(
                {
                    "label": f"{row['split']}-{row['representation']}",
                    "unknown": unknown,
                    "fraction": float(row["mean_fraction"]),
                }
            )
    plot = pd.DataFrame(rows)
    if plot.empty:
        return
    pivot = plot.pivot(index="unknown", columns="label", values="fraction")
    fig, ax = plt.subplots(figsize=(9, 4))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    fig.colorbar(image, ax=ax, label="mean attraction fraction")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_delivery6(
    paths: Delivery6Paths,
    *,
    device: str = "auto",
    num_workers: int = 4,
    smoke: bool = False,
    analyze_only: bool = False,
) -> None:
    """Run smoke or full Delivery 6 matrix sequentially."""

    assert_locked_configs()
    assert_manifest_hashes()
    if smoke:
        paths = Delivery6Paths(
            checkpoint_dir=paths.checkpoint_dir / "smoke",
            embedding_dir=paths.embedding_dir / "smoke",
            metrics_dir=paths.metrics_dir / "smoke",
            figures_dir=paths.figures_dir / "smoke",
            logs_dir=paths.logs_dir / "smoke",
        )
    paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = paths.metrics_dir / ("smoke_run_manifest.csv" if smoke else "run_manifest.csv")
    specs = [build_run_matrix()[0], build_run_matrix()[1]] if smoke else build_run_matrix()
    if not manifest_path.exists():
        write_run_manifest(manifest_path, specs)
    if not analyze_only:
        for spec in specs:
            update_run_status(manifest_path, spec, "running")
            try:
                checkpoint = train_delivery6_run(
                    spec,
                    paths,
                    device=device,
                    num_workers=num_workers,
                    smoke=smoke,
                    skip_completed=True,
                )
                export_delivery6_run(
                    spec,
                    checkpoint,
                    paths,
                    device=device,
                    num_workers=num_workers,
                    skip_existing=True,
                )
                update_run_status(manifest_path, spec, "completed", checkpoint_path=checkpoint)
            except Exception:
                update_run_status(manifest_path, spec, "failed")
                raise
    evaluate_all_completed(paths)
    if not smoke and len(pd.read_csv(paths.metrics_dir / "run_level_results.csv")) == 20:
        analyze_delivery6(paths)
