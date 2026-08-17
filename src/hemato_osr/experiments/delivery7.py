"""Delivery 7 external validation and Vietoris-Rips embedding topology."""

from __future__ import annotations

import hashlib
import json
import math
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from hemato_osr.data.manifest import manifest_hash
from hemato_osr.data.taxonomy import DEFAULT_KNOWN_CLASSES, KNOWN
from hemato_osr.embeddings.extract import EmbeddingExtractConfig, extract_embeddings
from hemato_osr.evaluation.metrics import closed_set_metrics, fpr_at_tpr, oscr
from hemato_osr.experiments.delivery4 import (
    EmbeddingArchive,
    ViMModel,
    l2_normalize,
    label_indices,
    load_embeddings,
    msp_anomaly_scores,
)
from hemato_osr.experiments.delivery6 import (
    DELIVERY6_OSR_METHODS,
    Delivery6RunSpec,
    build_run_matrix,
)
from hemato_osr.openset.thresholds import apply_threshold, calibrate_strict_open_set
from hemato_osr.topology.cache import config_hash
from hemato_osr.training.representation.train import _sha256_file, export_representation_embeddings
from hemato_osr.utils.tracking import write_json

VR_SAMPLE_SEED = 2026
N_VR = 192
H1_RELATIVE_PERSISTENCE_THRESHOLD = 0.10
DELIVERY7_OSR_METHODS = DELIVERY6_OSR_METHODS
VR_HARD_UNKNOWN_MLL23 = (
    "lymphocyte_large_granular",
    "lymphocyte_neoplastic",
    "lymphocyte_reactive",
    "hairy_cell",
    "neutrophil_band",
)
VR_LYMPHOID_CASE_CLASSES = (
    "lymphocyte",
    "lymphocyte_large_granular",
    "lymphocyte_neoplastic",
    "lymphocyte_reactive",
    "hairy_cell",
)
VR_NEUTROPHIL_CASE_CLASSES = ("neutrophil_segmented", "neutrophil_band")
VR_EPSILON_GRID = np.linspace(0.0, 2.0, 201)


@dataclass(frozen=True)
class Delivery7Paths:
    """Artifact roots for Delivery 7."""

    external_manifest_path: Path = Path("data/manifests/aml_lmu_external_delivery7.csv")
    external_data_root: Path = Path("data/external/aml_lmu/PKG - AML-Cytomorphology_LMU")
    external_annotations_path: Path = Path("data/external/aml_lmu/annotations.dat")
    external_taxonomy_path: Path = Path("configs/data/external_taxonomy_delivery7.yaml")
    internal_embedding_dir: Path = Path("artifacts/embeddings/delivery6")
    external_embedding_dir: Path = Path("artifacts/embeddings/delivery7/external")
    checkpoint_dir: Path = Path("artifacts/checkpoints/delivery6")
    metrics_dir: Path = Path("artifacts/metrics/delivery7")
    figures_dir: Path = Path("artifacts/figures/delivery7")
    topology_dir: Path = Path("artifacts/topology/delivery7")
    logs_dir: Path = Path("artifacts/logs/delivery7")
    data_audit_dir: Path = Path("artifacts/data_audit/delivery7")


def t_ci(values: pd.Series | np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """t-based CI over seed-level values."""

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    mean = float(np.mean(arr))
    sem = float(stats.sem(arr))
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, df=len(arr) - 1))
    return mean - critical * sem, mean + critical * sem


def sha256_text(text: str) -> str:
    """Stable SHA256 helper for cache keys and manifests."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_simple_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load the Delivery 7 taxonomy YAML through OmegaConf."""

    from omegaconf import OmegaConf

    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return cast(dict[str, Any], payload)


def read_aml_lmu_annotations(path: Path) -> pd.DataFrame:
    """Read TCIA AML-LMU annotations from DAT or ZIP."""

    lines: list[str]
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith(".dat")]
            if len(names) != 1:
                msg = f"Expected exactly one .dat in {path}, found {names}"
                raise ValueError(msg)
            lines = archive.read(names[0]).decode("utf-8").splitlines()
    else:
        lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) < 2:
            msg = f"Malformed AML-LMU annotation row: {line!r}"
            raise ValueError(msg)
        rows.append(
            {
                "relative_path": parts[0],
                "source_label": parts[1],
                "reannotation_1": parts[2] if len(parts) > 2 else "nan",
                "reannotation_2": parts[3] if len(parts) > 3 else "nan",
            }
        )
    return pd.DataFrame(rows)


def build_external_manifest(
    *,
    data_root: Path,
    annotations_path: Path,
    taxonomy_path: Path,
    output_path: Path,
    qc_path: Path,
) -> Path:
    """Build a test-only external manifest and deterministic QC table."""

    taxonomy = parse_simple_yaml_mapping(taxonomy_path)
    mapping = dict(taxonomy["mapping"])
    excluded = dict(taxonomy.get("excluded_labels", {}))
    annotations = read_aml_lmu_annotations(annotations_path)
    rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    checksums: Counter[str] = Counter()

    for record in annotations.to_dict("records"):
        source_label = str(record["source_label"])
        rel = str(record["relative_path"])
        full_path = data_root / rel
        status_record = mapping.get(source_label) or excluded.get(source_label)
        if status_record is None:
            msg = f"No external taxonomy mapping for source label {source_label}"
            raise ValueError(msg)
        known_or_unknown = str(status_record["known_or_unknown"])
        canonical = status_record.get("canonical_label")
        status = str(status_record["status"])
        readable = False
        width = height = channels = np.nan
        intensity_min = intensity_max = np.nan
        checksum = ""
        error = ""
        if full_path.exists():
            try:
                checksum = _sha256_file(full_path)
                checksums[checksum] += 1
                with Image.open(full_path) as image:
                    converted = image.convert("RGB")
                    arr = np.asarray(converted)
                    width, height = converted.size
                    channels = arr.shape[2] if arr.ndim == 3 else 1
                    intensity_min = float(arr.min())
                    intensity_max = float(arr.max())
                    readable = True
            except Exception as exc:  # pragma: no cover - exercised by integration QC
                error = str(exc)
        else:
            error = "missing_file"
        duplicate_path = rel in seen_paths
        seen_paths.add(rel)
        qc_rows.append(
            {
                "sample_id": f"AML_LMU:{Path(rel).with_suffix('').as_posix()}",
                "path": str(full_path),
                "relative_path": rel,
                "source_label": source_label,
                "readable": readable,
                "missing": not full_path.exists(),
                "duplicate_path": duplicate_path,
                "sha256": checksum,
                "width": width,
                "height": height,
                "channels": channels,
                "intensity_min": intensity_min,
                "intensity_max": intensity_max,
                "error": error,
            }
        )
        if known_or_unknown == "excluded":
            continue
        rows.append(
            {
                "sample_id": f"AML_LMU:{Path(rel).with_suffix('').as_posix()}",
                "path": str(full_path),
                "canonical_label": str(canonical),
                "known_status": KNOWN if known_or_unknown == "known" else "unknown",
                "split": "test",
                "source_label": source_label,
                "source_path": rel,
                "external_dataset": str(taxonomy["dataset"]),
                "mapping_status": status,
                "image_sha256": checksum,
            }
        )

    qc = pd.DataFrame(qc_rows)
    if not qc.empty:
        qc["duplicate_checksum"] = qc["sha256"].map(checksums).fillna(0).astype(int) > 1
        qc.loc[qc["sha256"] == "", "duplicate_checksum"] = False
    manifest = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    qc.to_csv(qc_path, index=False)
    return output_path


def assert_no_external_fit(archive: EmbeddingArchive) -> None:
    """External archives must contain test rows only."""

    if np.any(archive.split != "test"):
        msg = "External archive contains non-test split rows"
        raise ValueError(msg)


def export_external_run(
    spec: Delivery6RunSpec,
    *,
    checkpoint_path: Path,
    external_manifest_path: Path,
    output_path: Path,
    device: str = "auto",
    num_workers: int = 2,
    skip_existing: bool = True,
) -> Path:
    """Export logits and embeddings for one frozen run on the external manifest."""

    if skip_existing and output_path.exists():
        return output_path
    if spec.representation == "ce":
        return extract_embeddings(
            EmbeddingExtractConfig(
                manifest_path=external_manifest_path,
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                batch_size=64,
                num_workers=num_workers,
                image_size=224,
                device=device,
            )
        )
    return export_representation_embeddings(
        external_manifest_path,
        checkpoint_path,
        output_path,
        batch_size=64,
        num_workers=num_workers,
        image_size=224,
        device=device,
    )


def export_all_external_embeddings(
    paths: Delivery7Paths,
    *,
    device: str = "auto",
    num_workers: int = 2,
) -> list[Path]:
    """Export external embeddings for the locked 20-run matrix."""

    outputs = []
    for spec in build_run_matrix():
        checkpoint = paths.checkpoint_dir / spec.run_id / "best_checkpoint.pt"
        if not checkpoint.exists():
            msg = f"Missing checkpoint for {spec.run_id}: {checkpoint}"
            raise FileNotFoundError(msg)
        output_path = paths.external_embedding_dir / f"{spec.run_id}_external_embeddings.npz"
        outputs.append(
            export_external_run(
                spec,
                checkpoint_path=checkpoint,
                external_manifest_path=paths.external_manifest_path,
                output_path=output_path,
                device=device,
                num_workers=num_workers,
            )
        )
    return outputs


def _external_closed_metrics(archive: EmbeddingArchive) -> tuple[dict[str, Any], pd.DataFrame]:
    known = archive.known_status == KNOWN
    y_true = label_indices(archive.true_label[known], archive.label_to_index)
    y_pred = archive.logits[known].argmax(axis=1)
    metrics = closed_set_metrics(y_true.tolist(), y_pred.tolist(), DEFAULT_KNOWN_CLASSES)
    per_rows: list[dict[str, Any]] = []
    per_class = cast(dict[str, dict[str, Any]], metrics["per_class"])
    for class_name, values in per_class.items():
        per_rows.append({"class": class_name, **values})
    return {key: value for key, value in metrics.items() if key != "per_class"}, pd.DataFrame(
        per_rows
    )


def _fit_internal_score_external(
    internal: EmbeddingArchive,
    external: EmbeddingArchive,
    method: str,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    """Fit on MLL23 known-train when needed and score internal/external archives."""

    start = time.perf_counter()
    if method == "msp":
        internal_scores = msp_anomaly_scores(internal.logits)
        external_scores = msp_anomaly_scores(external.logits)
        state: dict[str, Any] = {"fit": "none"}
    elif method == "vim":
        fit_mask = (internal.split == "train") & (internal.known_status == KNOWN)
        model = ViMModel.fit(internal.embedding[fit_mask], internal.logits[fit_mask])
        internal_scores = model.score(internal.embedding, internal.logits)
        external_scores = model.score(external.embedding, external.logits)
        state = {
            "fit_split": "MLL23_known_train",
            "external_fit": False,
            "n_components": model.n_components,
            "explained_variance_ratio": model.explained_variance_ratio,
            "alpha": model.alpha,
        }
    else:
        msg = f"Unsupported Delivery 7 OSR method: {method}"
        raise ValueError(msg)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(1, len(external_scores))
    return internal_scores, external_scores, elapsed_ms, state


def _external_open_metrics(
    *,
    internal: EmbeddingArchive,
    external: EmbeddingArchive,
    internal_scores: np.ndarray,
    external_scores: np.ndarray,
    target_known_recall: float = 0.95,
) -> dict[str, float]:
    validation_known = (internal.split == "validation") & (internal.known_status == KNOWN)
    threshold = calibrate_strict_open_set(
        internal_scores[validation_known],
        target_known_recall=target_known_recall,
    )
    y_unknown = (external.known_status != KNOWN).astype(int)
    unknown_pred = apply_threshold(external_scores, threshold.threshold)
    pred_with_unknown = external.prediction.astype(object)
    pred_with_unknown[unknown_pred == 1] = "UNKNOWN"
    true_with_unknown = external.true_label.astype(object)
    true_with_unknown[y_unknown == 1] = "UNKNOWN"
    y_true = label_indices(external.true_label, external.label_to_index)
    metrics = {
        "AUROC": float(roc_auc_score(y_unknown, external_scores)),
        "AUPR_unknown": float(average_precision_score(y_unknown, external_scores)),
        "AUPR_known": float(average_precision_score(1 - y_unknown, -external_scores)),
        "FPR95": fpr_at_tpr(y_unknown, external_scores, target_tpr=0.95),
        "OSCR": oscr(y_true, external.logits.argmax(axis=1), -external_scores, y_unknown),
        "known_recall_frozen_threshold": float(np.mean(unknown_pred[y_unknown == 0] == 0)),
        "unknown_recall_frozen_threshold": float(np.mean(unknown_pred[y_unknown == 1] == 1)),
        "macro_f1_frozen_threshold": float(
            f1_score(true_with_unknown, pred_with_unknown, average="macro", zero_division=0)
        ),
        "threshold": threshold.threshold,
    }
    return metrics


def evaluate_external_run(
    spec: Delivery6RunSpec,
    *,
    internal_embedding_path: Path,
    external_embedding_path: Path,
    checkpoint_path: Path,
    external_manifest: pd.DataFrame,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Evaluate one frozen run externally with MSP and ViM."""

    internal = load_embeddings(internal_embedding_path)
    external = load_embeddings(external_embedding_path)
    assert_no_external_fit(external)
    closed, _per_class = _external_closed_metrics(external)
    rows: list[dict[str, Any]] = []
    per_unknown_frames = []
    checkpoint_hash = _sha256_file(checkpoint_path)
    external_known_n = int(np.sum(external.known_status == KNOWN))
    external_unknown_n = int(np.sum(external.known_status != KNOWN))
    manifest_lookup = external_manifest.set_index("sample_id")
    for method in DELIVERY7_OSR_METHODS:
        internal_scores, external_scores, score_ms, state = _fit_internal_score_external(
            internal,
            external,
            method,
        )
        metrics = _external_open_metrics(
            internal=internal,
            external=external,
            internal_scores=internal_scores,
            external_scores=external_scores,
        )
        rows.append(
            {
                "external_dataset": "AML-Cytomorphology_LMU",
                "training_split": spec.split,
                "seed": spec.seed,
                "representation": spec.representation,
                "osr_method": method,
                "external_known_n": external_known_n,
                "external_unknown_n": external_unknown_n,
                "closed_accuracy": float(closed["accuracy"]),
                "closed_balanced_accuracy": float(closed["balanced_accuracy"]),
                "closed_macro_f1": float(closed["macro_f1"]),
                "AUROC": metrics["AUROC"],
                "AUPR_unknown": metrics["AUPR_unknown"],
                "AUPR_known": metrics["AUPR_known"],
                "FPR95": metrics["FPR95"],
                "OSCR": metrics["OSCR"],
                "known_recall_frozen_threshold": metrics["known_recall_frozen_threshold"],
                "unknown_recall_frozen_threshold": metrics["unknown_recall_frozen_threshold"],
                "checkpoint_hash": checkpoint_hash,
                "threshold": metrics["threshold"],
                "scoring_ms_per_image": score_ms,
                "fit_state": json.dumps(state, sort_keys=True),
            }
        )
        per_unknown_frames.append(
            external_per_unknown_rows(
                external,
                external_scores,
                spec=spec,
                method=method,
                manifest_lookup=manifest_lookup,
            )
        )
    return rows, pd.concat(per_unknown_frames, ignore_index=True)


def external_per_unknown_rows(
    archive: EmbeddingArchive,
    scores: np.ndarray,
    *,
    spec: Delivery6RunSpec,
    method: str,
    manifest_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Per-source-label external unknown analysis."""

    known = archive.known_status == KNOWN
    rows: list[dict[str, Any]] = []
    sample_ids = pd.Series(archive.sample_id)
    source_labels = sample_ids.map(lambda sid: str(manifest_lookup.loc[str(sid), "source_label"]))
    for canonical_label in sorted(set(archive.true_label[~known].tolist())):
        group = (~known) & (archive.true_label == canonical_label)
        frame_mask = known | group
        y_unknown = (~known[frame_mask]).astype(int)
        group_scores = scores[frame_mask]
        if np.sum(group) > 0 and np.sum(known) > 0:
            auroc = float(roc_auc_score(y_unknown, group_scores))
            fpr95 = fpr_at_tpr(y_unknown, group_scores, target_tpr=0.95)
        else:
            auroc = float("nan")
            fpr95 = float("nan")
        predictions = archive.prediction[group]
        nearest = Counter(predictions.tolist()).most_common(1)[0][0] if len(predictions) else ""
        label_values = sorted(set(source_labels[group].tolist()))
        rows.append(
            {
                "source_label": ";".join(label_values),
                "canonical_label": str(canonical_label),
                "n": int(np.sum(group)),
                "training_split": spec.split,
                "seed": spec.seed,
                "representation": spec.representation,
                "osr_method": method,
                "AUROC": auroc,
                "FPR95": fpr95,
                "mean_score": float(np.mean(scores[group])),
                "median_score": float(np.median(scores[group])),
                "nearest_known_prediction": str(nearest),
            }
        )
    return pd.DataFrame(rows)


def summarize_external_results(run_level: pd.DataFrame) -> pd.DataFrame:
    """Seed-level summary for each split/representation/scorer/metric."""

    metrics = [
        "closed_accuracy",
        "closed_balanced_accuracy",
        "closed_macro_f1",
        "AUROC",
        "AUPR_unknown",
        "AUPR_known",
        "FPR95",
        "OSCR",
        "known_recall_frozen_threshold",
        "unknown_recall_frozen_threshold",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in run_level.groupby(["training_split", "representation", "osr_method"]):
        split, representation, method = keys
        for metric in metrics:
            values = group[metric].astype(float)
            low, high = t_ci(values)
            rows.append(
                {
                    "training_split": split,
                    "representation": representation,
                    "osr_method": method,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "median": float(values.median()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_seeds": int(values.count()),
                }
            )
    return pd.DataFrame(rows)


def matched_external_deltas(run_level: pd.DataFrame) -> pd.DataFrame:
    """Compute ArcFace minus CE external matched seed deltas."""

    rows: list[dict[str, Any]] = []
    metrics = ("AUROC", "FPR95")
    for (split, seed, method), group in run_level.groupby(["training_split", "seed", "osr_method"]):
        index = group.set_index("representation")
        if "ce" not in index.index or "arcface" not in index.index:
            continue
        for metric in metrics:
            ce_value = float(index.loc["ce", metric])
            arc_value = float(index.loc["arcface", metric])
            rows.append(
                {
                    "training_split": split,
                    "seed": int(seed),
                    "osr_method": method,
                    "metric": metric,
                    "ce_value": ce_value,
                    "arcface_value": arc_value,
                    "delta": arc_value - ce_value,
                }
            )
    return pd.DataFrame(rows)


def internal_external_shift(
    external_run_level: pd.DataFrame,
    delivery6_run_level: pd.DataFrame,
) -> pd.DataFrame:
    """Join internal and external metrics and compute domain shift."""

    rows: list[dict[str, Any]] = []
    internal_index = delivery6_run_level.set_index(["split", "seed", "representation"])
    for row in external_run_level.to_dict("records"):
        key = (row["training_split"], int(row["seed"]), row["representation"])
        if key not in internal_index.index:
            continue
        internal = internal_index.loc[key]
        method = str(row["osr_method"])
        rows.append(
            {
                "training_split": row["training_split"],
                "seed": int(row["seed"]),
                "representation": row["representation"],
                "osr_method": method,
                "internal_AUROC": float(internal[f"{method}_auroc"]),
                "external_AUROC": float(row["AUROC"]),
                "Delta_domain_AUROC": float(row["AUROC"]) - float(internal[f"{method}_auroc"]),
                "internal_FPR95": float(internal[f"{method}_fpr95"]),
                "external_FPR95": float(row["FPR95"]),
                "Delta_domain_FPR95": float(row["FPR95"]) - float(internal[f"{method}_fpr95"]),
            }
        )
    return pd.DataFrame(rows)


def run_external_evaluation(paths: Delivery7Paths) -> Path:
    """Evaluate all external embeddings and write required D7 external tables."""

    external_manifest = pd.read_csv(paths.external_manifest_path)
    rows: list[dict[str, Any]] = []
    per_unknown_frames = []
    for spec in build_run_matrix():
        checkpoint = paths.checkpoint_dir / spec.run_id / "best_checkpoint.pt"
        internal_embedding = paths.internal_embedding_dir / f"{spec.run_id}_embeddings.npz"
        external_embedding = paths.external_embedding_dir / f"{spec.run_id}_external_embeddings.npz"
        if (
            not checkpoint.exists()
            or not internal_embedding.exists()
            or not external_embedding.exists()
        ):
            msg = f"Missing frozen artifact for external evaluation of {spec.run_id}"
            raise FileNotFoundError(msg)
        run_rows, per_unknown = evaluate_external_run(
            spec,
            internal_embedding_path=internal_embedding,
            external_embedding_path=external_embedding,
            checkpoint_path=checkpoint,
            external_manifest=external_manifest,
        )
        rows.extend(run_rows)
        per_unknown_frames.append(per_unknown)
    paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    run_level = pd.DataFrame(rows)
    run_level.to_csv(paths.metrics_dir / "external_run_level_results.csv", index=False)
    summarize_external_results(run_level).to_csv(
        paths.metrics_dir / "external_multiseed_summary.csv",
        index=False,
    )
    matched_external_deltas(run_level).to_csv(
        paths.metrics_dir / "external_matched_seed_deltas.csv",
        index=False,
    )
    if per_unknown_frames:
        pd.concat(per_unknown_frames, ignore_index=True).to_csv(
            paths.metrics_dir / "external_per_unknown_class.csv",
            index=False,
        )
    delivery6 = pd.read_csv(Path("artifacts/metrics/delivery6/run_level_results.csv"))
    internal_external_shift(run_level, delivery6).to_csv(
        paths.metrics_dir / "internal_external_shift.csv",
        index=False,
    )
    return paths.metrics_dir / "external_run_level_results.csv"


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Cosine distance matrix after L2 normalization."""

    normed = l2_normalize(embeddings)
    distances = 1.0 - normed @ normed.T
    distances = np.clip(distances, 0.0, 2.0)
    np.fill_diagonal(distances, 0.0)
    return (distances + distances.T) / 2.0


def fixed_vr_sample_ids(
    frame: pd.DataFrame,
    *,
    dataset: str,
    class_name: str,
    n_vr: int = N_VR,
    seed: int = VR_SAMPLE_SEED,
) -> list[str]:
    """Select deterministic VR sample IDs for one dataset/class."""

    candidates = sorted(frame.loc[frame["canonical_label"] == class_name, "sample_id"].astype(str))
    class_hash = int(hashlib.sha256(f"{dataset}:{class_name}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed + class_hash)
    if len(candidates) <= n_vr:
        return candidates
    indices = rng.choice(len(candidates), size=n_vr, replace=False)
    return [candidates[int(idx)] for idx in sorted(indices)]


def write_vr_sample_manifest(
    populations: dict[str, pd.DataFrame],
    output_path: Path,
    *,
    classes: tuple[str, ...],
    n_vr: int = N_VR,
    seed: int = VR_SAMPLE_SEED,
) -> Path:
    """Persist fixed VR sample IDs for dataset/class populations."""

    rows: list[dict[str, Any]] = []
    for dataset, frame in populations.items():
        for class_name in classes:
            available_n = int(np.sum(frame["canonical_label"].astype(str) == class_name))
            ids = fixed_vr_sample_ids(
                frame,
                dataset=dataset,
                class_name=class_name,
                n_vr=n_vr,
                seed=seed,
            )
            rows.extend(
                {
                    "dataset": dataset,
                    "class": class_name,
                    "sample_id": sample_id,
                    "available_n": available_n,
                    "n_vr": n_vr,
                    "sample_seed": seed,
                }
                for sample_id in ids
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def _run_spec(split: str, seed: int, representation: str) -> Delivery6RunSpec:
    for spec in build_run_matrix():
        if spec.split == split and spec.seed == seed and spec.representation == representation:
            return spec
    msg = f"No Delivery 6 run spec for split={split}, seed={seed}, representation={representation}"
    raise ValueError(msg)


def _archive_frame(archive: EmbeddingArchive) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": archive.sample_id.astype(str),
            "canonical_label": archive.true_label.astype(str),
            "known_status": archive.known_status.astype(str),
            "split": archive.split.astype(str),
        }
    )


def _load_reference_archive(paths: Delivery7Paths, split: str) -> EmbeddingArchive:
    spec = _run_spec(split, seed=13, representation="ce")
    return load_embeddings(paths.internal_embedding_dir / f"{spec.run_id}_embeddings.npz")


def _mll23_common_test_population(paths: Delivery7Paths) -> pd.DataFrame:
    """Return fixed MLL23 test population eligible in both V1 and V2."""

    v1 = _archive_frame(_load_reference_archive(paths, "v1"))
    v2 = _archive_frame(_load_reference_archive(paths, "v2"))
    v1_test = v1.loc[v1["split"] == "test"].copy()
    v2_test = v2.loc[v2["split"] == "test"].copy()
    common_ids = set(v1_test["sample_id"]).intersection(v2_test["sample_id"])
    frame = v1_test.loc[v1_test["sample_id"].isin(common_ids)].copy()
    allowed = set(DEFAULT_KNOWN_CLASSES).union(VR_HARD_UNKNOWN_MLL23)
    return frame.loc[frame["canonical_label"].isin(allowed)].reset_index(drop=True)


def build_vr_sample_manifest(paths: Delivery7Paths) -> Path:
    """Build the fixed sample-id manifest for explanatory VR topology."""

    external = pd.read_csv(paths.external_manifest_path)
    external_classes = tuple(sorted(external["canonical_label"].astype(str).unique().tolist()))
    mll23_classes = DEFAULT_KNOWN_CLASSES + VR_HARD_UNKNOWN_MLL23
    rows: list[pd.DataFrame] = []
    for dataset, frame, classes in (
        ("MLL23_common_test", _mll23_common_test_population(paths), mll23_classes),
        ("AML_LMU_external_test", external, external_classes),
    ):
        temp = paths.topology_dir / f".{dataset}_vr_sample_manifest.csv"
        write_vr_sample_manifest(
            {dataset: frame},
            temp,
            classes=classes,
            n_vr=N_VR,
            seed=VR_SAMPLE_SEED,
        )
        rows.append(pd.read_csv(temp))
        temp.unlink(missing_ok=True)
    output_path = paths.topology_dir / "vr_sample_manifest.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(output_path, index=False)
    return output_path


def rips_diagrams(
    embeddings: np.ndarray,
    *,
    max_edge_length: float = 2.0,
    max_dimension: int = 1,
) -> dict[int, np.ndarray]:
    """Compute H0/H1 Vietoris-Rips diagrams from normalized cosine distances."""

    diagrams, _elapsed, _simplices = _rips_diagrams_with_metadata(
        embeddings,
        max_edge_length=max_edge_length,
        max_dimension=max_dimension,
    )
    return diagrams


def _rips_diagrams_with_metadata(
    embeddings: np.ndarray,
    *,
    max_edge_length: float = 2.0,
    max_dimension: int = 1,
) -> tuple[dict[int, np.ndarray], float, int]:
    """Compute VR diagrams and technical metadata."""

    import gudhi as gd

    start = time.perf_counter()
    distances = cosine_distance_matrix(embeddings)
    complex_ = gd.RipsComplex(distance_matrix=distances, max_edge_length=max_edge_length)
    simplex_tree = complex_.create_simplex_tree(max_dimension=max_dimension + 1)
    simplex_count = int(simplex_tree.num_simplices())
    simplex_tree.persistence(homology_coeff_field=2, min_persistence=0.0)
    diagrams: dict[int, np.ndarray] = {}
    for dim in range(max_dimension + 1):
        intervals = simplex_tree.persistence_intervals_in_dimension(dim)
        diagrams[dim] = np.asarray(intervals, dtype=float).reshape(-1, 2)
    elapsed = time.perf_counter() - start
    return diagrams, elapsed, simplex_count


def finite_bars(diagram: np.ndarray) -> np.ndarray:
    """Return finite persistence bars only."""

    if diagram.size == 0:
        return np.empty((0, 2), dtype=float)
    arr = np.asarray(diagram, dtype=float).reshape(-1, 2)
    return arr[np.isfinite(arr[:, 1])]


def diagram_summary(
    diagram: np.ndarray,
    *,
    homology_dim: int,
    h1_relative_threshold: float = H1_RELATIVE_PERSISTENCE_THRESHOLD,
) -> dict[str, float]:
    """Fixed persistence-diagram summaries."""

    finite = finite_bars(diagram)
    persistence = finite[:, 1] - finite[:, 0] if len(finite) else np.asarray([], dtype=float)
    total = float(np.sum(persistence)) if len(persistence) else 0.0
    probs = persistence / total if total > 0 else np.asarray([], dtype=float)
    entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12)))) if len(probs) else 0.0
    max_persistence = float(np.max(persistence)) if len(persistence) else 0.0
    summary = {
        "homology_dim": float(homology_dim),
        "n_finite_bars": float(len(finite)),
        "total_persistence": total,
        "mean_persistence": float(np.mean(persistence)) if len(persistence) else 0.0,
        "max_persistence": max_persistence,
        "persistence_entropy": entropy,
    }
    if homology_dim == 1:
        threshold = h1_relative_threshold * max_persistence if max_persistence > 0 else math.inf
        summary["h1_bars_above_relative_threshold"] = float(np.sum(persistence >= threshold))
    return summary


def bottleneck_distance(diagram_a: np.ndarray, diagram_b: np.ndarray) -> float:
    """Bottleneck distance between two diagrams, treating empty diagrams consistently."""

    import gudhi as gd

    a = finite_bars(diagram_a)
    b = finite_bars(diagram_b)
    if len(a) == 0 and len(b) == 0:
        return 0.0
    return float(gd.bottleneck_distance(a, b))


def betti_curve(diagram: np.ndarray, epsilon_grid: np.ndarray = VR_EPSILON_GRID) -> np.ndarray:
    """Evaluate a Betti curve on a fixed epsilon grid."""

    if diagram.size == 0:
        return np.zeros_like(epsilon_grid, dtype=float)
    intervals = np.asarray(diagram, dtype=float).reshape(-1, 2)
    births = intervals[:, 0]
    deaths = intervals[:, 1]
    values = []
    for epsilon in epsilon_grid:
        alive = (births <= epsilon) & ((epsilon < deaths) | ~np.isfinite(deaths))
        values.append(float(np.sum(alive)))
    return np.asarray(values, dtype=float)


def seed_pair_stability(diagram_rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pairwise seed bottleneck distances from serialized diagram paths."""

    rows: list[dict[str, Any]] = []
    for keys, group in diagram_rows.groupby(["dataset", "split", "representation", "class"]):
        dataset, split, representation, class_name = keys
        by_seed = {int(row["seed"]): row for row in group.to_dict("records")}
        for seed_a, seed_b in combinations(sorted(by_seed), 2):
            row_a = by_seed[seed_a]
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "representation": representation,
                    "class": class_name,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "h0_bottleneck": float(row_a["h0_bottleneck_to_" + str(seed_b)])
                    if "h0_bottleneck_to_" + str(seed_b) in row_a
                    else np.nan,
                    "h1_bottleneck": float(row_a["h1_bottleneck_to_" + str(seed_b)])
                    if "h1_bottleneck_to_" + str(seed_b) in row_a
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def cross_class_mixing(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    *,
    k: int = 10,
) -> dict[str, float]:
    """Fixed local cross-cloud neighbor mixing statistic."""

    from sklearn.neighbors import NearestNeighbors

    a = l2_normalize(embeddings_a)
    b = l2_normalize(embeddings_b)
    labels = np.asarray([0] * len(a) + [1] * len(b))
    cloud = np.vstack([a, b])
    neighbors = NearestNeighbors(n_neighbors=min(k + 1, len(cloud)), metric="cosine")
    neighbors.fit(cloud)
    distances, indices = neighbors.kneighbors(cloud)
    cross_fracs = []
    mean_knn = []
    for idx, row_indices in enumerate(indices):
        keep = row_indices[row_indices != idx][:k]
        cross_fracs.append(float(np.mean(labels[keep] != labels[idx])))
        mean_knn.append(float(np.mean(distances[idx, 1 : len(keep) + 1])))
    centroid_distance = float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))
    return {
        "k": float(k),
        "cross_neighbor_fraction": float(np.mean(cross_fracs)),
        "centroid_distance": centroid_distance,
        "mean_knn_distance": float(np.mean(mean_knn)),
    }


def _select_embeddings_by_ids(
    archive: EmbeddingArchive,
    sample_ids: pd.Series,
) -> np.ndarray:
    index = {sample_id: idx for idx, sample_id in enumerate(archive.sample_id.astype(str))}
    positions = [
        index[str(sample_id)] for sample_id in sample_ids.astype(str) if str(sample_id) in index
    ]
    if not positions:
        return np.empty((0, archive.embedding.shape[1]), dtype=float)
    return archive.embedding[np.asarray(positions, dtype=int)]


def _save_diagrams(path: Path, diagrams: dict[int, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, h0=diagrams[0], h1=diagrams[1])


def _load_saved_diagrams(path: Path) -> dict[int, np.ndarray]:
    with np.load(path) as data:
        return {0: np.asarray(data["h0"], dtype=float), 1: np.asarray(data["h1"], dtype=float)}


def _vr_library_version() -> str:
    import gudhi

    return str(gudhi.__version__)


def run_vr_feasibility_smoke(paths: Delivery7Paths) -> Path:
    """Record fixed VR feasibility timings before full explanatory extraction."""

    archive = _load_reference_archive(paths, "v1")
    frame = _archive_frame(archive)
    rows: list[dict[str, Any]] = []
    for sample_size in (64, 128, 192):
        candidates = frame.loc[
            (frame["split"] == "train")
            & (frame["known_status"] == KNOWN)
            & (frame["canonical_label"] == "neutrophil_segmented")
        ]
        ids = fixed_vr_sample_ids(
            candidates,
            dataset="MLL23_known_train_feasibility",
            class_name="neutrophil_segmented",
            n_vr=sample_size,
            seed=VR_SAMPLE_SEED,
        )
        embeddings = _select_embeddings_by_ids(archive, pd.Series(ids))
        diagrams, elapsed, simplex_count = _rips_diagrams_with_metadata(embeddings)
        rows.append(
            {
                "sample_size": len(ids),
                "requested_sample_size": sample_size,
                "dataset": "MLL23_known_train_feasibility",
                "class": "neutrophil_segmented",
                "runtime_seconds": elapsed,
                "simplex_count": simplex_count,
                "h0_intervals": int(len(diagrams[0])),
                "h1_intervals": int(len(diagrams[1])),
                "gudhi_version": _vr_library_version(),
                "vr_config_hash": vr_config_hash(),
            }
        )
    paths.topology_dir.mkdir(parents=True, exist_ok=True)
    output = paths.topology_dir / "vr_feasibility_smoke.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def _diagram_artifact_path(
    paths: Delivery7Paths,
    *,
    dataset: str,
    spec: Delivery6RunSpec,
    class_name: str,
) -> Path:
    safe_class = class_name.replace("/", "_")
    return (
        paths.topology_dir
        / "diagrams"
        / dataset
        / spec.split
        / spec.representation
        / f"seed{spec.seed}_{safe_class}.npz"
    )


def _diagram_summary_rows(
    *,
    dataset: str,
    spec: Delivery6RunSpec,
    class_name: str,
    n_samples: int,
    diagrams: dict[int, np.ndarray],
    diagram_path: Path,
    elapsed: float,
    simplex_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dim in (0, 1):
        row = diagram_summary(diagrams[dim], homology_dim=dim)
        rows.append(
            {
                "dataset": dataset,
                "training_split": spec.split,
                "seed": spec.seed,
                "representation": spec.representation,
                "class": class_name,
                "n_samples": n_samples,
                "runtime_seconds": elapsed,
                "simplex_count": simplex_count,
                "diagram_path": str(diagram_path),
                "gudhi_version": _vr_library_version(),
                "vr_config_hash": vr_config_hash(),
                **row,
            }
        )
    return rows


def _betti_curve_rows(
    *,
    dataset: str,
    spec: Delivery6RunSpec,
    class_name: str,
    diagrams: dict[int, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dim in (0, 1):
        values = betti_curve(diagrams[dim], VR_EPSILON_GRID)
        rows.extend(
            {
                "dataset": dataset,
                "training_split": spec.split,
                "seed": spec.seed,
                "representation": spec.representation,
                "class": class_name,
                "homology_dim": dim,
                "epsilon": float(epsilon),
                "betti": float(value),
            }
            for epsilon, value in zip(VR_EPSILON_GRID, values, strict=True)
        )
    return rows


def _pairwise_seed_bottleneck(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    h1 = summary.loc[summary["homology_dim"] == 1].copy()
    for keys, group in h1.groupby(["dataset", "training_split", "representation", "class"]):
        dataset, split, representation, class_name = keys
        by_seed = {int(row["seed"]): str(row["diagram_path"]) for row in group.to_dict("records")}
        for seed_a, seed_b in combinations(sorted(by_seed), 2):
            diag_a = _load_saved_diagrams(Path(by_seed[seed_a]))
            diag_b = _load_saved_diagrams(Path(by_seed[seed_b]))
            rows.append(
                {
                    "dataset": dataset,
                    "training_split": split,
                    "representation": representation,
                    "class": class_name,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "h0_bottleneck": bottleneck_distance(diag_a[0], diag_b[0]),
                    "h1_bottleneck": bottleneck_distance(diag_a[1], diag_b[1]),
                }
            )
    return pd.DataFrame(rows)


def _pairwise_split_bottleneck(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    h1 = summary.loc[summary["homology_dim"] == 1].copy()
    for keys, group in h1.groupby(["dataset", "representation", "seed", "class"]):
        dataset, representation, seed, class_name = keys
        by_split = {
            str(row["training_split"]): str(row["diagram_path"]) for row in group.to_dict("records")
        }
        if "v1" not in by_split or "v2" not in by_split:
            continue
        diag_v1 = _load_saved_diagrams(Path(by_split["v1"]))
        diag_v2 = _load_saved_diagrams(Path(by_split["v2"]))
        rows.append(
            {
                "dataset": dataset,
                "representation": representation,
                "seed": int(seed),
                "class": class_name,
                "split_a": "v1",
                "split_b": "v2",
                "h0_bottleneck": bottleneck_distance(diag_v1[0], diag_v2[0]),
                "h1_bottleneck": bottleneck_distance(diag_v1[1], diag_v2[1]),
            }
        )
    return pd.DataFrame(rows)


def _mixing_pairs_for_dataset(dataset: str, classes: set[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for class_a, class_b in (
        ("lymphocyte", "lymphocyte_large_granular"),
        ("lymphocyte", "lymphocyte_neoplastic"),
        ("lymphocyte", "lymphocyte_reactive"),
        ("lymphocyte", "hairy_cell"),
        ("neutrophil_segmented", "neutrophil_band"),
    ):
        if class_a in classes and class_b in classes:
            pairs.append((class_a, class_b))
    if dataset == "AML_LMU_external_test":
        for class_a, class_b in (
            ("lymphocyte", "lymphocyte_atypical"),
            ("neutrophil_segmented", "neutrophil_band"),
            ("monocyte", "monoblast"),
            ("neutrophil_segmented", "metamyelocyte"),
            ("neutrophil_segmented", "myelocyte"),
        ):
            if class_a in classes and class_b in classes and (class_a, class_b) not in pairs:
                pairs.append((class_a, class_b))
    return pairs


def _run_cloud_mixing(
    *,
    dataset: str,
    spec: Delivery6RunSpec,
    archive: EmbeddingArchive,
    sample_manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    classes = set(sample_manifest.loc[sample_manifest["dataset"] == dataset, "class"].astype(str))
    for class_a, class_b in _mixing_pairs_for_dataset(dataset, classes):
        ids_a = sample_manifest.loc[
            (sample_manifest["dataset"] == dataset) & (sample_manifest["class"] == class_a),
            "sample_id",
        ]
        ids_b = sample_manifest.loc[
            (sample_manifest["dataset"] == dataset) & (sample_manifest["class"] == class_b),
            "sample_id",
        ]
        embeddings_a = _select_embeddings_by_ids(archive, ids_a)
        embeddings_b = _select_embeddings_by_ids(archive, ids_b)
        if len(embeddings_a) < 2 or len(embeddings_b) < 2:
            continue
        stats_row = cross_class_mixing(embeddings_a, embeddings_b, k=10)
        rows.append(
            {
                "dataset": dataset,
                "training_split": spec.split,
                "seed": spec.seed,
                "representation": spec.representation,
                "class_a": class_a,
                "class_b": class_b,
                "n_a": len(embeddings_a),
                "n_b": len(embeddings_b),
                **stats_row,
            }
        )
    return pd.DataFrame(rows)


def _write_vr_figures(
    paths: Delivery7Paths,
    summary: pd.DataFrame,
    seed_pairs: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    h1 = summary.loc[summary["homology_dim"] == 1].copy()
    if not h1.empty:
        pivot = h1.pivot_table(
            index="class",
            columns="representation",
            values="max_persistence",
            aggfunc="mean",
        ).sort_index()
        fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(pivot))))
        pivot.plot(kind="barh", ax=ax)
        ax.set_xlabel("Mean H1 max persistence")
        ax.set_ylabel("Class")
        ax.set_title("Delivery 7 VR H1 persistence")
        fig.tight_layout()
        fig.savefig(paths.figures_dir / "vr_h1_max_persistence_by_class.png", dpi=180)
        plt.close(fig)
    if not seed_pairs.empty:
        stability = seed_pairs.pivot_table(
            index="class",
            columns="representation",
            values="h1_bottleneck",
            aggfunc="mean",
        ).sort_index()
        fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(stability))))
        stability.plot(kind="barh", ax=ax)
        ax.set_xlabel("Mean pairwise seed H1 bottleneck")
        ax.set_ylabel("Class")
        ax.set_title("Delivery 7 VR seed stability")
        fig.tight_layout()
        fig.savefig(paths.figures_dir / "vr_seed_stability_h1_bottleneck.png", dpi=180)
        plt.close(fig)


def topology_osr_correlations(paths: Delivery7Paths, summary: pd.DataFrame) -> pd.DataFrame:
    """Exploratory correlations between fixed VR summaries and frozen OSR metrics."""

    d6_path = Path("artifacts/metrics/delivery6/run_level_results.csv")
    if not d6_path.exists():
        return pd.DataFrame()
    h1 = summary.loc[
        (summary["dataset"] == "MLL23_common_test") & (summary["homology_dim"] == 1)
    ].copy()
    if h1.empty:
        return pd.DataFrame()
    topo_rows = []
    for keys, group in h1.groupby(["training_split", "seed", "representation"]):
        split, seed, representation = keys
        known = group.loc[group["class"].isin(DEFAULT_KNOWN_CLASSES)]
        row: dict[str, Any] = {
            "split": split,
            "seed": int(seed),
            "representation": representation,
            "mean_known_h1_max_persistence": float(known["max_persistence"].mean()),
            "mean_known_h1_total_persistence": float(known["total_persistence"].mean()),
        }
        for class_name in ("lymphocyte", "neutrophil_segmented"):
            class_rows = group.loc[group["class"] == class_name]
            if not class_rows.empty:
                row[f"{class_name}_h1_max_persistence"] = float(
                    class_rows["max_persistence"].iloc[0]
                )
        topo_rows.append(row)
    topo = pd.DataFrame(topo_rows)
    osr = pd.read_csv(d6_path)
    joined = topo.merge(osr, on=["split", "seed", "representation"], how="inner")
    rows: list[dict[str, Any]] = []
    topo_metrics = [
        "mean_known_h1_max_persistence",
        "mean_known_h1_total_persistence",
        "lymphocyte_h1_max_persistence",
        "neutrophil_segmented_h1_max_persistence",
    ]
    osr_metrics = ["msp_auroc", "vim_auroc", "msp_fpr95", "vim_fpr95"]
    for topo_metric in topo_metrics:
        if topo_metric not in joined:
            continue
        for osr_metric in osr_metrics:
            if osr_metric not in joined:
                continue
            valid = joined[[topo_metric, osr_metric]].dropna()
            if len(valid) < 3:
                continue
            pearson = stats.pearsonr(valid[topo_metric], valid[osr_metric])
            spearman = stats.spearmanr(valid[topo_metric], valid[osr_metric])
            rows.append(
                {
                    "dataset": "MLL23_common_test",
                    "topology_metric": topo_metric,
                    "osr_metric": osr_metric,
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "spearman_r": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                    "n_runs": int(len(valid)),
                    "role": "exploratory_no_model_selection",
                }
            )
    return pd.DataFrame(rows)


def run_vr_analysis(paths: Delivery7Paths) -> Path:
    """Run explanatory VR topology analysis over frozen Delivery 6 embeddings."""

    run_vr_feasibility_smoke(paths)
    sample_manifest_path = build_vr_sample_manifest(paths)
    sample_manifest = pd.read_csv(sample_manifest_path)
    summary_rows: list[dict[str, Any]] = []
    betti_rows: list[dict[str, Any]] = []
    mixing_frames: list[pd.DataFrame] = []
    for spec in build_run_matrix():
        internal = load_embeddings(paths.internal_embedding_dir / f"{spec.run_id}_embeddings.npz")
        external = load_embeddings(
            paths.external_embedding_dir / f"{spec.run_id}_external_embeddings.npz"
        )
        for dataset, archive in (
            ("MLL23_common_test", internal),
            ("AML_LMU_external_test", external),
        ):
            dataset_manifest = sample_manifest.loc[sample_manifest["dataset"] == dataset]
            for class_name, group in dataset_manifest.groupby("class"):
                embeddings = _select_embeddings_by_ids(archive, group["sample_id"])
                if len(embeddings) < 2:
                    continue
                diagrams, elapsed, simplex_count = _rips_diagrams_with_metadata(embeddings)
                diagram_path = _diagram_artifact_path(
                    paths,
                    dataset=dataset,
                    spec=spec,
                    class_name=str(class_name),
                )
                _save_diagrams(diagram_path, diagrams)
                summary_rows.extend(
                    _diagram_summary_rows(
                        dataset=dataset,
                        spec=spec,
                        class_name=str(class_name),
                        n_samples=len(embeddings),
                        diagrams=diagrams,
                        diagram_path=diagram_path,
                        elapsed=elapsed,
                        simplex_count=simplex_count,
                    )
                )
                betti_rows.extend(
                    _betti_curve_rows(
                        dataset=dataset,
                        spec=spec,
                        class_name=str(class_name),
                        diagrams=diagrams,
                    )
                )
            mixing = _run_cloud_mixing(
                dataset=dataset,
                spec=spec,
                archive=archive,
                sample_manifest=sample_manifest,
            )
            if not mixing.empty:
                mixing_frames.append(mixing)
    paths.topology_dir.mkdir(parents=True, exist_ok=True)
    paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    summary_path = paths.topology_dir / "vr_diagram_summaries.csv"
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(betti_rows).to_csv(paths.topology_dir / "vr_betti_curves.csv", index=False)
    seed_pairs = _pairwise_seed_bottleneck(summary)
    seed_pairs.to_csv(paths.topology_dir / "vr_seed_pair_bottleneck.csv", index=False)
    split_pairs = _pairwise_split_bottleneck(summary)
    split_pairs.to_csv(paths.topology_dir / "vr_split_pair_bottleneck.csv", index=False)
    if mixing_frames:
        pd.concat(mixing_frames, ignore_index=True).to_csv(
            paths.topology_dir / "vr_cross_class_mixing.csv",
            index=False,
        )
    correlations = topology_osr_correlations(paths, summary)
    if not correlations.empty:
        correlations.to_csv(paths.metrics_dir / "vr_topology_osr_correlations.csv", index=False)
    metadata = {
        "role": "explanatory_only_not_classifier_or_osr_score",
        "sample_manifest": str(sample_manifest_path),
        "N_VR": N_VR,
        "sample_seed": VR_SAMPLE_SEED,
        "distance": "cosine_on_l2_normalized_embeddings",
        "homology_dimensions": [0, 1],
        "epsilon_grid_size": len(VR_EPSILON_GRID),
        "gudhi_version": _vr_library_version(),
        "vr_config_hash": vr_config_hash(),
    }
    write_json(paths.topology_dir / "vr_metadata.json", metadata)
    _write_vr_figures(paths, summary, seed_pairs)
    return summary_path


def vr_config_hash() -> str:
    """Hash the fixed VR technical protocol."""

    return config_hash(
        {
            "N_VR": N_VR,
            "VR_SAMPLE_SEED": VR_SAMPLE_SEED,
            "metric": "cosine_on_l2_normalized_embeddings",
            "homology_dimensions": [0, 1],
            "max_edge_length": 2.0,
            "h1_relative_threshold": H1_RELATIVE_PERSISTENCE_THRESHOLD,
            "diagram_distance": "bottleneck",
        }
    )


def write_external_audit_update(paths: Delivery7Paths) -> None:
    """Record manifest hash after QC/materialization."""

    audit_path = paths.data_audit_dir / "external_dataset_audit.json"
    if not audit_path.exists() or not paths.external_manifest_path.exists():
        return
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(paths.external_manifest_path)
    audit["external_manifest"] = {
        "path": str(paths.external_manifest_path),
        "rows": int(len(frame)),
        "known_rows": int(np.sum(frame["known_status"] == KNOWN)),
        "unknown_rows": int(np.sum(frame["known_status"] != KNOWN)),
        "manifest_hash": manifest_hash(frame),
    }
    write_json(audit_path, audit)
