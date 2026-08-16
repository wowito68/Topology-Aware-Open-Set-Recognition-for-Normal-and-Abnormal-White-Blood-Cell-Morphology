"""Delivery 3 morphology-aware and feature-map TDA experiments."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from hemato_osr.data.manifest import manifest_hash
from hemato_osr.data.taxonomy import DEFAULT_KNOWN_CLASSES
from hemato_osr.evaluation.metrics import fpr_at_tpr, open_set_metrics, oscr
from hemato_osr.openset.thresholds import apply_threshold, calibrate_strict_open_set
from hemato_osr.topology.activation_maps import (
    ActivationConfig,
    activation_maps_for_manifest,
    activation_metadata,
)
from hemato_osr.topology.archive import save_diagram_archive
from hemato_osr.topology.cache import config_hash
from hemato_osr.topology.diagrams import TDAConfig, compute_diagram_from_array
from hemato_osr.topology.morphology import (
    MorphologyConfig,
    load_rgb,
    mask_distance_map,
    morphology_metadata_row,
    overlay_mask,
    segment_morphology,
    write_morphology_qc_sheet,
)
from hemato_osr.topology.pipeline import (
    TDAVectorizeConfig,
    sample_stratified_manifest,
    vectorize_diagram_archive,
)
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

DiagramKey = tuple[str, int]
DiagramBundle = dict[DiagramKey, list[np.ndarray]]


@dataclass(frozen=True)
class Delivery3Paths:
    """Common Delivery 3 paths."""

    manifest_path: Path
    embeddings_path: Path
    checkpoint_path: Path
    output_root: Path = Path("artifacts")
    seed: int = 37


@dataclass(frozen=True)
class TDAAnomalyConfig:
    """Class-statistics anomaly configuration."""

    osr_method: str = "centroid_distance"
    regularization: float = 1e-4
    known_classes: tuple[str, ...] = DEFAULT_KNOWN_CLASSES


@dataclass(frozen=True)
class LateFusionConfig:
    """Transparent late-fusion configuration."""

    alphas: tuple[float, ...] = (0.25, 0.5, 0.75)
    primary_alpha: float = 0.5


def sha256_file(path: Path) -> str:
    """Compute SHA256 without printing file contents."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _delivery3_figures_dir_from_artifact(path: Path) -> Path:
    parts = path.parts
    if "artifacts" in parts:
        idx = parts.index("artifacts")
        root = Path(*parts[: idx + 1])
        return root / "figures" / "delivery3"
    return path.parent / "figures" / "delivery3"


def sample_qc_subset(
    manifest: pd.DataFrame,
    *,
    per_class: int = 20,
    seed: int = 37,
) -> pd.DataFrame:
    """Select a deterministic known-train QC subset."""

    known_train = manifest.loc[
        (manifest["known_status"].astype(str) == "known")
        & (manifest["split"].astype(str) == "train")
    ]
    rows = []
    rng = np.random.default_rng(seed)
    for label in sorted(known_train["canonical_label"].astype(str).unique()):
        group = known_train.loc[known_train["canonical_label"].astype(str) == label]
        rows.append(
            group.sample(
                n=min(per_class, len(group)),
                random_state=int(rng.integers(0, 1_000_000)),
            )
        )
    return pd.concat(rows, ignore_index=True).sort_values("sample_id").reset_index(drop=True)


def run_morphology_qc(
    manifest_path: Path,
    output_dir: Path,
    *,
    per_class: int = 20,
    seed: int = 37,
    config: MorphologyConfig | None = None,
) -> Path:
    """Generate known-train morphology QC sheets and metadata."""

    cfg = config or MorphologyConfig()
    frame = pd.read_csv(manifest_path)
    subset = sample_qc_subset(frame, per_class=per_class, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_dir / "morphology_qc_subset.csv", index=False)
    metadata = write_morphology_qc_sheet(
        subset,
        output_dir / "morphology_candidate_qc.png",
        cfg,
        max_rows=len(subset),
    )
    _write_nucleus_candidate_examples(
        subset,
        output_dir / "nucleus_candidate_examples.png",
        cfg,
    )
    write_json(output_dir / "morphology_config.json", cfg.to_dict())
    summary = {
        "sample_count": int(len(metadata)),
        "classes": sorted(metadata["label"].astype(str).unique().tolist()),
        "status_counts": metadata["segmentation_status"].value_counts().to_dict(),
        "fraction_summary": metadata[
            ["foreground_fraction", "nucleus_fraction", "cytoplasm_fraction"]
        ]
        .describe()
        .to_dict(),
    }
    write_json(output_dir / "morphology_qc_summary.json", summary)
    return output_dir / "morphology_qc_summary.json"


def _write_nucleus_candidate_examples(
    frame: pd.DataFrame,
    output_path: Path,
    config: MorphologyConfig,
) -> None:
    n = min(8, len(frame))
    panel_w = config.image_size
    label_h = 24
    from PIL import Image, ImageDraw

    sheet = Image.new("RGB", (2 * panel_w, n * (panel_w + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    selected = frame.sort_values("sample_id").head(n).reset_index(drop=True)
    for row_idx, row in enumerate(selected.itertuples(index=False)):
        path = Path(str(row.path))
        rgb = load_rgb(path, config.image_size)
        result = segment_morphology(path, config)
        y = row_idx * (panel_w + label_h)
        sheet.paste(Image.fromarray((rgb * 255).astype(np.uint8)), (0, y))
        sheet.paste(overlay_mask(rgb, result.nucleus, (80, 80, 220)), (panel_w, y))
        text = f"{row.sample_id} {row.canonical_label} {result.status}"
        draw.text((4, y + panel_w + 2), text[:80], fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _morphology_maps(
    path: Path,
    config: MorphologyConfig,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    result = segment_morphology(path, config)
    valid = result.status != "invalid"
    whole_cell = result.whole_cell if valid else np.zeros_like(result.whole_cell)
    nucleus = result.nucleus if valid else np.zeros_like(result.nucleus)
    cytoplasm_valid = valid and result.cytoplasm_fraction >= config.min_cytoplasm_fraction
    cytoplasm = result.cytoplasm if cytoplasm_valid else np.zeros_like(result.cytoplasm)
    maps = {
        "cell": mask_distance_map(whole_cell),
        "nucleus": mask_distance_map(nucleus),
        "cytoplasm": mask_distance_map(cytoplasm),
    }
    metadata = morphology_metadata_row("", "", "", result)
    return maps, metadata


def _diagram_from_map(values: np.ndarray, filtration: str) -> dict[int, np.ndarray]:
    return compute_diagram_from_array(
        values,
        TDAConfig(
            filtration=filtration,
            image_size=int(values.shape[0]),
            homology_dimensions=(0, 1),
        ),
    )


def _plot_diagram(ax: Any, diagram: dict[int, np.ndarray]) -> None:
    colors = {0: "tab:blue", 1: "tab:orange"}
    for dim, points in diagram.items():
        arr = np.asarray(points, dtype=float).reshape(-1, 2)
        finite = arr[np.isfinite(arr).all(axis=1)]
        if finite.size:
            ax.scatter(
                finite[:, 0],
                finite[:, 1],
                s=8,
                alpha=0.75,
                label=f"H{dim}",
                color=colors.get(int(dim), "tab:gray"),
            )
    ax.plot([0, 1], [0, 1], color="black", linewidth=0.6, linestyle=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])


def _write_morphology_persistence_examples(
    frame: pd.DataFrame,
    output_path: Path,
    config: MorphologyConfig,
    *,
    n_examples: int = 4,
) -> None:
    import matplotlib.pyplot as plt

    selected = frame.sort_values("sample_id").head(n_examples).reset_index(drop=True)
    if selected.empty:
        return
    fig, axes = plt.subplots(len(selected), 4, figsize=(9, 2.2 * len(selected)))
    axes_arr = np.asarray(axes).reshape(len(selected), 4)
    for row_idx, row in enumerate(selected.itertuples(index=False)):
        path = Path(str(row.path))
        rgb = load_rgb(path, config.image_size)
        result = segment_morphology(path, config)
        distance_map = mask_distance_map(result.nucleus)
        diagram = _diagram_from_map(distance_map, "superlevel")
        axes_arr[row_idx, 0].imshow(rgb)
        axes_arr[row_idx, 1].imshow(overlay_mask(rgb, result.nucleus, (80, 80, 220)))
        axes_arr[row_idx, 2].imshow(distance_map, cmap="viridis", vmin=0, vmax=1)
        _plot_diagram(axes_arr[row_idx, 3], diagram)
        for col in range(3):
            axes_arr[row_idx, col].axis("off")
        axes_arr[row_idx, 0].set_ylabel(str(row.canonical_label)[:16], fontsize=8)
    axes_arr[0, 0].set_title("original")
    axes_arr[0, 1].set_title("nucleus candidate")
    axes_arr[0, 2].set_title("distance map")
    axes_arr[0, 3].set_title("PH diagram")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_feature_map_persistence_examples(
    frame: pd.DataFrame,
    maps: list[np.ndarray],
    output_path: Path,
    *,
    n_examples: int = 4,
) -> None:
    import matplotlib.pyplot as plt

    n = min(n_examples, len(frame), len(maps))
    if n == 0:
        return
    fig, axes = plt.subplots(n, 3, figsize=(7, 2.2 * n))
    axes_arr = np.asarray(axes).reshape(n, 3)
    selected = frame.sort_values("sample_id").head(n).reset_index(drop=True)
    for row_idx, row in enumerate(selected.itertuples(index=False)):
        image = plt.imread(str(row.path))
        values = maps[row_idx]
        diagram = _diagram_from_map(values, "superlevel")
        axes_arr[row_idx, 0].imshow(image)
        axes_arr[row_idx, 1].imshow(values, cmap="magma", vmin=0, vmax=1)
        _plot_diagram(axes_arr[row_idx, 2], diagram)
        axes_arr[row_idx, 0].axis("off")
        axes_arr[row_idx, 1].axis("off")
        axes_arr[row_idx, 0].set_ylabel(str(row.canonical_label)[:16], fontsize=8)
    axes_arr[0, 0].set_title("original")
    axes_arr[0, 1].set_title("activation energy")
    axes_arr[0, 2].set_title("PH diagram")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _morphology_worker(
    args: tuple[str, MorphologyConfig, tuple[str, ...]],
) -> tuple[DiagramBundle, dict[str, object]]:
    path_text, config, views = args
    maps, metadata = _morphology_maps(Path(path_text), config)
    diagrams: dict[tuple[str, int], list[np.ndarray]] = {}
    for view in views:
        for filtration in ("sublevel", "superlevel"):
            diagram = _diagram_from_map(maps[view], filtration)
            for dim, points in diagram.items():
                diagrams[(f"{view}_{filtration}", int(dim))] = [points]
    return diagrams, metadata


def benchmark_shape_tda(
    manifest_path: Path,
    output_dir: Path,
    *,
    sample_size: int = 500,
    seed: int = 37,
    config: MorphologyConfig | None = None,
) -> Path:
    """Benchmark morphology distance-map PH on a stratified subset."""

    cfg = config or MorphologyConfig()
    frame = pd.read_csv(manifest_path)
    sample = sample_stratified_manifest(frame, sample_size=sample_size, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_dir / "shape_tda_benchmark_manifest.csv", index=False)
    rows = []
    for view in ("cell", "nucleus", "cytoplasm"):
        runtimes = []
        diagram_points = []
        degenerate = []
        variances = []
        statuses = []
        for row in sample.itertuples(index=False):
            maps, metadata = _morphology_maps(Path(str(row.path)), cfg)
            statuses.append(str(metadata["segmentation_status"]))
            start = time.perf_counter()
            diagrams = [
                _diagram_from_map(maps[view], filtration)
                for filtration in ("sublevel", "superlevel")
            ]
            runtimes.append(time.perf_counter() - start)
            point_count = int(
                sum(
                    np.asarray(points).reshape(-1, 2).shape[0]
                    for diagram in diagrams
                    for points in diagram.values()
                )
            )
            diagram_points.append(point_count)
            degenerate.append(point_count <= 2 or float(np.var(maps[view])) <= 1e-10)
            variances.append(float(np.var(maps[view])))
        rows.append(
            {
                "view": view,
                "n_images": int(len(sample)),
                "mean_runtime_seconds": float(np.mean(runtimes)),
                "p95_runtime_seconds": float(np.quantile(runtimes, 0.95)),
                "mean_diagram_points": float(np.mean(diagram_points)),
                "degenerate_percent": float(np.mean(degenerate) * 100.0),
                "feature_variance": float(np.mean(variances)),
                "valid_percent": float(np.mean(np.asarray(statuses) == "valid") * 100.0),
            }
        )
    benchmark = pd.DataFrame(rows)
    benchmark.to_csv(output_dir / "shape_tda_benchmark.csv", index=False)
    write_json(output_dir / "shape_tda_benchmark_config.json", cfg.to_dict())
    return output_dir / "shape_tda_benchmark.csv"


def extract_morphology_tda(
    manifest_path: Path,
    diagram_output: Path,
    feature_output: Path,
    *,
    views: tuple[str, ...] = ("cell", "nucleus", "cytoplasm"),
    workers: int = 2,
    seed: int = 37,
    config: MorphologyConfig | None = None,
) -> Path:
    """Extract morphology-aware PH diagrams and vector features."""

    cfg = config or MorphologyConfig()
    frame = pd.read_csv(manifest_path).sort_values("sample_id").reset_index(drop=True)
    diagrams_by_key: dict[DiagramKey, list[np.ndarray | None]] = {
        (f"{view}_{filtration}", dim): [None] * len(frame)
        for view in views
        for filtration in ("sublevel", "superlevel")
        for dim in (0, 1)
    }
    metadata_rows: list[dict[str, object] | None] = [None] * len(frame)
    args = [(str(row.path), cfg, views) for row in frame.itertuples(index=False)]

    def store_result(idx: int, result: tuple[DiagramBundle, dict[str, object]]) -> None:
        diagrams, metadata = result
        row = frame.iloc[idx]
        metadata.update(
            {
                "sample_id": str(row["sample_id"]),
                "label": str(row["canonical_label"]),
                "split": str(row["split"]),
            }
        )
        metadata_rows[idx] = metadata
        for key, values in diagrams.items():
            diagrams_by_key[key][idx] = values[0]

    if workers <= 1:
        for idx, arg in enumerate(args):
            store_result(idx, _morphology_worker(arg))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_morphology_worker, arg): idx for idx, arg in enumerate(args)
            }
            for future in as_completed(futures):
                store_result(futures[future], future.result())

    complete_metadata = [metadata if metadata is not None else {} for metadata in metadata_rows]
    complete_diagrams: dict[DiagramKey, list[np.ndarray]] = {}
    for key, values in diagrams_by_key.items():
        complete_diagrams[key] = [
            value if value is not None else np.empty((0, 2), dtype=np.float64) for value in values
        ]

    diagram_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_frame = pd.DataFrame(complete_metadata)
    metadata_frame.to_csv(diagram_output.with_suffix(".metadata.csv"), index=False)
    archive_metadata: dict[str, Any] = {
        "sample_count": int(len(frame)),
        "manifest_hash": manifest_hash(frame),
        "views": list(views),
        "filtrations": [
            f"{view}_{filtration}" for view in views for filtration in ("sublevel", "superlevel")
        ],
        "homology_dimensions": [0, 1],
        "morphology_config": cfg.to_dict(),
        "morphology_config_hash": config_hash(cfg.to_dict()),
        "source": "candidate masks converted to normalized Euclidean distance maps",
        "invalid_policy": (
            "invalid candidate maps are encoded as zero distance maps and tracked in metadata"
        ),
        "seed": seed,
    }
    archive_metadata["config_hash"] = config_hash(archive_metadata)
    save_diagram_archive(
        diagram_output,
        sample_ids=frame["sample_id"].astype(str).tolist(),
        labels=frame["canonical_label"].astype(str).to_numpy(),
        known_status=frame["known_status"].astype(str).to_numpy(),
        splits=frame["split"].astype(str).to_numpy(),
        diagrams_by_key=complete_diagrams,
        metadata=archive_metadata,
    )
    vectorize_diagram_archive(
        TDAVectorizeConfig(
            diagram_path=diagram_output,
            output_path=feature_output,
            filtrations=tuple(archive_metadata["filtrations"]),
            seed=seed,
        )
    )
    _write_morphology_persistence_examples(
        frame,
        _delivery3_figures_dir_from_artifact(diagram_output)
        / "morphology_persistence_examples.png",
        cfg,
    )
    return feature_output


def benchmark_feature_maps(
    manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    *,
    sample_size: int = 500,
    seed: int = 37,
    layers: tuple[str, ...] = ("layer1", "layer2", "layer3", "layer4"),
    batch_size: int = 64,
    device: str = "auto",
) -> Path:
    """Benchmark PH over frozen ResNet activation-energy maps."""

    frame = pd.read_csv(manifest_path)
    sample = sample_stratified_manifest(frame, sample_size=sample_size, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_dir / "feature_map_benchmark_manifest.csv", index=False)
    rows = []
    for layer in layers:
        config = ActivationConfig(
            checkpoint_path=checkpoint_path,
            layer=layer,
            batch_size=batch_size,
            device=device,
        )
        sample_ids, maps, activation_shape = activation_maps_for_manifest(sample, config)
        runtimes = []
        point_counts = []
        degenerate = []
        variances = []
        for values in maps:
            start = time.perf_counter()
            diagrams = [
                _diagram_from_map(values, filtration) for filtration in ("sublevel", "superlevel")
            ]
            runtimes.append(time.perf_counter() - start)
            point_count = int(
                sum(
                    np.asarray(points).reshape(-1, 2).shape[0]
                    for diagram in diagrams
                    for points in diagram.values()
                )
            )
            point_counts.append(point_count)
            degenerate.append(point_count <= 2 or float(np.var(values)) <= 1e-10)
            variances.append(float(np.var(values)))
        rows.append(
            {
                "layer": layer,
                "activation_shape": "x".join(str(x) for x in activation_shape),
                "scalar_map_size": f"{maps[0].shape[0]}x{maps[0].shape[1]}",
                "n_images": int(len(sample_ids)),
                "mean_ph_runtime_seconds": float(np.mean(runtimes)),
                "p95_ph_runtime_seconds": float(np.quantile(runtimes, 0.95)),
                "mean_diagram_points": float(np.mean(point_counts)),
                "degenerate_percent": float(np.mean(degenerate) * 100.0),
                "feature_variance": float(np.mean(variances)),
            }
        )
        _write_feature_map_examples(sample, maps, output_dir / f"feature_map_examples_{layer}.png")
    benchmark = pd.DataFrame(rows)
    selected = _select_feature_layer(benchmark)
    benchmark["selected"] = benchmark["layer"] == selected
    benchmark["selection_reason"] = np.where(
        benchmark["selected"],
        "selected by spatial resolution, low degeneracy, runtime, and variance before OSR metrics",
        "not selected by pre-test feasibility criterion",
    )
    benchmark.to_csv(output_dir / "feature_map_layer_benchmark.csv", index=False)
    write_json(
        output_dir / "feature_map_layer_selection.json",
        {
            "selected_layer": selected,
            "criterion": (
                "prefer low degeneracy, nontrivial variance, moderate runtime, "
                "and at least 14x14 spatial map"
            ),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
    )
    return output_dir / "feature_map_layer_benchmark.csv"


def _select_feature_layer(benchmark: pd.DataFrame) -> str:
    candidates = benchmark.copy()
    candidates["height"] = candidates["scalar_map_size"].str.split("x").str[0].astype(int)
    feasible = candidates.loc[
        (candidates["height"] >= 14)
        & (candidates["degenerate_percent"] <= 25.0)
        & (candidates["feature_variance"] > 1e-6)
    ].copy()
    if feasible.empty:
        feasible = candidates.copy()
    feasible["score"] = (
        feasible["degenerate_percent"] / 100.0
        + feasible["mean_ph_runtime_seconds"]
        / max(float(feasible["mean_ph_runtime_seconds"].max()), 1e-9)
        - np.log1p(feasible["height"]) * 0.05
    )
    return str(feasible.sort_values("score").iloc[0]["layer"])


def _write_feature_map_examples(
    frame: pd.DataFrame,
    maps: list[np.ndarray],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    n = min(8, len(maps))
    fig, axes = plt.subplots(2, n, figsize=(max(8, 1.6 * n), 3.4))
    axes_arr = np.asarray(axes)
    for idx in range(n):
        image = plt.imread(str(frame.iloc[idx]["path"]))
        axes_arr[0, idx].imshow(image)
        axes_arr[0, idx].axis("off")
        axes_arr[1, idx].imshow(maps[idx], cmap="magma")
        axes_arr[1, idx].axis("off")
        axes_arr[1, idx].set_title(str(frame.iloc[idx]["canonical_label"])[:14], fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def extract_feature_map_tda(
    manifest_path: Path,
    checkpoint_path: Path,
    diagram_output: Path,
    feature_output: Path,
    *,
    layer: str,
    batch_size: int = 64,
    seed: int = 37,
    device: str = "auto",
) -> Path:
    """Extract feature-map activation-energy PH and vector features."""

    frame = pd.read_csv(manifest_path).sort_values("sample_id").reset_index(drop=True)
    activation_config = ActivationConfig(
        checkpoint_path=checkpoint_path,
        layer=layer,
        batch_size=batch_size,
        device=device,
    )
    sample_ids, maps, activation_shape = activation_maps_for_manifest(frame, activation_config)
    order = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    maps_ordered = [maps[order[str(sample_id)]] for sample_id in frame["sample_id"].astype(str)]
    diagrams_by_key: dict[tuple[str, int], list[np.ndarray]] = {
        (f"{layer}_{filtration}", dim): []
        for filtration in ("sublevel", "superlevel")
        for dim in (0, 1)
    }
    for values in maps_ordered:
        for filtration in ("sublevel", "superlevel"):
            diagram = _diagram_from_map(values, filtration)
            for dim, points in diagram.items():
                diagrams_by_key[(f"{layer}_{filtration}", int(dim))].append(points)
    metadata: dict[str, Any] = {
        "sample_count": int(len(frame)),
        "manifest_hash": manifest_hash(frame),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "activation": activation_metadata(activation_config, activation_shape),
        "filtrations": [f"{layer}_sublevel", f"{layer}_superlevel"],
        "homology_dimensions": [0, 1],
        "seed": seed,
    }
    metadata["config_hash"] = config_hash(metadata)
    save_diagram_archive(
        diagram_output,
        sample_ids=frame["sample_id"].astype(str).tolist(),
        labels=frame["canonical_label"].astype(str).to_numpy(),
        known_status=frame["known_status"].astype(str).to_numpy(),
        splits=frame["split"].astype(str).to_numpy(),
        diagrams_by_key=diagrams_by_key,
        metadata=metadata,
    )
    vectorize_diagram_archive(
        TDAVectorizeConfig(
            diagram_path=diagram_output,
            output_path=feature_output,
            filtrations=tuple(metadata["filtrations"]),
            seed=seed,
        )
    )
    _write_feature_map_persistence_examples(
        frame,
        maps_ordered,
        _delivery3_figures_dir_from_artifact(diagram_output)
        / "feature_map_persistence_examples.png",
    )
    return feature_output


def write_predeclared_matrix(
    output_path: Path,
    *,
    selected_layer: str,
    include_cytoplasm: bool,
) -> Path:
    """Persist the Delivery 3 matrix before final test evaluation."""

    methods: list[dict[str, object]] = [
        {
            "id": "deep_msp",
            "representation": "Deep MSP baseline",
            "source": "resnet18_logits",
            "osr_score": "msp_anomaly",
            "primary": True,
        },
        {
            "id": "raw_tda_centroid_historical",
            "representation": "Raw-image TDA historical",
            "source": "delivery2_raw_grayscale_tda",
            "osr_score": "centroid_distance",
            "primary": False,
        },
        {
            "id": "morph_cell_centroid",
            "representation": "Morph Cell",
            "source": "candidate_mask_distance_map",
            "tda_view": "cell",
            "osr_score": "centroid_distance",
            "primary": True,
        },
        {
            "id": "morph_cell_mahalanobis",
            "representation": "Morph Cell",
            "source": "candidate_mask_distance_map",
            "tda_view": "cell",
            "osr_score": "mahalanobis",
            "primary": False,
        },
        {
            "id": "morph_nucleus_centroid",
            "representation": "Morph Nucleus",
            "source": "candidate_mask_distance_map",
            "tda_view": "nucleus",
            "osr_score": "centroid_distance",
            "primary": True,
        },
        {
            "id": "morph_nucleus_mahalanobis",
            "representation": "Morph Nucleus",
            "source": "candidate_mask_distance_map",
            "tda_view": "nucleus",
            "osr_score": "mahalanobis",
            "primary": False,
        },
        {
            "id": "morph_cell_nucleus_centroid",
            "representation": "Morph Cell+Nucleus",
            "source": "candidate_mask_distance_map",
            "tda_view": "cell+nucleus",
            "osr_score": "centroid_distance",
            "primary": True,
        },
        {
            "id": "morph_cell_nucleus_mahalanobis",
            "representation": "Morph Cell+Nucleus",
            "source": "candidate_mask_distance_map",
            "tda_view": "cell+nucleus",
            "osr_score": "mahalanobis",
            "primary": False,
        },
        {
            "id": "feature_map_centroid",
            "representation": "Feature-map TDA",
            "source": "resnet18_activation_energy",
            "layer": selected_layer,
            "osr_score": "centroid_distance",
            "primary": True,
        },
        {
            "id": "feature_map_mahalanobis",
            "representation": "Feature-map TDA",
            "source": "resnet18_activation_energy",
            "layer": selected_layer,
            "osr_score": "mahalanobis",
            "primary": False,
        },
        {
            "id": "late_morph_cell_nucleus_alpha_0_50",
            "representation": "Deep+Morph late fusion",
            "source": "deep_msp_plus_morph_cell_nucleus",
            "fusion_rule": "fixed_alpha",
            "alpha": 0.5,
            "primary": True,
        },
        {
            "id": "late_feature_map_alpha_0_50",
            "representation": "Deep+Feature-map late fusion",
            "source": "deep_msp_plus_feature_map",
            "layer": selected_layer,
            "fusion_rule": "fixed_alpha",
            "alpha": 0.5,
            "primary": True,
        },
    ]
    for alpha in (0.25, 0.75):
        methods.append(
            {
                "id": f"late_morph_cell_nucleus_alpha_{str(alpha).replace('.', '_')}",
                "representation": "Deep+Morph late fusion",
                "source": "deep_msp_plus_morph_cell_nucleus",
                "fusion_rule": "fixed_alpha",
                "alpha": alpha,
                "primary": False,
            }
        )
        methods.append(
            {
                "id": f"late_feature_map_alpha_{str(alpha).replace('.', '_')}",
                "representation": "Deep+Feature-map late fusion",
                "source": "deep_msp_plus_feature_map",
                "layer": selected_layer,
                "fusion_rule": "fixed_alpha",
                "alpha": alpha,
                "primary": False,
            }
        )
    methods.extend(
        [
            {
                "id": "late_morph_cell_nucleus_max",
                "representation": "Deep+Morph late fusion",
                "source": "deep_msp_plus_morph_cell_nucleus",
                "fusion_rule": "max",
                "primary": False,
            },
            {
                "id": "late_feature_map_max",
                "representation": "Deep+Feature-map late fusion",
                "source": "deep_msp_plus_feature_map",
                "layer": selected_layer,
                "fusion_rule": "max",
                "primary": False,
            },
        ]
    )
    if include_cytoplasm:
        methods.append(
            {
                "id": "morph_cytoplasm_centroid",
                "representation": "Morph Cytoplasm",
                "source": "candidate_mask_distance_map",
                "tda_view": "cytoplasm",
                "osr_score": "centroid_distance",
                "primary": False,
            }
        )
        methods.append(
            {
                "id": "morph_cytoplasm_mahalanobis",
                "representation": "Morph Cytoplasm",
                "source": "candidate_mask_distance_map",
                "tda_view": "cytoplasm",
                "osr_score": "mahalanobis",
                "primary": False,
            }
        )
    payload = {
        "status": "predeclared_before_final_test_evaluation",
        "seed": 37,
        "split": "V1",
        "decision_criteria": {
            "criterion_a": "Delta overall AUROC >= +0.01 with favorable paired bootstrap CI",
            "criterion_b": "Delta FPR95 <= -0.05 with no meaningful overall AUROC degradation",
            "criterion_c": "Delta subgroup AUROC >= +0.03 without severe overall OSR degradation",
            "abandonment": (
                "If all defensible variants have delta AUROC <= 0 and delta FPR95 "
                ">= 0, and no subgroup effect, recommend stopping TDA development."
            ),
        },
        "unknown_groups": UNKNOWN_GROUPS,
        "methods": methods,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _load_feature_archive(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {
            "sample_id": np.asarray(data["sample_id"]).astype(str),
            "true_label": np.asarray(data["true_label"]).astype(str),
            "known_status": np.asarray(data["known_status"]).astype(str),
            "split": np.asarray(data["split"]).astype(str),
            "features": np.asarray(data["features"], dtype=np.float32),
            "feature_names": np.asarray(data["feature_names"]).astype(str),
            "metadata": json.loads(str(data["metadata"])),
        }


def _select_feature_columns(feature_names: np.ndarray, prefixes: tuple[str, ...]) -> np.ndarray:
    indices = [
        idx
        for idx, name in enumerate(feature_names.astype(str).tolist())
        if any(name.startswith(prefix) for prefix in prefixes)
    ]
    if not indices:
        msg = f"No feature columns found for prefixes {prefixes}"
        raise ValueError(msg)
    return np.asarray(indices, dtype=int)


def _label_mapping(known_classes: tuple[str, ...] = DEFAULT_KNOWN_CLASSES) -> dict[str, int]:
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
    regularization: float,
) -> np.ndarray:
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
    if not np.isfinite(precision).all():
        msg = "Regularized Mahalanobis precision matrix is not finite"
        raise ValueError(msg)
    distances = []
    for mean in means:
        delta = features - mean
        distances.append(np.sum(delta @ precision * delta, axis=1))
    return np.min(np.vstack(distances), axis=0)


def _tdafeature_scores(
    archive: dict[str, Any],
    prefixes: tuple[str, ...],
    config: TDAAnomalyConfig,
) -> np.ndarray:
    columns = _select_feature_columns(archive["feature_names"], prefixes)
    features = archive["features"][:, columns]
    labels = archive["true_label"].astype(str)
    splits = archive["split"].astype(str)
    label_to_index = _label_mapping(config.known_classes)
    y = np.asarray([label_to_index.get(label, -1) for label in labels], dtype=int)
    train_mask = (y >= 0) & (splits == "train")
    scaler = StandardScaler().fit(features[train_mask])
    scaled = scaler.transform(features)
    if not np.isfinite(scaled).all():
        msg = "Non-finite TDA scores after train-only scaling"
        raise ValueError(msg)
    if config.osr_method == "centroid_distance":
        return _centroid_scores(scaled[train_mask], y[train_mask], scaled)
    if config.osr_method == "mahalanobis":
        return _mahalanobis_scores(
            scaled[train_mask],
            y[train_mask],
            scaled,
            config.regularization,
        )
    msg = f"Unsupported Delivery 3 TDA score: {config.osr_method}"
    raise ValueError(msg)


def empirical_percentile_scores(scores: np.ndarray, reference_scores: np.ndarray) -> np.ndarray:
    """Map scores to empirical percentile against known-validation reference."""

    ref = np.sort(np.asarray(reference_scores, dtype=float))
    values = np.asarray(scores, dtype=float)
    if ref.size == 0:
        msg = "Calibration requires a non-empty known-validation reference"
        raise ValueError(msg)
    return np.searchsorted(ref, values, side="right") / float(ref.size)


def fixed_alpha_late_fusion(
    deep_scores: np.ndarray,
    tda_scores: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Combine calibrated scores with fixed alpha; larger remains unknown-like."""

    if not 0.0 <= alpha <= 1.0:
        msg = "Fixed late-fusion alpha must be in [0, 1]"
        raise ValueError(msg)
    return alpha * deep_scores + (1.0 - alpha) * tda_scores


def max_late_fusion(deep_scores: np.ndarray, tda_scores: np.ndarray) -> np.ndarray:
    """Predefined secondary fusion rule; larger remains unknown-like."""

    return np.maximum(np.asarray(deep_scores, dtype=float), np.asarray(tda_scores, dtype=float))


def representative_indices_by_score(
    labels: np.ndarray,
    scores: np.ndarray,
    target_label: str,
    *,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> list[int]:
    """Select deterministic score-quantile representatives for a label."""

    label_array = np.asarray(labels).astype(str)
    score_array = np.asarray(scores, dtype=float)
    candidates = np.flatnonzero(label_array == target_label)
    if candidates.size == 0:
        return []
    ordered = candidates[np.argsort(score_array[candidates], kind="mergesort")]
    selected: list[int] = []
    for quantile in quantiles:
        clipped = min(max(float(quantile), 0.0), 1.0)
        position = int(round(clipped * (len(ordered) - 1)))
        idx = int(ordered[position])
        if idx not in selected:
            selected.append(idx)
    return selected


def _load_embeddings(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _deep_msp_scores(logits: np.ndarray) -> np.ndarray:
    probabilities = softmax(logits, axis=1)
    return 1.0 - probabilities.max(axis=1)


def _score_metrics(
    embeddings: dict[str, Any],
    scores: np.ndarray,
    *,
    method_id: str,
    output_dir: Path,
) -> dict[str, float]:
    label_to_index = json.loads(str(embeddings["label_to_index"].tolist()))
    logits = embeddings["logits"].astype(float)
    true_label = embeddings["true_label"].astype(str)
    known_status = embeddings["known_status"].astype(str)
    split = embeddings["split"].astype(str)
    prediction = embeddings["prediction"].astype(str)
    numeric_true = np.asarray([label_to_index.get(label, -1) for label in true_label], dtype=int)
    test_mask = split == "test"
    validation_known = (split == "validation") & (known_status == "known")
    threshold = calibrate_strict_open_set(scores[validation_known], target_known_recall=0.95)
    test_unknown = (known_status[test_mask] != "known").astype(int)
    test_scores = scores[test_mask]
    unknown_pred = apply_threshold(test_scores, threshold.threshold)
    pred_with_unknown = prediction[test_mask].astype(object)
    pred_with_unknown[unknown_pred == 1] = "UNKNOWN"
    true_with_unknown = true_label[test_mask].astype(object)
    true_with_unknown[test_unknown == 1] = "UNKNOWN"
    metrics = open_set_metrics(test_unknown, test_scores, true_with_unknown, pred_with_unknown)
    metrics["aupr_known_positive"] = float(average_precision_score(1 - test_unknown, -test_scores))
    metrics["oscr"] = oscr(
        numeric_true[test_mask],
        logits[test_mask].argmax(axis=1),
        -test_scores,
        test_unknown,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sample_id": embeddings["sample_id"][test_mask].astype(str),
            "true_label": true_label[test_mask],
            "known_status": known_status[test_mask],
            "closed_set_prediction": prediction[test_mask],
            "open_set_prediction": pred_with_unknown,
            "unknown_score": test_scores,
            "threshold": threshold.threshold,
            "method": method_id,
        }
    ).to_csv(output_dir / f"{method_id}_predictions.csv", index=False)
    return {key: float(value) for key, value in metrics.items()}


def _predictions_for_method(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def evaluate_delivery3(
    embeddings_path: Path,
    morphology_features: Path,
    feature_map_features: Path,
    raw_tda_predictions: Path,
    matrix_path: Path,
    output_dir: Path,
    *,
    checkpoint_path: Path,
    seed: int = 37,
) -> Path:
    """Evaluate predeclared Delivery 3 scores on V1 test data."""

    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    embeddings = _load_embeddings(embeddings_path)
    morphology = _load_feature_archive(morphology_features)
    fmap = _load_feature_archive(feature_map_features)
    logits = embeddings["logits"].astype(float)
    split = embeddings["split"].astype(str)
    known_status = embeddings["known_status"].astype(str)
    validation_known = (split == "validation") & (known_status == "known")
    deep_scores = _deep_msp_scores(logits)
    deep_cal = empirical_percentile_scores(deep_scores, deep_scores[validation_known])
    score_by_id: dict[str, np.ndarray] = {
        "deep_msp": deep_scores,
    }
    tda_cfg = TDAAnomalyConfig()
    score_by_id["morph_cell_centroid"] = _tdafeature_scores(
        morphology,
        ("cell_sublevel_", "cell_superlevel_"),
        tda_cfg,
    )
    score_by_id["morph_nucleus_centroid"] = _tdafeature_scores(
        morphology,
        ("nucleus_sublevel_", "nucleus_superlevel_"),
        tda_cfg,
    )
    score_by_id["morph_cell_nucleus_centroid"] = _tdafeature_scores(
        morphology,
        ("cell_sublevel_", "cell_superlevel_", "nucleus_sublevel_", "nucleus_superlevel_"),
        tda_cfg,
    )
    fmap_prefixes = tuple(f"{filtration}_" for filtration in fmap["metadata"]["filtrations"])
    score_by_id["feature_map_centroid"] = _tdafeature_scores(
        fmap,
        fmap_prefixes,
        tda_cfg,
    )
    maha_cfg = TDAAnomalyConfig(osr_method="mahalanobis")
    score_by_id["morph_cell_mahalanobis"] = _tdafeature_scores(
        morphology,
        ("cell_sublevel_", "cell_superlevel_"),
        maha_cfg,
    )
    score_by_id["morph_nucleus_mahalanobis"] = _tdafeature_scores(
        morphology,
        ("nucleus_sublevel_", "nucleus_superlevel_"),
        maha_cfg,
    )
    score_by_id["morph_cell_nucleus_mahalanobis"] = _tdafeature_scores(
        morphology,
        ("cell_sublevel_", "cell_superlevel_", "nucleus_sublevel_", "nucleus_superlevel_"),
        maha_cfg,
    )
    score_by_id["feature_map_mahalanobis"] = _tdafeature_scores(
        fmap,
        fmap_prefixes,
        maha_cfg,
    )
    if any(method["id"] == "morph_cytoplasm_centroid" for method in matrix["methods"]):
        score_by_id["morph_cytoplasm_centroid"] = _tdafeature_scores(
            morphology,
            ("cytoplasm_sublevel_", "cytoplasm_superlevel_"),
            tda_cfg,
        )
        score_by_id["morph_cytoplasm_mahalanobis"] = _tdafeature_scores(
            morphology,
            ("cytoplasm_sublevel_", "cytoplasm_superlevel_"),
            maha_cfg,
        )
    morph_cal = empirical_percentile_scores(
        score_by_id["morph_cell_nucleus_centroid"],
        score_by_id["morph_cell_nucleus_centroid"][validation_known],
    )
    fmap_cal = empirical_percentile_scores(
        score_by_id["feature_map_centroid"],
        score_by_id["feature_map_centroid"][validation_known],
    )
    for method in matrix["methods"]:
        method_id = str(method["id"])
        if method_id.startswith("late_morph_cell_nucleus_alpha_"):
            score_by_id[method_id] = fixed_alpha_late_fusion(
                deep_cal,
                morph_cal,
                float(method["alpha"]),
            )
        if method_id.startswith("late_feature_map_alpha_"):
            score_by_id[method_id] = fixed_alpha_late_fusion(
                deep_cal,
                fmap_cal,
                float(method["alpha"]),
            )
        if method_id == "late_morph_cell_nucleus_max":
            score_by_id[method_id] = max_late_fusion(deep_cal, morph_cal)
        if method_id == "late_feature_map_max":
            score_by_id[method_id] = max_late_fusion(deep_cal, fmap_cal)

    output_dir.mkdir(parents=True, exist_ok=True)
    method_rows = []
    metrics_by_id: dict[str, dict[str, float]] = {}
    raw_predictions = _predictions_for_method(raw_tda_predictions)
    raw_output = raw_predictions.copy()
    raw_output["method"] = "raw_tda_centroid_historical"
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    raw_output.to_csv(
        output_dir / "predictions" / "raw_tda_centroid_historical_predictions.csv",
        index=False,
    )
    raw_metrics_path = raw_tda_predictions.parent / "metrics.json"
    for method in matrix["methods"]:
        method_id = str(method["id"])
        if method_id == "raw_tda_centroid_historical":
            metrics = _metrics_from_predictions(raw_predictions)
            if raw_metrics_path.exists():
                raw_json = json.loads(raw_metrics_path.read_text(encoding="utf-8"))
                metrics.update(
                    {
                        "auroc_known_unknown": float(raw_json["open_set"]["auroc_known_unknown"]),
                        "aupr_known_unknown": float(raw_json["open_set"]["aupr_known_unknown"]),
                        "fpr_at_95_tpr": float(raw_json["open_set"]["fpr_at_95_tpr"]),
                        "oscr": float(raw_json["open_set"].get("oscr", np.nan)),
                    }
                )
        else:
            metrics = _score_metrics(
                embeddings,
                score_by_id[method_id],
                method_id=method_id,
                output_dir=output_dir / "predictions",
            )
        metrics_by_id[method_id] = metrics
        method_rows.append(
            {
                "representation": method.get("representation", ""),
                "source": method.get("source", ""),
                "layer": method.get("layer", ""),
                "tda_view": method.get("tda_view", ""),
                "osr_score": method.get("osr_score", ""),
                "fusion_rule": method.get("fusion_rule", ""),
                "alpha": method.get("alpha", ""),
                "AUROC": metrics.get("auroc_known_unknown", np.nan),
                "AUPR_unknown": metrics.get("aupr_known_unknown", np.nan),
                "AUPR_known": metrics.get("aupr_known_positive", np.nan),
                "FPR95": metrics.get("fpr_at_95_tpr", np.nan),
                "OSCR": metrics.get("oscr", np.nan),
                "seed": seed,
                "split": "V1",
                "config_hash": config_hash(method),
                "checkpoint_hash": sha256_file(checkpoint_path),
                "method_id": method_id,
                "primary": bool(method.get("primary", False)),
            }
        )
    comparison = pd.DataFrame(method_rows)
    comparison.to_csv(output_dir / "method_comparison.csv", index=False)
    per_class = _per_unknown_class_table(embeddings, score_by_id, raw_predictions, comparison)
    per_class.to_csv(output_dir / "per_unknown_class.csv", index=False)
    subgroup = _subgroup_table(per_class)
    subgroup.to_csv(output_dir / "subgroup_results.csv", index=False)
    bootstrap = _bootstrap_late_fusion(embeddings, output_dir / "predictions", comparison)
    write_json(output_dir / "paired_bootstrap.json", bootstrap)
    decision = _decision_rule(comparison, subgroup, bootstrap)
    write_json(output_dir / "decision_rule.json", decision)
    _write_delivery3_figures(comparison, per_class, subgroup, output_dir)
    write_json(
        output_dir / "environment.json",
        environment_metadata(str(embeddings["manifest_hash"].tolist())),
    )
    return output_dir / "method_comparison.csv"


def _metrics_from_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    y_unknown = (predictions["known_status"].astype(str) != "known").astype(int).to_numpy()
    scores = predictions["unknown_score"].astype(float).to_numpy()
    true = predictions["true_label"].astype(str).to_numpy().astype(object)
    true[y_unknown == 1] = "UNKNOWN"
    pred = predictions["open_set_prediction"].astype(str).to_numpy()
    metrics = open_set_metrics(y_unknown, scores, true, pred)
    metrics["aupr_known_positive"] = float(average_precision_score(1 - y_unknown, -scores))
    return metrics


def _test_prediction_frame(
    embeddings: dict[str, Any],
    scores: np.ndarray,
    method_id: str,
) -> pd.DataFrame:
    split = embeddings["split"].astype(str)
    test_mask = split == "test"
    return pd.DataFrame(
        {
            "sample_id": embeddings["sample_id"][test_mask].astype(str),
            "true_label": embeddings["true_label"][test_mask].astype(str),
            "known_status": embeddings["known_status"][test_mask].astype(str),
            "unknown_score": scores[test_mask],
            "method": method_id,
        }
    )


def _per_unknown_class_table(
    embeddings: dict[str, Any],
    score_by_id: dict[str, np.ndarray],
    raw_predictions: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    deep = _test_prediction_frame(embeddings, score_by_id["deep_msp"], "deep_msp")
    deep_metrics = _per_unknown_for_predictions(deep)
    deep_lookup = deep_metrics.set_index("unknown_class")
    for method_id in comparison["method_id"].astype(str):
        if method_id == "raw_tda_centroid_historical":
            pred = raw_predictions[
                ["sample_id", "true_label", "known_status", "unknown_score"]
            ].copy()
            pred["method"] = method_id
        else:
            pred = _test_prediction_frame(embeddings, score_by_id[method_id], method_id)
        metrics = _per_unknown_for_predictions(pred)
        metrics["method"] = method_id
        metrics["group"] = metrics["unknown_class"].map(_unknown_group)
        metrics["delta_AUROC_vs_deep"] = metrics.apply(
            lambda row: float(row["AUROC"] - deep_lookup.loc[row["unknown_class"], "AUROC"]),
            axis=1,
        )
        metrics["delta_FPR95_vs_deep"] = metrics.apply(
            lambda row: float(row["FPR95"] - deep_lookup.loc[row["unknown_class"], "FPR95"]),
            axis=1,
        )
        frames.append(metrics)
    return pd.concat(frames, ignore_index=True)


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


def _unknown_group(label: str) -> str:
    for group, labels in UNKNOWN_GROUPS.items():
        if label in labels:
            return group
    return "unassigned"


def _subgroup_table(per_class: pd.DataFrame) -> pd.DataFrame:
    rows = []
    deep = per_class.loc[per_class["method"] == "deep_msp"].set_index("unknown_class")
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
    baseline = np.asarray(baseline_scores, dtype=float)
    candidate = np.asarray(candidate_scores, dtype=float)
    if len(y) != len(baseline) or len(y) != len(candidate):
        msg = "Paired bootstrap inputs must have identical length"
        raise ValueError(msg)
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
    if not auroc_delta or not fpr_delta:
        msg = "Paired bootstrap could not draw samples with both classes"
        raise ValueError(msg)
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


def _bootstrap_late_fusion(
    embeddings: dict[str, Any],
    predictions_dir: Path,
    comparison: pd.DataFrame,
    *,
    n_bootstraps: int = 1000,
    seed: int = 37,
) -> dict[str, Any]:
    baseline = pd.read_csv(predictions_dir / "deep_msp_predictions.csv")
    report: dict[str, Any] = {}
    for row in comparison.loc[comparison["primary"]].itertuples(index=False):
        method_id = str(row.method_id)
        if method_id == "deep_msp":
            continue
        candidate = pd.read_csv(predictions_dir / f"{method_id}_predictions.csv")
        merged = baseline[["sample_id", "known_status", "unknown_score"]].merge(
            candidate[["sample_id", "known_status", "unknown_score"]],
            on="sample_id",
            suffixes=("_deep", "_candidate"),
            validate="one_to_one",
        )
        y = (merged["known_status_deep"].astype(str) != "known").astype(int).to_numpy()
        deep_scores = merged["unknown_score_deep"].astype(float).to_numpy()
        cand_scores = merged["unknown_score_candidate"].astype(float).to_numpy()
        report[method_id] = paired_bootstrap_deltas(
            y,
            deep_scores,
            cand_scores,
            n_bootstraps=n_bootstraps,
            seed=seed,
        )
    _ = embeddings
    return report


def _decision_rule(
    comparison: pd.DataFrame,
    subgroup: pd.DataFrame,
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    deep = comparison.loc[comparison["method_id"] == "deep_msp"].iloc[0]
    late_mask = comparison["method_id"].astype(str).str.startswith("late_")
    candidates = comparison.loc[comparison["primary"] & late_mask]
    criterion_a = False
    criterion_b = False
    for row in candidates.itertuples(index=False):
        delta_auroc = float(row.AUROC - deep.AUROC)
        delta_fpr = float(row.FPR95 - deep.FPR95)
        boot = bootstrap.get(str(row.method_id), {})
        auroc_ci = boot.get("delta_auroc", {})
        fpr_ci = boot.get("delta_fpr95", {})
        criterion_a = criterion_a or (
            delta_auroc >= 0.01 and float(auroc_ci.get("lower", -1.0)) > 0.0
        )
        criterion_b = criterion_b or (
            delta_fpr <= -0.05 and delta_auroc > -0.005 and float(fpr_ci.get("upper", 1.0)) < 0.0
        )
    late_methods = set(candidates["method_id"].astype(str))
    subgroup_late = subgroup.loc[subgroup["method"].isin(late_methods)]
    criterion_c = bool(
        np.any(
            (subgroup_late["delta_AUROC"] >= 0.03) & (subgroup_late["method"].isin(late_methods))
        )
    )
    if criterion_a or criterion_b:
        decision = "Proceed to multiseed confirmation"
    elif criterion_c:
        decision = "Proceed only with subgroup-focused hypothesis"
    else:
        all_unfavorable = bool(
            np.all(candidates["AUROC"].to_numpy() - float(deep.AUROC) <= 0.0)
            and np.all(candidates["FPR95"].to_numpy() - float(deep.FPR95) >= 0.0)
        )
        decision = (
            "Stop TDA method development"
            if all_unfavorable
            else "Perform one narrowly justified refinement"
        )
    return {
        "criterion_a": "PASS" if criterion_a else "FAIL",
        "criterion_b": "PASS" if criterion_b else "FAIL",
        "criterion_c": "PASS" if criterion_c else "FAIL",
        "scientific_decision": decision,
    }


def _write_delivery3_figures(
    comparison: pd.DataFrame,
    per_class: pd.DataFrame,
    subgroup: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    figures = output_dir.parent.parent / "figures" / "delivery3"
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(comparison["method_id"], comparison["FPR95"])
    ax.tick_params(axis="x", rotation=70)
    ax.set_ylabel("FPR@95TPR")
    fig.tight_layout()
    fig.savefig(figures / "fpr95_delivery3.png", dpi=160)
    plt.close(fig)

    predictions_dir = output_dir / "predictions"
    roc_methods = [
        "deep_msp",
        "raw_tda_centroid_historical",
        "morph_cell_nucleus_centroid",
        "feature_map_centroid",
    ]
    best_late = (
        comparison.loc[comparison["method_id"].astype(str).str.startswith("late_")]
        .sort_values("AUROC", ascending=False)
        .iloc[0]["method_id"]
    )
    roc_methods.append(str(best_late))
    fig, ax = plt.subplots(figsize=(6, 5))
    for method_id in dict.fromkeys(roc_methods):
        path = predictions_dir / f"{method_id}_predictions.csv"
        if not path.exists():
            continue
        pred = pd.read_csv(path)
        y = (pred["known_status"].astype(str) != "known").astype(int).to_numpy()
        scores = pred["unknown_score"].astype(float).to_numpy()
        fpr, tpr, _thresholds = roc_curve(y, scores)
        ax.plot(fpr, tpr, label=str(method_id))
    ax.plot([0, 1], [0, 1], color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "open_set_roc_delivery3.png", dpi=160)
    plt.close(fig)

    deep_auroc = float(comparison.loc[comparison["method_id"] == "deep_msp", "AUROC"].iloc[0])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(comparison["method_id"], comparison["AUROC"] - deep_auroc)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.tick_params(axis="x", rotation=70)
    ax.set_ylabel("Delta AUROC vs Deep MSP")
    fig.tight_layout()
    fig.savefig(figures / "overall_delta_auroc_delivery3.png", dpi=160)
    plt.close(fig)

    best_per = per_class.loc[per_class["method"] == best_late].sort_values("delta_AUROC_vs_deep")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(best_per["unknown_class"], best_per["delta_AUROC_vs_deep"])
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Delta AUROC vs Deep MSP")
    fig.tight_layout()
    fig.savefig(figures / "per_unknown_delta_auroc.png", dpi=160)
    plt.close(fig)

    sub = subgroup.loc[subgroup["method"] == best_late]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(sub["group"], sub["delta_AUROC"])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel("Delta AUROC vs Deep MSP")
    fig.tight_layout()
    fig.savefig(figures / "subgroup_delta_auroc.png", dpi=160)
    plt.close(fig)

    methods_for_distributions = [
        "deep_msp",
        "raw_tda_centroid_historical",
        "morph_nucleus_centroid",
        "feature_map_centroid",
        str(best_late),
    ]
    _write_score_distribution_figure(
        predictions_dir,
        figures / "neutrophil_band_score_distributions.png",
        target_labels=("neutrophil_segmented", "neutrophil_band"),
        methods=methods_for_distributions,
    )
    _write_score_distribution_figure(
        predictions_dir,
        figures / "lymphocyte_related_score_distributions.png",
        target_labels=(
            "lymphocyte",
            "lymphocyte_large_granular",
            "lymphocyte_neoplastic",
            "lymphocyte_reactive",
        ),
        methods=methods_for_distributions,
    )


def _write_score_distribution_figure(
    predictions_dir: Path,
    output_path: Path,
    *,
    target_labels: tuple[str, ...],
    methods: list[str],
) -> None:
    import matplotlib.pyplot as plt

    available = [
        method
        for method in dict.fromkeys(methods)
        if (predictions_dir / f"{method}_predictions.csv").exists()
    ]
    if not available:
        return
    fig, axes = plt.subplots(
        len(available),
        1,
        figsize=(8, max(2.2, 1.8 * len(available))),
        sharex=True,
    )
    axes_arr = np.atleast_1d(axes)
    for ax, method in zip(axes_arr, available, strict=True):
        pred = pd.read_csv(predictions_dir / f"{method}_predictions.csv")
        subset = pred.loc[pred["true_label"].astype(str).isin(target_labels)]
        for label in target_labels:
            scores = subset.loc[
                subset["true_label"].astype(str) == label,
                "unknown_score",
            ].astype(float)
            if not scores.empty:
                ax.hist(scores, bins=30, alpha=0.45, density=True, label=label)
        ax.set_title(method, fontsize=9)
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
    axes_arr[-1].set_xlabel("Unknownness score")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
