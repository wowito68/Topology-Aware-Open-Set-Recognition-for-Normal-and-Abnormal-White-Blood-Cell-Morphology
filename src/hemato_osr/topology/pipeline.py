"""Benchmark, extract, and vectorize persistent-homology features."""

from __future__ import annotations

import importlib.metadata
import json
import resource
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hemato_osr.data.manifest import manifest_hash, save_manifest
from hemato_osr.topology.archive import (
    diagrams_for_filtration,
    load_diagram_archive,
    save_diagram_archive,
)
from hemato_osr.topology.cache import config_hash
from hemato_osr.topology.diagrams import TDAConfig, compute_diagram
from hemato_osr.topology.vectorizers import (
    BettiCurve,
    DiagramVectorizer,
    PersistenceEntropy,
    PersistenceImage,
    PersistenceLandscape,
)


@dataclass(frozen=True)
class TDAResolutionBenchmarkConfig:
    """Resolution benchmark configuration."""

    manifest_path: Path
    output_dir: Path
    sample_size: int = 1000
    seed: int = 37
    resolutions: tuple[int, ...] = (64, 96, 128)
    filtrations: tuple[str, ...] = ("sublevel", "superlevel")
    homology_dimensions: tuple[int, ...] = (0, 1)


@dataclass(frozen=True)
class TDADiagramExtractConfig:
    """Full diagram extraction configuration."""

    manifest_path: Path
    output_path: Path
    image_size: int
    filtrations: tuple[str, ...] = ("sublevel", "superlevel")
    homology_dimensions: tuple[int, ...] = (0, 1)
    workers: int = 2
    seed: int = 37


@dataclass(frozen=True)
class TDAVectorizeConfig:
    """Vectorization configuration for a diagram archive."""

    diagram_path: Path
    output_path: Path
    vectorizers: tuple[str, ...] = (
        "persistence-entropy",
        "betti-curve",
        "persistence-landscape",
        "persistence-image",
    )
    filtrations: tuple[str, ...] = ("sublevel", "superlevel")
    dimensions: tuple[int, ...] = (0, 1)
    seed: int = 37


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def sample_stratified_manifest(
    manifest: pd.DataFrame,
    *,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    """Sample approximately ``sample_size`` rows while covering all classes."""

    rng = np.random.default_rng(seed)
    classes = sorted(manifest["canonical_label"].astype(str).unique())
    base = max(1, sample_size // max(1, len(classes)))
    rows = []
    for label in classes:
        group = manifest.loc[manifest["canonical_label"].astype(str) == label]
        take = min(len(group), base)
        rows.append(group.sample(n=take, random_state=int(rng.integers(0, 1_000_000))))
    sampled = pd.concat(rows, ignore_index=True)
    remaining = sample_size - len(sampled)
    if remaining > 0:
        rest = manifest.loc[~manifest["sample_id"].isin(sampled["sample_id"])]
        if not rest.empty:
            sampled = pd.concat(
                [
                    sampled,
                    rest.sample(
                        n=min(remaining, len(rest)),
                        random_state=int(rng.integers(0, 1_000_000)),
                    ),
                ],
                ignore_index=True,
            )
    return sampled.sort_values("sample_id").reset_index(drop=True)


def _diagram_worker(args: tuple[str, str, int, tuple[int, ...]]) -> dict[int, np.ndarray]:
    path, filtration, image_size, dimensions = args
    return compute_diagram(
        Path(path),
        TDAConfig(
            filtration=filtration,
            image_size=image_size,
            homology_dimensions=dimensions,
        ),
    )


def _diagram_size(diagram: dict[int, np.ndarray]) -> int:
    return int(sum(np.asarray(points).reshape(-1, 2).shape[0] for points in diagram.values()))


def run_resolution_benchmark(config: TDAResolutionBenchmarkConfig) -> Path:
    """Run a seeded TDA resolution benchmark and write CSV/JSON/figures."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(config.manifest_path)
    sample = sample_stratified_manifest(
        manifest,
        sample_size=config.sample_size,
        seed=config.seed,
    )
    sample_path = config.output_dir / "tda_benchmark_manifest_seed37.csv"
    save_manifest(sample, sample_path)

    rows: list[dict[str, object]] = []
    entropy_by_key: dict[tuple[int, str], np.ndarray] = {}
    for resolution in config.resolutions:
        for filtration in config.filtrations:
            runtimes = []
            diagram_sizes = []
            diagrams = []
            rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            for row in sample.itertuples(index=False):
                start = time.perf_counter()
                diagram = compute_diagram(
                    Path(str(row.path)),
                    TDAConfig(
                        filtration=filtration,
                        image_size=resolution,
                        homology_dimensions=config.homology_dimensions,
                    ),
                )
                runtimes.append(time.perf_counter() - start)
                diagram_sizes.append(_diagram_size(diagram))
                diagrams.append(diagram)
            rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            entropy = PersistenceEntropy(dimensions=config.homology_dimensions).fit_transform(
                diagrams
            )
            entropy_by_key[(resolution, filtration)] = entropy
            rows.append(
                {
                    "resolution": resolution,
                    "filtration": filtration,
                    "n_images": int(len(sample)),
                    "mean_runtime_seconds": float(np.mean(runtimes)),
                    "median_runtime_seconds": float(np.median(runtimes)),
                    "p95_runtime_seconds": float(np.quantile(runtimes, 0.95)),
                    "rss_delta_kb": int(rss_after - rss_before),
                    "mean_serialized_diagram_points": float(np.mean(diagram_sizes)),
                    "p95_serialized_diagram_points": float(np.quantile(diagram_sizes, 0.95)),
                }
            )

    benchmark = pd.DataFrame(rows)
    max_resolution = max(config.resolutions)
    stability_rows = []
    for resolution in config.resolutions:
        for filtration in config.filtrations:
            current = entropy_by_key[(resolution, filtration)]
            reference = entropy_by_key[(max_resolution, filtration)]
            stability = np.linalg.norm(current - reference, axis=1)
            stability_rows.append(
                {
                    "resolution": resolution,
                    "filtration": filtration,
                    "feature_stability_l2_to_max_resolution": float(np.mean(stability)),
                }
            )
    stability_frame = pd.DataFrame(stability_rows)
    benchmark = benchmark.merge(stability_frame, on=["resolution", "filtration"])
    benchmark["estimated_full_dataset_runtime_seconds"] = benchmark["mean_runtime_seconds"] * len(
        manifest
    )
    csv_path = config.output_dir / "tda_resolution_benchmark.csv"
    benchmark.to_csv(csv_path, index=False)
    selected = select_resolution(benchmark)
    (config.output_dir / "tda_resolution_selection.json").write_text(
        json.dumps(
            {
                "selected_resolution": selected,
                "criterion": "min normalized(mean_runtime + stability_to_max_resolution)",
                "config": asdict(config),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    write_benchmark_figures(benchmark, config.output_dir)
    return csv_path


def select_resolution(benchmark: pd.DataFrame) -> int:
    """Select a resolution from runtime plus stability measurements."""

    grouped = benchmark.groupby("resolution").agg(
        mean_runtime=("mean_runtime_seconds", "mean"),
        stability=("feature_stability_l2_to_max_resolution", "mean"),
    )
    normalized = grouped.copy()
    for column in ("mean_runtime", "stability"):
        values = grouped[column].to_numpy(dtype=float)
        denom = float(values.max() - values.min())
        normalized[column] = 0.0 if denom <= 1e-12 else (values - values.min()) / denom
    normalized["score"] = normalized["mean_runtime"] + normalized["stability"]
    return int(normalized["score"].idxmin())


def write_benchmark_figures(benchmark: pd.DataFrame, output_dir: Path) -> None:
    """Write runtime and stability figures."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for filtration, group in benchmark.groupby("filtration"):
        ax.plot(group["resolution"], group["mean_runtime_seconds"], marker="o", label=filtration)
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Mean runtime / image (s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "tda_runtime_vs_resolution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for filtration, group in benchmark.groupby("filtration"):
        ax.plot(
            group["resolution"],
            group["feature_stability_l2_to_max_resolution"],
            marker="o",
            label=filtration,
        )
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Mean entropy-feature L2 to max resolution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "tda_stability_vs_resolution.png", dpi=160)
    plt.close(fig)


def extract_diagram_archive(config: TDADiagramExtractConfig) -> Path:
    """Extract full persistence diagrams for all manifest rows."""

    frame = pd.read_csv(config.manifest_path).sort_values("sample_id").reset_index(drop=True)
    diagrams_by_key: dict[tuple[str, int], list[np.ndarray]] = {
        (filtration, dim): []
        for filtration in config.filtrations
        for dim in config.homology_dimensions
    }
    failures: list[dict[str, str]] = []
    for filtration in config.filtrations:
        args = [
            (str(row.path), filtration, config.image_size, config.homology_dimensions)
            for row in frame.itertuples(index=False)
        ]
        results: list[dict[int, np.ndarray] | None]
        if config.workers <= 1:
            results = [_diagram_worker(arg) for arg in args]
        else:
            results = [None] * len(args)
            with ProcessPoolExecutor(max_workers=config.workers) as executor:
                future_to_idx = {
                    executor.submit(_diagram_worker, arg): idx for idx, arg in enumerate(args)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as exc:  # pragma: no cover - defensive runtime path
                        failures.append(
                            {
                                "sample_id": str(frame.iloc[idx]["sample_id"]),
                                "error": str(exc),
                            }
                        )
                        results[idx] = {
                            dim: np.empty((0, 2), dtype=float) for dim in config.homology_dimensions
                        }
        for diagram in results:
            assert diagram is not None
            for dim in config.homology_dimensions:
                diagrams_by_key[(filtration, dim)].append(diagram[int(dim)])

    metadata: dict[str, Any] = {
        "sample_count": int(len(frame)),
        "manifest_hash": manifest_hash(frame),
        "source_checksums_sha256": True,
        "preprocessing": {
            "grayscale": True,
            "resize": config.image_size,
            "intensity_normalization": "[0,1]",
        },
        "filtrations": list(config.filtrations),
        "homology_dimensions": list(config.homology_dimensions),
        "infinite_interval_policy": "keep in diagrams; finite vectorizers exclude essential bars",
        "gudhi_version": _package_version("gudhi"),
        "numpy_version": _package_version("numpy"),
        "config": asdict(config),
        "failed": failures,
    }
    metadata["config_hash"] = config_hash(metadata)
    save_diagram_archive(
        config.output_path,
        sample_ids=frame["sample_id"].astype(str).tolist(),
        labels=frame["canonical_label"].astype(str).to_numpy(),
        known_status=frame["known_status"].astype(str).to_numpy(),
        splits=frame["split"].astype(str).to_numpy(),
        diagrams_by_key=diagrams_by_key,
        metadata=metadata,
    )
    return config.output_path


def _make_vectorizer(name: str, dimensions: tuple[int, ...]) -> DiagramVectorizer:
    normalized = name.lower().replace("_", "-")
    if normalized == "persistence-entropy":
        return PersistenceEntropy(dimensions=dimensions)
    if normalized == "betti-curve":
        return BettiCurve(dimensions=dimensions, n_bins=16)
    if normalized == "persistence-landscape":
        return PersistenceLandscape(dimensions=dimensions, n_bins=16, n_layers=3)
    if normalized == "persistence-image":
        return PersistenceImage(dimensions=dimensions, resolution=8, sigma=0.1)
    msg = f"Unsupported TDA vectorizer: {name}"
    raise ValueError(msg)


def vectorize_diagram_archive(config: TDAVectorizeConfig) -> Path:
    """Vectorize a diagram archive with train-only fitting."""

    archive = load_diagram_archive(config.diagram_path)
    train_mask = (archive.splits == "train") & (archive.known_status == "known")
    feature_blocks: list[np.ndarray] = []
    feature_names: list[str] = []
    component_slices: dict[str, tuple[int, int]] = {}
    start = 0
    for filtration in config.filtrations:
        diagrams = diagrams_for_filtration(archive, filtration)
        train_diagrams = [
            diagram for diagram, is_train in zip(diagrams, train_mask, strict=True) if is_train
        ]
        for vectorizer_name in config.vectorizers:
            vectorizer = _make_vectorizer(vectorizer_name, config.dimensions)
            vectorizer.fit(train_diagrams)
            block = vectorizer.transform(diagrams)
            if not np.isfinite(block).all():
                msg = f"Non-finite TDA features from {filtration}/{vectorizer_name}"
                raise ValueError(msg)
            feature_blocks.append(block.astype(np.float32))
            end = start + block.shape[1]
            component = f"{filtration}_{vectorizer_name}"
            component_slices[component] = (start, end)
            feature_names.extend(f"{component}_{idx}" for idx in range(block.shape[1]))
            start = end
    features = np.concatenate(feature_blocks, axis=1) if feature_blocks else np.empty((0, 0))
    metadata: dict[str, Any] = {
        "diagram_path": str(config.diagram_path),
        "diagram_config_hash": archive.metadata.get("config_hash"),
        "manifest_hash": archive.metadata.get("manifest_hash"),
        "fit_split": "known train only",
        "normalization": "none in feature file; models fit scalers on train only",
        "vectorizers": list(config.vectorizers),
        "filtrations": list(config.filtrations),
        "dimensions": list(config.dimensions),
        "component_slices": component_slices,
        "seed": config.seed,
    }
    metadata["config_hash"] = config_hash(metadata)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.output_path,
        sample_id=np.asarray(archive.sample_ids, dtype=object),
        true_label=archive.labels.astype(object),
        known_status=archive.known_status.astype(object),
        split=archive.splits.astype(object),
        features=features,
        feature_names=np.asarray(feature_names, dtype=object),
        metadata=json.dumps(metadata, sort_keys=True, default=str),
    )
    return config.output_path


def load_tda_feature_archive(path: Path) -> dict[str, Any]:
    """Load vectorized TDA features."""

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
