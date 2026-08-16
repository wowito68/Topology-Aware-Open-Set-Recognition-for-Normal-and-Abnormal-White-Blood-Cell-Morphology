"""End-to-end synthetic smoke test pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hemato_osr.data.audit import audit_manifest
from hemato_osr.data.manifest import ManifestOptions, build_manifest, save_manifest
from hemato_osr.data.splitting import SplitConfig, assert_no_split_overlap, split_manifest
from hemato_osr.data.synthetic import SyntheticConfig, generate_synthetic_dataset
from hemato_osr.data.taxonomy import Taxonomy
from hemato_osr.embeddings.extract import EmbeddingExtractConfig, extract_embeddings
from hemato_osr.evaluation.pipeline import OpenSetEvaluationConfig, evaluate_open_set
from hemato_osr.topology.extract import TDAExtractConfig, extract_tda_features
from hemato_osr.training.fusion_train import FusionTrainConfig, train_fusion
from hemato_osr.training.train import TrainConfig, train_closed_set


@dataclass(frozen=True)
class SmokeConfig:
    """Synthetic smoke-test configuration."""

    output_dir: Path
    seed: int = 13


def run_smoke_test(config: SmokeConfig) -> dict[str, Path]:
    """Run synthetic images through manifest, training, embeddings, TDA, and evaluation."""

    root = config.output_dir
    raw_dir = root / "raw"
    manifest_path = root / "manifest.csv"
    checkpoint_dir = root / "checkpoints"
    embeddings_path = root / "embeddings" / "embeddings.npz"
    tda_path = root / "topology" / "tda_features.npz"
    fusion_path = root / "checkpoints" / "fusion.pt"
    evaluation_dir = root / "evaluation"

    generate_synthetic_dataset(
        SyntheticConfig(
            output_dir=raw_dir,
            samples_per_known_class=6,
            samples_per_unknown_class=4,
            image_size=64,
            seed=config.seed,
            overwrite=True,
        )
    )
    taxonomy = Taxonomy()
    manifest = build_manifest(ManifestOptions(root=raw_dir, dataset="synthetic"), taxonomy)
    split = split_manifest(manifest, SplitConfig(method="stratified", seed=config.seed))
    save_manifest(split, manifest_path)
    loaded = pd.read_csv(manifest_path)
    assert_no_split_overlap(loaded)
    audit_manifest(loaded, taxonomy, abort_on_leakage=True)

    checkpoint_path = train_closed_set(
        TrainConfig(
            manifest_path=manifest_path,
            output_dir=checkpoint_dir,
            backbone="tiny_cnn",
            pretrained=False,
            image_size=64,
            batch_size=8,
            epochs=1,
            early_stopping_patience=2,
            seed=config.seed,
            device="cpu",
        )
    )
    extract_embeddings(
        EmbeddingExtractConfig(
            manifest_path=manifest_path,
            checkpoint_path=checkpoint_path,
            output_path=embeddings_path,
            batch_size=8,
            image_size=64,
            device="cpu",
        )
    )
    extract_tda_features(
        TDAExtractConfig(
            manifest_path=manifest_path,
            output_path=tda_path,
            image_size=32,
            vectorizer="persistence-entropy",
        )
    )
    train_fusion(
        FusionTrainConfig(
            embeddings_path=embeddings_path,
            tda_features_path=tda_path,
            output_path=fusion_path,
            epochs=1,
            seed=config.seed,
            device="cpu",
        )
    )
    metrics_path = evaluate_open_set(
        OpenSetEvaluationConfig(
            embeddings_path=embeddings_path,
            output_dir=evaluation_dir,
            open_set_method="msp",
        )
    )
    return {
        "manifest": manifest_path,
        "checkpoint": checkpoint_path,
        "embeddings": embeddings_path,
        "tda_features": tda_path,
        "fusion_checkpoint": fusion_path,
        "metrics": metrics_path,
    }
