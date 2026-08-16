"""TDA extraction orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hemato_osr.topology.cache import config_hash, save_feature_cache
from hemato_osr.topology.diagrams import TDAConfig, compute_diagram
from hemato_osr.topology.vectorizers import create_vectorizer


@dataclass(frozen=True)
class TDAExtractConfig:
    """Configuration for precomputing topological features."""

    manifest_path: Path
    output_path: Path
    filtration: str = "sublevel"
    image_size: int = 64
    vectorizer: str = "persistence-entropy"


def extract_tda_features(config: TDAExtractConfig) -> Path:
    """Compute diagrams and vectorized features for a manifest."""

    frame = pd.read_csv(config.manifest_path).sort_values("sample_id")
    tda_config = TDAConfig(filtration=config.filtration, image_size=config.image_size)
    diagrams = [
        compute_diagram(Path(str(row.path)), tda_config) for row in frame.itertuples(index=False)
    ]
    vectorizer = create_vectorizer(config.vectorizer)
    train_mask = frame["split"].astype(str) == "train"
    train_diagrams = [
        diagram for diagram, is_train in zip(diagrams, train_mask, strict=True) if is_train
    ]
    vectorizer.fit(train_diagrams)
    features = vectorizer.transform(diagrams)
    metadata = {
        "sample_count": int(len(frame)),
        "dataset": sorted(set(frame["dataset"].astype(str))),
        "preprocessing": {"image_size": config.image_size},
        "tda": tda_config.to_dict(),
        "vectorizer": config.vectorizer,
    }
    metadata["config_hash"] = config_hash(metadata)
    save_feature_cache(
        config.output_path,
        sample_ids=[str(item) for item in frame["sample_id"].tolist()],
        features=features,
        metadata=metadata,
    )
    return config.output_path
