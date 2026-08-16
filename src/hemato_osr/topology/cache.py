"""Cache helpers for TDA diagrams and features."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def config_hash(config: dict[str, Any]) -> str:
    """Hash a preprocessing/TDA/code config dictionary."""

    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def save_feature_cache(
    path: Path,
    *,
    sample_ids: list[str],
    features: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Save topological features with invalidation metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_ids=np.asarray(sample_ids, dtype=object),
        features=features,
        metadata=json.dumps(metadata, sort_keys=True, default=str),
    )


def load_feature_cache(
    path: Path,
    expected_hash: str,
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    """Load cache and validate its config hash."""

    with np.load(path, allow_pickle=True) as data:
        metadata = json.loads(str(data["metadata"]))
        found_hash = str(metadata.get("config_hash", ""))
        if found_hash != expected_hash:
            msg = f"Cache config hash mismatch: expected {expected_hash}, found {found_hash}"
            raise ValueError(msg)
        sample_ids = [str(item) for item in data["sample_ids"].tolist()]
        features = np.asarray(data["features"], dtype=float)
    return sample_ids, features, metadata
