from __future__ import annotations

import numpy as np
import pytest

from hemato_osr.topology.cache import config_hash, load_feature_cache, save_feature_cache
from hemato_osr.topology.diagrams import TDAConfig, compute_diagram
from hemato_osr.topology.vectorizers import BettiCurve, PersistenceEntropy, PersistenceImage


def test_persistence_feature_shapes(synthetic_manifest) -> None:  # type: ignore[no-untyped-def]
    path = synthetic_manifest.iloc[0]["path"]
    diagram = compute_diagram(path, TDAConfig(image_size=16))
    diagrams = [diagram, diagram]

    assert set(diagram) == {0, 1}
    assert PersistenceEntropy().fit_transform(diagrams).shape == (2, 2)
    assert BettiCurve(n_bins=5).fit_transform(diagrams).shape == (2, 10)
    assert PersistenceImage(resolution=4).fit_transform(diagrams).shape == (2, 32)


def test_cache_invalidation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    good_hash = config_hash({"a": 1})
    path = tmp_path / "features.npz"
    save_feature_cache(
        path,
        sample_ids=["s1"],
        features=np.asarray([[1.0, 2.0]]),
        metadata={"config_hash": good_hash},
    )

    _, features, _ = load_feature_cache(path, good_hash)
    assert features.shape == (1, 2)
    with pytest.raises(ValueError, match="mismatch"):
        load_feature_cache(path, config_hash({"a": 2}))
