from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hemato_osr.data.manifest import ManifestOptions, build_manifest
from hemato_osr.data.splitting import SplitConfig, split_manifest
from hemato_osr.data.synthetic import SyntheticConfig, generate_synthetic_dataset
from hemato_osr.data.taxonomy import Taxonomy


@pytest.fixture()
def synthetic_root(tmp_path: Path) -> Path:
    return generate_synthetic_dataset(
        SyntheticConfig(
            output_dir=tmp_path / "raw",
            samples_per_known_class=6,
            samples_per_unknown_class=3,
            image_size=32,
            seed=13,
        )
    )


@pytest.fixture()
def synthetic_manifest(synthetic_root: Path) -> pd.DataFrame:
    manifest = build_manifest(ManifestOptions(root=synthetic_root, dataset="synthetic"), Taxonomy())
    return split_manifest(manifest, SplitConfig(seed=13))
