from __future__ import annotations

from hemato_osr.data.manifest import (
    MANIFEST_COLUMNS,
    ManifestOptions,
    build_manifest,
    manifest_hash,
)
from hemato_osr.data.taxonomy import Taxonomy


def test_manifest_generation_is_deterministic(synthetic_root) -> None:  # type: ignore[no-untyped-def]
    taxonomy = Taxonomy()
    first = build_manifest(ManifestOptions(root=synthetic_root, dataset="synthetic"), taxonomy)
    second = build_manifest(ManifestOptions(root=synthetic_root, dataset="synthetic"), taxonomy)

    assert list(first.columns) == list(MANIFEST_COLUMNS)
    assert len(first) == len(second)
    assert manifest_hash(first) == manifest_hash(second)
    assert {"known", "unknown"} <= set(first["known_status"])
