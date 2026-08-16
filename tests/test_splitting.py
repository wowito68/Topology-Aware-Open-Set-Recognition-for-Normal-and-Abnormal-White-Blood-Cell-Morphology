from __future__ import annotations

from hemato_osr.data.splitting import SplitConfig, assert_no_split_overlap, split_manifest


def test_split_is_deterministic_and_has_no_overlap(synthetic_manifest) -> None:  # type: ignore[no-untyped-def]
    first = split_manifest(synthetic_manifest, SplitConfig(seed=37))
    second = split_manifest(synthetic_manifest, SplitConfig(seed=37))

    assert first["split"].tolist() == second["split"].tolist()
    assert set(first["split"]) == {"train", "validation", "test"}
    assert_no_split_overlap(first)


def test_unknown_samples_are_not_training(synthetic_manifest) -> None:  # type: ignore[no-untyped-def]
    unknown = synthetic_manifest.loc[synthetic_manifest["known_status"] == "unknown"]

    assert not unknown.empty
    assert set(unknown["split"]) == {"test"}
