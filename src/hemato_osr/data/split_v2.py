"""Conservative near-duplicate-aware split construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hemato_osr.data.manifest import MANIFEST_COLUMNS, manifest_hash, save_manifest
from hemato_osr.data.near_duplicates import count_cross_split_candidates
from hemato_osr.data.taxonomy import KNOWN


@dataclass(frozen=True)
class ConservativeSplitConfig:
    """Configuration for Split V2."""

    seed: int = 37
    train_size: float = 0.70
    validation_size: float = 0.15
    test_size: float = 0.15


def _target_counts(known: pd.DataFrame, config: ConservativeSplitConfig) -> dict[str, pd.Series]:
    counts = known["canonical_label"].value_counts().sort_index()
    return {
        "train": counts * config.train_size,
        "validation": counts * config.validation_size,
        "test": counts * config.test_size,
    }


def _group_table(known: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    component_lookup = dict(
        zip(
            components["sample_id"].astype(str),
            components["component_id"].astype(str),
            strict=False,
        )
    )
    grouped = known.copy()
    grouped["_split_group"] = [
        component_lookup.get(str(row.sample_id), str(row.sample_id))
        for row in grouped.itertuples(index=False)
    ]
    rows = []
    for group_id, group in grouped.groupby("_split_group", sort=True):
        label_counts = group["canonical_label"].value_counts().to_dict()
        rows.append(
            {
                "group_id": str(group_id),
                "sample_ids": tuple(group["sample_id"].astype(str).tolist()),
                "size": int(len(group)),
                "label_counts": {str(k): int(v) for k, v in label_counts.items()},
                "largest_label_count": int(max(label_counts.values())),
            }
        )
    return pd.DataFrame(rows)


def _score_assignment(
    current: dict[str, pd.Series],
    targets: dict[str, pd.Series],
    label_counts: dict[str, int],
    split: str,
) -> float:
    score = 0.0
    for candidate_split in ("train", "validation", "test"):
        projected = current[candidate_split].copy()
        if candidate_split == split:
            for label, count in label_counts.items():
                projected.loc[label] = projected.get(label, 0.0) + float(count)
        target = targets[candidate_split]
        all_labels = sorted(set(projected.index) | set(target.index))
        diff = projected.reindex(all_labels, fill_value=0.0) - target.reindex(
            all_labels,
            fill_value=0.0,
        )
        score += float(np.sum(np.square(diff.to_numpy(dtype=float))))
    return score


def build_conservative_split_v2(
    manifest: pd.DataFrame,
    components: pd.DataFrame,
    config: ConservativeSplitConfig | None = None,
) -> pd.DataFrame:
    """Assign known rows to group-aware approximate stratified splits.

    Unknown classes remain test-only. Only components supplied by the caller are
    kept intact; callers should therefore pass high-confidence components only.
    """

    cfg = config or ConservativeSplitConfig()
    result = manifest.copy()
    known = result.loc[result["known_status"] == KNOWN].copy()
    unknown = result.loc[result["known_status"] != KNOWN].copy()
    if known.empty:
        msg = "Split V2 requires at least one known sample"
        raise ValueError(msg)

    labels = sorted(known["canonical_label"].astype(str).unique())
    targets = _target_counts(known, cfg)
    current = {}
    for split in ("train", "validation", "test"):
        split_counts = (
            result.loc[(result["known_status"] == KNOWN) & (result["split"] == split)]
            .groupby("canonical_label")
            .size()
        )
        current[split] = split_counts.reindex(labels, fill_value=0.0).astype(float)

    singleton_free = components.loc[components["component_size"].astype(int) > 1].copy()
    component_lookup = dict(
        zip(
            singleton_free["sample_id"].astype(str),
            singleton_free["component_id"].astype(str),
            strict=False,
        )
    )
    grouped = known.loc[known["sample_id"].astype(str).isin(component_lookup)].copy()
    grouped["_component_id"] = grouped["sample_id"].astype(str).map(component_lookup)
    for _, component in grouped.groupby("_component_id", sort=True):
        existing_splits = component["split"].astype(str).tolist()
        if len(set(existing_splits)) <= 1:
            continue
        label_counts = component["canonical_label"].value_counts().astype(int).to_dict()
        for row in component.itertuples():
            current[str(row.split)].loc[str(row.canonical_label)] -= 1.0
        majority_order = (
            component["split"].value_counts().sort_values(ascending=False).index.tolist()
        )
        best_split = min(
            ("train", "validation", "test"),
            key=lambda split: (
                _score_assignment(current, targets, label_counts, split),
                0 if split in majority_order[:1] else 1,
                split,
            ),
        )
        for row in component.itertuples():
            result.at[row.Index, "split"] = best_split
        for label, count in label_counts.items():
            current[best_split].loc[str(label)] += float(count)

    result.loc[unknown.index, "split"] = "test"
    return result.loc[:, MANIFEST_COLUMNS].sort_values("sample_id").reset_index(drop=True)


def compare_splits(
    v1: pd.DataFrame,
    v2: pd.DataFrame,
    candidates: pd.DataFrame,
    components: pd.DataFrame,
) -> dict[str, object]:
    """Create a structured V1 vs V2 split comparison."""

    v1_lookup = dict(zip(v1["sample_id"].astype(str), v1["split"].astype(str), strict=True))
    v2_lookup = dict(zip(v2["sample_id"].astype(str), v2["split"].astype(str), strict=True))
    grouped_candidates = candidates.loc[
        candidates["evidence_level"].isin(["confirmed", "probable_level1"])
    ]
    aligned = v1[["sample_id", "split", "canonical_label", "known_status"]].merge(
        v2[["sample_id", "split"]],
        on="sample_id",
        suffixes=("_v1", "_v2"),
        validate="one_to_one",
    )
    moved = aligned.loc[aligned["split_v1"].astype(str) != aligned["split_v2"].astype(str)]
    component_sizes = (
        components["component_size"].astype(int) if not components.empty else pd.Series(dtype=int)
    )
    return {
        "moved_samples": int(len(moved)),
        "components": int(components["component_id"].nunique()) if not components.empty else 0,
        "non_singleton_components": int(
            components.loc[components["component_size"].astype(int) > 1, "component_id"].nunique()
        )
        if not components.empty
        else 0,
        "largest_component": int(component_sizes.max()) if not component_sizes.empty else 0,
        "class_distribution_v1": _nested_counts(v1, ["canonical_label", "split"]),
        "class_distribution_v2": _nested_counts(v2, ["canonical_label", "split"]),
        "split_distribution_v1": v1["split"].value_counts().sort_index().astype(int).to_dict(),
        "split_distribution_v2": v2["split"].value_counts().sort_index().astype(int).to_dict(),
        "cross_split_candidate_count_before": count_cross_split_candidates(candidates, v1_lookup),
        "cross_split_candidate_count_after": count_cross_split_candidates(candidates, v2_lookup),
        "cross_split_grouped_candidate_count_before": count_cross_split_candidates(
            grouped_candidates,
            v1_lookup,
        ),
        "cross_split_grouped_candidate_count_after": count_cross_split_candidates(
            grouped_candidates,
            v2_lookup,
        ),
        "manifest_hash_v2": manifest_hash(v2),
    }


def _nested_counts(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, int]]:
    counts = frame.groupby(columns).size()
    nested: dict[str, dict[str, int]] = {}
    for keys, value in counts.items():
        outer, inner = keys
        nested.setdefault(str(outer), {})[str(inner)] = int(value)
    return nested


def write_v2_manifest(frame: pd.DataFrame, output_path: Path) -> str:
    """Write Split V2 and return its manifest hash."""

    save_manifest(frame, output_path)
    return manifest_hash(frame)
