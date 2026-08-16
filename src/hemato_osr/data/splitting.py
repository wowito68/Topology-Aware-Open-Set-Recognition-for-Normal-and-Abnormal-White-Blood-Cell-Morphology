"""Deterministic train/validation/test splitting protocols."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from hemato_osr.data.taxonomy import KNOWN


@dataclass(frozen=True)
class SplitConfig:
    """Parameters for reproducible splitting."""

    method: str = "stratified"
    train_size: float = 0.7
    validation_size: float = 0.15
    test_size: float = 0.15
    seed: int = 13

    def validate(self) -> None:
        total = self.train_size + self.validation_size + self.test_size
        if abs(total - 1.0) > 1e-6:
            msg = f"Split ratios must sum to 1.0, got {total}"
            raise ValueError(msg)
        if self.method not in {"stratified", "group-stratified", "predefined"}:
            msg = f"Unsupported split method: {self.method}"
            raise ValueError(msg)


def _assign(frame: pd.DataFrame, indices: list[int], split: str) -> None:
    frame.loc[indices, "split"] = split


def _split_stratified(frame: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    labels = frame["canonical_label"].astype(str)
    indices = list(frame.index)
    try:
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=config.train_size,
            random_state=config.seed,
            stratify=labels,
        )
        temp_labels = frame.loc[temp_idx, "canonical_label"].astype(str)
        relative_val = config.validation_size / (config.validation_size + config.test_size)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=relative_val,
            random_state=config.seed,
            stratify=temp_labels,
        )
    except ValueError:
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=config.train_size,
            random_state=config.seed,
            shuffle=True,
        )
        relative_val = config.validation_size / (config.validation_size + config.test_size)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=relative_val,
            random_state=config.seed,
            shuffle=True,
        )
    result = frame.copy()
    _assign(result, list(train_idx), "train")
    _assign(result, list(val_idx), "validation")
    _assign(result, list(test_idx), "test")
    return result


def _split_group_stratified(frame: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    groups = frame["group_id"].fillna("").astype(str)
    if (groups == "").any():
        msg = "group-stratified splitting requires non-empty group_id for every row"
        raise ValueError(msg)

    indices = list(frame.index)
    splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=config.train_size,
        random_state=config.seed,
    )
    train_pos, temp_pos = next(splitter.split(frame, groups=groups))
    train_idx = [indices[pos] for pos in train_pos]
    temp_idx = [indices[pos] for pos in temp_pos]
    temp_groups = frame.loc[temp_idx, "group_id"].astype(str)
    relative_val = config.validation_size / (config.validation_size + config.test_size)
    second = GroupShuffleSplit(n_splits=1, train_size=relative_val, random_state=config.seed)
    val_pos, test_pos = next(second.split(frame.loc[temp_idx], groups=temp_groups))
    val_idx = [temp_idx[pos] for pos in val_pos]
    test_idx = [temp_idx[pos] for pos in test_pos]

    result = frame.copy()
    _assign(result, train_idx, "train")
    _assign(result, val_idx, "validation")
    _assign(result, test_idx, "test")
    return result


def split_manifest(frame: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    """Assign deterministic splits, keeping unknown examples out of training."""

    config.validate()
    if config.method == "predefined":
        missing = set(frame["split"].fillna("").astype(str)) - {"train", "validation", "test", ""}
        if missing:
            msg = f"Unknown predefined split labels: {sorted(missing)}"
            raise ValueError(msg)
        return frame.copy()

    result = frame.copy()
    result["split"] = ""
    known = result.loc[result["known_status"] == KNOWN].copy()
    unknown = result.loc[result["known_status"] != KNOWN].copy()

    if not known.empty:
        if config.method == "group-stratified":
            known_split = _split_group_stratified(known, config)
        else:
            known_split = _split_stratified(known, config)
        result.loc[known_split.index, "split"] = known_split["split"]

    if not unknown.empty:
        # Unknown morphologies are never placed in training by default.
        result.loc[unknown.index, "split"] = "test"

    return result.sort_values("sample_id").reset_index(drop=True)


def assert_no_split_overlap(frame: pd.DataFrame) -> None:
    """Raise when sample, checksum, path, or group overlap across splits."""

    for column in ("sample_id", "checksum", "path"):
        for _, group in frame.groupby(column):
            splits = set(group["split"].fillna("").astype(str)) - {""}
            if len(splits) > 1:
                msg = f"{column} overlaps across splits: {sorted(splits)}"
                raise ValueError(msg)
    if "group_id" in frame.columns:
        groups = frame.loc[frame["group_id"].fillna("").astype(str) != ""]
        for group_id, group in groups.groupby("group_id"):
            splits = set(group["split"].fillna("").astype(str)) - {""}
            if len(splits) > 1:
                msg = f"group_id={group_id} overlaps across splits: {sorted(splits)}"
                raise ValueError(msg)
