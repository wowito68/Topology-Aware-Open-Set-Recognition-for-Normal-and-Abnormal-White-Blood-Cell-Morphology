"""Leakage and data-quality audit for manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from hemato_osr.data.taxonomy import Taxonomy


class LeakageError(RuntimeError):
    """Raised when unequivocal leakage is found."""


@dataclass
class AuditReport:
    """Structured audit result."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.errors


def average_hash(path: Path, hash_size: int = 8) -> str:
    """Compute a compact perceptual average hash for near-duplicate screening."""

    with Image.open(path) as image:
        pixels = image.convert("L").resize((hash_size, hash_size))
        arr = np.asarray(pixels, dtype=np.float32)
    bits = arr >= float(arr.mean())
    return "".join("1" if bit else "0" for bit in bits.flatten())


def hamming_distance(left: str, right: str) -> int:
    """Return Hamming distance between equal-length bit strings."""

    if len(left) != len(right):
        msg = "Cannot compute Hamming distance for different length strings"
        raise ValueError(msg)
    return sum(a != b for a, b in zip(left, right, strict=True))


def _cross_split_duplicates(frame: pd.DataFrame, column: str) -> list[str]:
    issues: list[str] = []
    valid = frame.loc[frame[column].fillna("").astype(str) != ""]
    for value, group in valid.groupby(column):
        splits = sorted(set(group["split"].fillna("").astype(str)) - {""})
        if len(splits) > 1:
            issues.append(f"{column}={value} appears in multiple splits: {splits}")
    return issues


def audit_manifest(
    frame: pd.DataFrame,
    taxonomy: Taxonomy,
    *,
    near_duplicate_hamming: int = 4,
    abort_on_leakage: bool = True,
    max_hamming_pairs: int = 2_000_000,
) -> AuditReport:
    """Audit a manifest before training or threshold calibration."""

    report = AuditReport()
    report.details["rows"] = int(len(frame))
    if frame.empty:
        report.errors.append("Manifest is empty")

    duplicated_paths = int(frame["path"].duplicated().sum())
    duplicated_checksums = int(frame["checksum"].duplicated().sum())
    report.details["duplicated_paths"] = duplicated_paths
    report.details["duplicated_checksums"] = duplicated_checksums
    if duplicated_paths:
        report.errors.append(f"Found {duplicated_paths} duplicate paths")
    if duplicated_checksums:
        report.warnings.append(f"Found {duplicated_checksums} duplicate checksums")

    for issue in _cross_split_duplicates(frame, "sample_id"):
        report.errors.append(f"Sample leakage: {issue}")
    for issue in _cross_split_duplicates(frame, "checksum"):
        report.errors.append(f"Identical-image leakage: {issue}")

    if "group_id" in frame.columns:
        nonempty_groups = frame.loc[frame["group_id"].fillna("").astype(str) != ""]
        for group_id, group in nonempty_groups.groupby("group_id"):
            splits = sorted(set(group["split"].fillna("").astype(str)) - {""})
            if len(splits) > 1:
                report.errors.append(f"group_id={group_id} appears in multiple splits: {splits}")
        if nonempty_groups.empty:
            report.warnings.append(
                "No reliable group_id/patient identifier is present; "
                "patient-level leakage cannot be excluded."
            )

    known_universe = set(taxonomy.canonical_classes)
    unknown_labels = sorted(set(frame["canonical_label"]) - known_universe)
    if unknown_labels:
        report.warnings.append(
            "Labels outside the configured taxonomy will be treated as unknown: "
            + ", ".join(unknown_labels)
        )

    unreadable = 0
    hashes: list[tuple[str, str, str]] = []
    for row in frame.itertuples(index=False):
        path = Path(str(row.path))
        try:
            with Image.open(path) as image:
                image.verify()
            hashes.append((str(row.sample_id), str(row.split), average_hash(path)))
        except (OSError, UnidentifiedImageError, FileNotFoundError):
            unreadable += 1
    report.details["unreadable_images"] = unreadable
    if unreadable:
        report.errors.append(f"Found {unreadable} missing/corrupt/unreadable images")

    near_duplicate_pairs = 0
    hash_groups: dict[str, list[tuple[str, str]]] = {}
    for sample_id, split, hash_value in hashes:
        hash_groups.setdefault(hash_value, []).append((sample_id, split))

    for hash_value, group in hash_groups.items():
        if len(group) < 2:
            continue
        pairs = len(group) * (len(group) - 1) // 2
        near_duplicate_pairs += pairs
        splits = sorted({split for _, split in group if split})
        if len(splits) > 1:
            report.warnings.append(
                f"Potential cross-split perceptual-hash duplicate hash={hash_value}: "
                f"{len(group)} samples across {splits}"
            )

    total_pairs = len(hashes) * (len(hashes) - 1) // 2
    if near_duplicate_hamming > 0 and total_pairs <= max_hamming_pairs:
        for idx, (left_id, left_split, left_hash) in enumerate(hashes):
            for right_id, right_split, right_hash in hashes[idx + 1 :]:
                if left_hash == right_hash:
                    continue
                if left_split and right_split and left_split == right_split:
                    continue
                if hamming_distance(left_hash, right_hash) <= near_duplicate_hamming:
                    near_duplicate_pairs += 1
                    if left_split != right_split:
                        report.warnings.append(
                            "Potential cross-split near-duplicate: "
                            f"{left_id} ({left_split}) vs {right_id} ({right_split})"
                        )
    elif near_duplicate_hamming > 0:
        report.warnings.append(
            "Pairwise near-duplicate Hamming audit skipped because "
            f"{total_pairs} pairs exceeds max_hamming_pairs={max_hamming_pairs}; "
            "exact perceptual-hash buckets were still checked."
        )
    report.details["potential_near_duplicate_pairs"] = near_duplicate_pairs

    if report.errors and abort_on_leakage:
        raise LeakageError("; ".join(report.errors))
    return report
