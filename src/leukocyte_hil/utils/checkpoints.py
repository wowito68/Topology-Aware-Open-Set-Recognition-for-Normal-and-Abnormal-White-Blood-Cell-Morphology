"""Checkpoint checksum verification without loading model weights."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class CheckpointVerification:
    """Result of an exact SHA256 checkpoint check."""

    path: Path
    expected_sha256: str
    observed_sha256: str | None
    exact_match: bool
    exists: bool


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 for a file by streaming bytes."""

    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected_sha256: str) -> CheckpointVerification:
    """Verify a file SHA256 exactly without interpreting file contents."""

    if not path.exists():
        return CheckpointVerification(
            path=path,
            expected_sha256=expected_sha256,
            observed_sha256=None,
            exact_match=False,
            exists=False,
        )
    observed = sha256_file(path)
    return CheckpointVerification(
        path=path,
        expected_sha256=expected_sha256,
        observed_sha256=observed,
        exact_match=observed == expected_sha256,
        exists=True,
    )
