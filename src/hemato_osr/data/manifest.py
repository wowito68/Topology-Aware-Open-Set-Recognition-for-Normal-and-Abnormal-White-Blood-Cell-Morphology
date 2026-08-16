"""Reproducible dataset manifest generation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from hemato_osr.data.taxonomy import Taxonomy

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MANIFEST_COLUMNS = (
    "sample_id",
    "path",
    "dataset",
    "original_label",
    "canonical_label",
    "known_status",
    "split",
    "group_id",
    "checksum",
    "width",
    "height",
)


@dataclass(frozen=True)
class ManifestOptions:
    """Options for indexing a directory tree into a manifest."""

    root: Path
    dataset: str = "mll23"
    split: str = ""
    group_from_parent_depth: int | None = None


def iter_image_paths(root: Path) -> Iterable[Path]:
    """Yield image files under ``root`` in deterministic order."""

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a SHA-256 checksum for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sample_id(dataset: str, rel_path: str, checksum: str) -> str:
    """Create a deterministic sample identifier from stable file metadata."""

    payload = f"{dataset}:{rel_path}:{checksum}".encode()
    return hashlib.sha1(payload).hexdigest()[:20]


def manifest_hash(frame: pd.DataFrame) -> str:
    """Hash manifest content in a stable row/column order."""

    existing_cols = [col for col in MANIFEST_COLUMNS if col in frame.columns]
    normalized = frame.loc[:, existing_cols].sort_values(existing_cols).to_csv(index=False)
    return hashlib.sha256(normalized.encode()).hexdigest()


def _class_from_path(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    if len(rel.parts) < 2:
        msg = f"Image path must be nested under a class directory: {path}"
        raise ValueError(msg)
    return rel.parts[0]


def _group_id(root: Path, path: Path, depth: int | None) -> str:
    if depth is None:
        return ""
    rel_parts = path.relative_to(root).parts
    if depth < 0 or depth >= len(rel_parts):
        return ""
    return rel_parts[depth]


def build_manifest(options: ManifestOptions, taxonomy: Taxonomy) -> pd.DataFrame:
    """Index a raw image tree into an explicit manifest."""

    root = options.root.resolve()
    if not root.exists():
        msg = f"Dataset root does not exist: {root}"
        raise FileNotFoundError(msg)

    rows: list[dict[str, object]] = []
    for path in iter_image_paths(root):
        original_label = _class_from_path(root, path)
        canonical = taxonomy.canonicalize(original_label)
        checksum = file_sha256(path)
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
        except (OSError, UnidentifiedImageError) as exc:
            msg = f"Image is not readable during indexing: {path}"
            raise ValueError(msg) from exc

        rel_path = path.relative_to(root).as_posix()
        rows.append(
            {
                "sample_id": stable_sample_id(options.dataset, rel_path, checksum),
                "path": str(path.resolve()),
                "dataset": options.dataset,
                "original_label": original_label,
                "canonical_label": canonical,
                "known_status": taxonomy.known_status(canonical),
                "split": options.split,
                "group_id": _group_id(root, path, options.group_from_parent_depth),
                "checksum": checksum,
                "width": width,
                "height": height,
            }
        )

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def save_manifest(frame: pd.DataFrame, output_path: Path) -> None:
    """Save a manifest as CSV with a stable column order."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    missing = [col for col in MANIFEST_COLUMNS if col not in frame.columns]
    if missing:
        msg = f"Manifest is missing required columns: {missing}"
        raise ValueError(msg)
    frame.loc[:, MANIFEST_COLUMNS].sort_values("sample_id").to_csv(output_path, index=False)
