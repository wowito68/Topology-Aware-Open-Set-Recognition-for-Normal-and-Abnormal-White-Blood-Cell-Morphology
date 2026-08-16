"""Near-duplicate candidate generation and conservative component grouping."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from hemato_osr.data.audit import average_hash, hamming_distance


@dataclass(frozen=True)
class NearDuplicateConfig:
    """Thresholds for candidate generation and evidence labels.

    Perceptual hashes are only a screening signal. Only exact checksum matches are
    considered confirmed duplicates.
    """

    hash_size: int = 8
    candidate_hamming_max: int = 0
    max_pairwise_pairs: int = 2_000_000
    level1_hamming_max: int = 1
    level1_ssim_min: float = 0.985
    level1_correlation_min: float = 0.985
    level2_hamming_max: int = 4
    level2_ssim_min: float = 0.95
    level2_correlation_min: float = 0.95
    group_evidence_levels: tuple[str, ...] = ("confirmed", "probable_level1")
    compare_image_size: int = 128

    def to_dict(self) -> dict[str, object]:
        """Return a serializable config."""

        return asdict(self)


def load_normalized_grayscale(path: Path, image_size: int) -> np.ndarray:
    """Load an image as normalized grayscale with a common shape."""

    with Image.open(path) as image:
        gray = image.convert("L").resize((image_size, image_size))
        return np.asarray(gray, dtype=np.float64) / 255.0


def structural_similarity_index(left: np.ndarray, right: np.ndarray) -> float:
    """Compute a deterministic global SSIM approximation for same-shaped arrays."""

    if left.shape != right.shape:
        msg = f"SSIM requires same shape, got {left.shape} and {right.shape}"
        raise ValueError(msg)
    c1 = 0.01**2
    c2 = 0.03**2
    mean_x = float(left.mean())
    mean_y = float(right.mean())
    var_x = float(left.var())
    var_y = float(right.var())
    cov_xy = float(((left - mean_x) * (right - mean_y)).mean())
    numerator = (2.0 * mean_x * mean_y + c1) * (2.0 * cov_xy + c2)
    denominator = (mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2)
    return float(numerator / denominator) if denominator > 0 else 1.0


def normalized_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Compute Pearson normalized correlation for same-shaped image arrays."""

    if left.shape != right.shape:
        msg = f"Correlation requires same shape, got {left.shape} and {right.shape}"
        raise ValueError(msg)
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denom = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denom <= 1e-12:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.sum(left_centered * right_centered) / denom)


def _evidence_level(
    *,
    checksum_equal: bool,
    phash_hamming_distance: int,
    ssim: float,
    normalized_corr: float,
    config: NearDuplicateConfig,
) -> str:
    if checksum_equal:
        return "confirmed"
    if (
        phash_hamming_distance <= config.level1_hamming_max
        and ssim >= config.level1_ssim_min
        and normalized_corr >= config.level1_correlation_min
    ):
        return "probable_level1"
    if (
        phash_hamming_distance <= config.level2_hamming_max
        and ssim >= config.level2_ssim_min
        and normalized_corr >= config.level2_correlation_min
    ):
        return "probable_level2"
    return "candidate_level3"


def compute_phash_frame(frame: pd.DataFrame, config: NearDuplicateConfig) -> pd.DataFrame:
    """Compute average perceptual hashes for manifest rows."""

    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            {
                "sample_id": str(row.sample_id),
                "phash": average_hash(Path(str(row.path)), hash_size=config.hash_size),
            }
        )
    return pd.DataFrame(rows)


def _candidate_index_pairs(
    phashes: pd.DataFrame,
    config: NearDuplicateConfig,
) -> list[tuple[int, int]]:
    total_pairs = len(phashes) * (len(phashes) - 1) // 2
    pairs: list[tuple[int, int]] = []
    if total_pairs <= config.max_pairwise_pairs:
        hashes = phashes["phash"].astype(str).tolist()
        for left, right in combinations(range(len(hashes)), 2):
            if hamming_distance(hashes[left], hashes[right]) <= config.candidate_hamming_max:
                pairs.append((left, right))
        return pairs

    for _, group in phashes.groupby("phash", sort=True):
        indices = group.index.tolist()
        pairs.extend(combinations(indices, 2))
    return pairs


def generate_candidate_pairs(
    frame: pd.DataFrame,
    config: NearDuplicateConfig | None = None,
) -> pd.DataFrame:
    """Generate pHash-screened candidate pairs with structural similarity signals."""

    cfg = config or NearDuplicateConfig()
    manifest = frame.reset_index(drop=True).copy()
    phashes = compute_phash_frame(manifest, cfg)
    manifest = manifest.merge(phashes, on="sample_id", how="left", validate="one_to_one")
    pair_indices = _candidate_index_pairs(manifest[["sample_id", "phash"]], cfg)
    rows: list[dict[str, object]] = []
    image_cache: dict[str, np.ndarray] = {}

    def image_for(path_str: str) -> np.ndarray:
        if path_str not in image_cache:
            image_cache[path_str] = load_normalized_grayscale(
                Path(path_str),
                cfg.compare_image_size,
            )
        return image_cache[path_str]

    for left_idx, right_idx in pair_indices:
        left = manifest.iloc[left_idx]
        right = manifest.iloc[right_idx]
        phash_distance = hamming_distance(str(left["phash"]), str(right["phash"]))
        if phash_distance > cfg.candidate_hamming_max:
            continue
        left_image = image_for(str(left["path"]))
        right_image = image_for(str(right["path"]))
        ssim = structural_similarity_index(left_image, right_image)
        corr = normalized_correlation(left_image, right_image)
        checksum_equal = str(left["checksum"]) == str(right["checksum"])
        evidence = _evidence_level(
            checksum_equal=checksum_equal,
            phash_hamming_distance=phash_distance,
            ssim=ssim,
            normalized_corr=corr,
            config=cfg,
        )
        rows.append(
            {
                "sample_id_a": str(left["sample_id"]),
                "sample_id_b": str(right["sample_id"]),
                "path_a": str(left["path"]),
                "path_b": str(right["path"]),
                "label_a": str(left["canonical_label"]),
                "label_b": str(right["canonical_label"]),
                "split_a": str(left["split"]),
                "split_b": str(right["split"]),
                "phash_a": str(left["phash"]),
                "phash_b": str(right["phash"]),
                "phash_hamming_distance": int(phash_distance),
                "checksum_equal": bool(checksum_equal),
                "ssim": float(ssim),
                "normalized_correlation": float(corr),
                "evidence_level": evidence,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["phash_hamming_distance", "evidence_level", "sample_id_a", "sample_id_b"]
    )


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def build_candidate_components(
    frame: pd.DataFrame,
    candidates: pd.DataFrame,
    config: NearDuplicateConfig | None = None,
) -> pd.DataFrame:
    """Build connected components from conservative high-confidence candidate edges."""

    cfg = config or NearDuplicateConfig()
    uf = _UnionFind()
    for sample_id in frame["sample_id"].astype(str):
        uf.find(sample_id)

    usable = candidates.loc[candidates["evidence_level"].isin(cfg.group_evidence_levels)]
    evidence_rank = {"confirmed": 3, "probable_level1": 2, "probable_level2": 1}
    edge_strength: dict[str, int] = {}
    for row in usable.itertuples(index=False):
        left = str(row.sample_id_a)
        right = str(row.sample_id_b)
        uf.union(left, right)
        strength = evidence_rank.get(str(row.evidence_level), 0)
        edge_strength[left] = max(edge_strength.get(left, 0), strength)
        edge_strength[right] = max(edge_strength.get(right, 0), strength)

    groups: dict[str, list[str]] = {}
    for sample_id in frame["sample_id"].astype(str):
        groups.setdefault(uf.find(sample_id), []).append(sample_id)

    component_lookup: dict[str, int] = {}
    component_rows: list[dict[str, object]] = []
    for component_idx, (_, members) in enumerate(
        sorted(groups.items(), key=lambda item: (-len(item[1]), sorted(item[1])[0]))
    ):
        component_id = f"ndc_{component_idx:06d}"
        for sample_id in members:
            component_lookup[sample_id] = component_idx
            component_rows.append(
                {
                    "component_id": component_id,
                    "sample_id": sample_id,
                    "component_size": len(members),
                    "evidence_strength": int(edge_strength.get(sample_id, 0)),
                }
            )
    return pd.DataFrame(component_rows).sort_values(["component_id", "sample_id"])


def count_cross_split_candidates(candidates: pd.DataFrame, split_lookup: dict[str, str]) -> int:
    """Count candidate pairs whose samples currently sit in different splits."""

    if candidates.empty:
        return 0
    total = 0
    for row in candidates.itertuples(index=False):
        if split_lookup[str(row.sample_id_a)] != split_lookup[str(row.sample_id_b)]:
            total += 1
    return total


def _sort_for_suspicion(candidates: pd.DataFrame) -> pd.DataFrame:
    rank = {"confirmed": 0, "probable_level1": 1, "probable_level2": 2, "candidate_level3": 3}
    result = candidates.copy()
    result["_rank"] = result["evidence_level"].map(rank).fillna(9)
    return result.sort_values(
        [
            "_rank",
            "phash_hamming_distance",
            "checksum_equal",
            "ssim",
            "normalized_correlation",
        ],
        ascending=[True, True, False, False, False],
    ).drop(columns=["_rank"])


def write_contact_sheets(
    candidates: pd.DataFrame,
    output_dir: Path,
    *,
    top_n: int = 100,
    pairs_per_sheet: int = 20,
    image_size: int = 160,
) -> list[Path]:
    """Create contact sheets for human inspection without covering cell imagery."""

    output_dir.mkdir(parents=True, exist_ok=True)
    cross_split = candidates.loc[
        candidates["split_a"].astype(str) != candidates["split_b"].astype(str)
    ]
    selected = _sort_for_suspicion(cross_split).head(top_n)
    if selected.empty:
        return []

    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    sheet_paths: list[Path] = []
    text_height = 58
    row_height = image_size + text_height
    width = image_size * 2 + 24
    for sheet_idx, start in enumerate(range(0, len(selected), pairs_per_sheet)):
        chunk = selected.iloc[start : start + pairs_per_sheet]
        sheet = Image.new("RGB", (width, row_height * len(chunk)), "white")
        draw = ImageDraw.Draw(sheet)
        for row_idx, row in enumerate(chunk.itertuples(index=False)):
            y = row_idx * row_height
            with Image.open(str(row.path_a)) as img_a:
                left_img = img_a.convert("RGB").resize((image_size, image_size))
            with Image.open(str(row.path_b)) as img_b:
                right_img = img_b.convert("RGB").resize((image_size, image_size))
            sheet.paste(left_img, (0, y))
            sheet.paste(right_img, (image_size + 24, y))
            text = (
                f"{row.sample_id_a[:10]} {row.label_a}/{row.split_a} vs "
                f"{row.sample_id_b[:10]} {row.label_b}/{row.split_b}\n"
                f"pHash d={row.phash_hamming_distance} "
                f"SSIM={float(row.ssim):.4f} NCC={float(row.normalized_correlation):.4f} "
                f"{row.evidence_level}"
            )
            draw.text((0, y + image_size + 3), text, fill="black", font=font)
        path = output_dir / f"near_duplicate_contact_sheet_{sheet_idx:03d}.png"
        sheet.save(path)
        sheet_paths.append(path)
    return sheet_paths
