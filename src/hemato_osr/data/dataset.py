"""Torch dataset backed by explicit manifests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from hemato_osr.data.taxonomy import KNOWN
from hemato_osr.data.transforms import ImageTransform


@dataclass(frozen=True)
class Sample:
    """Single manifest sample returned by ``ManifestImageDataset``."""

    image: torch.Tensor
    label: int
    sample_id: str
    canonical_label: str
    known_status: str
    path: str


class ManifestImageDataset(Dataset[Sample]):
    """Load images and labels from an explicit manifest dataframe."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        label_to_index: dict[str, int],
        transform: ImageTransform,
        split: str | None = None,
        include_unknown: bool = True,
    ) -> None:
        frame = manifest.copy()
        if split is not None:
            frame = frame.loc[frame["split"] == split].copy()
        if not include_unknown:
            frame = frame.loc[frame["known_status"] == KNOWN].copy()
        self.frame = frame.sort_values("sample_id").reset_index(drop=True)
        self.label_to_index = label_to_index
        self.transform = transform

    def __len__(self) -> int:
        return int(len(self.frame))

    def __getitem__(self, index: int) -> Sample:
        row = self.frame.iloc[index]
        path = Path(str(row["path"]))
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        canonical_label = str(row["canonical_label"])
        label = self.label_to_index.get(canonical_label, -1)
        return Sample(
            image=tensor,
            label=label,
            sample_id=str(row["sample_id"]),
            canonical_label=canonical_label,
            known_status=str(row["known_status"]),
            path=str(row["path"]),
        )


def make_label_mapping(classes: Sequence[str]) -> dict[str, int]:
    """Create a stable label-to-index mapping."""

    return {label: idx for idx, label in enumerate(classes)}


def collate_samples(samples: Sequence[Sample]) -> dict[str, object]:
    """Collate samples into a dict batch while preserving metadata."""

    images = torch.stack([sample.image for sample in samples], dim=0)
    labels = torch.tensor([sample.label for sample in samples], dtype=torch.long)
    return {
        "image": images,
        "label": labels,
        "sample_id": [sample.sample_id for sample in samples],
        "canonical_label": [sample.canonical_label for sample in samples],
        "known_status": [sample.known_status for sample in samples],
        "path": [sample.path for sample in samples],
    }
