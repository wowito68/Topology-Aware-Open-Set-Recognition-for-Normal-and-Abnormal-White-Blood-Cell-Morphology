"""Class-aware batch samplers for supervised contrastive learning."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from torch.utils.data import Sampler


class PKBatchSampler(Sampler[list[int]]):
    """Yield batches with P classes and K examples per represented class."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        label_column: str = "canonical_label",
        p_classes: int = 5,
        k_per_class: int = 12,
        seed: int = 37,
        drop_last: bool = False,
    ) -> None:
        if p_classes <= 0 or k_per_class <= 0:
            msg = "p_classes and k_per_class must be positive"
            raise ValueError(msg)
        if "known_status" in frame and np.any(frame["known_status"].astype(str) != "known"):
            msg = "PKBatchSampler received unknown rows"
            raise ValueError(msg)
        if "split" in frame and np.any(frame["split"].astype(str) != "train"):
            msg = "PKBatchSampler received non-train rows"
            raise ValueError(msg)
        self.p_classes = int(p_classes)
        self.k_per_class = int(k_per_class)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.class_to_indices = {
            str(label): group.index.to_numpy(dtype=int)
            for label, group in frame.reset_index(drop=True).groupby(label_column, sort=True)
        }
        if not self.class_to_indices:
            msg = "PKBatchSampler requires non-empty data"
            raise ValueError(msg)
        self.labels = sorted(self.class_to_indices)
        self.batch_size = min(self.p_classes, len(self.labels)) * self.k_per_class
        self._length = len(frame) // self.batch_size
        if not self.drop_last and len(frame) % self.batch_size:
            self._length += 1
        self._length = max(1, self._length)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed)
        for _ in range(len(self)):
            selected = rng.choice(
                self.labels,
                size=min(self.p_classes, len(self.labels)),
                replace=False,
            )
            batch: list[int] = []
            for label in selected:
                indices = self.class_to_indices[str(label)]
                replace = len(indices) < self.k_per_class
                chosen = rng.choice(indices, size=self.k_per_class, replace=replace)
                batch.extend(int(idx) for idx in chosen.tolist())
            rng.shuffle(batch)
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        return self._length
