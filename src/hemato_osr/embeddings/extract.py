"""Export logits and embeddings from a trained image classifier."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hemato_osr.data.dataset import ManifestImageDataset, collate_samples
from hemato_osr.data.manifest import manifest_hash
from hemato_osr.data.transforms import TransformConfig, build_transforms
from hemato_osr.models.backbones import ModelConfig, create_classifier
from hemato_osr.training.checkpoint import load_checkpoint


@dataclass(frozen=True)
class EmbeddingExtractConfig:
    """Embedding export configuration."""

    manifest_path: Path
    checkpoint_path: Path
    output_path: Path
    batch_size: int = 32
    num_workers: int = 0
    image_size: int | None = None
    device: str = "auto"


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _checkpoint_model_config(checkpoint: dict[str, Any]) -> ModelConfig:
    cfg = dict(checkpoint.get("config", {}))
    label_to_index = dict(checkpoint["label_to_index"])
    return ModelConfig(
        backbone=str(cfg.get("backbone", "resnet18")),
        num_classes=len(label_to_index),
        pretrained=False,
    )


def extract_embeddings(config: EmbeddingExtractConfig) -> Path:
    """Run inference over a manifest and save efficient embedding artifacts."""

    checkpoint = load_checkpoint(config.checkpoint_path, map_location="cpu")
    label_to_index = {str(k): int(v) for k, v in dict(checkpoint["label_to_index"]).items()}
    train_cfg = dict(checkpoint.get("config", {}))
    image_size = config.image_size or int(train_cfg.get("image_size", 224))
    model = create_classifier(_checkpoint_model_config(checkpoint))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = _device(config.device)
    model.to(device)
    model.eval()

    frame = pd.read_csv(config.manifest_path).sort_values("sample_id")
    source_manifest_hash = manifest_hash(frame)
    dataset = ManifestImageDataset(
        frame,
        label_to_index=label_to_index,
        transform=build_transforms("test", TransformConfig(image_size=image_size)),
        include_unknown=True,
    )
    loader: DataLoader[Any] = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_samples,
    )

    sample_ids: list[str] = []
    true_labels: list[str] = []
    known_status: list[str] = []
    splits: list[str] = []
    logits_rows: list[np.ndarray] = []
    embedding_rows: list[np.ndarray] = []
    predictions: list[str] = []
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)  # type: ignore[union-attr]
            outputs = model(images)
            logits = outputs["logits"].detach().cpu().numpy()
            embeddings = outputs["embedding"].detach().cpu().numpy()
            pred_idx = logits.argmax(axis=1)
            sample_ids.extend(str(item) for item in batch["sample_id"])  # type: ignore[union-attr]
            true_labels.extend(str(item) for item in batch["canonical_label"])  # type: ignore[union-attr]
            known_status.extend(str(item) for item in batch["known_status"])  # type: ignore[union-attr]
            logits_rows.append(logits)
            embedding_rows.append(embeddings)
            predictions.extend(index_to_label.get(int(idx), "UNKNOWN") for idx in pred_idx)

    split_lookup = dict(
        zip(frame["sample_id"].astype(str), frame["split"].astype(str), strict=True)
    )
    splits = [split_lookup[sample_id] for sample_id in sample_ids]
    logits_arr = np.vstack(logits_rows)
    embeddings_arr = np.vstack(embedding_rows)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.output_path,
        sample_id=np.asarray(sample_ids, dtype=object),
        true_label=np.asarray(true_labels, dtype=object),
        known_status=np.asarray(known_status, dtype=object),
        split=np.asarray(splits, dtype=object),
        embedding=embeddings_arr,
        logits=logits_arr,
        prediction=np.asarray(predictions, dtype=object),
        label_to_index=json.dumps(label_to_index, sort_keys=True),
        manifest_hash=source_manifest_hash,
    )
    csv_path = config.output_path.with_suffix(".csv")
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "true_label": true_labels,
            "known_status": known_status,
            "split": splits,
            "prediction": predictions,
        }
    ).to_csv(csv_path, index=False)
    return config.output_path
