"""Train a lightweight classifier over deep embeddings plus TDA features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hemato_osr.models.fusion import DeepTDAFusionMLP, FusionConfig
from hemato_osr.topology.pipeline import load_tda_feature_archive
from hemato_osr.training.seed import seed_everything


@dataclass(frozen=True)
class FusionTrainConfig:
    """Fusion training configuration."""

    embeddings_path: Path
    tda_features_path: Path
    output_path: Path
    epochs: int = 5
    learning_rate: float = 1e-3
    seed: int = 13
    device: str = "auto"


@dataclass(frozen=True)
class FrozenFusionExperimentConfig:
    """Train a frozen-feature Deep+TDA fusion MLP and export predictions."""

    embeddings_path: Path
    tda_features_path: Path
    output_dir: Path
    feature_components: tuple[str, ...] = ()
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    hidden_dim: int = 128
    dropout: float = 0.1
    early_stopping_patience: int = 8
    seed: int = 37
    device: str = "auto"


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_aligned(
    config: FusionTrainConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(config.embeddings_path, allow_pickle=True) as emb_data:
        sample_ids = [str(item) for item in emb_data["sample_id"].tolist()]
        embeddings = np.asarray(emb_data["embedding"], dtype=np.float32)
        splits = emb_data["split"].astype(str)
        labels = emb_data["true_label"].astype(str)
        label_to_index = json.loads(str(emb_data["label_to_index"].tolist()))
    with np.load(config.tda_features_path, allow_pickle=True) as tda_data:
        tda_ids = [str(item) for item in tda_data["sample_ids"].tolist()]
        tda_features = np.asarray(tda_data["features"], dtype=np.float32)
    order = {sample_id: idx for idx, sample_id in enumerate(tda_ids)}
    aligned_tda = np.vstack([tda_features[order[sample_id]] for sample_id in sample_ids])
    y = np.asarray([label_to_index.get(label, -1) for label in labels], dtype=np.int64)
    known = y >= 0
    return embeddings[known], aligned_tda[known], y[known], splits[known]


def _safe_scale(scaler: StandardScaler, values: np.ndarray) -> np.ndarray:
    scaled = scaler.transform(values)
    if not np.isfinite(scaled).all():
        msg = "Non-finite features after train-only standardization"
        raise ValueError(msg)
    return scaled.astype(np.float32)


def _component_indices(metadata: dict[str, object], components: tuple[str, ...]) -> np.ndarray:
    if not components:
        return np.asarray([], dtype=int)
    slices = metadata.get("component_slices", {})
    if not isinstance(slices, dict):
        msg = "TDA feature metadata is missing component_slices"
        raise ValueError(msg)
    indices: list[int] = []
    for component in components:
        start, end = slices[component]  # type: ignore[index]
        indices.extend(range(int(start), int(end)))
    return np.asarray(indices, dtype=int)


def _load_fusion_arrays(config: FrozenFusionExperimentConfig) -> dict[str, object]:
    with np.load(config.embeddings_path, allow_pickle=True) as emb_data:
        sample_ids = np.asarray(emb_data["sample_id"]).astype(str)
        deep = np.asarray(emb_data["embedding"], dtype=np.float32)
        labels = np.asarray(emb_data["true_label"]).astype(str)
        known_status = np.asarray(emb_data["known_status"]).astype(str)
        splits = np.asarray(emb_data["split"]).astype(str)
        label_to_index = json.loads(str(emb_data["label_to_index"].tolist()))
        manifest_hash = str(emb_data["manifest_hash"].tolist())
    tda = load_tda_feature_archive(config.tda_features_path)
    tda_ids = list(tda["sample_id"])
    tda_order = {sample_id: idx for idx, sample_id in enumerate(tda_ids)}
    selected = _component_indices(tda["metadata"], config.feature_components)
    tda_features = tda["features"]
    if selected.size:
        tda_features = tda_features[:, selected]
    aligned_tda = np.vstack([tda_features[tda_order[sample_id]] for sample_id in sample_ids])
    y = np.asarray([label_to_index.get(label, -1) for label in labels], dtype=np.int64)
    return {
        "sample_id": sample_ids,
        "deep": deep,
        "tda": aligned_tda.astype(np.float32),
        "labels": labels,
        "known_status": known_status,
        "splits": splits,
        "y": y,
        "label_to_index": label_to_index,
        "manifest_hash": manifest_hash,
        "tda_metadata": tda["metadata"],
    }


def train_fusion(config: FusionTrainConfig) -> Path:
    """Train fusion MLP on cached deep and TDA features."""

    seed_everything(config.seed)
    deep, tda, labels, splits = _load_aligned(config)
    train_mask = splits == "train"
    if not np.any(train_mask):
        msg = "Fusion training requires train split features"
        raise ValueError(msg)
    device = _device(config.device)
    model = DeepTDAFusionMLP(
        FusionConfig(
            deep_dim=deep.shape[1],
            tda_dim=tda.shape[1],
            num_classes=int(labels.max()) + 1,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    x_deep = torch.tensor(deep[train_mask], dtype=torch.float32, device=device)
    x_tda = torch.tensor(tda[train_mask], dtype=torch.float32, device=device)
    y = torch.tensor(labels[train_mask], dtype=torch.long, device=device)
    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_deep, x_tda)["logits"], y)
        loss.backward()
        optimizer.step()
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config.__dict__},
        config.output_path,
    )
    return config.output_path


def train_frozen_fusion(config: FrozenFusionExperimentConfig) -> Path:
    """Train a frozen-feature fusion MLP and export an embedding-style NPZ."""

    seed_everything(config.seed)
    arrays = _load_fusion_arrays(config)
    sample_ids = arrays["sample_id"]  # type: ignore[assignment]
    deep = arrays["deep"]  # type: ignore[assignment]
    tda = arrays["tda"]  # type: ignore[assignment]
    splits = arrays["splits"]  # type: ignore[assignment]
    known_status = arrays["known_status"]  # type: ignore[assignment]
    y = arrays["y"]  # type: ignore[assignment]
    labels = arrays["labels"]  # type: ignore[assignment]
    label_to_index = arrays["label_to_index"]  # type: ignore[assignment]
    manifest_hash = str(arrays["manifest_hash"])
    assert isinstance(sample_ids, np.ndarray)
    assert isinstance(deep, np.ndarray)
    assert isinstance(tda, np.ndarray)
    assert isinstance(splits, np.ndarray)
    assert isinstance(known_status, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(label_to_index, dict)

    known = y >= 0
    train_mask = known & (splits == "train")
    val_mask = known & (splits == "validation")
    if not np.any(train_mask) or not np.any(val_mask):
        msg = "Fusion requires non-empty known train and validation splits"
        raise ValueError(msg)
    deep_scaler = StandardScaler().fit(deep[train_mask])
    tda_scaler = StandardScaler().fit(tda[train_mask])
    deep_scaled = _safe_scale(deep_scaler, deep)
    tda_scaled = _safe_scale(tda_scaler, tda)

    device = _device(config.device)
    model = DeepTDAFusionMLP(
        FusionConfig(
            deep_dim=deep_scaled.shape[1],
            tda_dim=tda_scaled.shape[1],
            num_classes=len(label_to_index),
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
    ).to(device)
    labels_train = y[train_mask]
    class_counts = np.bincount(labels_train, minlength=len(label_to_index)).astype(np.float32)
    weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    weights = weights * (len(weights) / weights.sum())
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_dataset = TensorDataset(
        torch.tensor(deep_scaled[train_mask], dtype=torch.float32),
        torch.tensor(tda_scaled[train_mask], dtype=torch.float32),
        torch.tensor(y[train_mask], dtype=torch.long),
    )
    loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_deep = torch.tensor(deep_scaled[val_mask], dtype=torch.float32, device=device)
    val_tda = torch.tensor(tda_scaled[val_mask], dtype=torch.float32, device=device)
    val_y = y[val_mask]

    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.output_dir / "training_log.csv"
    best_state = None
    best_macro_f1 = -1.0
    stale = 0
    rows = []
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for batch_deep, batch_tda, batch_y in loader:
            batch_deep = batch_deep.to(device)
            batch_tda = batch_tda.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_deep, batch_tda)["logits"], batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_logits = model(val_deep, val_tda)["logits"].detach().cpu().numpy()
        val_pred = val_logits.argmax(axis=1)
        macro_f1 = float(f1_score(val_y, val_pred, average="macro", zero_division=0))
        rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else 0.0,
                "validation_macro_f1": macro_f1,
                "seed": config.seed,
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
            }
        )
        pd.DataFrame(rows).to_csv(log_path, index=False)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            stale = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= config.early_stopping_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    all_logits = []
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sample_ids), config.batch_size):
            end = start + config.batch_size
            out = model(
                torch.tensor(deep_scaled[start:end], dtype=torch.float32, device=device),
                torch.tensor(tda_scaled[start:end], dtype=torch.float32, device=device),
            )
            all_logits.append(out["logits"].detach().cpu().numpy())
            all_embeddings.append(out["embedding"].detach().cpu().numpy())
    logits = np.vstack(all_logits)
    fused = np.vstack(all_embeddings)
    index_to_label = {int(idx): label for label, idx in label_to_index.items()}
    predictions = np.asarray(
        [index_to_label[int(idx)] for idx in logits.argmax(axis=1)],
        dtype=object,
    )
    output_path = config.output_dir / "fusion_embeddings.npz"
    np.savez_compressed(
        output_path,
        sample_id=sample_ids.astype(object),
        true_label=labels.astype(object),
        known_status=known_status.astype(object),
        split=splits.astype(object),
        embedding=fused.astype(np.float32),
        logits=logits.astype(np.float32),
        prediction=predictions,
        label_to_index=json.dumps(label_to_index, sort_keys=True),
        manifest_hash=manifest_hash,
        fusion_config=json.dumps(config.__dict__, sort_keys=True, default=str),
        tda_metadata=json.dumps(arrays["tda_metadata"], sort_keys=True, default=str),
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "deep_scaler_mean": deep_scaler.mean_,
            "deep_scaler_scale": deep_scaler.scale_,
            "tda_scaler_mean": tda_scaler.mean_,
            "tda_scaler_scale": tda_scaler.scale_,
            "label_to_index": label_to_index,
        },
        config.output_dir / "fusion.pt",
    )
    return output_path
