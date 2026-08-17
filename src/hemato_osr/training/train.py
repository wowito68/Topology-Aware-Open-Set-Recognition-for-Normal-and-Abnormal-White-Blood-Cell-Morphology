"""Closed-set training loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from hemato_osr.data.dataset import (
    ManifestImageDataset,
    collate_samples,
    make_label_mapping,
)
from hemato_osr.data.taxonomy import DEFAULT_KNOWN_CLASSES
from hemato_osr.data.transforms import TransformConfig, build_transforms
from hemato_osr.models.backbones import ModelConfig, create_classifier
from hemato_osr.training.checkpoint import save_checkpoint
from hemato_osr.training.seed import seed_everything


@dataclass(frozen=True)
class TrainConfig:
    """Minimal closed-set training configuration."""

    manifest_path: Path
    output_dir: Path
    known_classes: tuple[str, ...] = DEFAULT_KNOWN_CLASSES
    backbone: str = "resnet18"
    pretrained: bool = True
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 0
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    imbalance_strategy: str = "none"
    gradient_clip_norm: float | None = 1.0
    early_stopping_patience: int = 5
    seed: int = 13
    device: str = "auto"
    precision: str = "fp32"
    log_path: Path | None = None
    smoke_max_train_per_class: int | None = None


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _class_weights(frame: pd.DataFrame, label_to_index: dict[str, int]) -> torch.Tensor:
    counts = frame["canonical_label"].map(label_to_index).value_counts().sort_index()
    weights = []
    for idx in range(len(label_to_index)):
        count = float(counts.get(idx, 1))
        weights.append(1.0 / count)
    arr = np.asarray(weights, dtype=np.float32)
    arr = arr * (len(arr) / arr.sum())
    return torch.tensor(arr, dtype=torch.float32)


def _sampler(frame: pd.DataFrame, label_to_index: dict[str, int]) -> WeightedRandomSampler:
    labels = frame["canonical_label"].map(label_to_index).astype(int)
    counts = labels.value_counts()
    weights = labels.map(lambda label: 1.0 / float(counts[label])).to_numpy(dtype=np.float64)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def _subset_per_class(frame: pd.DataFrame, per_class: int | None, seed: int) -> pd.DataFrame:
    if per_class is None:
        return frame
    return (
        frame.groupby("canonical_label", group_keys=False, sort=True)
        .apply(lambda group: group.sample(n=min(per_class, len(group)), random_state=seed))
        .sort_values("sample_id")
        .reset_index(drop=True)
    )


def _balanced_accuracy(y_true: list[int], y_pred: list[int]) -> float:
    labels = sorted(set(y_true))
    if not labels:
        return 0.0
    recalls = []
    true_arr = np.asarray(y_true)
    pred_arr = np.asarray(y_pred)
    for label in labels:
        mask = true_arr == label
        recalls.append(float(np.mean(pred_arr[mask] == label)))
    return float(np.mean(recalls))


def _macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _append_epoch_log(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=write_header, index=False)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    scaler: torch.cuda.amp.GradScaler | None,
    gradient_clip_norm: float | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    preds: list[int] = []
    labels_all: list[int] = []
    for batch in loader:
        images = batch["image"].to(device)  # type: ignore[union-attr]
        labels = batch["label"].to(device)  # type: ignore[union-attr]
        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            use_amp = scaler is not None and device.type == "cuda"
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs["logits"], labels)
            if training:
                assert optimizer is not None
                if scaler is not None and device.type == "cuda":
                    scaler.scale(loss).backward()
                    if gradient_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if gradient_clip_norm is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    optimizer.step()
        losses.append(float(loss.detach().cpu()))
        preds.extend(outputs["logits"].argmax(dim=1).detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().tolist())
    accuracy = float(np.mean(np.asarray(preds) == np.asarray(labels_all))) if labels_all else 0.0
    return float(np.mean(losses)) if losses else 0.0, accuracy


def train_closed_set(config: TrainConfig) -> Path:
    """Train a closed-set classifier and return the best checkpoint path."""

    seed_everything(config.seed)
    device = _device(config.device)
    frame = pd.read_csv(config.manifest_path)
    known_frame = frame.loc[frame["canonical_label"].isin(config.known_classes)].copy()
    train_frame = known_frame.loc[known_frame["split"] == "train"].copy()
    val_frame = known_frame.loc[known_frame["split"] == "validation"].copy()
    train_frame = _subset_per_class(train_frame, config.smoke_max_train_per_class, config.seed)
    if train_frame.empty or val_frame.empty:
        msg = "Training requires non-empty known train and validation splits"
        raise ValueError(msg)

    label_to_index = make_label_mapping(config.known_classes)
    transform_cfg = TransformConfig(image_size=config.image_size)
    train_dataset = ManifestImageDataset(
        train_frame,
        label_to_index=label_to_index,
        transform=build_transforms("train", transform_cfg),
        include_unknown=False,
    )
    val_dataset = ManifestImageDataset(
        val_frame,
        label_to_index=label_to_index,
        transform=build_transforms("validation", transform_cfg),
        include_unknown=False,
    )
    sampler = None
    shuffle = True
    if config.imbalance_strategy == "class-aware-sampler":
        sampler = _sampler(train_frame, label_to_index)
        shuffle = False
    elif config.imbalance_strategy not in {"none", "weighted-cross-entropy"}:
        msg = f"Unknown imbalance strategy: {config.imbalance_strategy}"
        raise ValueError(msg)

    train_loader: DataLoader[Any] = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=collate_samples,
    )
    val_loader: DataLoader[Any] = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_samples,
    )

    model = create_classifier(
        ModelConfig(
            backbone=config.backbone,
            num_classes=len(config.known_classes),
            pretrained=config.pretrained,
        )
    ).to(device)
    class_weights = None
    if config.imbalance_strategy == "weighted-cross-entropy":
        class_weights = _class_weights(train_frame, label_to_index).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler() if config.precision == "amp" else None

    best_macro_f1 = -1.0
    stale_epochs = 0
    best_path = config.output_dir / "best_checkpoint.pt"
    last_path = config.output_dir / "last_checkpoint.pt"
    log_path = config.log_path or config.output_dir / "training_log.csv"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(config.output_dir / "tensorboard"))
    for epoch in range(config.epochs):
        train_loss, train_accuracy = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        val_loss, val_accuracy = _run_epoch(
            model,
            val_loader,
            criterion,
            None,
            device,
            scaler=None,
            gradient_clip_norm=None,
        )
        val_preds, val_labels = predict_labels(model, val_loader, device)
        balanced = _balanced_accuracy(val_labels, val_preds)
        macro_f1 = _macro_f1(val_labels, val_preds)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/validation", val_loss, epoch)
        writer.add_scalar("accuracy/train", train_accuracy, epoch)
        writer.add_scalar("accuracy/validation", val_accuracy, epoch)
        writer.add_scalar("balanced_accuracy/validation", balanced, epoch)
        writer.add_scalar("macro_f1/validation", macro_f1, epoch)
        _append_epoch_log(
            log_path,
            {
                "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "train_accuracy": train_accuracy,
                "validation_accuracy": val_accuracy,
                "validation_macro_f1": macro_f1,
                "validation_balanced_accuracy": balanced,
                "learning_rate": learning_rate,
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
                "seed": config.seed,
            },
        )
        save_checkpoint(
            last_path,
            model,
            label_to_index=label_to_index,
            config=asdict(config),
            metrics={
                "epoch": float(epoch),
                "validation_accuracy": val_accuracy,
                "validation_macro_f1": macro_f1,
                "validation_balanced_accuracy": balanced,
            },
            optimizer=optimizer,
            scheduler=None,
            epoch=epoch,
            seed=config.seed,
        )
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            stale_epochs = 0
            save_checkpoint(
                best_path,
                model,
                label_to_index=label_to_index,
                config=asdict(config),
                metrics={
                    "epoch": float(epoch),
                    "validation_accuracy": val_accuracy,
                    "validation_macro_f1": macro_f1,
                    "validation_balanced_accuracy": balanced,
                },
                optimizer=optimizer,
                scheduler=None,
                epoch=epoch,
                seed=config.seed,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= config.early_stopping_patience:
            break
    writer.close()
    return best_path


def predict_labels(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Return predicted and true labels for a loader."""

    model.eval()
    preds: list[int] = []
    labels_all: list[int] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)  # type: ignore[union-attr]
            labels = batch["label"].to(device)  # type: ignore[union-attr]
            outputs = model(images)
            preds.extend(outputs["logits"].argmax(dim=1).cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
    return preds, labels_all
