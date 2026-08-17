"""Training and embedding export for Delivery 5 representations."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from hemato_osr.data.dataset import ManifestImageDataset, collate_samples, make_label_mapping
from hemato_osr.data.manifest import manifest_hash
from hemato_osr.data.taxonomy import DEFAULT_KNOWN_CLASSES
from hemato_osr.data.transforms import TransformConfig, build_transforms
from hemato_osr.evaluation.metrics import calibration_metrics, closed_set_metrics
from hemato_osr.training.checkpoint import load_checkpoint, save_checkpoint
from hemato_osr.training.representation.losses import SupervisedContrastiveLoss
from hemato_osr.training.representation.models import (
    AngularMarginModel,
    RepresentationModelSpec,
    SupConClassifier,
    create_representation_model,
)
from hemato_osr.training.representation.samplers import PKBatchSampler
from hemato_osr.training.seed import seed_everything
from hemato_osr.utils.tracking import write_json


@dataclass(frozen=True)
class RepresentationTrainConfig:
    """Delivery 5 representation training configuration."""

    manifest_path: Path
    output_dir: Path
    representation: str
    known_classes: tuple[str, ...] = DEFAULT_KNOWN_CLASSES
    backbone: str = "resnet18"
    pretrained: bool = True
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 7
    gradient_clip_norm: float | None = 1.0
    seed: int = 37
    device: str = "auto"
    precision: str = "amp"
    supcon_lambda: float = 0.50
    supcon_temperature: float = 0.10
    pk_classes: int = 5
    pk_examples: int = 12
    angular_scale: float = 30.0
    angular_margin: float = 0.30
    ce_weighting: str = "auto"
    smoke_max_train_per_class: int | None = None
    log_path: Path | None = None


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_known_training_frame(frame: pd.DataFrame) -> None:
    if np.any(frame["known_status"].astype(str) != "known"):
        msg = "Training frame contains unknown rows"
        raise ValueError(msg)
    if np.any(frame["split"].astype(str) != "train"):
        msg = "Training frame contains non-train rows"
        raise ValueError(msg)


def _class_weights(frame: pd.DataFrame, label_to_index: dict[str, int]) -> torch.Tensor:
    labels = frame["canonical_label"].map(label_to_index).astype(int)
    counts = labels.value_counts().sort_index()
    weights = []
    for idx in range(len(label_to_index)):
        weights.append(1.0 / float(counts.get(idx, 1)))
    arr = np.asarray(weights, dtype=np.float32)
    arr = arr * (len(arr) / arr.sum())
    return torch.tensor(arr, dtype=torch.float32)


def _subset_per_class(frame: pd.DataFrame, per_class: int | None, seed: int) -> pd.DataFrame:
    if per_class is None:
        return frame
    return (
        frame.groupby("canonical_label", group_keys=False, sort=True)
        .apply(lambda group: group.sample(n=min(per_class, len(group)), random_state=seed))
        .sort_values("sample_id")
        .reset_index(drop=True)
    )


def _make_model(config: RepresentationTrainConfig, num_classes: int) -> nn.Module:
    if config.representation == "supcon":
        return SupConClassifier(
            backbone=config.backbone,
            num_classes=num_classes,
            pretrained=config.pretrained,
        )
    if config.representation in {"arcface", "cosface"}:
        return AngularMarginModel(
            backbone=config.backbone,
            num_classes=num_classes,
            pretrained=config.pretrained,
            scale=config.angular_scale,
            margin=config.angular_margin,
            loss_type=config.representation,  # type: ignore[arg-type]
        )
    msg = f"Unknown representation: {config.representation}"
    raise ValueError(msg)


def _build_loaders(
    config: RepresentationTrainConfig,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    label_to_index: dict[str, int],
) -> tuple[DataLoader[Any], DataLoader[Any]]:
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
    if config.representation == "supcon":
        batch_sampler = PKBatchSampler(
            train_frame,
            p_classes=config.pk_classes,
            k_per_class=config.pk_examples,
            seed=config.seed,
        )
        train_loader: DataLoader[Any] = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=config.num_workers,
            collate_fn=collate_samples,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
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
    return train_loader, val_loader


def _forward_for_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    training: bool,
) -> dict[str, torch.Tensor]:
    if isinstance(model, AngularMarginModel):
        return model(images, labels if training else None, apply_margin=training)
    return model(images)  # type: ignore[no-any-return]


def _inference_logits(model: nn.Module, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(model, AngularMarginModel):
        return model.classifier(outputs["embedding"], None, apply_margin=False)
    return outputs["logits"]


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    ce_loss: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    scaler: torch.cuda.amp.GradScaler | None,
    gradient_clip_norm: float | None,
    supcon_loss: SupervisedContrastiveLoss | None,
    supcon_lambda: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    ce_losses: list[float] = []
    supcon_losses: list[float] = []
    preds: list[int] = []
    labels_all: list[int] = []
    for batch in loader:
        images = batch["image"].to(device)  # type: ignore[union-attr]
        labels = batch["label"].to(device)  # type: ignore[union-attr]
        if torch.any(labels < 0):
            msg = "Representation training received an unknown label"
            raise ValueError(msg)
        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            use_amp = scaler is not None and device.type == "cuda"
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = _forward_for_loss(model, images, labels, training=training)
                loss_ce = ce_loss(outputs["logits"], labels)
                loss_sup = torch.zeros((), device=device)
                if training and supcon_loss is not None:
                    loss_sup = supcon_loss(outputs["projection"], labels)
                loss = loss_ce + supcon_lambda * loss_sup
            if training:
                assert optimizer is not None
                if scaler is not None and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if gradient_clip_norm is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if gradient_clip_norm is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    optimizer.step()
        logits_for_pred = _inference_logits(model, outputs)
        losses.append(float(loss.detach().cpu()))
        ce_losses.append(float(loss_ce.detach().cpu()))
        supcon_losses.append(float(loss_sup.detach().cpu()))
        preds.extend(logits_for_pred.argmax(dim=1).detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().tolist())
    accuracy = float(np.mean(np.asarray(preds) == np.asarray(labels_all))) if labels_all else 0.0
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "ce_loss": float(np.mean(ce_losses)) if ce_losses else 0.0,
        "supcon_loss": float(np.mean(supcon_losses)) if supcon_losses else 0.0,
        "accuracy": accuracy,
    }


def _collect_outputs(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_rows: list[np.ndarray] = []
    embedding_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)  # type: ignore[union-attr]
            labels = batch["label"].to(device)  # type: ignore[union-attr]
            outputs = _forward_for_loss(model, images, labels, training=False)
            logits = _inference_logits(model, outputs)
            logits_rows.append(logits.detach().cpu().numpy())
            embedding_rows.append(outputs["embedding"].detach().cpu().numpy())
            label_rows.append(labels.detach().cpu().numpy())
    return np.vstack(logits_rows), np.vstack(embedding_rows), np.concatenate(label_rows)


def _geometry_tiebreaker(embeddings: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    centroids = []
    within = []
    cosine = []
    for label in sorted(set(labels.tolist())):
        class_embeddings = embeddings[labels == label]
        centroid = class_embeddings.mean(axis=0)
        centroids.append(centroid)
        within.append(float(np.mean(np.linalg.norm(class_embeddings - centroid, axis=1))))
        normed = class_embeddings / np.maximum(
            np.linalg.norm(class_embeddings, axis=1, keepdims=True),
            1e-12,
        )
        proto = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
        cosine.append(float(np.mean(normed @ proto)))
    centroid_arr = np.vstack(centroids)
    distances = np.linalg.norm(centroid_arr[:, None, :] - centroid_arr[None, :, :], axis=2)
    upper = distances[np.triu_indices_from(distances, k=1)]
    return {
        "within_class_dispersion": float(np.mean(within)),
        "between_class_separation": float(np.mean(upper)),
        "fisher_ratio": float(np.mean(upper) / max(float(np.mean(within)), 1e-12)),
        "mean_cosine_compactness": float(np.mean(cosine)),
    }


def _validation_report(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    logits, embeddings, labels = _collect_outputs(model, loader, device)
    preds = logits.argmax(axis=1)
    closed = closed_set_metrics(
        labels.astype(int).tolist(),
        preds.astype(int).tolist(),
        class_names,
    )
    cal = calibration_metrics(logits, labels)
    geometry = _geometry_tiebreaker(embeddings, labels)
    return {
        **{key: value for key, value in closed.items() if key != "per_class"},
        **cal,
        **geometry,
        "per_class": closed["per_class"],
    }


def _is_better(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    macro_delta = float(candidate["macro_f1"]) - float(best["macro_f1"])
    if abs(macro_delta) > 1e-12:
        return macro_delta > 0
    fisher_delta = float(candidate["fisher_ratio"]) - float(best["fisher_ratio"])
    if abs(fisher_delta) > 1e-12:
        return fisher_delta > 0
    return float(candidate["within_class_dispersion"]) < float(best["within_class_dispersion"])


def train_representation_model(config: RepresentationTrainConfig) -> Path:
    """Train SupCon/ArcFace/CosFace representations on known train only."""

    seed_everything(config.seed)
    device = _device(config.device)
    frame = pd.read_csv(config.manifest_path)
    known_frame = frame.loc[frame["canonical_label"].isin(config.known_classes)].copy()
    train_frame = known_frame.loc[known_frame["split"] == "train"].copy()
    val_frame = known_frame.loc[known_frame["split"] == "validation"].copy()
    train_frame = _subset_per_class(train_frame, config.smoke_max_train_per_class, config.seed)
    _validate_known_training_frame(train_frame)
    if val_frame.empty:
        msg = "Representation training requires known validation rows"
        raise ValueError(msg)
    label_to_index = make_label_mapping(config.known_classes)
    train_loader, val_loader = _build_loaders(config, train_frame, val_frame, label_to_index)
    model = _make_model(config, len(label_to_index)).to(device)
    use_weighted_ce = config.representation != "supcon" and config.ce_weighting == "weighted"
    class_weights = (
        _class_weights(train_frame, label_to_index).to(device) if use_weighted_ce else None
    )
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    supcon_loss = (
        SupervisedContrastiveLoss(config.supcon_temperature).to(device)
        if config.representation == "supcon"
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler() if config.precision == "amp" else None

    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.log_path or config.output_dir / "training_log.csv"
    writer = SummaryWriter(log_dir=str(config.output_dir / "tensorboard"))
    best_path = config.output_dir / "best_checkpoint.pt"
    last_path = config.output_dir / "last_checkpoint.pt"
    best_report: dict[str, Any] | None = None
    stale_epochs = 0
    start_time = time.perf_counter()
    peak_vram_mb = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(config.epochs):
        train_stats = _run_epoch(
            model,
            train_loader,
            ce_loss,
            optimizer,
            device,
            scaler=scaler,
            gradient_clip_norm=config.gradient_clip_norm,
            supcon_loss=supcon_loss,
            supcon_lambda=config.supcon_lambda,
        )
        val_report = _validation_report(model, val_loader, device, config.known_classes)
        if device.type == "cuda":
            peak_vram_mb = max(peak_vram_mb, torch.cuda.max_memory_allocated(device) / 1024**2)
        row = {
            "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_ce_loss": train_stats["ce_loss"],
            "train_supcon_loss": train_stats["supcon_loss"],
            "train_accuracy": train_stats["accuracy"],
            "validation_accuracy": val_report["accuracy"],
            "validation_balanced_accuracy": val_report["balanced_accuracy"],
            "validation_macro_f1": val_report["macro_f1"],
            "validation_fisher_ratio": val_report["fisher_ratio"],
            "validation_within_class_dispersion": val_report["within_class_dispersion"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
            "seed": config.seed,
        }
        pd.DataFrame([row]).to_csv(
            log_path,
            mode="a",
            header=not log_path.exists(),
            index=False,
        )
        writer.add_scalar("loss/train", train_stats["loss"], epoch)
        writer.add_scalar("macro_f1/validation", float(val_report["macro_f1"]), epoch)
        metrics = {
            "epoch": float(epoch),
            "validation_accuracy": float(val_report["accuracy"]),
            "validation_macro_f1": float(val_report["macro_f1"]),
            "validation_balanced_accuracy": float(val_report["balanced_accuracy"]),
            "validation_fisher_ratio": float(val_report["fisher_ratio"]),
        }
        save_checkpoint(
            last_path,
            model,
            label_to_index=label_to_index,
            config=asdict(config),
            metrics=metrics,
            optimizer=optimizer,
            epoch=epoch,
            seed=config.seed,
        )
        if _is_better(val_report, best_report):
            best_report = val_report
            stale_epochs = 0
            save_checkpoint(
                best_path,
                model,
                label_to_index=label_to_index,
                config=asdict(config),
                metrics=metrics,
                optimizer=optimizer,
                epoch=epoch,
                seed=config.seed,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= config.early_stopping_patience:
            break
    writer.close()
    wall_time_seconds = time.perf_counter() - start_time
    summary = {
        "representation": config.representation,
        "checkpoint": str(best_path),
        "checkpoint_sha256": _sha256_file(best_path),
        "epochs_requested": config.epochs,
        "epochs_completed": int(epoch + 1),
        "best_epoch": float(load_checkpoint(best_path, map_location="cpu")["metrics"]["epoch"]),
        "wall_time_seconds": wall_time_seconds,
        "gpu_hours": wall_time_seconds / 3600.0 if device.type == "cuda" else 0.0,
        "peak_vram_mb": peak_vram_mb,
        "selection_rule": (
            "known-validation macro-F1; tie-breaker validation Fisher ratio; "
            "second tie-breaker within-class dispersion"
        ),
        "unknown_used_during_training": False,
        "ce_weighting": "none_with_class_aware_sampler"
        if config.representation == "supcon"
        else ("weighted_cross_entropy" if use_weighted_ce else "none"),
    }
    write_json(config.output_dir / "training_summary.json", summary)
    return best_path


def _model_from_checkpoint(checkpoint: dict[str, Any]) -> nn.Module:
    cfg = dict(checkpoint["config"])
    label_to_index = dict(checkpoint["label_to_index"])
    representation = str(cfg["representation"])
    spec = RepresentationModelSpec(
        representation=representation,
        backbone=str(cfg.get("backbone", "resnet18")),
        num_classes=len(label_to_index),
        pretrained=False,
        scale=float(cfg.get("angular_scale", 30.0)),
        margin=float(cfg.get("angular_margin", 0.30)),
    )
    return create_representation_model(spec)


def export_representation_embeddings(
    manifest_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    *,
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int | None = None,
    device: str = "auto",
) -> Path:
    """Export logits/embeddings for a Delivery 5 representation checkpoint."""

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    label_to_index = {str(k): int(v) for k, v in dict(checkpoint["label_to_index"]).items()}
    train_cfg = dict(checkpoint.get("config", {}))
    eval_image_size = image_size or int(train_cfg.get("image_size", 224))
    model = _model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    dev = _device(device)
    model.to(dev)
    model.eval()

    frame = pd.read_csv(manifest_path).sort_values("sample_id")
    dataset = ManifestImageDataset(
        frame,
        label_to_index=label_to_index,
        transform=build_transforms("test", TransformConfig(image_size=eval_image_size)),
        include_unknown=True,
    )
    loader: DataLoader[Any] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_samples,
    )
    sample_ids: list[str] = []
    true_labels: list[str] = []
    known_status: list[str] = []
    logits_rows: list[np.ndarray] = []
    embedding_rows: list[np.ndarray] = []
    projection_rows: list[np.ndarray] = []
    predictions: list[str] = []
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    forward_start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(dev)  # type: ignore[union-attr]
            labels = batch["label"].to(dev)  # type: ignore[union-attr]
            outputs = _forward_for_loss(model, images, labels.clamp_min(0), training=False)
            logits = _inference_logits(model, outputs)
            logits_arr = logits.detach().cpu().numpy()
            logits_rows.append(logits_arr)
            embedding_rows.append(outputs["embedding"].detach().cpu().numpy())
            if "projection" in outputs:
                projection_rows.append(outputs["projection"].detach().cpu().numpy())
            predictions.extend(
                index_to_label.get(int(idx), "UNKNOWN") for idx in logits_arr.argmax(axis=1)
            )
            sample_ids.extend(str(item) for item in batch["sample_id"])  # type: ignore[union-attr]
            true_labels.extend(str(item) for item in batch["canonical_label"])  # type: ignore[union-attr]
            known_status.extend(str(item) for item in batch["known_status"])  # type: ignore[union-attr]
    forward_seconds = time.perf_counter() - forward_start
    split_lookup = dict(
        zip(frame["sample_id"].astype(str), frame["split"].astype(str), strict=True)
    )
    splits = [split_lookup[sample_id] for sample_id in sample_ids]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "sample_id": np.asarray(sample_ids, dtype=object),
        "true_label": np.asarray(true_labels, dtype=object),
        "known_status": np.asarray(known_status, dtype=object),
        "split": np.asarray(splits, dtype=object),
        "embedding": np.vstack(embedding_rows),
        "logits": np.vstack(logits_rows),
        "prediction": np.asarray(predictions, dtype=object),
        "label_to_index": json.dumps(label_to_index, sort_keys=True),
        "manifest_hash": manifest_hash(frame),
        "representation": str(train_cfg.get("representation", "unknown")),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "forward_ms_per_image": forward_seconds * 1000.0 / max(1, len(sample_ids)),
    }
    if projection_rows:
        payload["projection"] = np.vstack(projection_rows)
    np.savez_compressed(output_path, **payload)
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "true_label": true_labels,
            "known_status": known_status,
            "split": splits,
            "prediction": predictions,
        }
    ).to_csv(output_path.with_suffix(".csv"), index=False)
    write_json(
        output_path.with_suffix(".metadata.json"),
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "representation": str(train_cfg.get("representation", "unknown")),
            "forward_ms_per_image": payload["forward_ms_per_image"],
            "manifest_hash": manifest_hash(frame),
        },
    )
    return output_path
