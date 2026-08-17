from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from hemato_osr.data.taxonomy import DEFAULT_KNOWN_CLASSES
from hemato_osr.experiments.delivery4 import EmbeddingArchive
from hemato_osr.experiments.delivery5 import (
    CLOSED_MACRO_F1_FLOOR,
    _known_geometry,
    _progression,
    _score_representation,
    _unknown_geometry,
)
from hemato_osr.training.representation.losses import SupervisedContrastiveLoss
from hemato_osr.training.representation.models import AngularMarginHead, ProjectionHead
from hemato_osr.training.representation.samplers import PKBatchSampler
from hemato_osr.training.representation.train import _is_better


def _toy_archive(offset: float = 0.0) -> EmbeddingArchive:
    rng = np.random.default_rng(37)
    dim = 8
    label_to_index = {label: idx for idx, label in enumerate(DEFAULT_KNOWN_CLASSES)}
    centers = np.eye(len(DEFAULT_KNOWN_CLASSES), dim) * (4.0 + offset)
    rows: list[dict[str, object]] = []
    embeddings: list[np.ndarray] = []
    logits: list[np.ndarray] = []

    def add(label: str, split: str, status: str, embedding: np.ndarray) -> None:
        rows.append(
            {
                "sample_id": f"s{len(rows)}",
                "true_label": label,
                "known_status": status,
                "split": split,
                "prediction": label if status == "known" else DEFAULT_KNOWN_CLASSES[2],
            }
        )
        embeddings.append(embedding)
        logit = np.zeros(len(DEFAULT_KNOWN_CLASSES), dtype=float)
        if status == "known":
            logit[label_to_index[label]] = 6.0
        logits.append(logit)

    for idx, label in enumerate(DEFAULT_KNOWN_CLASSES):
        for _ in range(3):
            add(label, "train", "known", centers[idx] + rng.normal(0, 0.04, dim))
        add(label, "validation", "known", centers[idx] + rng.normal(0, 0.04, dim))
        add(label, "test", "known", centers[idx] + rng.normal(0, 0.04, dim))
    for label, center in {
        "lymphocyte_neoplastic": centers[2] + np.asarray([0.0, 0.0, 1.0, 0, 0, 0, 0, 0]),
        "neutrophil_band": centers[4] + np.asarray([0.0, 0.0, 0.0, 0, 1.0, 0, 0, 0]),
    }.items():
        for _ in range(2):
            add(label, "test", "unknown", center + rng.normal(0, 0.04, dim))
    frame = pd.DataFrame(rows)
    return EmbeddingArchive(
        sample_id=frame["sample_id"].to_numpy(str),
        true_label=frame["true_label"].to_numpy(str),
        known_status=frame["known_status"].to_numpy(str),
        split=frame["split"].to_numpy(str),
        embedding=np.vstack(embeddings),
        logits=np.vstack(logits),
        prediction=frame["prediction"].to_numpy(str),
        label_to_index=label_to_index,
        manifest_hash="toy",
    )


def test_supcon_loss_finite_and_no_positive_edge() -> None:
    projections = F.normalize(torch.randn(6, 4), dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = SupervisedContrastiveLoss(temperature=0.10)(projections, labels)
    assert torch.isfinite(loss)
    assert float(loss) >= 0

    no_positive = SupervisedContrastiveLoss()(projections[:3], torch.tensor([0, 1, 2]))
    assert torch.isfinite(no_positive)
    assert float(no_positive) == pytest.approx(0.0)


def test_projection_head_normalizes_outputs() -> None:
    head = ProjectionHead(input_dim=4, hidden_dim=3, output_dim=2)
    projected = head(torch.randn(5, 4))
    assert torch.allclose(projected.norm(dim=1), torch.ones(5), atol=1e-5)


def test_pk_sampler_enforces_positive_pairs_and_rejects_leakage() -> None:
    frame = pd.DataFrame(
        {
            "canonical_label": ["a", "a", "b", "b", "c", "c"],
            "known_status": ["known"] * 6,
            "split": ["train"] * 6,
        }
    )
    sampler = PKBatchSampler(frame, p_classes=2, k_per_class=2, seed=37)
    batch = next(iter(sampler))
    labels = frame.iloc[batch]["canonical_label"].value_counts()
    assert set(labels.tolist()) == {2}

    leaked = frame.copy()
    leaked.loc[0, "known_status"] = "unknown"
    with pytest.raises(ValueError, match="unknown"):
        PKBatchSampler(leaked)
    leaked = frame.copy()
    leaked.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="non-train"):
        PKBatchSampler(leaked)


def test_arcface_margin_and_inference_without_labels() -> None:
    head = AngularMarginHead(embedding_dim=3, num_classes=3, scale=30.0, margin=0.30)
    with torch.no_grad():
        head.weight.copy_(torch.eye(3))
    embeddings = torch.eye(3)
    inference = head(embeddings, labels=None, apply_margin=False)
    training = head(embeddings, labels=torch.tensor([0, 1, 2]), apply_margin=True)
    assert torch.allclose(head.normalized_weight().norm(dim=1), torch.ones(3))
    assert torch.all(training.diag() < inference.diag())
    assert torch.allclose(
        inference, head(embeddings, labels=torch.tensor([2, 1, 0]), apply_margin=False)
    )


def test_delivery5_scorers_refit_and_orient_scores() -> None:
    archive = _toy_archive()
    msp, _state, _ms = _score_representation(archive, "msp")
    energy, _state, _ms = _score_representation(archive, "energy")
    knn, state, _ms = _score_representation(archive, "knn_cosine_k5")
    vim_a, vim_state_a, _ms = _score_representation(archive, "vim")
    vim_b, vim_state_b, _ms = _score_representation(_toy_archive(offset=0.5), "vim")
    assert np.isfinite(msp).all()
    assert np.isfinite(energy).all()
    assert np.isfinite(knn).all()
    assert state["fit_split"] == "known_train"
    assert vim_state_a["n_components"] != 0
    assert not np.allclose(vim_a, vim_b)


def test_geometry_and_attractor_matrix() -> None:
    archive = _toy_archive()
    vim, _state, _ms = _score_representation(archive, "vim")
    known_geometry, summary = _known_geometry(archive, "toy")
    sample_geometry, attractor, lookup = _unknown_geometry(archive, "toy", vim)
    assert set(known_geometry["class"]) == set(DEFAULT_KNOWN_CLASSES)
    assert {"train", "validation", "test"}.issubset(set(summary["split"]))
    assert "lymphocyte_neoplastic" in attractor.index
    assert lookup["lymphocyte_neoplastic"] == "lymphocyte"
    assert np.isfinite(sample_geometry["knn_distance"]).all()


def test_progression_criteria_and_closed_set_safeguard() -> None:
    comparison = pd.DataFrame(
        [
            {
                "representation": "ce",
                "osr_method": "vim",
                "AUROC": 0.872,
                "FPR95": 0.514,
                "closed_macro_f1": 0.971,
            },
            {
                "representation": "supcon",
                "osr_method": "vim",
                "AUROC": 0.890,
                "FPR95": 0.430,
                "closed_macro_f1": CLOSED_MACRO_F1_FLOOR,
            },
            {
                "representation": "arcface",
                "osr_method": "vim",
                "AUROC": 0.850,
                "FPR95": 0.600,
                "closed_macro_f1": 0.950,
            },
        ]
    )
    subgroup = pd.DataFrame(
        [
            {
                "representation": "ce",
                "osr_method": "vim",
                "group": "lymphoid_related",
                "AUROC": 0.80,
            },
            {
                "representation": "supcon",
                "osr_method": "vim",
                "group": "lymphoid_related",
                "AUROC": 0.84,
            },
        ]
    )
    bootstrap = {
        "supcon__vim": {
            "delta_auroc": {"lower": 0.001},
            "delta_fpr95": {"upper": -0.01},
        }
    }
    decision = _progression(comparison, subgroup, bootstrap)
    assert decision["closed_set_safeguard"] == "PASS"
    assert decision["criterion_a"] == "PASS"
    assert decision["criterion_b"] == "PASS"
    assert decision["criterion_c"] == "PASS"


def test_checkpoint_selection_rule_uses_known_validation_metrics() -> None:
    best = {
        "macro_f1": 0.90,
        "fisher_ratio": 2.0,
        "within_class_dispersion": 1.0,
    }
    assert _is_better({"macro_f1": 0.91, "fisher_ratio": 1.0, "within_class_dispersion": 2.0}, best)
    assert _is_better({"macro_f1": 0.90, "fisher_ratio": 2.1, "within_class_dispersion": 2.0}, best)
    assert _is_better({"macro_f1": 0.90, "fisher_ratio": 2.0, "within_class_dispersion": 0.9}, best)
    assert not _is_better(
        {"macro_f1": 0.89, "fisher_ratio": 10.0, "within_class_dispersion": 0.1},
        best,
    )
