from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hemato_osr.data.taxonomy import DEFAULT_KNOWN_CLASSES
from hemato_osr.experiments.delivery4 import (
    EmbeddingArchive,
    PooledMahalanobis,
    RelativeMahalanobis,
    ViMModel,
    _score_metrics,
    apply_progression_rule,
    compute_embedding_geometry,
    energy_scores,
    knn_cosine_scores,
    label_indices,
    msp_anomaly_scores,
    ncm_cosine_scores,
    ncm_euclidean_scores,
    paired_bootstrap_deltas,
    predictive_entropy_scores,
    react_clipped_logits,
    react_scores,
    validate_train_only_fit_mask,
    write_predeclared_matrix,
)


def _toy_archive() -> EmbeddingArchive:
    rng = np.random.default_rng(37)
    dim = 8
    label_to_index = {label: idx for idx, label in enumerate(DEFAULT_KNOWN_CLASSES)}
    centers = np.eye(len(DEFAULT_KNOWN_CLASSES), dim) * 5.0
    rows: list[dict[str, object]] = []
    embeddings: list[np.ndarray] = []
    logits: list[np.ndarray] = []

    def add_row(label: str, split: str, status: str, embedding: np.ndarray) -> None:
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
            class_idx = label_to_index[label]
            logit[class_idx] = 8.0
        else:
            logit[:] = 0.25
        logits.append(logit)

    for idx, label in enumerate(DEFAULT_KNOWN_CLASSES):
        for repeat in range(4):
            offset = rng.normal(0.0, 0.08, size=dim)
            offset[-1] += repeat * 0.03
            add_row(label, "train", "known", centers[idx] + offset)
        add_row(label, "validation", "known", centers[idx] + rng.normal(0.0, 0.06, size=dim))
        add_row(label, "test", "known", centers[idx] + rng.normal(0.0, 0.06, size=dim))

    unknown_specs = {
        "lymphocyte_neoplastic": centers[2] + np.asarray([0.2, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "neutrophil_band": centers[4] + np.asarray([0.0, 0.0, 0.0, 0.2, 0.8, 0.0, 0.0, 0.0]),
        "normoblast": np.asarray([8.0, 8.0, 8.0, 8.0, 8.0, 2.0, 2.0, 2.0]),
    }
    for label, center in unknown_specs.items():
        for _ in range(3):
            add_row(label, "test", "unknown", center + rng.normal(0.0, 0.05, size=dim))

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


def test_logit_score_orientation_and_energy_temperature() -> None:
    logits = np.asarray([[8.0, 0.0, 0.0], [0.1, 0.1, 0.1]])
    assert msp_anomaly_scores(logits)[1] > msp_anomaly_scores(logits)[0]
    assert predictive_entropy_scores(logits)[1] > predictive_entropy_scores(logits)[0]
    assert energy_scores(logits, temperature=1.0)[1] > energy_scores(logits, temperature=1.0)[0]
    assert energy_scores(logits, temperature=0.5).shape == (2,)
    assert energy_scores(logits, temperature=2.0).shape == (2,)
    with pytest.raises(ValueError, match="positive"):
        energy_scores(logits, temperature=0.0)


def test_ncm_and_knn_scores_are_oriented_as_unknownness() -> None:
    archive = _toy_archive()
    train = (archive.split == "train") & (archive.known_status == "known")
    labels = label_indices(archive.true_label, archive.label_to_index)
    train_embeddings = archive.embedding[train]
    train_labels = labels[train]
    near_known = archive.embedding[archive.true_label == DEFAULT_KNOWN_CLASSES[0]][:1]
    far_unknown = np.asarray([[9.0, 9.0, 9.0, 9.0, 9.0, 0.0, 0.0, 0.0]])

    euclidean, _means = ncm_euclidean_scores(
        train_embeddings,
        train_labels,
        np.vstack([near_known, far_unknown]),
    )
    cosine, _prototypes = ncm_cosine_scores(
        train_embeddings,
        train_labels,
        np.vstack([near_known, far_unknown]),
    )
    knn_k5, _neighbors = knn_cosine_scores(
        train_embeddings,
        np.vstack([near_known, far_unknown]),
        k=5,
    )
    knn_k10, _neighbors = knn_cosine_scores(
        train_embeddings,
        np.vstack([near_known, far_unknown]),
        k=10,
    )
    assert euclidean[1] > euclidean[0]
    assert cosine[1] > cosine[0]
    assert knn_k5[1] > knn_k5[0]
    assert np.isfinite(knn_k10).all()
    with pytest.raises(ValueError, match="positive"):
        knn_cosine_scores(train_embeddings, near_known, k=0)


def test_mahalanobis_and_relative_mahalanobis_are_finite() -> None:
    archive = _toy_archive()
    train = (archive.split == "train") & (archive.known_status == "known")
    labels = label_indices(archive.true_label, archive.label_to_index)
    train_embeddings = archive.embedding[train]
    train_labels = labels[train]
    query = np.vstack(
        [
            train_embeddings[:1],
            np.asarray([[9.0, 9.0, 9.0, 9.0, 9.0, 0.0, 0.0, 0.0]]),
        ]
    )

    historical = PooledMahalanobis.fit_historical(train_embeddings, train_labels).score(query)
    regularized = PooledMahalanobis.fit_ledoit_wolf(train_embeddings, train_labels).score(query)
    relative = RelativeMahalanobis.fit(train_embeddings, train_labels).score(query)
    assert np.isfinite(historical).all()
    assert np.isfinite(regularized).all()
    assert np.isfinite(relative).all()
    assert historical[1] > historical[0]
    assert regularized[1] > regularized[0]


def test_react_uses_train_percentile_and_recomputed_energy() -> None:
    train_embeddings = np.asarray([[0.0, 2.0], [4.0, 6.0]])
    embeddings = np.asarray([[1.0, 7.0]])
    weight = np.eye(2)
    bias = np.asarray([0.0, 0.5])
    clipped = react_clipped_logits(embeddings, weight, bias, clip_value=3.0)
    assert np.allclose(clipped, [[1.0, 3.5]])

    scores, clip_value = react_scores(
        train_embeddings,
        embeddings,
        weight,
        bias,
        percentile=50.0,
    )
    assert clip_value == pytest.approx(3.0)
    assert scores.shape == (1,)


def test_vim_fit_uses_id_statistics_and_scores_residuals() -> None:
    archive = _toy_archive()
    train = (archive.split == "train") & (archive.known_status == "known")
    model = ViMModel.fit(archive.embedding[train], archive.logits[train], explained_variance=0.90)
    scores = model.score(archive.embedding, archive.logits)
    assert 1 <= model.n_components < archive.embedding.shape[1]
    assert 0.0 < model.explained_variance_ratio <= 1.0
    assert np.isfinite(scores).all()


def test_known_train_fit_mask_rejects_leakage() -> None:
    archive = _toy_archive()
    valid = (archive.split == "train") & (archive.known_status == "known")
    validate_train_only_fit_mask(archive.split, archive.known_status, valid)
    with pytest.raises(ValueError, match="validation"):
        validate_train_only_fit_mask(
            archive.split,
            archive.known_status,
            valid | (archive.split == "validation"),
        )
    with pytest.raises(ValueError, match="unknown"):
        validate_train_only_fit_mask(
            np.asarray(["train", "train"]),
            np.asarray(["known", "unknown"]),
            np.asarray([True, True]),
        )


def test_thresholding_metrics_and_prediction_export(tmp_path) -> None:
    archive = _toy_archive()
    scores = np.where(archive.known_status == "known", 0.1, 1.0)
    metrics = _score_metrics(archive, scores, method_id="toy", output_dir=tmp_path)
    assert metrics["auroc_known_unknown"] == pytest.approx(1.0)
    assert (tmp_path / "toy_predictions.csv").exists()


def test_geometry_attractors_and_compactness_are_written_from_train_statistics() -> None:
    archive = _toy_archive()
    geometry = compute_embedding_geometry(archive)
    assert "lymphocyte_neoplastic" in geometry["nearest_lookup"]
    assert geometry["nearest_lookup"]["lymphocyte_neoplastic"] == "lymphocyte"
    assert set(geometry["compactness"]["class"]) == set(DEFAULT_KNOWN_CLASSES)
    assert np.isfinite(geometry["class_distance"].to_numpy()).all()
    assert "neutrophil_segmented" in geometry["attractor"].columns


def test_bootstrap_and_progression_rule() -> None:
    y_unknown = np.asarray([0, 0, 0, 1, 1, 1])
    baseline = np.asarray([0.1, 0.2, 0.9, 0.3, 0.5, 0.8])
    candidate = np.asarray([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    report = paired_bootstrap_deltas(
        y_unknown,
        baseline,
        candidate,
        n_bootstraps=50,
        seed=37,
    )
    assert report["delta_auroc"]["estimate"] > 0

    comparison = pd.DataFrame(
        [
            {"method_id": "msp_primary", "primary": True, "AUROC": 0.80, "FPR95": 0.60},
            {"method_id": "knn_cosine_k5", "primary": True, "AUROC": 0.81, "FPR95": 0.50},
        ]
    )
    subgroup = pd.DataFrame(
        [
            {"group": "lymphoid_related", "method": "knn_cosine_k5", "delta_AUROC": 0.01},
        ]
    )
    decision = apply_progression_rule(comparison, subgroup, {"knn_cosine_k5": report})
    assert decision["criterion_b"] == "PASS"


def test_predeclared_matrix_documents_primary_methods_and_skipped_dice(tmp_path) -> None:
    output = write_predeclared_matrix(tmp_path / "delivery4_predeclared_matrix.json")
    text = output.read_text(encoding="utf-8")
    assert "larger means more UNKNOWN-like" in text
    assert '"method": "knn_cosine"' in text
    assert '"variant": "k=5"' in text
    assert '"method": "dice"' in text
    assert '"status": "not implemented"' in text
