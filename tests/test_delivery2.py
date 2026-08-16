from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from hemato_osr.data.manifest import MANIFEST_COLUMNS, save_manifest
from hemato_osr.data.near_duplicates import (
    NearDuplicateConfig,
    build_candidate_components,
    generate_candidate_pairs,
    normalized_correlation,
    structural_similarity_index,
)
from hemato_osr.data.split_v2 import ConservativeSplitConfig, build_conservative_split_v2
from hemato_osr.evaluation.delivery2 import paired_bootstrap_delta, per_unknown_class_metrics
from hemato_osr.models.tda_baseline import TDAOnlyExperimentConfig, evaluate_tda_only
from hemato_osr.topology.archive import load_diagram_archive
from hemato_osr.topology.pipeline import (
    TDADiagramExtractConfig,
    TDAVectorizeConfig,
    extract_diagram_archive,
    vectorize_diagram_archive,
)
from hemato_osr.topology.vectorizers import PersistenceEntropy


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((24, 24), value, dtype=np.uint8)).save(path)


def _small_manifest(tmp_path: Path) -> pd.DataFrame:
    rows = []
    labels = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil_segmented"]
    for label_idx, label in enumerate(labels):
        for split_idx, split in enumerate(["train", "validation", "test"]):
            path = tmp_path / label / f"{split}.png"
            _write_image(path, 20 + label_idx * 30 + split_idx)
            checksum = f"{label_idx}-{split_idx}"
            rows.append(
                {
                    "sample_id": f"{label}_{split}",
                    "path": str(path),
                    "dataset": "synthetic",
                    "original_label": label,
                    "canonical_label": label,
                    "known_status": "known",
                    "split": split,
                    "group_id": "",
                    "checksum": checksum,
                    "width": 24,
                    "height": 24,
                }
            )
    for idx, label in enumerate(["myeloblast", "promyelocyte"]):
        path = tmp_path / label / "test.png"
        _write_image(path, 200 + idx)
        rows.append(
            {
                "sample_id": f"{label}_test",
                "path": str(path),
                "dataset": "synthetic",
                "original_label": label,
                "canonical_label": label,
                "known_status": "unknown",
                "split": "test",
                "group_id": "",
                "checksum": f"u-{idx}",
                "width": 24,
                "height": 24,
            }
        )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def test_near_duplicate_candidates_and_components(tmp_path: Path) -> None:
    first = tmp_path / "a" / "first.png"
    second = tmp_path / "a" / "second.png"
    third = tmp_path / "b" / "third.png"
    _write_image(first, 80)
    _write_image(second, 80)
    _write_image(third, 170)
    frame = pd.DataFrame(
        [
            ["s1", str(first), "synthetic", "a", "basophil", "known", "train", "", "same", 24, 24],
            ["s2", str(second), "synthetic", "a", "basophil", "known", "test", "", "same", 24, 24],
            [
                "s3",
                str(third),
                "synthetic",
                "b",
                "eosinophil",
                "known",
                "test",
                "",
                "other",
                24,
                24,
            ],
        ],
        columns=MANIFEST_COLUMNS,
    )
    cfg = NearDuplicateConfig(candidate_hamming_max=0, max_pairwise_pairs=10)
    candidates = generate_candidate_pairs(frame, cfg)
    assert {"sample_id_a", "sample_id_b", "ssim", "normalized_correlation"} <= set(candidates)
    assert (candidates["checksum_equal"] == True).any()  # noqa: E712
    assert "confirmed" in set(candidates["evidence_level"])
    components = build_candidate_components(frame, candidates, cfg)
    pair_component = components.loc[components["sample_id"].isin(["s1", "s2"]), "component_id"]
    assert pair_component.nunique() == 1


def test_structural_similarity_and_correlation_are_bounded() -> None:
    arr = np.eye(8)
    assert structural_similarity_index(arr, arr) > 0.99
    assert normalized_correlation(arr, arr) > 0.99


def test_conservative_split_v2_keeps_component_together(tmp_path: Path) -> None:
    frame = _small_manifest(tmp_path)
    components = pd.DataFrame(
        [
            {
                "component_id": "ndc_000000",
                "sample_id": "basophil_train",
                "component_size": 2,
                "evidence_strength": 3,
            },
            {
                "component_id": "ndc_000000",
                "sample_id": "basophil_test",
                "component_size": 2,
                "evidence_strength": 3,
            },
        ]
    )
    result = build_conservative_split_v2(frame, components, ConservativeSplitConfig(seed=37))
    splits = result.set_index("sample_id")["split"]
    assert splits["basophil_train"] == splits["basophil_test"]
    assert set(result.loc[result["known_status"] == "unknown", "split"]) == {"test"}


def test_tda_infinite_policy_and_vectorizer() -> None:
    diagram = {0: np.asarray([[0.0, np.inf], [0.1, 0.5]]), 1: np.asarray([[0.2, 0.4]])}
    features = PersistenceEntropy().fit_transform([diagram])
    assert features.shape == (1, 2)
    assert np.isfinite(features).all()


def test_tda_archive_vectorize_and_tda_only(tmp_path: Path) -> None:
    frame = _small_manifest(tmp_path / "raw")
    manifest_path = tmp_path / "manifest.csv"
    save_manifest(frame, manifest_path)
    diagram_path = tmp_path / "diagrams.npz"
    extract_diagram_archive(
        TDADiagramExtractConfig(
            manifest_path=manifest_path,
            output_path=diagram_path,
            image_size=8,
            filtrations=("sublevel", "superlevel"),
            workers=1,
        )
    )
    archive = load_diagram_archive(diagram_path)
    assert archive.filtrations == ("sublevel", "superlevel")
    features_path = tmp_path / "features.npz"
    vectorize_diagram_archive(
        TDAVectorizeConfig(
            diagram_path=diagram_path,
            output_path=features_path,
            vectorizers=("persistence-entropy", "betti-curve"),
        )
    )
    output = evaluate_tda_only(
        TDAOnlyExperimentConfig(
            features_path=features_path,
            output_dir=tmp_path / "tda_eval",
            feature_components=("sublevel_persistence-entropy", "superlevel_persistence-entropy"),
        )
    )
    metrics = json.loads(output.read_text())
    assert metrics["feature_dimension"] == 4
    assert "open_set" in metrics


def test_per_unknown_and_paired_bootstrap() -> None:
    baseline = pd.DataFrame(
        {
            "sample_id": ["k1", "k2", "u1", "u2"],
            "true_label": ["basophil", "eosinophil", "myeloblast", "myeloblast"],
            "known_status": ["known", "known", "unknown", "unknown"],
            "closed_set_prediction": ["basophil", "eosinophil", "basophil", "eosinophil"],
            "unknown_score": [0.1, 0.2, 0.8, 0.9],
        }
    )
    candidate = baseline.copy()
    candidate["unknown_score"] = [0.05, 0.1, 0.9, 0.95]
    per_class = per_unknown_class_metrics(baseline, method="demo")
    delta = paired_bootstrap_delta(baseline, candidate, metric="auroc", n_bootstraps=20, seed=37)
    assert per_class.loc[0, "unknown_class"] == "myeloblast"
    assert delta["metric"] == "delta_auroc"
