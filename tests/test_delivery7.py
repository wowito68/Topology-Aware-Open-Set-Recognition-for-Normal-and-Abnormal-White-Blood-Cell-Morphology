from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from hemato_osr.experiments.delivery4 import EmbeddingArchive
from hemato_osr.experiments.delivery7 import (
    assert_no_external_fit,
    betti_curve,
    bottleneck_distance,
    build_external_manifest,
    cosine_distance_matrix,
    diagram_summary,
    fixed_vr_sample_ids,
    rips_diagrams,
    write_vr_sample_manifest,
)


def _write_taxonomy(path: Path) -> None:
    path.write_text(
        """
dataset: AML-Cytomorphology_LMU
mapping:
  BAS:
    canonical_label: basophil
    status: known
    known_or_unknown: known
  NGB:
    canonical_label: neutrophil_band
    status: unknown_morphology
    known_or_unknown: unknown
excluded_labels:
  UNC:
    canonical_label: null
    status: excluded_ambiguous
    known_or_unknown: excluded
""",
        encoding="utf-8",
    )


def test_external_manifest_mapping_and_ambiguous_exclusion(tmp_path: Path) -> None:
    data_root = tmp_path / "aml"
    (data_root / "BAS").mkdir(parents=True)
    (data_root / "NGB").mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(data_root / "BAS" / "BAS_0001.tiff")
    Image.new("RGB", (8, 8), color=(40, 50, 60)).save(data_root / "NGB" / "NGB_0001.tiff")
    annotations = tmp_path / "annotations.zip"
    with zipfile.ZipFile(annotations, "w") as archive:
        archive.writestr(
            "annotations.dat",
            "\n".join(
                [
                    "BAS/BAS_0001.tiff BAS nan nan",
                    "NGB/NGB_0001.tiff NGB nan nan",
                    "UNC/UNC_0001.tiff UNC nan nan",
                ]
            ),
        )
    taxonomy = tmp_path / "taxonomy.yaml"
    _write_taxonomy(taxonomy)
    manifest_path = tmp_path / "manifest.csv"
    qc_path = tmp_path / "qc.csv"

    build_external_manifest(
        data_root=data_root,
        annotations_path=annotations,
        taxonomy_path=taxonomy,
        output_path=manifest_path,
        qc_path=qc_path,
    )

    manifest = pd.read_csv(manifest_path)
    qc = pd.read_csv(qc_path)
    assert set(manifest["canonical_label"]) == {"basophil", "neutrophil_band"}
    assert set(manifest["known_status"]) == {"known", "unknown"}
    assert "UNC" in set(qc["source_label"])
    assert len(manifest) == 2


def test_no_external_fit_rejects_non_test_split() -> None:
    archive = EmbeddingArchive(
        sample_id=np.asarray(["a"]),
        true_label=np.asarray(["basophil"]),
        known_status=np.asarray(["known"]),
        split=np.asarray(["train"]),
        embedding=np.zeros((1, 2)),
        logits=np.zeros((1, 2)),
        prediction=np.asarray(["basophil"]),
        label_to_index={"basophil": 0},
        manifest_hash="hash",
    )
    with pytest.raises(ValueError, match="non-test"):
        assert_no_external_fit(archive)


def test_cosine_distance_matrix_properties() -> None:
    values = np.asarray([[3.0, 0.0], [0.0, 4.0], [3.0, 0.0]])
    distances = cosine_distance_matrix(values)
    assert np.allclose(np.diag(distances), 0.0)
    assert np.allclose(distances, distances.T)
    assert distances[0, 2] == pytest.approx(0.0)
    assert distances[0, 1] == pytest.approx(1.0)


def test_vr_sample_ids_are_reproducible_and_class_scoped() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(20)],
            "canonical_label": ["a"] * 10 + ["b"] * 10,
        }
    )
    first = fixed_vr_sample_ids(frame, dataset="synthetic", class_name="a", n_vr=5, seed=2026)
    second = fixed_vr_sample_ids(frame, dataset="synthetic", class_name="a", n_vr=5, seed=2026)
    other = fixed_vr_sample_ids(frame, dataset="synthetic", class_name="b", n_vr=5, seed=2026)
    assert first == second
    assert len(first) == 5
    assert set(first).isdisjoint(other)


def test_write_vr_sample_manifest_records_available_n(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(7)],
            "canonical_label": ["a"] * 4 + ["b"] * 3,
        }
    )
    output = tmp_path / "vr_sample_manifest.csv"

    write_vr_sample_manifest(
        {"synthetic": frame},
        output,
        classes=("a", "b"),
        n_vr=3,
        seed=2026,
    )

    manifest = pd.read_csv(output)
    assert set(manifest["dataset"]) == {"synthetic"}
    assert set(manifest["class"]) == {"a", "b"}
    assert manifest.loc[manifest["class"] == "a", "available_n"].unique().tolist() == [4]
    assert len(manifest.loc[manifest["class"] == "a"]) == 3


def test_betti_curve_counts_alive_intervals() -> None:
    diagram = np.asarray([[0.0, 1.0], [0.25, 0.75], [0.5, np.inf]])
    grid = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])

    curve = betti_curve(diagram, grid)

    assert curve.tolist() == [1.0, 2.0, 3.0, 2.0, 1.0]


def test_rips_circle_has_h1_and_bottleneck_identity_zero() -> None:
    theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
    diagrams = rips_diagrams(circle, max_edge_length=2.0, max_dimension=1)
    h1_summary = diagram_summary(diagrams[1], homology_dim=1)
    assert h1_summary["n_finite_bars"] >= 1
    assert bottleneck_distance(diagrams[1], diagrams[1]) == pytest.approx(0.0)


def test_single_cluster_has_weak_h1_relative_to_circle() -> None:
    rng = np.random.default_rng(13)
    cluster = np.asarray([1.0, 0.0, 0.0]) + rng.normal(scale=0.02, size=(24, 3))
    theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
    cluster_h1 = diagram_summary(rips_diagrams(cluster, max_edge_length=2.0)[1], homology_dim=1)
    circle_h1 = diagram_summary(rips_diagrams(circle, max_edge_length=2.0)[1], homology_dim=1)
    assert circle_h1["max_persistence"] >= cluster_h1["max_persistence"]
