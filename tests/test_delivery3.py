from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from hemato_osr.experiments.delivery3 import (
    UNKNOWN_GROUPS,
    empirical_percentile_scores,
    fixed_alpha_late_fusion,
    max_late_fusion,
    paired_bootstrap_deltas,
    representative_indices_by_score,
    sha256_file,
)
from hemato_osr.models.backbones import TinyCNN
from hemato_osr.topology.activation_maps import (
    ActivationCollector,
    activation_energy_map,
    normalize_activation_map,
    resolve_layer,
)
from hemato_osr.topology.cache import config_hash
from hemato_osr.topology.diagrams import TDAConfig, compute_diagram_from_array
from hemato_osr.topology.morphology import (
    MorphologyConfig,
    mask_distance_map,
    segment_morphology,
)


def _synthetic_cell(path) -> None:
    size = 96
    yy, xx = np.mgrid[:size, :size]
    image = np.full((size, size, 3), 245, dtype=np.uint8)
    cell = (yy - 48) ** 2 + (xx - 48) ** 2 <= 34**2
    nucleus = (yy - 45) ** 2 + (xx - 50) ** 2 <= 15**2
    image[cell] = np.asarray([205, 170, 215], dtype=np.uint8)
    image[nucleus] = np.asarray([70, 45, 155], dtype=np.uint8)
    Image.fromarray(image).save(path)


def test_morphology_candidates_and_cytoplasm_relation(tmp_path) -> None:
    path = tmp_path / "cell.png"
    _synthetic_cell(path)
    result = segment_morphology(path, MorphologyConfig())
    assert result.status in {"valid", "warning"}
    assert result.whole_cell.any()
    assert result.nucleus.any()
    assert np.all(result.cytoplasm <= result.whole_cell)
    assert not np.any(result.cytoplasm & result.nucleus)
    assert result.foreground_fraction > result.nucleus_fraction


def test_distance_transform_filtration_and_known_topology() -> None:
    yy, xx = np.mgrid[:48, :48]
    disk = (yy - 24) ** 2 + (xx - 24) ** 2 <= 12**2
    two_disks = ((yy - 16) ** 2 + (xx - 16) ** 2 <= 6**2) | (
        (yy - 32) ** 2 + (xx - 32) ** 2 <= 6**2
    )
    ring = disk & ((yy - 24) ** 2 + (xx - 24) ** 2 >= 6**2)
    disk_map = 1.0 - disk.astype(float)
    two_disk_map = 1.0 - two_disks.astype(float)
    ring_map = 1.0 - ring.astype(float)
    disk_diagram = compute_diagram_from_array(disk_map, TDAConfig(image_size=48))
    two_disk_diagram = compute_diagram_from_array(two_disk_map, TDAConfig(image_size=48))
    ring_diagram = compute_diagram_from_array(ring_map, TDAConfig(image_size=48))
    assert disk_diagram[0].shape[1] == 2
    assert len(two_disk_diagram[0]) >= len(disk_diagram[0])
    assert ring_diagram[0].shape[1] == 2
    assert len(ring_diagram[1]) >= len(disk_diagram[1])
    distance = mask_distance_map(disk)
    assert distance.max() == 1.0
    assert distance[24, 24] > distance[24, 12]


def test_activation_hooks_energy_and_normalization() -> None:
    model = TinyCNN(num_classes=2)
    layer = resolve_layer(model, "features")
    assert layer is model.features
    with ActivationCollector(model, "features") as collector:
        with torch.inference_mode():
            _ = model(torch.ones(2, 3, 32, 32))
    assert collector.output is not None
    energy = activation_energy_map(collector.output)
    assert energy.ndim == 3
    normalized = normalize_activation_map(energy[0])
    assert normalized.shape == energy[0].shape
    assert 0.0 <= float(normalized.min()) <= float(normalized.max()) <= 1.0


def test_score_orientation_calibration_and_late_fusion() -> None:
    reference = np.asarray([0.1, 0.2, 0.4, 0.8])
    scores = np.asarray([0.05, 0.2, 0.9])
    calibrated = empirical_percentile_scores(scores, reference)
    assert calibrated.tolist() == [0.0, 0.5, 1.0]
    fused = fixed_alpha_late_fusion(calibrated, np.asarray([1.0, 0.5, 0.0]), 0.5)
    assert np.allclose(fused, [0.5, 0.5, 0.5])
    max_fused = max_late_fusion(calibrated, np.asarray([1.0, 0.4, 0.0]))
    assert np.allclose(max_fused, [1.0, 0.5, 1.0])


def test_subgroup_mapping_is_predeclared() -> None:
    assert "neutrophil_band" in UNKNOWN_GROUPS["immature_related_myeloid"]
    assert "lymphocyte_neoplastic" in UNKNOWN_GROUPS["lymphoid_related"]
    assert "normoblast" in UNKNOWN_GROUPS["other"]


def test_representative_selection_uses_score_quantiles() -> None:
    labels = np.asarray(["a", "b", "a", "a", "b"])
    scores = np.asarray([0.4, 0.0, 0.1, 0.9, 0.2])
    selected = representative_indices_by_score(labels, scores, "a")
    assert selected == [2, 0, 3]


def test_paired_bootstrap_deltas_preserve_pairing() -> None:
    y_unknown = np.asarray([0, 0, 0, 1, 1, 1])
    baseline = np.asarray([0.1, 0.2, 0.8, 0.3, 0.4, 0.9])
    candidate = np.asarray([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    report = paired_bootstrap_deltas(
        y_unknown,
        baseline,
        candidate,
        n_bootstraps=50,
        seed=37,
    )
    assert report["delta_auroc"]["estimate"] > 0
    assert report["delta_auroc"]["n_bootstraps"] == 50


def test_feature_map_cache_identity_changes_with_checkpoint_hash(tmp_path) -> None:
    checkpoint_a = tmp_path / "a.pt"
    checkpoint_b = tmp_path / "b.pt"
    checkpoint_a.write_bytes(b"checkpoint-a")
    checkpoint_b.write_bytes(b"checkpoint-b")
    metadata_a = {
        "checkpoint_sha256": sha256_file(checkpoint_a),
        "layer": "layer2",
        "activation": "rms",
    }
    metadata_b = {
        "checkpoint_sha256": sha256_file(checkpoint_b),
        "layer": "layer2",
        "activation": "rms",
    }
    assert metadata_a["checkpoint_sha256"] != metadata_b["checkpoint_sha256"]
    assert config_hash(metadata_a) != config_hash(metadata_b)
