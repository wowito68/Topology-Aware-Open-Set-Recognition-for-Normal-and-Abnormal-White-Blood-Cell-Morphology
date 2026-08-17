from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from hemato_osr.experiments.delivery6 import (
    ARCFACE_CONFIG_HASH,
    CE_CONFIG_HASH,
    DELIVERY6_SEEDS,
    LOCKED_ARCFACE_CONFIG,
    assert_locked_configs,
    assert_resume_compatible,
    build_run_matrix,
    decision_from_deltas,
    matched_seed_deltas,
    multiseed_summary,
    run_metadata,
    t_ci,
)


def _run_level_frame(v2_delta: float = 0.0) -> pd.DataFrame:
    rows = []
    for split in ["v1", "v2"]:
        for seed in DELIVERY6_SEEDS:
            ce_auroc = 0.86 + seed * 0.00001
            arc_delta = 0.03 if split == "v1" else v2_delta
            for representation in ["ce", "arcface"]:
                is_arc = representation == "arcface"
                rows.append(
                    {
                        "split": split,
                        "manifest_hash": "hash",
                        "seed": seed,
                        "representation": representation,
                        "checkpoint_hash": f"{split}-{representation}-{seed}",
                        "best_epoch": 1.0,
                        "closed_macro_f1": 0.972 if not is_arc else 0.971,
                        "closed_balanced_accuracy": 0.96,
                        "msp_auroc": ce_auroc + (arc_delta if is_arc else 0.0),
                        "msp_fpr95": 0.55 + (-0.10 if is_arc and split == "v1" else 0.0),
                        "msp_oscr": 0.85 + (0.02 if is_arc else 0.0),
                        "vim_auroc": 0.87 + (0.01 if is_arc else 0.0),
                        "vim_fpr95": 0.51 + (-0.02 if is_arc else 0.0),
                        "vim_oscr": 0.86,
                        "fisher_ratio": 2.6 + (0.4 if is_arc else 0.0),
                        "within_dispersion": 6.8 + (-0.5 if is_arc else 0.0),
                        "between_separation": 18.0,
                        "cosine_compactness": 0.91,
                        "mean_unknown_nearest_distance": 7.0 + (0.2 if is_arc else 0.0),
                        "training_seconds": 1.0,
                        "peak_vram_mb": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def _per_unknown_frame() -> pd.DataFrame:
    rows = []
    unknowns = ["neutrophil_band", "hairy_cell"]
    for split in ["v1", "v2"]:
        for seed in DELIVERY6_SEEDS:
            for representation in ["ce", "arcface"]:
                for method in ["msp", "vim"]:
                    for unknown in unknowns:
                        rows.append(
                            {
                                "split": split,
                                "seed": seed,
                                "representation": representation,
                                "osr_method": method,
                                "unknown_class": unknown,
                                "group": "immature_related_myeloid"
                                if unknown == "neutrophil_band"
                                else "lymphoid_related",
                                "n": 10,
                                "AUROC": 0.8 + (0.05 if representation == "arcface" else 0.0),
                                "FPR95": 0.5 + (-0.05 if representation == "arcface" else 0.0),
                                "mean_score": 0.0,
                                "median_score": 0.0,
                                "nearest_known_class": "lymphocyte",
                                "delta_AUROC_vs_CE": 0.0,
                                "delta_FPR95_vs_CE": 0.0,
                            }
                        )
    return pd.DataFrame(rows)


def test_delivery6_run_matrix_is_exact() -> None:
    matrix = build_run_matrix()
    assert len(matrix) == 20
    assert DELIVERY6_SEEDS == (13, 37, 73, 101, 137)
    assert {(spec.split, spec.seed, spec.representation) for spec in matrix} == {
        (split, seed, representation)
        for split in ["v1", "v2"]
        for seed in DELIVERY6_SEEDS
        for representation in ["ce", "arcface"]
    }


def test_delivery6_locked_configs_are_immutable() -> None:
    assert_locked_configs()
    assert CE_CONFIG_HASH == "2d9d5aa0027e6f49"
    assert ARCFACE_CONFIG_HASH == "bda304d847e6cd2c"
    assert LOCKED_ARCFACE_CONFIG["margin"] == 0.30
    assert LOCKED_ARCFACE_CONFIG["scale"] == 30.0


def test_resume_metadata_rejects_cross_seed_reuse(tmp_path) -> None:
    spec = build_run_matrix()[0]
    metadata = run_metadata(spec)
    metadata["seed"] = 999
    path = tmp_path / "run_metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Resume metadata mismatch"):
        assert_resume_compatible(path, spec)


def test_t_ci_uses_seed_level_t_interval() -> None:
    low, high = t_ci(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert low < 3.0 < high
    assert high - low > 0


def test_matched_seed_deltas_pair_ce_and_arcface() -> None:
    matched = matched_seed_deltas(_run_level_frame(), _per_unknown_frame())
    v1_auroc = matched[(matched["split"] == "v1") & (matched["metric"] == "msp_auroc")]
    assert len(v1_auroc) == 5
    assert np.allclose(v1_auroc["delta"], 0.03)


def test_multiseed_summary_has_required_statistics() -> None:
    summary = multiseed_summary(_run_level_frame())
    row = summary[
        (summary["split"] == "v1")
        & (summary["representation"] == "arcface")
        & (summary["metric"] == "msp_auroc")
    ].iloc[0]
    assert row["n"] == 5
    assert row["ci95_low"] <= row["mean"] <= row["ci95_high"]


def test_decision_detects_split_sensitivity() -> None:
    run_level = _run_level_frame(v2_delta=-0.05)
    matched = matched_seed_deltas(run_level, _per_unknown_frame())
    decision = decision_from_deltas(run_level, matched)
    assert decision["criterion_a"] == "PASS"
    assert decision["criterion_c"] == "PASS"
    assert decision["criterion_d"] == "FAIL"
    assert decision["decision"] == "ArcFace improves V1 but is split-sensitive"
