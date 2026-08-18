#!/usr/bin/env python
# ruff: noqa: E501
"""Consolidate Delivery 7.5 article-direction evidence from existing artifacts.

This script only reads completed Delivery 2-7 outputs and writes lightweight
summary/audit tables. It does not train, tune, re-export embeddings, or recompute
Vietoris-Rips filtrations.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(".")
METRICS7 = ROOT / "artifacts" / "metrics" / "delivery7"
TOPO7 = ROOT / "artifacts" / "topology" / "delivery7"
FIGURES = ROOT / "artifacts" / "figures"
REPORT = METRICS7 / "delivery75_final_scientific_consolidation.md"

KNOWN_CLASSES = [
    "basophil",
    "eosinophil",
    "lymphocyte",
    "monocyte",
    "neutrophil_segmented",
]
PRIMARY_SYSTEM = ("ce", "vim")


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def ci95(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (math.nan, math.nan)
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))
    mean = float(arr.mean())
    sem = float(stats.sem(arr))
    half_width = float(stats.t.ppf(0.975, len(arr) - 1) * sem)
    return (mean - half_width, mean + half_width)


def summarize(
    frame: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
    *,
    metric_name_col: str = "metric",
    n_name: str = "n_seeds",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys, strict=True))
        for metric in metric_cols:
            values = group[metric].astype(float).to_numpy()
            low, high = ci95(values)
            rows.append(
                {
                    **base,
                    metric_name_col: metric,
                    "mean": float(np.nanmean(values)),
                    "sd": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(np.nanmedian(values)),
                    "min": float(np.nanmin(values)),
                    "max": float(np.nanmax(values)),
                    "ci95_low": low,
                    "ci95_high": high,
                    n_name: int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def format_ci(row: pd.Series, digits: int = 3) -> str:
    return f"{row['mean']:.{digits}f} +/- {row['sd']:.{digits}f} [{row['ci95_low']:.{digits}f}, {row['ci95_high']:.{digits}f}]"


def markdown_table(frame: pd.DataFrame, *, index: bool = False) -> str:
    display = frame.reset_index() if index else frame.copy()
    display.columns = [str(col) for col in display.columns]
    rows = [display.columns.tolist()]
    rows.extend([[str(value) for value in row] for row in display.to_numpy().tolist()])
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |"

    header = fmt(rows[0])
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [fmt(row) for row in rows[1:]]
    return "\n".join([header, separator, *body])


def make_final_scientific_summary() -> pd.DataFrame:
    external = read_csv(METRICS7 / "external_multiseed_summary.csv").copy()
    external.insert(0, "domain", "AML_LMU_external")
    external = external.rename(columns={"n_seeds": "n_seeds"})
    external = external[
        [
            "domain",
            "training_split",
            "representation",
            "osr_method",
            "metric",
            "mean",
            "sd",
            "ci95_low",
            "ci95_high",
            "n_seeds",
        ]
    ]

    d6 = read_csv("artifacts/metrics/delivery6/run_level_results.csv")
    internal_rows: list[dict[str, object]] = []
    metric_specs = [
        ("closed_macro_f1", "closed", "closed_macro_f1"),
        ("closed_balanced_accuracy", "closed", "closed_balanced_accuracy"),
        ("msp_auroc", "msp", "AUROC"),
        ("msp_fpr95", "msp", "FPR95"),
        ("msp_oscr", "msp", "OSCR"),
        ("vim_auroc", "vim", "AUROC"),
        ("vim_fpr95", "vim", "FPR95"),
        ("vim_oscr", "vim", "OSCR"),
    ]
    for (split, rep), group in d6.groupby(["split", "representation"]):
        for source_metric, method, metric in metric_specs:
            low, high = ci95(group[source_metric])
            values = group[source_metric].astype(float).to_numpy()
            internal_rows.append(
                {
                    "domain": "MLL23_internal",
                    "training_split": split,
                    "representation": rep,
                    "osr_method": method,
                    "metric": metric,
                    "mean": float(np.nanmean(values)),
                    "sd": float(np.nanstd(values, ddof=1)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_seeds": int(len(values)),
                }
            )
    combined = pd.concat([external, pd.DataFrame(internal_rows)], ignore_index=True)
    combined.to_csv(METRICS7 / "final_scientific_summary.csv", index=False)
    return combined


def make_closed_summary() -> pd.DataFrame:
    run = read_csv(METRICS7 / "external_run_level_results.csv")
    closed = run.drop_duplicates(["training_split", "seed", "representation"])
    out = summarize(
        closed,
        ["training_split", "representation"],
        ["closed_accuracy", "closed_balanced_accuracy", "closed_macro_f1"],
    )
    out.to_csv(METRICS7 / "delivery75_external_closed_summary.csv", index=False)
    return out


def make_seed_level_open() -> pd.DataFrame:
    run = read_csv(METRICS7 / "external_run_level_results.csv")
    out = run[
        [
            "training_split",
            "seed",
            "representation",
            "osr_method",
            "AUROC",
            "FPR95",
            "OSCR",
            "AUPR_unknown",
            "AUPR_known",
            "closed_macro_f1",
        ]
    ].sort_values(["training_split", "seed", "representation", "osr_method"])
    out.to_csv(METRICS7 / "delivery75_external_seed_level_results.csv", index=False)
    return out


def make_arcface_deltas() -> pd.DataFrame:
    deltas = read_csv(METRICS7 / "external_matched_seed_deltas.csv")
    rows: list[dict[str, object]] = []
    for keys, group in deltas.groupby(["training_split", "osr_method", "metric"]):
        split, method, metric = keys
        values = group["delta"].astype(float).to_numpy()
        low, high = ci95(values)
        favorable = int((values > 0).sum()) if metric == "AUROC" else int((values < 0).sum())
        rows.append(
            {
                "training_split": split,
                "osr_method": method,
                "metric": metric,
                "mean_delta": float(values.mean()),
                "sd_delta": float(values.std(ddof=1)),
                "ci95_low": low,
                "ci95_high": high,
                "favorable_seeds": favorable,
                "n_seeds": int(len(values)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(METRICS7 / "delivery75_arcface_external_matched_deltas.csv", index=False)
    return out


def make_domain_shift_summary() -> pd.DataFrame:
    shift = read_csv(METRICS7 / "internal_external_shift.csv")
    d6 = read_csv("artifacts/metrics/delivery6/run_level_results.csv")
    ext = read_csv(METRICS7 / "external_run_level_results.csv")
    ext_closed = ext.drop_duplicates(["training_split", "seed", "representation"])[
        ["training_split", "seed", "representation", "closed_macro_f1"]
    ].rename(columns={"closed_macro_f1": "external_closed_macro_f1"})
    int_closed = d6[["split", "seed", "representation", "closed_macro_f1"]].rename(
        columns={"split": "training_split", "closed_macro_f1": "internal_closed_macro_f1"}
    )
    closed_shift = int_closed.merge(ext_closed, on=["training_split", "seed", "representation"])
    closed_shift["Delta_domain_closed_macro_f1"] = (
        closed_shift["external_closed_macro_f1"] - closed_shift["internal_closed_macro_f1"]
    )
    merged = shift.merge(
        closed_shift,
        on=["training_split", "seed", "representation"],
        how="left",
    )
    rows: list[dict[str, object]] = []
    for keys, group in merged.groupby(["training_split", "representation", "osr_method"]):
        split, rep, method = keys
        row: dict[str, object] = {
            "training_split": split,
            "representation": rep,
            "osr_method": method,
            "system": f"{rep}+{method}",
            "n_seeds": int(len(group)),
        }
        for metric in ["AUROC", "FPR95", "closed_macro_f1"]:
            if metric == "closed_macro_f1":
                internal = group["internal_closed_macro_f1"].astype(float)
                external = group["external_closed_macro_f1"].astype(float)
                delta = group["Delta_domain_closed_macro_f1"].astype(float)
            else:
                internal = group[f"internal_{metric}"].astype(float)
                external = group[f"external_{metric}"].astype(float)
                delta = group[f"Delta_domain_{metric}"].astype(float)
            low, high = ci95(delta)
            row[f"internal_{metric}"] = float(internal.mean())
            row[f"external_{metric}"] = float(external.mean())
            row[f"delta_{metric}"] = float(delta.mean())
            row[f"delta_{metric}_sd"] = float(delta.std(ddof=1))
            row[f"delta_{metric}_ci95_low"] = low
            row[f"delta_{metric}_ci95_high"] = high
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(METRICS7 / "delivery75_internal_external_domain_shift_summary.csv", index=False)
    return out


def make_unknown_summary() -> pd.DataFrame:
    per = read_csv(METRICS7 / "external_per_unknown_class.csv")
    primary = per[
        (per["representation"] == PRIMARY_SYSTEM[0]) & (per["osr_method"] == PRIMARY_SYSTEM[1])
    ].copy()
    rows: list[dict[str, object]] = []
    for label, group in primary.groupby("canonical_label"):
        attractor_counts = group["nearest_known_prediction"].astype(str).value_counts()
        mode = str(attractor_counts.index[0])
        rows.append(
            {
                "canonical_label": label,
                "source_label": ",".join(sorted(group["source_label"].astype(str).unique())),
                "n": int(group["n"].max()),
                "AUROC_mean": float(group["AUROC"].mean()),
                "AUROC_sd": float(group["AUROC"].std(ddof=1)),
                "FPR95_mean": float(group["FPR95"].mean()),
                "FPR95_sd": float(group["FPR95"].std(ddof=1)),
                "mean_score_mean": float(group["mean_score"].mean()),
                "median_score_mean": float(group["median_score"].mean()),
                "dominant_known_prediction": mode,
                "dominant_known_prediction_fraction": float(attractor_counts.iloc[0] / len(group)),
                "n_runs": int(len(group)),
            }
        )
    out = pd.DataFrame(rows).sort_values("AUROC_mean", ascending=False)
    out.to_csv(METRICS7 / "delivery75_external_unknown_morphology_summary_ce_vim.csv", index=False)
    return out


def make_vr_summaries() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = read_csv(TOPO7 / "vr_seed_pair_bottleneck.csv")
    split = read_csv(TOPO7 / "vr_split_pair_bottleneck.csv")
    diag = read_csv(TOPO7 / "vr_diagram_summaries.csv")
    betti = read_csv(TOPO7 / "vr_betti_curves.csv")

    seed_known = seed[seed["class"].isin(KNOWN_CLASSES)]
    seed_summary = summarize(
        seed_known,
        ["dataset", "training_split", "representation"],
        ["h0_bottleneck", "h1_bottleneck"],
        n_name="n_pairs",
    )
    seed_summary.to_csv(METRICS7 / "delivery75_vr_seed_stability_summary.csv", index=False)

    split_known = split[split["class"].isin(KNOWN_CLASSES)]
    split_summary = summarize(
        split_known,
        ["dataset", "representation"],
        ["h0_bottleneck", "h1_bottleneck"],
        n_name="n_pairs",
    )
    split_summary.to_csv(METRICS7 / "delivery75_vr_split_stability_summary.csv", index=False)

    # Domain shift from already generated summaries and Betti curves. Bottleneck
    # requires local diagram caches; this lightweight table reports available
    # total-persistence deltas and Betti-curve L2 distances.
    domain_rows: list[dict[str, object]] = []
    known_diag = diag[diag["class"].isin(KNOWN_CLASSES)]
    for keys, group in known_diag.groupby(
        ["training_split", "seed", "representation", "class", "homology_dim"]
    ):
        split_name, seed_value, rep, klass, dim = keys
        by_dataset = {str(row["dataset"]): row for row in group.to_dict("records")}
        if {"MLL23_common_test", "AML_LMU_external_test"} <= set(by_dataset):
            mll = by_dataset["MLL23_common_test"]
            aml = by_dataset["AML_LMU_external_test"]
            domain_rows.append(
                {
                    "training_split": split_name,
                    "seed": seed_value,
                    "representation": rep,
                    "class": klass,
                    "homology_dim": int(dim),
                    "mll23_total_persistence": float(mll["total_persistence"]),
                    "aml_lmu_total_persistence": float(aml["total_persistence"]),
                    "delta_total_persistence": float(aml["total_persistence"])
                    - float(mll["total_persistence"]),
                    "mll23_max_persistence": float(mll["max_persistence"]),
                    "aml_lmu_max_persistence": float(aml["max_persistence"]),
                    "delta_max_persistence": float(aml["max_persistence"])
                    - float(mll["max_persistence"]),
                    "bottleneck_available": False,
                }
            )
    domain = pd.DataFrame(domain_rows)

    betti_rows: list[dict[str, object]] = []
    known_betti = betti[betti["class"].isin(KNOWN_CLASSES)]
    for keys, group in known_betti.groupby(
        ["training_split", "seed", "representation", "class", "homology_dim"]
    ):
        split_name, seed_value, rep, klass, dim = keys
        piv = group.pivot_table(index="epsilon", columns="dataset", values="betti")
        if {"MLL23_common_test", "AML_LMU_external_test"} <= set(piv.columns):
            diff = piv["AML_LMU_external_test"].to_numpy() - piv["MLL23_common_test"].to_numpy()
            betti_rows.append(
                {
                    "training_split": split_name,
                    "seed": seed_value,
                    "representation": rep,
                    "class": klass,
                    "homology_dim": int(dim),
                    "betti_l2_distance": float(np.linalg.norm(diff)),
                    "betti_mean_abs_difference": float(np.mean(np.abs(diff))),
                    "betti_max_abs_difference": float(np.max(np.abs(diff))),
                }
            )
    betti_domain = pd.DataFrame(betti_rows)
    domain = domain.merge(
        betti_domain,
        on=["training_split", "seed", "representation", "class", "homology_dim"],
        how="left",
    )
    domain.to_csv(METRICS7 / "delivery75_vr_domain_shift_summary_proxy.csv", index=False)

    domain_agg = summarize(
        domain,
        ["representation", "homology_dim"],
        ["delta_total_persistence", "delta_max_persistence", "betti_l2_distance"],
        n_name="n_cloud_pairs",
    )
    domain_agg.to_csv(METRICS7 / "delivery75_vr_domain_shift_aggregate_proxy.csv", index=False)
    return seed_summary, split_summary, domain, domain_agg


def make_case_studies() -> tuple[pd.DataFrame, pd.DataFrame]:
    per6 = read_csv("artifacts/metrics/delivery6/per_unknown_seed_results.csv")
    attract = read_csv("artifacts/metrics/delivery6/attractor_seed_results.csv")
    mixing = read_csv(TOPO7 / "vr_cross_class_mixing.csv")
    diag = read_csv(TOPO7 / "vr_diagram_summaries.csv")

    lymphoid = [
        "lymphocyte_large_granular",
        "lymphocyte_neoplastic",
        "lymphocyte_reactive",
        "hairy_cell",
    ]
    lymph_rows: list[dict[str, object]] = []
    for klass in lymphoid:
        for rep in ["ce", "arcface"]:
            sub = per6[
                (per6["unknown_class"] == klass)
                & (per6["representation"] == rep)
                & (per6["osr_method"].isin(["msp", "vim"]))
            ]
            att = attract[
                (attract["unknown_class"] == klass)
                & (attract["representation"] == rep)
                & (attract["known_class"] == "lymphocyte")
            ]
            mix = mixing[
                (mixing["dataset"] == "MLL23_common_test")
                & (mixing["class_a"] == "lymphocyte")
                & (mixing["class_b"] == klass)
                & (mixing["representation"] == rep)
            ]
            vr = diag[
                (diag["dataset"] == "MLL23_common_test")
                & (diag["class"] == klass)
                & (diag["representation"] == rep)
                & (diag["homology_dim"] == 1)
            ]
            lymph_rows.append(
                {
                    "unknown_class": klass,
                    "representation": rep,
                    "msp_AUROC_mean": float(sub[sub["osr_method"] == "msp"]["AUROC"].mean()),
                    "vim_AUROC_mean": float(sub[sub["osr_method"] == "vim"]["AUROC"].mean()),
                    "msp_FPR95_mean": float(sub[sub["osr_method"] == "msp"]["FPR95"].mean()),
                    "vim_FPR95_mean": float(sub[sub["osr_method"] == "vim"]["FPR95"].mean()),
                    "attraction_to_lymphocyte_mean": float(att["fraction"].mean()),
                    "cross_neighbor_fraction_mean": float(mix["cross_neighbor_fraction"].mean()),
                    "centroid_distance_mean": float(mix["centroid_distance"].mean()),
                    "mean_knn_distance_mean": float(mix["mean_knn_distance"].mean()),
                    "h1_max_persistence_mean": float(vr["max_persistence"].mean()),
                    "h1_total_persistence_mean": float(vr["total_persistence"].mean()),
                }
            )
    lymph = pd.DataFrame(lymph_rows)
    lymph.to_csv(METRICS7 / "delivery75_lymphocyte_failure_summary.csv", index=False)

    nb_rows: list[dict[str, object]] = []
    for rep in ["ce", "arcface"]:
        for split_name in ["v1", "v2"]:
            sub = per6[
                (per6["unknown_class"] == "neutrophil_band")
                & (per6["representation"] == rep)
                & (per6["split"] == split_name)
                & (per6["osr_method"].isin(["msp", "vim"]))
            ]
            att = attract[
                (attract["unknown_class"] == "neutrophil_band")
                & (attract["representation"] == rep)
                & (attract["split"] == split_name)
                & (attract["known_class"] == "neutrophil_segmented")
            ]
            mix = mixing[
                (mixing["dataset"] == "MLL23_common_test")
                & (mixing["class_a"] == "neutrophil_segmented")
                & (mixing["class_b"] == "neutrophil_band")
                & (mixing["representation"] == rep)
                & (mixing["training_split"] == split_name)
            ]
            vr = diag[
                (diag["dataset"] == "MLL23_common_test")
                & (diag["class"].isin(["neutrophil_segmented", "neutrophil_band"]))
                & (diag["representation"] == rep)
                & (diag["training_split"] == split_name)
            ]
            nb_rows.append(
                {
                    "training_split": split_name,
                    "representation": rep,
                    "msp_AUROC_mean": float(sub[sub["osr_method"] == "msp"]["AUROC"].mean()),
                    "vim_AUROC_mean": float(sub[sub["osr_method"] == "vim"]["AUROC"].mean()),
                    "msp_FPR95_mean": float(sub[sub["osr_method"] == "msp"]["FPR95"].mean()),
                    "vim_FPR95_mean": float(sub[sub["osr_method"] == "vim"]["FPR95"].mean()),
                    "attraction_to_neutrophil_segmented_mean": float(att["fraction"].mean()),
                    "cross_neighbor_fraction_mean": float(mix["cross_neighbor_fraction"].mean()),
                    "centroid_distance_mean": float(mix["centroid_distance"].mean()),
                    "mean_knn_distance_mean": float(mix["mean_knn_distance"].mean()),
                    "h0_total_persistence_mean": float(
                        vr[vr["homology_dim"] == 0]["total_persistence"].mean()
                    ),
                    "h1_total_persistence_mean": float(
                        vr[vr["homology_dim"] == 1]["total_persistence"].mean()
                    ),
                }
            )
    nb = pd.DataFrame(nb_rows)
    nb.to_csv(METRICS7 / "delivery75_neutrophil_band_summary.csv", index=False)
    return lymph, nb


def make_article_key_results(
    final_summary: pd.DataFrame,
    closed: pd.DataFrame,
    deltas: pd.DataFrame,
    domain_shift: pd.DataFrame,
    unknown: pd.DataFrame,
    seed_vr: pd.DataFrame,
    split_vr: pd.DataFrame,
    domain_vr: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(
        result_id: str,
        question: str,
        comparison: str,
        metric: str,
        value: str,
        uncertainty: str,
        interpretation: str,
        source: str,
    ) -> None:
        rows.append(
            {
                "result_id": result_id,
                "question": question,
                "comparison": comparison,
                "metric": metric,
                "value": value,
                "uncertainty": uncertainty,
                "interpretation": interpretation,
                "source_artifact": source,
            }
        )

    ext = final_summary[final_summary["domain"] == "AML_LMU_external"]
    for split_name in ["v1", "v2"]:
        for rep, method in [("ce", "vim"), ("ce", "msp"), ("arcface", "msp")]:
            auroc = ext[
                (ext["training_split"] == split_name)
                & (ext["representation"] == rep)
                & (ext["osr_method"] == method)
                & (ext["metric"] == "AUROC")
            ].iloc[0]
            fpr = ext[
                (ext["training_split"] == split_name)
                & (ext["representation"] == rep)
                & (ext["osr_method"] == method)
                & (ext["metric"] == "FPR95")
            ].iloc[0]
            add(
                f"external_{split_name}_{rep}_{method}",
                "External OSR generalization",
                f"{rep}+{method} on AML-LMU, trained on {split_name}",
                "AUROC / FPR95",
                f"{auroc['mean']:.3f} / {fpr['mean']:.3f}",
                f"AUROC CI {auroc['ci95_low']:.3f}-{auroc['ci95_high']:.3f}; FPR95 CI {fpr['ci95_low']:.3f}-{fpr['ci95_high']:.3f}",
                "External OSR is substantially weaker than internal MLL23; CE+ViM is not externally best by mean AUROC.",
                "artifacts/metrics/delivery7/final_scientific_summary.csv",
            )

    best_unknown = unknown.sort_values("AUROC_mean", ascending=False).iloc[0]
    hard_unknown = unknown.sort_values("AUROC_mean", ascending=True).iloc[0]
    add(
        "unknown_range_primary",
        "Morphology-dependent unknown difficulty",
        "CE+ViM external unknown classes",
        "AUROC range",
        f"{hard_unknown['canonical_label']} {hard_unknown['AUROC_mean']:.3f} to {best_unknown['canonical_label']} {best_unknown['AUROC_mean']:.3f}",
        "Means across 10 CE+ViM external runs",
        "Unknown rejection difficulty is strongly class-dependent.",
        "artifacts/metrics/delivery7/delivery75_external_unknown_morphology_summary_ce_vim.csv",
    )

    ce_vim_shift = domain_shift[
        (domain_shift["representation"] == "ce") & (domain_shift["osr_method"] == "vim")
    ]
    add(
        "domain_shift_ce_vim",
        "Internal to external shift",
        "CE+ViM MLL23 to AML-LMU",
        "Delta AUROC / Delta FPR95",
        f"{ce_vim_shift['delta_AUROC'].mean():.3f} / {ce_vim_shift['delta_FPR95'].mean():.3f}",
        "Mean of V1/V2 seed-level deltas",
        "Domain shift dominates the practical open-set reliability story.",
        "artifacts/metrics/delivery7/delivery75_internal_external_domain_shift_summary.csv",
    )

    af_msp = deltas[(deltas["osr_method"] == "msp") & (deltas["metric"] == "AUROC")]
    add(
        "arcface_external_msp",
        "ArcFace external comparison",
        "ArcFace MSP minus CE MSP, matched seeds",
        "Delta AUROC",
        f"V1 {af_msp[af_msp['training_split'] == 'v1']['mean_delta'].iloc[0]:.3f}; V2 {af_msp[af_msp['training_split'] == 'v2']['mean_delta'].iloc[0]:.3f}",
        "Seed-level 95% CIs in matched-delta table",
        "External evidence is mixed: ArcFace MSP helps V2 but not consistently enough to erase D6 failure.",
        "artifacts/metrics/delivery7/delivery75_arcface_external_matched_deltas.csv",
    )

    h1_seed = seed_vr[(seed_vr["metric"] == "h1_bottleneck")]
    add(
        "vr_seed_stability",
        "VR seed stability",
        "Known-class pairwise seed bottleneck",
        "H1 bottleneck",
        f"CE mean {h1_seed[h1_seed['representation'] == 'ce']['mean'].mean():.3f}; ArcFace mean {h1_seed[h1_seed['representation'] == 'arcface']['mean'].mean():.3f}",
        "Aggregated over datasets and splits",
        "ArcFace is not globally more H1 seed-unstable; signal is mixed by dataset/split/dimension.",
        "artifacts/metrics/delivery7/delivery75_vr_seed_stability_summary.csv",
    )

    split_h1 = split_vr[(split_vr["metric"] == "h1_bottleneck")]
    add(
        "vr_split_stability",
        "VR split stability",
        "V1 vs V2 same-seed bottleneck",
        "H1 bottleneck",
        f"CE mean {split_h1[split_h1['representation'] == 'ce']['mean'].mean():.3f}; ArcFace mean {split_h1[split_h1['representation'] == 'arcface']['mean'].mean():.3f}",
        "Aggregated over datasets",
        "Split-topology sensitivity is small in H1 and mixed; not a standalone causal explanation.",
        "artifacts/metrics/delivery7/delivery75_vr_split_stability_summary.csv",
    )

    add(
        "vr_domain_proxy",
        "VR domain shift",
        "MLL23 vs AML-LMU known-class Betti curves",
        "Betti L2 proxy",
        f"CE mean {domain_vr[(domain_vr['representation'] == 'ce') & (domain_vr['metric'] == 'betti_l2_distance')]['mean'].mean():.2f}; ArcFace mean {domain_vr[(domain_vr['representation'] == 'arcface') & (domain_vr['metric'] == 'betti_l2_distance')]['mean'].mean():.2f}",
        "Bottleneck domain table unavailable without local diagram cache at consolidation time",
        "Topology shows a domain signal, but current strongest claim remains domain/morphology reliability, not topological causality.",
        "artifacts/metrics/delivery7/delivery75_vr_domain_shift_aggregate_proxy.csv",
    )

    out = pd.DataFrame(rows)
    out.to_csv(METRICS7 / "article_key_results.csv", index=False)
    return out


def make_claim_matrix() -> pd.DataFrame:
    rows = [
        {
            "claim_id": "C1",
            "candidate_claim": "Closed-set WBC morphology classification remains high internally but degrades externally.",
            "support_status": "SUPPORTED",
            "supporting_result": "Internal D6 closed macro-F1 near 0.97; external closed macro-F1 about 0.71-0.75 depending representation/split.",
            "contradicting_result": "External closed accuracy can remain high due to class imbalance.",
            "external_support": "Yes",
            "recommended_wording": "Closed-set accuracy alone overstates external reliability; macro-F1/balanced accuracy reveal domain degradation.",
            "paper_section": "External Validation",
        },
        {
            "claim_id": "C2",
            "candidate_claim": "MSP is limited but externally can outperform ViM in this domain.",
            "support_status": "PARTIALLY_SUPPORTED",
            "supporting_result": "External MSP AUROC exceeds ViM for CE and ArcFace in V1/V2 means.",
            "contradicting_result": "Internal D6 favored CE+ViM stability.",
            "external_support": "Yes",
            "recommended_wording": "No OSR score is uniformly reliable; ViM stability internally does not transfer cleanly to AML-LMU.",
            "paper_section": "Open-Set Results",
        },
        {
            "claim_id": "C3",
            "candidate_claim": "ViM is the externally best open-set method.",
            "support_status": "NOT_SUPPORTED",
            "supporting_result": "Internal CE+ViM was stable.",
            "contradicting_result": "External mean AUROC is lower for ViM than MSP in both CE and ArcFace systems.",
            "external_support": "No",
            "recommended_wording": "CE+ViM is the conservative internal baseline, not the external winner.",
            "paper_section": "Discussion",
        },
        {
            "claim_id": "C4",
            "candidate_claim": "Cubical persistent homology improves predictive OSR utility.",
            "support_status": "NOT_SUPPORTED",
            "supporting_result": "None in confirmatory artifacts.",
            "contradicting_result": "Delivery 3 decision rule stopped TDA method development.",
            "external_support": "No external predictive TDA test",
            "recommended_wording": "Cubical PH is a negative ablation and should not be promoted as a predictive method.",
            "paper_section": "Supplement",
        },
        {
            "claim_id": "C5",
            "candidate_claim": "ArcFace is superior to CE.",
            "support_status": "NOT_SUPPORTED",
            "supporting_result": "Some external ArcFace+MSP means are favorable.",
            "contradicting_result": "Delivery 6 internal multiseed/split confirmation failed; neutrophil-band degradation persisted.",
            "external_support": "Mixed",
            "recommended_wording": "ArcFace is an informative instability/negative ablation, not a confirmed main method.",
            "paper_section": "Representation Analysis",
        },
        {
            "claim_id": "C6",
            "candidate_claim": "Open-set reliability is morphology-dependent.",
            "support_status": "SUPPORTED",
            "supporting_result": "External per-unknown AUROC varies widely by mapped unknown class; internal D6 also shows class-specific failures.",
            "contradicting_result": "Small-n external classes require cautious wording.",
            "external_support": "Yes",
            "recommended_wording": "Unknown rejection difficulty is strongly morphology-dependent, with small-class uncertainty acknowledged.",
            "paper_section": "Failure Analysis",
        },
        {
            "claim_id": "C7",
            "candidate_claim": "External domain shift is a dominant limitation.",
            "support_status": "SUPPORTED",
            "supporting_result": "All major systems lose AUROC and closed macro-F1 externally and increase FPR95.",
            "contradicting_result": "Relative system ranking changes by split/method.",
            "external_support": "Yes",
            "recommended_wording": "External acquisition/domain shift materially changes both closed-set and OSR behavior.",
            "paper_section": "External Validation",
        },
        {
            "claim_id": "C8",
            "candidate_claim": "Embedding geometry explains OSR failures better than VR alone.",
            "support_status": "PARTIALLY_SUPPORTED",
            "supporting_result": "Attractor and cross-class mixing align with lymphocyte-like and neutrophil-band failures.",
            "contradicting_result": "VR captures complementary stability/domain summaries but weak OSR correlations.",
            "external_support": "Partial",
            "recommended_wording": "Conventional geometry provides the primary failure explanation; VR is a secondary descriptive lens.",
            "paper_section": "Representation and Failure Analysis",
        },
        {
            "claim_id": "C9",
            "candidate_claim": "Vietoris-Rips topology causally explains OSR instability.",
            "support_status": "NOT_SUPPORTED",
            "supporting_result": "VR artifacts show measurable seed/split/domain variation.",
            "contradicting_result": "Topology-OSR correlations are weak/inconsistent and exploratory.",
            "external_support": "Limited",
            "recommended_wording": "VR is explanatory/descriptive only; avoid causal or predictive claims.",
            "paper_section": "Supplement or Secondary Analysis",
        },
        {
            "claim_id": "C10",
            "candidate_claim": "The system externally generalizes robustly.",
            "support_status": "NOT_SUPPORTED",
            "supporting_result": "Some closed accuracy remains high.",
            "contradicting_result": "External AUROC is modest and FPR95 high for all main systems.",
            "external_support": "No",
            "recommended_wording": "External validation reveals limited OSR generalization under domain shift.",
            "paper_section": "Discussion",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(METRICS7 / "article_claim_evidence_matrix.csv", index=False)
    return out


def make_figure_inventory() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    main_candidates = {
        "artifacts/figures/delivery6/auroc_distribution_by_representation_split.png",
        "artifacts/figures/delivery6/matched_delta_auroc.png",
        "artifacts/figures/delivery6/neutrophil_band_by_seed.png",
        "artifacts/figures/delivery6/unknown_attractor_stability.png",
        "artifacts/figures/delivery7/vr_h1_max_persistence_by_class.png",
        "artifacts/figures/delivery7/vr_seed_stability_h1_bottleneck.png",
        "artifacts/figures/delivery5/geometry_within_between_comparison.png",
        "artifacts/figures/delivery5/lymphoid_auroc_comparison.png",
    }
    for path in sorted(FIGURES.glob("delivery*/**/*.png")):
        delivery = path.parts[path.parts.index("figures") + 1]
        rel = str(path)
        name = path.name
        if "tda" in name or delivery in {"delivery2", "delivery3"}:
            question = "TDA negative ablation / early protocol"
            main = False
            supplement = True
            action = "Move to supplement unless needed for negative ablation summary."
        elif rel in main_candidates:
            question = "Core robustness, geometry, failure, or VR evidence"
            main = True
            supplement = False
            action = "Reuse after publication-quality redesign."
        elif delivery in {"delivery6", "delivery7"}:
            question = "Confirmatory robustness or topology support"
            main = False
            supplement = True
            action = "Supplement or combine into multi-panel figure."
        else:
            question = "Exploratory representation or OSR support"
            main = False
            supplement = True
            action = "Supplement; do not crowd main text."
        rows.append(
            {
                "figure_path": rel,
                "delivery": delivery,
                "scientific_question": question,
                "main_text_candidate": main,
                "supplement_candidate": supplement,
                "redundant": False if main else name.startswith(("pca_", "umap_")),
                "quality_issue": "Needs consistent manuscript styling" if main else "",
                "recommended_action": action,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(METRICS7 / "article_figure_inventory.csv", index=False)
    return out


def row_lookup(
    summary: pd.DataFrame, domain: str, split: str, rep: str, method: str, metric: str
) -> pd.Series:
    return summary[
        (summary["domain"] == domain)
        & (summary["training_split"] == split)
        & (summary["representation"] == rep)
        & (summary["osr_method"] == method)
        & (summary["metric"] == metric)
    ].iloc[0]


def write_report(
    final_summary: pd.DataFrame,
    closed: pd.DataFrame,
    seed_level: pd.DataFrame,
    deltas: pd.DataFrame,
    domain_shift: pd.DataFrame,
    unknown: pd.DataFrame,
    seed_vr: pd.DataFrame,
    split_vr: pd.DataFrame,
    domain_vr_agg: pd.DataFrame,
    lymph: pd.DataFrame,
    nb: pd.DataFrame,
    claims: pd.DataFrame,
    figures: pd.DataFrame,
) -> None:
    audit = json.loads(
        (ROOT / "artifacts/data_audit/delivery7/external_dataset_audit.json").read_text()
    )
    mapping = read_csv(METRICS7 / "external_taxonomy_mapping.csv")
    corr = read_csv(METRICS7 / "vr_topology_osr_correlations.csv")
    vr_meta = json.loads((TOPO7 / "vr_metadata.json").read_text())
    d6_decision = json.loads(
        (ROOT / "artifacts/metrics/delivery6/confirmation_decision.json").read_text()
    )
    d3_decision = json.loads((ROOT / "artifacts/metrics/delivery3/decision_rule.json").read_text())

    lines: list[str] = []
    add = lines.append
    add("# Delivery 7.5 Final Scientific Consolidation")
    add("")
    add("## A. Repository State")
    add("- starting commit: 857e33b Delivery 6: reset epoch logs for fresh rebuilds")
    add("- actual Delivery 7 commits: a469d0f, a77c7ae, 567a3b1, 8487119, c5158fa")
    add("- new consolidation commit: pending if this script is committed")
    add("- worktree: clean before Delivery 7.5 source edit; generated artifacts are ignored")
    add("- ruff/mypy/pytest: see command output; expected 64 tests")
    add("- Delivery 7 was previously committed; the later D6 commit was maintenance.")
    add("")

    add("## B. External Dataset Audit")
    add(
        f"- dataset: {audit['dataset_name']}; version: {audit['version_or_release']['version_number']}"
    )
    add(
        f"- images / manifest samples: {audit['image_count']} / {audit['external_manifest']['rows']}"
    )
    add(
        f"- known mapped / unknown mapped: {audit['external_manifest']['known_rows']} / {audit['external_manifest']['unknown_rows']}"
    )
    add(
        "- excluded/ambiguous: UNC and nan have 0 samples in the gold-standard column; excluded by policy."
    )
    add(f"- patient IDs: {audit['patient_ids_available']}")
    add(f"- format/domain: {audit['image_format']}; {audit['domain_and_acquisition']}")
    add("")

    add("## C. External Taxonomy")
    add(
        markdown_table(
            mapping[["source_label", "canonical_label", "known_or_unknown", "n_samples", "reason"]]
        )
    )
    add("")

    add("## D. External Closed-Set Summary")
    closed_piv = closed.pivot_table(
        index=["training_split", "representation"],
        columns="metric",
        values=["mean", "sd", "ci95_low", "ci95_high"],
    )
    add(markdown_table(closed_piv.round(4), index=True))
    add("")

    add("## E. External Open-Set Summary")
    ext = final_summary[final_summary["domain"] == "AML_LMU_external"]
    osr_rows = ext[
        ext["metric"].isin(["AUROC", "FPR95", "OSCR", "AUPR_unknown", "AUPR_known"])
    ].copy()
    add(
        markdown_table(
            osr_rows[
                [
                    "training_split",
                    "representation",
                    "osr_method",
                    "metric",
                    "mean",
                    "sd",
                    "ci95_low",
                    "ci95_high",
                ]
            ].round(4)
        )
    )
    add("")

    add("## F. External Seed-Level Results")
    add(
        markdown_table(
            seed_level[
                ["training_split", "seed", "representation", "osr_method", "AUROC", "FPR95"]
            ].round(4)
        )
    )
    add("")

    add("## G. ArcFace vs CE External Comparison")
    add(markdown_table(deltas.round(4)))
    add("")

    add("## H. Internal -> External Domain Shift")
    add(
        markdown_table(
            domain_shift[
                [
                    "system",
                    "training_split",
                    "internal_AUROC",
                    "external_AUROC",
                    "delta_AUROC",
                    "internal_FPR95",
                    "external_FPR95",
                    "delta_FPR95",
                    "internal_closed_macro_f1",
                    "external_closed_macro_f1",
                    "delta_closed_macro_f1",
                ]
            ].round(4)
        )
    )
    add("")

    add("## I. External Unknown Morphology Analysis")
    add("- easiest CE+ViM unknowns: " + ", ".join(unknown.head(5)["canonical_label"].astype(str)))
    add(
        "- hardest CE+ViM unknowns: "
        + ", ".join(unknown.tail(5).sort_values("AUROC_mean")["canonical_label"].astype(str))
    )
    add(markdown_table(unknown.round(4)))
    add("")

    add("## J. Practical Primary System")
    add(
        "CE+MSP is the strongest practical external baseline by mean external AUROC/FPR95, but CE+ViM remains the conservative internal-stability baseline. For the article, present CE+ViM as the internally selected baseline and CE+MSP as the external comparator; do not promote a single peak run."
    )
    add("")

    add("## K. Delivery 6 ArcFace Conclusion After External Validation")
    add(
        "EXTERNAL EVIDENCE PARTIALLY REHABILITATES ARCFACE BUT DOES NOT CONFIRM IT. External ArcFace+MSP is competitive, especially V2, but D6 internal confirmation remains failed."
    )
    add(f"- D6 decision: {d6_decision.get('decision', d6_decision)}")
    add("")

    add("## L. Vietoris-Rips Protocol")
    add(f"- library/version: GUDHI {vr_meta['gudhi_version']}")
    add(f"- sample size/seed: N_VR={vr_meta['N_VR']}, sample_seed={vr_meta['sample_seed']}")
    add(f"- normalization/metric: {vr_meta['distance']}")
    add(f"- H0/H1: {vr_meta['homology_dimensions']}; filtration: Vietoris-Rips on cosine distances")
    add("- diagram distance: bottleneck for seed/split tables")
    add(
        "- number of cloud analyses: 500; number of homology summary rows: 1000 (H0 and H1 per cloud)."
    )
    add("")

    add("## M. VR Seed Stability")
    add(markdown_table(seed_vr.round(4)))
    add(
        "Answer: MIXED. ArcFace is lower in some H0 summaries but not uniformly lower/higher in H1 across domain/split."
    )
    add("")

    add("## N. VR Split Stability")
    add(markdown_table(split_vr.round(4)))
    add("Answer: MIXED; split sensitivity is present but not a clean ArcFace-only explanation.")
    add("")

    add("## O. VR External Domain Shift")
    add(
        "Bottleneck domain distances were not present as a precomputed local table. The generated proxy table uses existing total persistence and Betti curves, not new filtrations."
    )
    add(markdown_table(domain_vr_agg.round(4)))
    add("")

    add("## P. Topology vs OSR")
    add(markdown_table(corr.round(4)))
    add("Answer: WEAK / INCONSISTENT. Do not claim topology causes OSR instability.")
    add("")

    add("## Q. Lymphocyte Failure Analysis")
    add(markdown_table(lymph.round(4)))
    add(
        "Answer: PARTIALLY. VR adds descriptive structure, but attraction/mixing/kNN geometry give the clearer explanation."
    )
    add("")

    add("## R. Neutrophil-Band Analysis")
    add(markdown_table(nb.round(4)))
    add(
        "Neutrophil band remains a robust failure case: ArcFace internal MSP degradation persists despite compact known embeddings; VR does not explain it better than overlap/attraction geometry."
    )
    add("")

    add("## S. Role of Conventional Geometry")
    geom = read_csv("artifacts/metrics/delivery6/geometry_stability.csv")
    add(
        markdown_table(
            geom.groupby(["split", "representation"])[
                [
                    "fisher_ratio",
                    "within_dispersion",
                    "between_separation",
                    "mean_unknown_nearest_distance",
                ]
            ]
            .mean()
            .round(4),
            index=True,
        )
    )
    add(
        "Conventional geometry and attractor patterns explain failures more directly than VR correlations."
    )
    add("")

    add("## T. Scientific Role of Vietoris-Rips")
    add(
        "2. Secondary analysis. It is useful as a descriptive topology/stability audit, not as the main article engine."
    )
    add("")

    add("## U. Scientific Role of Cubical Persistent Homology")
    add(
        f"brief main-text mention + supplement. Delivery 3 decision: {d3_decision['scientific_decision']}. Cubical PH did NOT improve the predictive OSR system."
    )
    add("")

    add("## V. Strongest Defensible Article Contribution")
    add(
        "Primary: external validation shows that high closed-set leukocyte morphology recognition does not imply reliable open-set rejection under domain and morphology shift."
    )
    add(
        "Secondary: multiseed/split confirmation prevents overclaiming ArcFace; morphology-specific failure analysis identifies entangled unknowns; VR offers a secondary topology audit."
    )
    add("")

    add("## W. Candidate Article Thesis")
    add(
        "High internal closed-set performance in white-blood-cell morphology recognition does not guarantee externally reliable rejection of unseen morphologies; reliability is constrained by acquisition-domain shift, morphology-specific embedding entanglement, and representation instability."
    )
    add("")

    add("## X. Claim-Evidence Audit")
    add(
        markdown_table(
            claims[["claim_id", "support_status", "candidate_claim", "recommended_wording"]]
        )
    )
    add("")

    add("## Y. Paper Readiness")
    add(
        "1. READY FOR PAPER DRAFTING. No further model-development Delivery is warranted; remaining work is manuscript construction and figure consolidation."
    )
    add("")

    add("## Z. Proposed Paper Structure")
    add(
        "Title candidate: Domain and Morphology Limits of Open-Set Recognition in White Blood Cell Morphology."
    )
    add(
        "Abstract thesis: rigorous internal confirmation plus external AML-LMU validation reveals limited open-set generalization despite strong closed-set performance."
    )
    add(
        "Sections: Introduction; Related Work; Dataset and Open-Set Protocol; Methods; Internal Experiments; Representation and Failure Analysis; External Validation; Secondary Vietoris-Rips Topology Audit; Discussion; Limitations; Conclusion. Shorten TDA method details and move exhaustive VR diagrams to supplement."
    )
    add("")

    add("## AA. Main Figures")
    main_figs = figures[figures["main_text_candidate"]].head(8)
    add(markdown_table(main_figs[["figure_path", "scientific_question", "recommended_action"]]))
    add("")

    add("## AB. Main Tables")
    add(
        "Main tables: taxonomy/protocol summary; external closed/open multiseed summary; domain-shift summary; ArcFace matched-delta summary; key failure-class summary."
    )
    add("")

    add("## AC. Supplementary Material")
    add(
        "Full seed tables, per-class metrics, TDA negative ablations, full persistence diagrams, full taxonomy mapping, additional ROC/PR curves, and figure inventory."
    )
    add("")

    add("## AD. Next Step")
    add(
        "1. Begin manuscript drafting. Next task: Delivery 8 - Manuscript Construction and Publication-Quality Figure Consolidation."
    )
    add("")

    add("## AE. AWS Status")
    add(
        "AWS status must be verified live at handoff. Last verified after Delivery 7: no tmux sessions, GPU idle, EC2 may be stopped manually to avoid further charges."
    )
    add("")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    METRICS7.mkdir(parents=True, exist_ok=True)
    final_summary = make_final_scientific_summary()
    closed = make_closed_summary()
    seed_level = make_seed_level_open()
    deltas = make_arcface_deltas()
    domain_shift = make_domain_shift_summary()
    unknown = make_unknown_summary()
    seed_vr, split_vr, _domain_vr, domain_vr_agg = make_vr_summaries()
    lymph, nb = make_case_studies()
    claims = make_claim_matrix()
    figures = make_figure_inventory()
    keys = make_article_key_results(
        final_summary,
        closed,
        deltas,
        domain_shift,
        unknown,
        seed_vr,
        split_vr,
        domain_vr_agg,
    )
    write_report(
        final_summary,
        closed,
        seed_level,
        deltas,
        domain_shift,
        unknown,
        seed_vr,
        split_vr,
        domain_vr_agg,
        lymph,
        nb,
        claims,
        figures,
    )
    print("wrote", METRICS7 / "final_scientific_summary.csv")
    print("wrote", METRICS7 / "article_key_results.csv", keys.shape)
    print("wrote", METRICS7 / "article_claim_evidence_matrix.csv", claims.shape)
    print("wrote", METRICS7 / "article_figure_inventory.csv", figures.shape)
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
