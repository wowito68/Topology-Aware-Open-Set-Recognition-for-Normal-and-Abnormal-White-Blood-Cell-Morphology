#!/usr/bin/env python3
"""Build Delivery 2 comparison tables, figures, and bootstrap reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from hemato_osr.evaluation.delivery2 import (
    MethodArtifact,
    write_bootstrap_report,
    write_confusion_matrix_figure,
    write_delivery2_figures,
    write_model_comparison,
    write_per_unknown_class_table,
)


def _method(
    *,
    split: str,
    representation: str,
    classifier: str,
    osr_method: str,
    root: Path,
) -> MethodArtifact:
    return MethodArtifact(
        split_protocol=split,
        representation=representation,
        classifier=classifier,
        osr_method=osr_method,
        metrics_path=root / "metrics.json",
        predictions_path=root / "predictions.csv",
        config_hash=_config_hash(root),
    )


def _config_hash(root: Path) -> str:
    env = root / "environment.json"
    if env.exists():
        try:
            return str(json.loads(env.read_text()).get("manifest_hash", ""))
        except json.JSONDecodeError:
            return ""
    config = root / "experiment_config.json"
    if config.exists():
        return hashlib.sha256(config.read_bytes()).hexdigest()
    return ""


def _existing(methods: list[MethodArtifact]) -> list[MethodArtifact]:
    return [
        method
        for method in methods
        if method.metrics_path.exists() and method.predictions_path.exists()
    ]


def _first_existing(methods: list[MethodArtifact]) -> MethodArtifact | None:
    existing = _existing(methods)
    return existing[0] if existing else None


def _write_sensitivity(
    v1_msp: Path,
    v1_energy: Path,
    v2_msp: Path,
    v2_energy: Path,
    output_path: Path,
) -> None:
    v1_msp_metrics = json.loads((v1_msp / "metrics.json").read_text())
    v1_energy_metrics = json.loads((v1_energy / "metrics.json").read_text())
    v2_msp_metrics = json.loads((v2_msp / "metrics.json").read_text())
    v2_energy_metrics = json.loads((v2_energy / "metrics.json").read_text())
    rows = [
        (
            "macro-F1",
            v1_msp_metrics["closed_set"]["macro_f1"],
            v2_msp_metrics["closed_set"]["macro_f1"],
        ),
        (
            "balanced accuracy",
            v1_msp_metrics["closed_set"]["balanced_accuracy"],
            v2_msp_metrics["closed_set"]["balanced_accuracy"],
        ),
        (
            "MSP AUROC",
            v1_msp_metrics["open_set"]["auroc_known_unknown"],
            v2_msp_metrics["open_set"]["auroc_known_unknown"],
        ),
        (
            "MSP FPR95",
            v1_msp_metrics["open_set"]["fpr_at_95_tpr"],
            v2_msp_metrics["open_set"]["fpr_at_95_tpr"],
        ),
        (
            "Energy AUROC",
            v1_energy_metrics["open_set"]["auroc_known_unknown"],
            v2_energy_metrics["open_set"]["auroc_known_unknown"],
        ),
        (
            "Energy FPR95",
            v1_energy_metrics["open_set"]["fpr_at_95_tpr"],
            v2_energy_metrics["open_set"]["fpr_at_95_tpr"],
        ),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["metric", "V1", "V2"]).assign(
        delta=lambda frame: frame["V2"] - frame["V1"]
    ).to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()

    root = args.root
    metrics_dir = root / "artifacts/metrics/delivery2"
    figures_dir = root / "artifacts/figures/delivery2"
    v1_metrics = root / "artifacts/metrics"

    methods = [
        method
        for method in [
            _first_existing(
                [
                    _method(
                        split="V1",
                        representation="Deep",
                        classifier="ResNet18",
                        osr_method="MSP",
                        root=metrics_dir / "v1_resnet18_msp",
                    ),
                    _method(
                        split="V1",
                        representation="Deep",
                        classifier="ResNet18",
                        osr_method="MSP",
                        root=v1_metrics / "resnet18_seed37_msp",
                    ),
                ]
            ),
            _method(
                split="V1",
                representation="Deep",
                classifier="ResNet18",
                osr_method="Energy",
                root=v1_metrics / "resnet18_seed37_energy",
            ),
            _method(
                split="V1",
                representation="Deep",
                classifier="ResNet18",
                osr_method="Mahalanobis",
                root=v1_metrics / "resnet18_seed37_mahalanobis",
            ),
            _method(
                split="V2",
                representation="Deep",
                classifier="ResNet18",
                osr_method="MSP",
                root=metrics_dir / "v2_resnet18_msp",
            ),
            _method(
                split="V2",
                representation="Deep",
                classifier="ResNet18",
                osr_method="Energy",
                root=metrics_dir / "v2_resnet18_energy",
            ),
            _method(
                split="V1",
                representation="TDA",
                classifier="LogisticRegression",
                osr_method="centroid_distance",
                root=metrics_dir / "tda_only_centroid",
            ),
            _method(
                split="V1",
                representation="Deep+TDA_sublevel",
                classifier="FrozenFusionMLP",
                osr_method="MSP",
                root=metrics_dir / "fusion_sublevel_msp",
            ),
            _method(
                split="V1",
                representation="Deep+TDA_superlevel",
                classifier="FrozenFusionMLP",
                osr_method="MSP",
                root=metrics_dir / "fusion_superlevel_msp",
            ),
            _method(
                split="V1",
                representation="Deep+TDA_full",
                classifier="FrozenFusionMLP",
                osr_method="MSP",
                root=metrics_dir / "fusion_full_msp",
            ),
            _method(
                split="V1",
                representation="Deep+TDA_full",
                classifier="FrozenFusionMLP",
                osr_method="Energy",
                root=metrics_dir / "fusion_full_energy",
            ),
            _method(
                split="V1",
                representation="Deep+TDA_full",
                classifier="FrozenFusionMLP",
                osr_method="Mahalanobis",
                root=metrics_dir / "fusion_full_mahalanobis",
            ),
        ]
        if method is not None
    ]
    methods = _existing(methods)
    comparison_path = write_model_comparison(methods, metrics_dir / "model_comparison.csv")
    per_unknown_path = write_per_unknown_class_table(methods, metrics_dir / "per_unknown_class.csv")

    _write_sensitivity(
        v1_metrics / "resnet18_seed37_msp",
        v1_metrics / "resnet18_seed37_energy",
        metrics_dir / "v2_resnet18_msp",
        metrics_dir / "v2_resnet18_energy",
        metrics_dir / "v1_vs_v2_sensitivity.csv",
    )

    comparison = pd.read_csv(comparison_path)
    fusion_rows = comparison.loc[
        comparison["representation"].str.contains("Deep\\+TDA", regex=True)
    ]
    best_fusion = None
    if not fusion_rows.empty:
        best_idx = fusion_rows["open_auroc"].astype(float).idxmax()
        best_fusion_name = fusion_rows.loc[best_idx, "representation"]
        best_fusion_osr = fusion_rows.loc[best_idx, "osr_method"]
        for method in methods:
            if method.representation == best_fusion_name and method.osr_method == best_fusion_osr:
                best_fusion = method
                break
    baseline = next(
        method
        for method in methods
        if method.representation == "Deep"
        and method.osr_method == "MSP"
        and method.split_protocol == "V1"
    )
    if best_fusion is not None:
        write_bootstrap_report(
            pd.read_csv(baseline.predictions_path),
            pd.read_csv(best_fusion.predictions_path),
            metrics_dir / "bootstrap_deep_vs_best_fusion.json",
            n_bootstraps=args.n_bootstraps,
            seed=args.seed,
        )

    figure_methods = [baseline]
    energy = [
        method
        for method in methods
        if method.representation == "Deep"
        and method.osr_method == "Energy"
        and method.split_protocol == "V1"
    ]
    tda = [method for method in methods if method.representation == "TDA"]
    if energy:
        figure_methods.append(energy[0])
    if tda:
        figure_methods.append(tda[0])
    if best_fusion is not None:
        figure_methods.append(best_fusion)
    write_delivery2_figures(figure_methods, figures_dir, per_unknown_path=per_unknown_path)
    write_confusion_matrix_figure(
        baseline.predictions_path,
        figures_dir / "closed_set_confusion_matrix_deep.png",
    )
    if tda:
        write_confusion_matrix_figure(
            tda[0].predictions_path,
            figures_dir / "closed_set_confusion_matrix_tda.png",
        )
    if best_fusion is not None:
        write_confusion_matrix_figure(
            best_fusion.predictions_path,
            figures_dir / "closed_set_confusion_matrix_fusion.png",
        )
    runtime = root / "artifacts/benchmarks/tda_resolution/tda_runtime_vs_resolution.png"
    stability = root / "artifacts/benchmarks/tda_resolution/tda_stability_vs_resolution.png"
    if runtime.exists():
        shutil.copy2(runtime, figures_dir / "tda_runtime_vs_resolution.png")
    if stability.exists():
        shutil.copy2(stability, figures_dir / "tda_stability_vs_resolution.png")
    print(comparison_path)


if __name__ == "__main__":
    main()
