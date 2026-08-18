# ruff: noqa: E501, T201
"""Build the Delivery 8 LaTeX manuscript from frozen artifacts.

This script is a publication-artifact generator. It does not train models,
recompute embeddings, tune thresholds, alter splits, or change taxonomy.
"""

from __future__ import annotations

import csv
import json
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
SECTIONS = PAPER / "sections"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"
SUPPLEMENT = PAPER / "supplement"
BUILD = PAPER / "build"
METRICS7 = ROOT / "artifacts" / "metrics" / "delivery7"
METRICS6 = ROOT / "artifacts" / "metrics" / "delivery6"
METRICS5 = ROOT / "artifacts" / "metrics" / "delivery5"
METRICS4 = ROOT / "artifacts" / "metrics" / "delivery4"
METRICS3 = ROOT / "artifacts" / "metrics" / "delivery3"
METRICS2 = ROOT / "artifacts" / "metrics" / "delivery2"


TITLE = "Open-Set Recognition of Unseen Hematological Cell Morphologies under Domain Shift"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def mkdirs() -> None:
    for path in [PAPER, SECTIONS, FIGURES, TABLES, SUPPLEMENT, BUILD]:
        path.mkdir(parents=True, exist_ok=True)
    (BUILD / ".gitkeep").touch()


def tex_escape(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def mean_sd(mean: float, sd: float) -> str:
    return f"{mean:.3f} $\\pm$ {sd:.3f}"


def metric_row(df: pd.DataFrame, **kwargs: str) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for key, value in kwargs.items():
        mask &= df[key].astype(str).str.lower() == str(value).lower()
    matches = df[mask]
    if matches.empty:
        raise KeyError(kwargs)
    return matches.iloc[0]


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def latex_table(
    df: pd.DataFrame, columns: list[str], headers: list[str], align: str | None = None
) -> str:
    align = align or ("l" * len(columns))
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule"]
    lines.append(" & ".join(tex_escape(h) for h in headers) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join(tex_escape(row[col]) for col in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def table_float(caption: str, label: str, body: str, resize: bool = True) -> str:
    wrapped = rf"\resizebox{{\textwidth}}{{!}}{{%{chr(10)}{body}{chr(10)}}}" if resize else body
    return f"""
\\begin{{table}}[htbp]
\\centering
{wrapped}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{table}}
"""


def selected_results(summary: pd.DataFrame) -> dict[str, pd.Series]:
    picks: dict[str, pd.Series] = {}
    for split in ["v1", "v2"]:
        for rep in ["ce", "arcface"]:
            for method in ["msp", "vim"]:
                for metric in ["AUROC", "FPR95", "OSCR", "closed_macro_f1"]:
                    key = f"external_{split}_{rep}_{method}_{metric}"
                    picks[key] = metric_row(
                        summary,
                        domain="AML_LMU_external",
                        training_split=split,
                        representation=rep,
                        osr_method=method,
                        metric=metric,
                    )
    return picks


def build_results_freeze() -> dict:
    d6 = read_csv(METRICS6 / "multiseed_summary.csv")
    d7 = read_csv(METRICS7 / "final_scientific_summary.csv")
    shift = read_csv(METRICS7 / "delivery75_internal_external_domain_shift_summary.csv")
    arc_deltas = read_csv(METRICS6 / "matched_seed_deltas.csv")
    external_unknown = read_csv(
        METRICS7 / "delivery75_external_unknown_morphology_summary_ce_vim.csv"
    )
    freeze: dict[str, dict] = {}

    def add(
        result_id: str,
        value: object,
        uncertainty: object,
        source_file: str,
        source_row: str,
        interpretation: str,
    ) -> None:
        freeze[result_id] = {
            "value": value,
            "uncertainty": uncertainty,
            "source_file": source_file,
            "source_row": source_row,
            "interpretation": interpretation,
        }

    for split in ["v1", "v2"]:
        for rep in ["ce", "arcface"]:
            row = metric_row(d6, split=split, representation=rep, metric="closed_macro_f1")
            add(
                f"internal_{split}_{rep}_closed_macro_f1",
                float(row["mean"]),
                {
                    "sd": float(row["sd"]),
                    "ci95_low": float(row["ci95_low"]),
                    "ci95_high": float(row["ci95_high"]),
                },
                "artifacts/metrics/delivery6/multiseed_summary.csv",
                f"split={split}; representation={rep}; metric=closed_macro_f1",
                "Internal closed-set classification is high across five seeds.",
            )
            for method in ["msp", "vim"]:
                for metric in ["auroc", "fpr95", "oscr"]:
                    row = metric_row(
                        d6, split=split, representation=rep, metric=f"{method}_{metric}"
                    )
                    add(
                        f"internal_{split}_{rep}_{method}_{metric}",
                        float(row["mean"]),
                        {
                            "sd": float(row["sd"]),
                            "ci95_low": float(row["ci95_low"]),
                            "ci95_high": float(row["ci95_high"]),
                        },
                        "artifacts/metrics/delivery6/multiseed_summary.csv",
                        f"split={split}; representation={rep}; metric={method}_{metric}",
                        "Internal open-set result over five training seeds.",
                    )
    for key, row in selected_results(d7).items():
        add(
            key,
            float(row["mean"]),
            {
                "sd": float(row["sd"]),
                "ci95_low": float(row["ci95_low"]),
                "ci95_high": float(row["ci95_high"]),
            },
            "artifacts/metrics/delivery7/final_scientific_summary.csv",
            f"domain=AML_LMU_external; split={row['training_split']}; representation={row['representation']}; osr_method={row['osr_method']}; metric={row['metric']}",
            "External AML-LMU result without external training or tuning.",
        )
    for _, row in shift.iterrows():
        add(
            f"domain_shift_{row['training_split']}_{row['representation']}_{row['osr_method']}",
            {
                "delta_AUROC": float(row["delta_AUROC"]),
                "delta_FPR95": float(row["delta_FPR95"]),
                "delta_closed_macro_f1": float(row["delta_closed_macro_f1"]),
            },
            {
                "delta_AUROC_ci95": [
                    float(row["delta_AUROC_ci95_low"]),
                    float(row["delta_AUROC_ci95_high"]),
                ],
                "delta_FPR95_ci95": [
                    float(row["delta_FPR95_ci95_low"]),
                    float(row["delta_FPR95_ci95_high"]),
                ],
                "delta_closed_macro_f1_ci95": [
                    float(row["delta_closed_macro_f1_ci95_low"]),
                    float(row["delta_closed_macro_f1_ci95_high"]),
                ],
            },
            "artifacts/metrics/delivery7/delivery75_internal_external_domain_shift_summary.csv",
            f"split={row['training_split']}; representation={row['representation']}; osr_method={row['osr_method']}",
            "Internal-to-external degradation quantifies domain shift.",
        )
    for split in ["v1", "v2"]:
        for metric in ["msp_auroc", "msp_fpr95"]:
            sub = arc_deltas[(arc_deltas["split"] == split) & (arc_deltas["metric"] == metric)]
            add(
                f"arcface_delta_{split}_{metric}",
                float(sub["delta"].mean()),
                {
                    "sd": float(sub["delta"].std(ddof=1)),
                    "favorable_seeds": int((sub["delta"] > 0).sum()),
                    "n": int(sub.shape[0]),
                },
                "artifacts/metrics/delivery6/matched_seed_deltas.csv",
                f"split={split}; metric={metric}; aggregate=mean_delta",
                "ArcFace-minus-CE matched seed delta.",
            )
    for _, row in external_unknown.iterrows():
        add(
            f"external_unknown_{row['canonical_label']}",
            {"AUROC_mean": float(row["AUROC_mean"]), "FPR95_mean": float(row["FPR95_mean"])},
            {
                "AUROC_sd": float(row["AUROC_sd"]),
                "FPR95_sd": float(row["FPR95_sd"]),
                "n": int(row["n"]),
            },
            "artifacts/metrics/delivery7/delivery75_external_unknown_morphology_summary_ce_vim.csv",
            f"canonical_label={row['canonical_label']}",
            "External CE+ViM per-morphology open-set result.",
        )
    return freeze


def write_claim_files() -> None:
    claims = read_csv(METRICS7 / "article_claim_evidence_matrix.csv")
    lines = [
        "# Claim Freeze",
        "",
        "The manuscript may use SUPPORTED claims as main conclusions. PARTIALLY_SUPPORTED claims require cautious wording. NOT_SUPPORTED claims must not be stated as conclusions.",
        "",
    ]
    for status in ["SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED"]:
        lines.append(f"## {status}")
        status_df = claims[claims["support_status"] == status]
        for _, row in status_df.iterrows():
            lines.append(f"- **{row['claim_id']}**: {row['candidate_claim']}")
            lines.append(f"  - Recommended wording: {row['recommended_wording']}")
            lines.append(f"  - Evidence: {row['supporting_result']}")
            if isinstance(row.get("contradicting_result"), str) and row["contradicting_result"]:
                lines.append(f"  - Constraint: {row['contradicting_result']}")
        lines.append("")
    write(PAPER / "CLAIMS.md", "\n".join(lines))

    trace_rows = []
    for _, row in claims.iterrows():
        trace_rows.append(
            {
                "claim_id": row["claim_id"],
                "paper_section": row["paper_section"],
                "claim_text": row["recommended_wording"],
                "evidence_artifact": row["supporting_result"],
                "metric": "see source artifact",
                "support_status": row["support_status"],
            }
        )
    with (PAPER / "claim_traceability.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trace_rows)


def write_freeze_yaml() -> None:
    freeze = build_results_freeze()
    (PAPER / "results_freeze.yaml").write_text(
        yaml.safe_dump(freeze, sort_keys=True, allow_unicode=True), encoding="utf-8"
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()


def make_figures() -> None:
    plt.style.use("default")
    colors = {"ce": "#2f6f9f", "arcface": "#b45f2a", "msp": "#4062bb", "vim": "#59a14f"}

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    boxes = [
        ("MLL23\n41,621 images", 0.05, 0.72),
        ("Known train\n12,793", 0.28, 0.82),
        ("Known validation\n2,741", 0.28, 0.62),
        ("Known test\n2,742", 0.52, 0.82),
        ("Unknown test\n23,345", 0.52, 0.62),
        ("Frozen CE / ArcFace\nResNet18 runs", 0.30, 0.34),
        ("Post-hoc OSR\nMSP and ViM", 0.55, 0.34),
        ("AML-LMU external\n18,365 test-only images", 0.77, 0.62),
        ("Geometry and VR\nexplanatory analysis", 0.77, 0.34),
    ]
    for text, x, y in boxes:
        ax.add_patch(
            plt.Rectangle(
                (x, y), 0.18, 0.12, edgecolor="#303030", facecolor="#f4f6f8", linewidth=1.2
            )
        )
        ax.text(x + 0.09, y + 0.06, text, ha="center", va="center", fontsize=10)
    arrows = [
        ((0.23, 0.78), (0.28, 0.88)),
        ((0.23, 0.78), (0.28, 0.68)),
        ((0.46, 0.88), (0.52, 0.88)),
        ((0.46, 0.68), (0.52, 0.68)),
        ((0.37, 0.62), (0.39, 0.46)),
        ((0.61, 0.62), (0.61, 0.46)),
        ((0.48, 0.40), (0.55, 0.40)),
        ((0.73, 0.40), (0.77, 0.40)),
        ((0.70, 0.68), (0.77, 0.68)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#303030"},
        )
    ax.text(
        0.05,
        0.12,
        "UNKNOWN = outside the known training taxonomy only; not a diagnostic label.",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_title("Experimental protocol and taxonomy freeze", fontsize=14)
    savefig(FIGURES / "figure1_protocol.png")

    d6 = read_csv(METRICS6 / "run_level_results.csv")
    systems = [("ce", "msp"), ("ce", "vim"), ("arcface", "msp"), ("arcface", "vim")]
    labels = ["CE+MSP", "CE+ViM", "ArcFace+MSP", "ArcFace+ViM"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for ax, metric, ylabel in zip(axes, ["auroc", "fpr95"], ["AUROC", "FPR@95TPR"], strict=True):
        positions = np.arange(len(systems))
        for i, (rep, method) in enumerate(systems):
            for j, split in enumerate(["v1", "v2"]):
                vals = d6[(d6["representation"] == rep) & (d6["split"] == split)][
                    f"{method}_{metric}"
                ].to_numpy()
                x = i + (j - 0.5) * 0.18
                ax.scatter(
                    np.full(vals.shape, x),
                    vals,
                    s=32,
                    alpha=0.85,
                    label=split.upper() if i == 0 else None,
                )
                ax.hlines(vals.mean(), x - 0.08, x + 0.08, color="black", linewidth=1.4)
        ax.set_xticks(positions, labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    axes[0].legend(frameon=False, title="Split")
    fig.suptitle("Internal multiseed open-set performance")
    savefig(FIGURES / "figure2_internal_performance.png")

    deltas = read_csv(METRICS6 / "matched_seed_deltas.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=False)
    for ax, metric, ylabel in zip(
        axes, ["msp_auroc", "msp_fpr95"], ["Delta AUROC", "Delta FPR@95TPR"], strict=True
    ):
        sub = deltas[deltas["metric"] == metric]
        for i, split in enumerate(["v1", "v2"]):
            vals = sub[sub["split"] == split]["delta"].to_numpy()
            ax.scatter(np.full(vals.shape, i), vals, s=42, color="#7a3b2e")
            ax.hlines(vals.mean(), i - 0.18, i + 0.18, color="black", linewidth=1.6)
        ax.axhline(0, color="#404040", linewidth=1, linestyle="--")
        ax.set_xticks([0, 1], ["V1", "V2"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    fig.suptitle("Matched ArcFace-minus-CE seed deltas for MSP")
    savefig(FIGURES / "figure3_arcface_stability.png")

    shift = read_csv(METRICS7 / "delivery75_internal_external_domain_shift_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    plot_metrics = [
        ("AUROC", "internal_AUROC", "external_AUROC"),
        ("FPR@95TPR", "internal_FPR95", "external_FPR95"),
        ("Closed macro-F1", "internal_closed_macro_f1", "external_closed_macro_f1"),
    ]
    x = np.arange(len(shift))
    tick_labels = [f"{r.system}\n{r.training_split.upper()}" for r in shift.itertuples()]
    for ax, (title, internal_col, external_col) in zip(axes, plot_metrics, strict=True):
        ax.bar(x - 0.18, shift[internal_col], width=0.36, label="MLL23", color="#5975a4")
        ax.bar(x + 0.18, shift[external_col], width=0.36, label="AML-LMU", color="#cc8963")
        ax.set_xticks(x, tick_labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    add_panel_label(axes[2], "C")
    axes[0].legend(frameon=False)
    fig.suptitle("Internal-to-external domain shift")
    savefig(FIGURES / "figure4_domain_shift.png")

    unknown = read_csv(
        METRICS7 / "delivery75_external_unknown_morphology_summary_ce_vim.csv"
    ).sort_values("AUROC_mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(unknown))
    ax.errorbar(
        unknown["AUROC_mean"], y, xerr=unknown["AUROC_sd"], fmt="o", color="#2f6f9f", label="AUROC"
    )
    ax.errorbar(
        unknown["FPR95_mean"],
        y + 0.18,
        xerr=unknown["FPR95_sd"],
        fmt="s",
        color="#b45f2a",
        label="FPR@95TPR",
    )
    ax.set_yticks(y, [label.replace("_", " ") for label in unknown["canonical_label"]])
    ax.set_xlabel("Mean across CE+ViM external runs")
    ax.set_xlim(0.35, 1.02)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("External morphology-dependent rejection difficulty")
    savefig(FIGURES / "figure5_external_morphology.png")

    band = read_csv(METRICS7 / "delivery75_neutrophil_band_summary.csv")
    lymph = read_csv(METRICS7 / "delivery75_lymphocyte_failure_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    band_labels = [f"{r.representation}\n{r.training_split.upper()}" for r in band.itertuples()]
    axes[0].bar(
        np.arange(len(band)), band["attraction_to_neutrophil_segmented_mean"], color="#b45f2a"
    )
    axes[0].set_xticks(np.arange(len(band)), band_labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Dominant known-prediction fraction")
    axes[0].set_title("Neutrophil band to segmented-neutrophil attraction")
    axes[0].grid(axis="y", alpha=0.25)
    lymph_plot = lymph.groupby("unknown_class", as_index=False).agg(
        attraction=("attraction_to_lymphocyte_mean", "mean"),
        msp=("msp_AUROC_mean", "mean"),
        vim=("vim_AUROC_mean", "mean"),
    )
    x2 = np.arange(len(lymph_plot))
    axes[1].bar(
        x2 - 0.18,
        lymph_plot["attraction"],
        width=0.36,
        color="#59a14f",
        label="Lymphocyte attraction",
    )
    axes[1].bar(x2 + 0.18, lymph_plot["vim"], width=0.36, color="#4062bb", label="ViM AUROC")
    axes[1].set_xticks(
        x2,
        [v.replace("lymphocyte_", "").replace("_", " ") for v in lymph_plot["unknown_class"]],
        rotation=20,
        ha="right",
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Lymphoid-like failures")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    fig.suptitle("Local attractor failure analysis")
    savefig(FIGURES / "figure6_failure_cases.png")

    vr = read_csv(METRICS7 / "delivery75_vr_seed_stability_summary.csv")
    corr = read_csv(METRICS7 / "vr_topology_osr_correlations.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    agg = vr.groupby(["representation", "metric"], as_index=False).agg(
        mean=("mean", "mean"), sd=("mean", "std")
    )
    for i, rep in enumerate(["ce", "arcface"]):
        sub = agg[agg["representation"] == rep]
        axes[0].bar(
            i - 0.17,
            sub[sub["metric"] == "h0_bottleneck"]["mean"].iloc[0],
            0.34,
            color=colors[rep],
            alpha=0.85,
        )
        axes[0].bar(
            i + 0.17,
            sub[sub["metric"] == "h1_bottleneck"]["mean"].iloc[0],
            0.34,
            color=colors[rep],
            alpha=0.45,
        )
    axes[0].set_xticks([0, 1], ["CE", "ArcFace"])
    axes[0].set_ylabel("Mean seed bottleneck proxy")
    axes[0].set_title("VR seed stability summaries")
    axes[0].grid(axis="y", alpha=0.25)
    corr_plot = corr.copy()
    corr_plot["label"] = (
        corr_plot["topology_metric"].str.replace("_", " ")
        + "\n"
        + corr_plot["osr_metric"].str.replace("_", " ")
    )
    order = corr_plot["spearman_r"].abs().sort_values(ascending=False).index[:8]
    corr_top = corr_plot.loc[order].sort_values("spearman_r")
    axes[1].barh(np.arange(len(corr_top)), corr_top["spearman_r"], color="#6b6b6b")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_yticks(np.arange(len(corr_top)), corr_top["label"], fontsize=7)
    axes[1].set_xlabel("Spearman rho")
    axes[1].set_title("Exploratory topology--OSR associations")
    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    fig.suptitle("Embedding geometry and Vietoris-Rips topology")
    savefig(FIGURES / "figure7_topology.png")


def write_tables() -> None:
    d6_summary = read_csv(METRICS6 / "multiseed_summary.csv")
    d7_summary = read_csv(METRICS7 / "final_scientific_summary.csv")
    claims = read_csv(METRICS7 / "article_claim_evidence_matrix.csv")

    datasets = pd.DataFrame(
        [
            {
                "dataset": "MLL23",
                "role": "Internal development and multiseed evaluation",
                "images": "41,621",
                "known": "5",
                "unknown": "13",
                "format": "TIFF, 288x288",
                "training_use": "Known train only",
                "external_use": "No",
            },
            {
                "dataset": "AML-LMU",
                "role": "Independent external test-only validation",
                "images": "18,365",
                "known": "5 mapped",
                "unknown": "10 mapped",
                "format": "TIFF, 400x400x3",
                "training_use": "None",
                "external_use": "Test only",
            },
        ]
    )
    body = latex_table(
        datasets,
        list(datasets.columns),
        ["Dataset", "Role", "Images", "Known", "Unknown", "Format", "Training use", "External use"],
        "llllllll",
    )
    write(
        TABLES / "table1_datasets_protocol.tex",
        table_float(
            "Datasets and open-set protocol roles. UNKNOWN denotes only outside the known training taxonomy.",
            "tab:datasets",
            body,
        ),
    )

    rows = []
    for rep, method, label in [
        ("ce", "msp", "CE+MSP"),
        ("ce", "vim", "CE+ViM"),
        ("arcface", "msp", "ArcFace+MSP"),
        ("arcface", "vim", "ArcFace+ViM"),
    ]:
        row = {"system": label}
        for split in ["v1", "v2"]:
            au = metric_row(d6_summary, split=split, representation=rep, metric=f"{method}_auroc")
            fp = metric_row(d6_summary, split=split, representation=rep, metric=f"{method}_fpr95")
            cf = metric_row(d6_summary, split=split, representation=rep, metric="closed_macro_f1")
            row[f"{split}_auroc"] = mean_sd(au["mean"], au["sd"])
            row[f"{split}_fpr95"] = mean_sd(fp["mean"], fp["sd"])
            row[f"{split}_closed"] = mean_sd(cf["mean"], cf["sd"])
        rows.append(row)
    internal = pd.DataFrame(rows)
    body = latex_table(
        internal,
        list(internal.columns),
        ["System", "V1 AUROC", "V1 FPR95", "V1 closed F1", "V2 AUROC", "V2 FPR95", "V2 closed F1"],
        "lllllll",
    )
    write(
        TABLES / "table2_internal_multiseed.tex",
        table_float(
            "Internal MLL23 multiseed results. Values are mean $\\pm$ SD over five training seeds.",
            "tab:internal",
            body,
        ),
    )

    rows = []
    for rep, method, label in [
        ("ce", "msp", "CE+MSP"),
        ("ce", "vim", "CE+ViM"),
        ("arcface", "msp", "ArcFace+MSP"),
        ("arcface", "vim", "ArcFace+ViM"),
    ]:
        for split in ["v1", "v2"]:
            au = metric_row(
                d7_summary,
                domain="AML_LMU_external",
                training_split=split,
                representation=rep,
                osr_method=method,
                metric="AUROC",
            )
            fp = metric_row(
                d7_summary,
                domain="AML_LMU_external",
                training_split=split,
                representation=rep,
                osr_method=method,
                metric="FPR95",
            )
            os = metric_row(
                d7_summary,
                domain="AML_LMU_external",
                training_split=split,
                representation=rep,
                osr_method=method,
                metric="OSCR",
            )
            cf = metric_row(
                d7_summary,
                domain="AML_LMU_external",
                training_split=split,
                representation=rep,
                osr_method=method,
                metric="closed_macro_f1",
            )
            rows.append(
                {
                    "system": label,
                    "split": split.upper(),
                    "auroc": mean_sd(au["mean"], au["sd"]),
                    "fpr95": mean_sd(fp["mean"], fp["sd"]),
                    "oscr": mean_sd(os["mean"], os["sd"]),
                    "closed": mean_sd(cf["mean"], cf["sd"]),
                }
            )
    external = pd.DataFrame(rows)
    body = latex_table(
        external,
        list(external.columns),
        ["System", "Train split", "AUROC", "FPR95", "OSCR", "Closed macro-F1"],
        "llllll",
    )
    write(
        TABLES / "table3_external_results.tex",
        table_float(
            "AML-LMU external test-only results. Values are mean $\\pm$ SD over five training seeds.",
            "tab:external",
            body,
        ),
    )

    claim_rows = claims[claims["claim_id"].isin(["C1", "C2", "C5", "C6", "C7", "C8", "C9", "C10"])][
        ["claim_id", "support_status", "recommended_wording"]
    ].copy()
    body = latex_table(
        claim_rows, list(claim_rows.columns), ["Claim", "Support", "Manuscript wording"], "lll"
    )
    write(
        TABLES / "table4_claim_evidence.tex",
        table_float(
            "Claim-evidence summary used to freeze manuscript conclusions.", "tab:claims", body
        ),
    )


def write_references() -> None:
    write(
        PAPER / "references.bib",
        r"""
@article{mll23_nature_2025,
  title = {A large expert-annotated single-cell peripheral blood dataset for hematological disease diagnostics},
  author = {Shetab Boushehri, Sayedali and others},
  journal = {Scientific Data},
  year = {2025},
  doi = {10.1038/s41597-025-06223-x},
  url = {https://www.nature.com/articles/s41597-025-06223-x}
}

@misc{mll23_zenodo_2024,
  title = {A large publicly available single-cell peripheral blood dataset (MLL23)},
  author = {Shetab Boushehri, Sayedali and others},
  publisher = {Zenodo},
  year = {2024},
  doi = {10.5281/zenodo.14277609},
  url = {https://zenodo.org/records/14277609}
}

@misc{aml_lmu_tcia_2019,
  title = {A Single-cell Morphological Dataset of Leukocytes from AML Patients and Non-malignant Controls},
  author = {{The Cancer Imaging Archive}},
  publisher = {The Cancer Imaging Archive},
  year = {2019},
  doi = {10.7937/tcia.2019.36f5o9ld},
  url = {https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/}
}

@inproceedings{he2016resnet,
  title = {Deep Residual Learning for Image Recognition},
  author = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year = {2016},
  url = {https://arxiv.org/abs/1512.03385}
}

@inproceedings{bendale2016openmax,
  title = {Towards Open Set Deep Networks},
  author = {Bendale, Abhijit and Boult, Terrance},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year = {2016},
  url = {https://arxiv.org/abs/1511.06233}
}

@inproceedings{hendrycks2017baseline,
  title = {A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks},
  author = {Hendrycks, Dan and Gimpel, Kevin},
  booktitle = {International Conference on Learning Representations},
  year = {2017},
  url = {https://arxiv.org/abs/1610.02136}
}

@inproceedings{liu2020energy,
  title = {Energy-based Out-of-distribution Detection},
  author = {Liu, Weitang and Wang, Xiaoyun and Owens, John D. and Li, Yixuan},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2020},
  url = {https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html}
}

@inproceedings{sun2021react,
  title = {ReAct: Out-of-distribution Detection With Rectified Activations},
  author = {Sun, Yiyou and Guo, Chuan and Li, Yixuan},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2021},
  url = {https://proceedings.neurips.cc/paper/2021/hash/01894d6f048493d2cacde3c579c315a3-Abstract.html}
}

@inproceedings{wang2022vim,
  title = {ViM: Out-Of-Distribution with Virtual-logit Matching},
  author = {Wang, Haoqi and Li, Zhizhong and Feng, Litong and Zhang, Wayne},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2022},
  url = {https://arxiv.org/abs/2203.10807}
}

@inproceedings{deng2019arcface,
  title = {ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
  author = {Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2019},
  url = {https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html}
}

@inproceedings{khosla2020supcon,
  title = {Supervised Contrastive Learning},
  author = {Khosla, Prannay and Teterwak, Piotr and Wang, Chen and Sarna, Aaron and Tian, Yonglong and Isola, Phillip and Maschinot, Aaron and Liu, Ce and Krishnan, Dilip},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2020},
  url = {https://arxiv.org/abs/2004.11362}
}

@article{edelsbrunner2002persistence,
  title = {Topological Persistence and Simplification},
  author = {Edelsbrunner, Herbert and Letscher, David and Zomorodian, Afra},
  journal = {Discrete and Computational Geometry},
  volume = {28},
  number = {4},
  pages = {511--533},
  year = {2002},
  doi = {10.1007/s00454-002-2885-2}
}

@article{carlsson2009topology,
  title = {Topology and Data},
  author = {Carlsson, Gunnar},
  journal = {Bulletin of the American Mathematical Society},
  volume = {46},
  number = {2},
  pages = {255--308},
  year = {2009},
  doi = {10.1090/S0273-0979-09-01249-X}
}

@incollection{maria2014gudhi,
  title = {The Gudhi Library: Simplicial Complexes and Persistent Homology},
  author = {Maria, Clement and Boissonnat, Jean-Daniel and Glisse, Marc and Yvinec, Mariette},
  booktitle = {Mathematical Software -- ICMS 2014},
  series = {Lecture Notes in Computer Science},
  volume = {8592},
  pages = {167--174},
  publisher = {Springer},
  year = {2014},
  doi = {10.1007/978-3-662-44199-2_28}
}

@misc{TODO_hematology_ai_review,
  title = {TODO: Add a verified review citation for automated hematological cell classification},
  author = {TODO},
  year = {9999},
  note = {Placeholder; replace before submission}
}
""",
    )


def write_main_tex() -> None:
    write(
        PAPER / "main.tex",
        rf"""
\documentclass[11pt]{{article}}

\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{lmodern}}
\usepackage{{microtype}}
\usepackage{{geometry}}
\geometry{{margin=1in}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{mathtools}}
\usepackage{{graphicx}}
\usepackage{{subcaption}}
\usepackage{{booktabs}}
\usepackage{{multirow}}
\usepackage{{array}}
\usepackage{{tabularx}}
\usepackage{{longtable}}
\usepackage{{hyperref}}
\usepackage{{cleveref}}
\usepackage{{xcolor}}

\title{{{TITLE}}}
\author{{Author information to be inserted}}
\date{{}}

\begin{{document}}
\maketitle

\begin{{abstract}}
Open-set recognition is a central but under-tested requirement for hematological image classifiers: a model trained on a fixed taxonomy may encounter morphologies outside that taxonomy while still being forced to assign every image to a known class. We evaluated this problem using a frozen, reproducible protocol for white blood cell morphology recognition. The internal development domain was MLL23, with 41,621 single-cell images, five known mature leukocyte classes, and 13 held-out unknown morphologies. Across five seeds and two split protocols, conventional ResNet18 classifiers achieved high internal closed-set macro-F1 near 0.97, while internal open-set performance remained only moderately strong. CE+ViM achieved AUROC 0.877 $\pm$ 0.004 on V1 and 0.874 $\pm$ 0.008 on V2, but its advantage did not transfer externally. On AML-LMU, an independent 18,365-image test-only dataset, all principal systems degraded: external AUROC ranged from 0.563 to 0.696 across V1 systems and from 0.573 to 0.696 across V2 systems, with high FPR@95TPR. The ranking of post-hoc methods changed under domain shift, with MSP outperforming ViM externally in the principal comparisons. Failure analysis showed strong morphology dependence, including difficult rejection of neutrophil-band, monoblast, myeloblast, lymphocyte-atypical, and smudge-cell morphologies. ArcFace improved some known-class geometry summaries but failed multiseed/split confirmation as a robust superior representation. Cubical persistent homology and Vietoris-Rips topology were informative as negative and explanatory analyses, respectively, but not as confirmed predictors of open-set performance. These results show that strong closed-set classification does not guarantee reliable rejection of unseen hematological morphologies, especially under morphological similarity and acquisition-domain shift.
\end{{abstract}}

\input{{sections/01_introduction}}
\input{{sections/02_related_work}}
\input{{sections/03_data_protocol}}
\input{{sections/04_methods}}
\input{{sections/05_internal_results}}
\input{{sections/06_failure_analysis}}
\input{{sections/07_external_validation}}
\input{{sections/08_topological_analysis}}
\input{{sections/09_discussion}}
\input{{sections/10_limitations}}
\input{{sections/11_conclusion}}

\bibliographystyle{{plain}}
\bibliography{{references}}

\end{{document}}
""",
    )


def write_sections() -> None:
    write(
        SECTIONS / "01_introduction.tex",
        r"""
\section{Introduction}

Automated recognition of white blood cell morphology is often framed as a closed-set image classification task: every evaluated image is assumed to belong to one of the cell types observed during training. Under that framing, modern convolutional classifiers can achieve strong performance on curated single-cell microscopy datasets. This is scientifically useful, but it does not answer a harder operational question. A system trained to distinguish a fixed set of mature leukocyte morphologies may encounter immature, atypical, rare, or otherwise out-of-taxonomy morphologies at test time. In that setting, assigning every input to the closest known class can be misleading as an evaluation behavior, even for research-only systems.

This study investigates open-set recognition for hematological cell morphology. We define the known taxonomy as five mature leukocyte classes: basophil, eosinophil, lymphocyte, monocyte, and segmented neutrophil. All other evaluated morphologies are treated as unknown only in the protocol sense: they are outside the known training taxonomy. This terminology is deliberately narrow. An unknown sample is not a cancer label, leukemia label, malignancy label, diagnostic label, or clinically positive label. The task is not to diagnose disease. The task is to ask whether a classifier trained on a known morphology taxonomy can detect that a test image should not be forced into that taxonomy.

The distinction matters because closed-set classification and unknown rejection measure different behavior. A model may learn a compact representation for the known classes and still place an unseen but morphologically related cell close to a known class. For example, a band neutrophil may be visually and semantically close to segmented neutrophils, and lymphoid-like unknowns may be attracted toward the known lymphocyte class. Such behavior can leave closed-set accuracy high while open-set reliability remains limited. The scientific question is therefore not whether deep networks can classify known leukocytes, but whether the confidence and embedding structure of those networks support reliable rejection of unseen morphologies.

We approach this question as a reproducible evaluation study rather than as a claim of a new superior method. The internal dataset is MLL23, a large expert-annotated single-cell peripheral blood dataset with 18 morphology classes \cite{mll23_nature_2025,mll23_zenodo_2024}. We use five mature leukocyte classes as the known training taxonomy and hold the remaining classes out for open-set testing. We evaluate two split protocols: the original deterministic split and a conservative near-duplicate-aware sensitivity split. Because reliable patient or acquisition-group identifiers are not available in the working MLL23 manifests, the conservative split should be interpreted as a leakage sensitivity analysis rather than as proof of patient-level separation.

The external dataset is AML-Cytomorphology\_LMU, an independent TCIA collection of 18,365 expert-labeled single-cell images \cite{aml_lmu_tcia_2019}. We map only exact source labels to the five known classes and keep immature or atypical morphologies outside the known taxonomy. The external dataset is used only for testing. There is no external fine-tuning, external threshold calibration, external ViM fitting, external model selection, or taxonomy modification. This makes AML-LMU central to the paper: it evaluates whether conclusions drawn internally survive a change in acquisition domain and class composition.

The experiments cover three families of approaches. First, we evaluate a conventional ImageNet-pretrained ResNet18 classifier with post-hoc open-set scores, including maximum softmax probability (MSP), energy, distance-based scores, ReAct, and ViM. Second, we evaluate representation-learning alternatives, including supervised contrastive learning and ArcFace-style angular-margin training. Third, we evaluate persistent homology in two distinct roles. Cubical persistent homology over image, morphology-candidate, and feature-map filtrations is treated as a predictive ablation. Vietoris-Rips persistent homology over learned embedding clouds is treated only as an explanatory analysis of representation topology. These roles are intentionally separated, because using topology as a classifier feature is a different claim from using topology to describe learned geometry.

The central finding is conservative but important: strong known-class classification does not imply reliable open-set recognition of unseen hematological morphologies. Across internal five-seed experiments, closed-set macro-F1 remained near 0.97, yet open-set metrics were more variable and morphology dependent. ArcFace improved some known-class geometry summaries and showed promising behavior in selected conditions, but it failed multiseed and split confirmation as a robust superior representation. Under external AML-LMU validation, every principal system degraded substantially, and the internal ranking of post-hoc open-set methods did not transfer cleanly. In particular, CE+ViM was the stable internal baseline, but MSP outperformed ViM externally in the principal blocks.

Our contribution is therefore a claim-disciplined evaluation framework and evidence base for open-set hematological morphology recognition under representation instability and acquisition-domain shift. The paper makes the following contributions:
\begin{itemize}
\item We construct a reproducible open-set evaluation protocol for hematological cell morphology, with explicit known and unknown taxonomies, strict train/validation/test separation, and frozen rules for unknown terminology.
\item We perform multiseed and split-sensitivity analysis of post-hoc open-set scoring and representation learning, showing that closed-set geometry improvements do not automatically translate into robust unknown rejection.
\item We externally evaluate frozen systems on AML-LMU without external training or tuning, showing substantial domain-shift degradation and changed method rankings.
\item We analyze morphology-dependent failure modes using per-class open-set metrics, known-class attractors, embedding geometry, and secondary Vietoris-Rips topology.
\end{itemize}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/figure1_protocol.png}
\caption{Frozen open-set evaluation protocol. MLL23 supports internal multiseed and split-sensitivity analysis; AML-LMU is used as an external test-only domain.}
\label{fig:protocol}
\end{figure}
""",
    )
    write(
        SECTIONS / "02_related_work.tex",
        r"""
\section{Related Work}

\subsection{Automated Hematological Cell Classification}

Deep learning has been widely applied to microscopic blood cell classification, where the dominant evaluation setup is often closed-set recognition over a fixed label inventory. In that setting, the model is evaluated only on classes that appear during training, and every image is expected to receive one of those labels. The MLL23 dataset provides a recent large-scale expert-annotated resource for peripheral blood single-cell morphology research \cite{mll23_nature_2025,mll23_zenodo_2024}. The AML-LMU collection provides an independent single-cell leukocyte morphology dataset from a different source and acquisition context \cite{aml_lmu_tcia_2019}. A broader review citation for automated hematological cell classification remains to be verified before submission \cite{TODO_hematology_ai_review}.

Our study differs from standard closed-set classification work in two ways. First, the unknown classes are not incidental anomalies but named hematological morphologies deliberately excluded from the known training taxonomy. Second, the central evaluation asks whether systems trained on mature leukocytes can reject out-of-taxonomy morphologies, not whether they can assign a disease status. This places the work closer to open-set recognition and out-of-distribution detection than to diagnostic classification.

\subsection{Open-Set Recognition and OOD Detection}

Open-set recognition formalizes the setting in which test inputs may not belong to any class observed during training. OpenMax and related work emphasized that deep networks trained under closed-set assumptions can assign high confidence to inputs outside the training taxonomy \cite{bendale2016openmax}. Maximum softmax probability remains an important baseline because it is simple, model-agnostic, and often difficult to beat robustly \cite{hendrycks2017baseline}. Energy-based scores, ReAct, and other post-hoc approaches attempt to improve unknown detection by using logits, activations, or feature-space structure \cite{liu2020energy,sun2021react}.

The present work uses these methods as controlled baselines rather than as a venue for open-ended score engineering. The main post-hoc methods are fit using known-training data only, with thresholds calibrated from known-validation samples where thresholds are reported. Unknown-test samples are reserved for final reporting and analysis. This is important because small choices in score orientation, calibration, or PCA fitting can otherwise leak unknown-test information into method selection.

\subsection{Representation Learning for Open-Set Recognition}

Representation learning can improve the structure of known-class embeddings. Supervised contrastive learning encourages same-class samples to cluster while separating different classes \cite{khosla2020supcon}. ArcFace uses an additive angular margin during training to improve class separability in normalized embedding space \cite{deng2019arcface}. Such losses are attractive for open-set recognition because unknown rejection often depends on how known and unknown samples are arranged in feature space.

However, improved known-class geometry is not equivalent to improved open-set reliability. A representation can compress the known classes while also pulling morphologically related unknowns toward a known prototype. This study therefore treats ArcFace and supervised contrastive training as representation-learning ablations. ArcFace is not presented as a confirmed proposed method; its apparent gains in selected internal or external conditions must be interpreted together with the multiseed and split-sensitivity evidence.

\subsection{Topological Data Analysis of Learned Representations}

Persistent homology summarizes connected components, loops, and higher-dimensional topological features across filtrations \cite{edelsbrunner2002persistence,carlsson2009topology}. The GUDHI library provides computational tools for cubical and simplicial persistent homology \cite{maria2014gudhi}. In image analysis, cubical filtrations can be applied to intensity maps, morphology-derived distance fields, or feature maps. For learned embeddings, Vietoris-Rips complexes can be constructed from pairwise distances in a point cloud.

This paper separates two topological roles. Cubical persistent homology is evaluated as a potential predictive representation or open-set score. Vietoris-Rips topology is used only to characterize learned embedding clouds after the predictive experiments are frozen. The evidence supports a cautious conclusion: topology provides useful descriptive summaries and rigorous negative ablations, but the current artifacts do not support a claim that persistent topology predicts or causes open-set performance.
""",
    )
    write(
        SECTIONS / "03_data_protocol.tex",
        r"""
\section{Data and Open-Set Protocol}

\subsection{Known and Unknown Taxonomies}

Let $\mathcal{Y}_K$ denote the known training taxonomy. In this study,
\[
\mathcal{Y}_K=\{\text{basophil},\text{eosinophil},\text{lymphocyte},\text{monocyte},\text{neutrophil\_segmented}\}.
\]
All remaining evaluated morphologies are outside $\mathcal{Y}_K$ and are treated as unknown for open-set evaluation. This definition is protocol-specific. It does not imply that an unknown image is malignant, leukemic, cancerous, clinically abnormal, or diagnostically positive.

\subsection{MLL23 Internal Dataset}

MLL23 contains 41,621 images in the audited manifest used here, with 18 canonical classes, TIFF format, and 288 by 288 pixel images. The split contains 12,793 known-training images, 2,741 known-validation images, 2,742 known-test images, and 23,345 unknown-test images. The five known classes are used for model fitting and closed-set testing. The 13 unknown classes are held out from training and used only for open-set reporting and explanatory analysis.

Two split protocols are evaluated. V1 is the original deterministic seed-37 split. V2 is a conservative near-duplicate-aware sensitivity split produced from checksum and perceptual-hash candidate audits. The near-duplicate audit identified high-confidence connected components and moved only two samples while preserving the split totals. Because the working manifests do not contain reliable patient identifiers, V2 is not described as patient-level validation. It is a conservative split-sensitivity audit against obvious or high-confidence near-duplicate leakage.

\subsection{AML-LMU External Dataset}

AML-Cytomorphology\_LMU is used as the independent external validation dataset. The audited manifest contains 18,365 TIFF images, all readable in the Delivery 7 quality-control artifact, with 400 by 400 RGB arrays after loading. The source collection documents expert-labeled single-cell peripheral blood smear images from 100 AML patients and 100 non-malignant controls. Patient identifiers are not available in the audited TCIA API fields or annotation file used by this project; subject counts are documented, but patient-level splitting is not part of this external test-only protocol.

External mapping is frozen before model evaluation. BAS, EOS, LYT, MON, and NGS map exactly to the five known classes. EBO, KSC, LYA, MMZ, MOB, MYB, MYO, NGB, PMB, and PMO are external protocol unknowns. UNC and missing re-annotation labels are excluded by policy and have zero samples in the audited gold-standard column. Band neutrophils are not collapsed into segmented neutrophils, atypical lymphocytes are not collapsed into typical lymphocytes, and blast or immature labels are not mapped to mature leukocyte classes.

AML-LMU is never used for training, checkpoint selection, fine-tuning, threshold selection, PCA fitting, ViM fitting, scorer selection, or taxonomy revision. Each frozen model is evaluated on AML-LMU as a test-only domain. This is a stricter external validation setting than one in which external data are used for adaptation.

\input{tables/table1_datasets_protocol}
""",
    )
    write(
        SECTIONS / "04_methods.tex",
        r"""
\section{Methods}

\subsection{Closed-Set Classifier}

The principal classifier family is an ImageNet-pretrained ResNet18 \cite{he2016resnet}. Images are resized to 224 by 224, normalized with the training pipeline, and passed through the network to obtain logits and a 512-dimensional pre-logit embedding. Training uses AdamW, learning rate 0.0003, weight decay 0.0001, batch size 64, automatic mixed precision, and known-validation macro-F1 checkpoint selection. Weighted cross entropy is used where specified by the frozen configuration. These details are taken from the predeclared Delivery 6 configuration and are not reconstructed from memory.

\subsection{Open-Set Prediction and Metrics}

For input $x$, the closed-set classifier outputs logits $f_c(x)$ for $c\in\mathcal{Y}_K$ and predicts
\[
\hat{y}(x)=\arg\max_{c\in\mathcal{Y}_K} f_c(x).
\]
Open-set evaluation additionally defines an anomaly score $S(x)$, oriented so that larger values are more unknown-like. Threshold-free metrics are computed over known-test and unknown-test samples. AUROC and AUPR treat unknown samples as the positive class unless explicitly stated otherwise. FPR@95TPR is the false-positive rate among known samples at 95\% true-positive rate for unknown samples. OSCR summarizes the tradeoff between correct known classification and unknown rejection. Closed-set accuracy, balanced accuracy, macro-F1, ECE, NLL, and Brier score are used to characterize known-class classification and calibration.

\subsection{Post-Hoc Open-Set Scores}

MSP uses the negative maximum softmax probability as an unknownness score. Energy uses the negative log-sum-exp energy convention after orientation so that larger values indicate greater unknownness. Cosine kNN uses the mean cosine distance to the $k=5$ nearest known-training embeddings. Nearest-class-mean and Mahalanobis variants compare embeddings to class prototypes or covariance-normalized residuals fitted from known-training data. ReAct clips pre-classifier activations at a known-training percentile and recomputes an energy-like score. ViM fits a known-training PCA residual subspace and combines residual magnitude with logit evidence to form an unknownness score \cite{wang2022vim}. All method statistics are fitted using known-training data only.

\subsection{Representation-Learning Ablations}

Supervised contrastive learning is evaluated as a representation-learning alternative that did not improve open-set performance under the frozen protocol. The SupCon branch uses a projection head during training but evaluates the backbone embedding for open-set scoring. ArcFace uses normalized features and normalized class weights. If $z$ is the embedding and $w_c$ is the class weight, then
\[
\hat{z}=\frac{z}{\lVert z\rVert},\qquad \hat{w}_c=\frac{w_c}{\lVert w_c\rVert},\qquad \cos\theta_c=\hat{w}_c^\top \hat{z}.
\]
For the target class $y$, the training logit is $s\cos(\theta_y+m)$ with scale $s=30$ and angular margin $m=0.30$. At inference time there is no target margin because the target label is unavailable. ArcFace configuration is frozen before multiseed confirmation and is treated as a major ablation, not as a confirmed primary method.

\subsection{Persistent Homology}

Cubical persistent homology is computed over image-derived filtrations: raw grayscale sublevel and superlevel fields, morphology-candidate distance maps, and frozen CNN activation-energy maps. Vectorized persistence features and TDA anomaly scores are fitted with training-only statistics. These experiments test whether persistent homology provides predictive complementary information for open-set detection.

Vietoris-Rips persistent homology is used differently. For a point cloud $X=\{z_1,\ldots,z_n\}$ of L2-normalized embeddings and cosine distance $d$, the Vietoris-Rips complex at scale $\epsilon$ is
\[
VR_\epsilon(X)=\{\sigma\subseteq X: d(z_i,z_j)\leq \epsilon\;\forall z_i,z_j\in\sigma\}.
\]
The analysis computes $H_0$ and $H_1$ summaries with GUDHI 3.13.0, using $N_{VR}=192$ samples per cloud and a fixed seed. These summaries are explanatory only. They are not used for prediction, thresholding, model selection, or scorer tuning.
""",
    )
    write(
        SECTIONS / "05_internal_results.tex",
        r"""
\section{Internal Results}

\subsection{Closed-Set Classification Is Strong Internally}

Internal closed-set recognition is consistently strong across five training seeds. For CE, closed-set macro-F1 is 0.970 $\pm$ 0.004 on V1 and 0.970 $\pm$ 0.004 on V2. For ArcFace, closed-set macro-F1 is 0.970 $\pm$ 0.004 on V1 and 0.969 $\pm$ 0.005 on V2. These values show that the models learn the known mature leukocyte taxonomy well. They do not, by themselves, establish open-set reliability.

\subsection{Internal Open-Set Performance}

The internal MLL23 open-set results are stronger than the external results but remain imperfect. CE+MSP reaches AUROC 0.868 $\pm$ 0.014 on V1 and 0.872 $\pm$ 0.019 on V2. CE+ViM reaches AUROC 0.877 $\pm$ 0.004 on V1 and 0.874 $\pm$ 0.008 on V2. The small standard deviation of CE+ViM supports its role as the conservative internal-stability baseline. However, the corresponding FPR@95TPR values remain high: 0.484 $\pm$ 0.027 on V1 and 0.528 $\pm$ 0.050 on V2. Thus, even the best internal baseline requires many known samples to be falsely flagged when the unknown true-positive rate is fixed at 95\%.

\input{tables/table2_internal_multiseed}

\subsection{ArcFace Improves Some Geometry but Fails Confirmation}

ArcFace shows the pattern that motivated a confirmatory evaluation. On V1 with MSP, ArcFace reaches AUROC 0.881 $\pm$ 0.012, compared with 0.868 $\pm$ 0.014 for CE+MSP. The matched V1 AUROC delta is +0.013 across five seed pairs, with four of five seeds favorable. However, the seed-level confidence interval crosses zero in the frozen Delivery 6 decision artifact, and the V2 result reverses direction. On V2, ArcFace+MSP reaches AUROC 0.852 $\pm$ 0.030, compared with 0.872 $\pm$ 0.019 for CE+MSP; the matched V2 delta is -0.020 and only one of five seeds is favorable.

ArcFace also changes the known-class embedding geometry. The Fisher ratio is higher for ArcFace than CE on both split protocols: 3.210 $\pm$ 0.181 versus 2.703 $\pm$ 0.065 on V1 and 3.234 $\pm$ 0.323 versus 2.691 $\pm$ 0.092 on V2. Within-class dispersion is much smaller because the ArcFace representation is normalized and angular. These geometric changes are real, but they do not provide a robust open-set improvement. Across the 20 runs, Fisher ratio has weak association with MSP AUROC: Pearson $r=-0.138$ and Spearman $\rho=0.030$. The result supports a limited conclusion: better known-class geometry, as measured here, does not guarantee better unknown rejection.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/figure2_internal_performance.png}
\caption{Internal MLL23 open-set performance over five seeds. Points are individual runs; horizontal bars are seed means.}
\label{fig:internal-performance}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\textwidth]{figures/figure3_arcface_stability.png}
\caption{Matched ArcFace-minus-CE seed deltas for MSP. Positive AUROC deltas favor ArcFace; negative FPR@95TPR deltas favor ArcFace. The V1 trend does not reproduce under V2.}
\label{fig:arcface-stability}
\end{figure}
""",
    )
    write(
        SECTIONS / "06_failure_analysis.tex",
        r"""
\section{Morphology-Dependent Failure Analysis}

Unknown rejection difficulty varies strongly by morphology. This is visible in both internal analyses and external AML-LMU results. The practical implication is that a single aggregate AUROC can hide qualitatively different failure modes. Some unknown classes are rejected with relatively high AUROC, while others are repeatedly absorbed by a semantically related known class.

\subsection{External Unknown Morphologies}

Under the external CE+ViM analysis, the easiest unknown morphologies by mean AUROC are promyelocyte (0.897), myelocyte (0.880), metamyelocyte (0.867), promyelocyte-bilobled (0.856), and erythroblast (0.792). The hardest are neutrophil-band (0.526), monoblast (0.575), myeloblast (0.592), lymphocyte-atypical (0.597), and smudge-cell (0.782). The associated FPR@95TPR values are especially high for neutrophil-band (0.944) and myeloblast (0.819). These values are morphology-specific open-set results, not clinical difficulty rankings.

\subsection{Attractor Behavior}

The hardest classes align with known-class attractors. External neutrophil-band images have segmented neutrophil as the dominant known prediction in all CE+ViM external runs. Monoblast maps dominantly to monocyte. Myeloblast, lymphocyte-atypical, and smudge-cell often map to lymphocyte. This indicates that unknown errors are not uniformly distributed. They cluster along morphologically plausible known-class neighborhoods, which helps explain why high closed-set classification accuracy can coexist with weak rejection.

\subsection{Neutrophil Band and Lymphoid-Like Cases}

Neutrophil-band is the clearest persistent failure family. In internal Delivery 6, ArcFace consistently degraded MSP rejection of neutrophil-band, with mean AUROC deltas of -0.120 on V1 and -0.112 on V2, and zero of five favorable seeds in both split protocols. In the external analysis, the same class remains difficult for CE+ViM, with AUROC 0.526 and FPR@95TPR 0.944. ViM improves band-neutrophil AUROC for ArcFace in the dedicated case summary, but that improvement does not make ViM the best external method overall.

Lymphoid-like unknowns show a different pattern. Large granular lymphocyte, neoplastic lymphocyte, reactive lymphocyte, and hairy-cell morphologies retain high attraction to the known lymphocyte class. For large granular lymphocyte, the mean attraction to lymphocyte is 0.972 under CE and 0.978 under ArcFace. ArcFace improves some MSP AUROCs in this family, but attraction remains high. Thus, a better rejection score in selected cases does not necessarily imply a disentangled representation.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.86\textwidth]{figures/figure5_external_morphology.png}
\caption{External morphology-dependent unknown detection for CE+ViM. Error bars show standard deviation across external CE+ViM runs.}
\label{fig:external-morphology}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/figure6_failure_cases.png}
\caption{Known-class attractor failure analysis for neutrophil-band and lymphoid-like unknown morphologies.}
\label{fig:failure-cases}
\end{figure}
""",
    )
    write(
        SECTIONS / "07_external_validation.tex",
        r"""
\section{External Validation}

\subsection{AML-LMU Test-Only Evaluation}

AML-LMU is the central external validation domain. The dataset is independent of MLL23 and is used only after the models, splits, scorers, and taxonomy mapping are frozen. No external images are used for fine-tuning, scorer selection, threshold calibration, PCA fitting, ViM fitting, kNN reference construction, or preprocessing decisions. The external experiment therefore asks whether the frozen internal systems generalize to a different acquisition domain and a different source taxonomy.

\subsection{Closed-Set Degradation}

External closed-set accuracy remains superficially high, but macro-F1 and balanced accuracy reveal substantial degradation. V1 ArcFace has the best external closed-set macro-F1 among the principal representations, 0.755 $\pm$ 0.009, while V1 CE reaches 0.711 $\pm$ 0.051. V2 ArcFace reaches 0.708 $\pm$ 0.036 and V2 CE reaches 0.705 $\pm$ 0.030. These values are far below the internal closed-set macro-F1 near 0.97. The result shows why class imbalance and aggregate accuracy should not be the only external validation lens.

\subsection{External Open-Set Results}

External open-set performance is substantially weaker than internal MLL23 performance. V1 CE+MSP reaches AUROC 0.696 $\pm$ 0.074 and FPR@95TPR 0.796 $\pm$ 0.134. V1 CE+ViM reaches AUROC 0.641 $\pm$ 0.090 and FPR@95TPR 0.785 $\pm$ 0.116. V1 ArcFace+MSP reaches AUROC 0.690 $\pm$ 0.045 and FPR@95TPR 0.689 $\pm$ 0.107, while V1 ArcFace+ViM reaches AUROC 0.563 $\pm$ 0.084 and FPR@95TPR 0.879 $\pm$ 0.083.

The V2 pattern is similarly limited. V2 CE+MSP reaches AUROC 0.630 $\pm$ 0.094 and FPR@95TPR 0.887 $\pm$ 0.100. V2 CE+ViM reaches AUROC 0.573 $\pm$ 0.123 and FPR@95TPR 0.867 $\pm$ 0.138. V2 ArcFace+MSP reaches AUROC 0.696 $\pm$ 0.069 and FPR@95TPR 0.723 $\pm$ 0.162. V2 ArcFace+ViM reaches AUROC 0.632 $\pm$ 0.087 and FPR@95TPR 0.851 $\pm$ 0.062.

\input{tables/table3_external_results}

\subsection{Method Rankings Do Not Transfer Cleanly}

The internal CE+ViM baseline is stable, but ViM does not remain best externally. MSP outperforms ViM by mean AUROC for both CE and ArcFace across V1 and V2 external comparisons. This is a central result because it shows that post-hoc method rankings themselves can be domain-sensitive. The practical external baseline is therefore CE+MSP: it is simple, reproducible, and competitive externally. CE+ViM remains the conservative internal-stability baseline, not the external winner.

\subsection{Internal-to-External Domain Shift}

All principal systems degrade from MLL23 to AML-LMU. For CE+ViM, AUROC drops from 0.877 to 0.641 on V1 and from 0.874 to 0.573 on V2. FPR@95TPR increases from 0.484 to 0.785 on V1 and from 0.528 to 0.867 on V2. Closed macro-F1 drops by 0.260 on V1 and 0.264 on V2. ArcFace+MSP is externally competitive, especially on V2, but it also degrades relative to internal performance. These exact deltas support the paper's main external-validity conclusion: domain shift materially changes both closed-set and open-set behavior.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/figure4_domain_shift.png}
\caption{Internal-to-external domain shift for principal systems. AML-LMU is test-only; no external training or tuning is used.}
\label{fig:domain-shift}
\end{figure}
""",
    )
    write(
        SECTIONS / "08_topological_analysis.tex",
        r"""
\section{Topological Analysis}

\subsection{Cubical Persistent Homology as a Predictive Ablation}

Cubical persistent homology was evaluated over raw images, morphology-conditioned distance maps, nucleus and cytoplasm candidates, CNN feature-map filtrations, early fusion, and late fusion. These experiments were useful because they tested whether topological summaries add predictive unknown-detection signal beyond deep embeddings. The answer under the evaluated protocol is negative. TDA-only features encode some class information, with the Delivery 2 TDA-only closed macro-F1 reaching 0.828, but open-set AUROC remains approximately 0.55--0.58 for TDA-only variants, and Deep+TDA fusion degrades the principal deep baseline. Delivery 3 morphology-aware and feature-map variants also fail the predeclared progression rule.

This is not a claim that topology contains no biological or morphological information. It is a narrower claim: the evaluated cubical persistent-homology representations did not provide robust complementary unknown-detection utility under the frozen protocol. The detailed ablations belong in the supplement because they support claim discipline without displacing the central external-validation result.

\subsection{Vietoris-Rips Embedding Topology}

The Vietoris-Rips analysis is secondary and explanatory. It comprises 500 cloud analyses over MLL23 and AML-LMU, V1 and V2 training splits, CE and ArcFace representations, five seeds, and known plus predeclared hard unknown morphology clouds. Each cloud has $H_0$ and $H_1$ summaries, producing 1,000 homology summary rows. The construction uses L2-normalized embeddings, cosine distance, $N_{VR}=192$, a fixed sample seed, and GUDHI 3.13.0.

The VR results are mixed. H0 bottleneck summaries are generally larger for ArcFace than CE across seed and split comparisons, but H1 behavior is weaker and not a clean explanation of open-set performance. The topology--OSR correlations are exploratory and inconsistent. The strongest observed associations involve mean known H1 total persistence and ViM metrics, but the broader pattern does not support predictive or causal claims. For MSP AUROC, mean known H1 max persistence has Pearson $r=0.148$ and Spearman $\rho=0.014$, while Fisher ratio has Pearson $r=-0.138$ and Spearman $\rho=0.030$ in the conventional geometry correlation table.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/figure7_topology.png}
\caption{Vietoris-Rips topology summaries and exploratory associations with OSR metrics. The analysis is descriptive only and is not used for prediction.}
\label{fig:topology}
\end{figure}
""",
    )
    write(
        SECTIONS / "09_discussion.tex",
        r"""
\section{Discussion}

\subsection{Closed-Set Accuracy Is Not Open-Set Reliability}

The study shows a consistent gap between closed-set classification and open-set recognition. Internally, closed-set macro-F1 is near 0.97 across the principal CE and ArcFace models. Externally, accuracy remains high enough that a closed-set-only report might appear favorable. Yet macro-F1, AUROC, FPR@95TPR, and morphology-specific analyses show that the open-set task is much harder. A classifier can learn the known mature leukocyte taxonomy and still accept out-of-taxonomy morphologies that are close to known classes.

\subsection{Morphological Similarity as a Persistent Failure Mode}

The error structure is not random. Neutrophil-band is attracted toward segmented neutrophils, monoblast toward monocyte, and several lymphoid-like unknowns toward lymphocyte. These results are consistent with morphology-dependent recognition difficulty, not with a clinical diagnosis claim. The finding argues for per-morphology reporting in hematological open-set studies. Aggregate AUROC is necessary, but it is insufficient when the unknown set contains visually related classes with different failure modes.

\subsection{Representation Compactness Does Not Guarantee Unknown Separation}

ArcFace is an informative ablation because it improves some known-class geometry summaries while failing robust open-set confirmation. This separates two hypotheses that are often conflated: compact known classes can help classification, but compactness alone does not ensure that unseen related morphologies are pushed away. The V1 ArcFace trend is promising, but V2 reverses the AUROC direction for MSP and the seed-level evidence does not support a global superiority claim.

\subsection{Acquisition-Domain Shift Dominates External Generalization}

AML-LMU external validation is the strongest stress test in the paper. The external dataset differs from MLL23 in source, acquisition context, image dimensions, class composition, and source taxonomy. The observed degradation is therefore plausibly related to acquisition-domain and taxonomy-domain differences, but the study does not isolate each contributor causally. Possible contributors include staining, scanner, preprocessing, population, and class-definition differences. The evidence supports a domain-shift limitation, not a mechanistic attribution to any single factor.

\subsection{What Persistent Homology Reveals and Does Not Reveal}

Persistent homology plays two disciplined roles. Cubical PH is a negative predictive ablation: it was tested across multiple image-derived filtrations and did not provide robust complementary unknown-detection utility. Vietoris-Rips PH is a descriptive analysis of learned embedding clouds. It reveals measurable seed, split, and domain variation, but its associations with open-set metrics are weak and inconsistent. This prevents the paper from claiming that topology improves open-set hematology while still preserving the scientific value of the topological experiments.

\subsection{Implications for Evaluation of Hematological AI Systems}

The results argue for evaluation protocols that include unknown morphologies, multiple seeds, split-sensitivity analysis, external validation, and claim-level traceability. They also argue against presenting a single favorable run or a single aggregate metric as evidence of readiness. This repository is research software, not a clinical tool. The outputs studied here should be interpreted as morphology-recognition and unknown-rejection measurements on research datasets, not as patient-level diagnostic endpoints.

\input{tables/table4_claim_evidence}
""",
    )
    write(
        SECTIONS / "10_limitations.tex",
        r"""
\section{Limitations}

This study has several important limitations. First, MLL23 is the only internal development domain, and AML-LMU is the only external validation domain. A stronger external-validity claim would require multiple independent sources with harmonized metadata. Second, reliable patient or acquisition-group identifiers are not available in the working MLL23 manifests, and AML-LMU patient identifiers are not documented in the audited fields used here. The near-duplicate-aware split reduces a specific leakage risk but does not prove patient-level independence.

Third, the unknown classes are defined by the protocol. They are outside the known training taxonomy, not necessarily clinically abnormal or diagnostically positive. Fourth, class imbalance is substantial in both internal and external datasets, making macro-F1, balanced accuracy, per-class metrics, and AUPR orientation important. Fifth, the multiseed analysis uses five training seeds and one ResNet18 backbone family. This is stronger than a single-run report but not exhaustive over architectures, optimizers, augmentations, or loss configurations.

Sixth, ArcFace is evaluated with a single frozen scale and margin. The study intentionally does not tune ArcFace after the confirmatory result because doing so would reopen method development. Seventh, the MLL23 unknown set was inspected during earlier deliveries, so later internal analyses are not independent discovery experiments. Eighth, the AML-LMU mapping is defensible but still a protocol mapping between source abbreviations and the known taxonomy. Ninth, no external adaptation is performed, which strengthens the test-only claim but may underestimate what could be achieved by explicitly domain-adaptive methods.

Finally, persistent homology is not exhausted as a mathematical tool. The negative cubical PH result applies to the evaluated filtrations, vectorizers, and fusion designs. The Vietoris-Rips analysis is explanatory and exploratory, with weak and inconsistent topology--OSR correlations. It should not be interpreted as a validated topological predictor of open-set reliability.
""",
    )
    write(
        SECTIONS / "11_conclusion.tex",
        r"""
\section{Conclusion}

This paper provides a claim-disciplined evaluation of open-set recognition for unseen hematological cell morphologies. The main lesson is that strong closed-set classification is not enough. Across internal MLL23 experiments, ResNet18 models classify known mature leukocytes with macro-F1 near 0.97, yet unknown rejection remains morphology-dependent and imperfect. ArcFace improves some known-class geometry summaries but does not reproduce as a robust superior open-set representation across seeds and split protocols. Cubical persistent homology provides a rigorous negative predictive ablation, and Vietoris-Rips topology provides descriptive information about learned embedding clouds without supporting a causal or predictive claim.

External AML-LMU validation is the decisive stress test. Without external training or tuning, all principal systems degrade, FPR@95TPR remains high, and post-hoc method rankings change. The most defensible conclusion is therefore not that topology or ArcFace solves open-set hematology, but that open-set reliability is substantially harder than closed-set classification and is shaped by morphology similarity, representation stability, and acquisition-domain shift. Future work should prioritize patient-aware datasets where available, additional independent external validation, and domain-robust representations evaluated under frozen, leakage-controlled protocols.
""",
    )


def write_supplement() -> None:
    write(
        SUPPLEMENT / "supplement.tex",
        r"""
\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{hyperref}
\title{Supplementary Material: Open-Set Recognition of Unseen Hematological Cell Morphologies under Domain Shift}
\author{Author information to be inserted}
\date{}
\begin{document}
\maketitle
\input{supplementary_methods}
\input{supplementary_tables}
\input{supplementary_figures}
\end{document}
""",
    )
    write(
        SUPPLEMENT / "supplementary_methods.tex",
        r"""
\section{Supplementary Methods}

The supplement records details that are necessary for reproducibility but too extensive for the main text. It includes the full taxonomy mapping, V1/V2 near-duplicate-aware split notes, post-hoc open-set score definitions, cubical persistent-homology methodology, morphology-aware and feature-map persistent-homology variants, SupCon and ArcFace training details, the 20-run Delivery 6 matrix, external seed-level results, external per-class metrics, Vietoris-Rips protocol details, and additional reproducibility metadata.

All supplementary analyses retain the same leakage rules as the main manuscript: unknown-test samples are never used for training, threshold selection, hyperparameter tuning, vectorizer fitting, PCA fitting, ViM fitting, kNN reference construction, or model selection.
""",
    )
    write(
        SUPPLEMENT / "supplementary_tables.tex",
        r"""
\section{Supplementary Tables}

Full machine-readable supplementary tables are stored in the repository artifacts and are referenced here rather than duplicated as oversized typeset tables:
\begin{itemize}
\item Full external taxonomy mapping: \path{artifacts/metrics/delivery7/external_taxonomy_mapping.csv}.
\item Full Delivery 6 run matrix: \path{artifacts/metrics/delivery6/run_level_results.csv}.
\item Full external seed-level results: \path{artifacts/metrics/delivery7/delivery75_external_seed_level_results.csv}.
\item Full external per-unknown morphology results: \path{artifacts/metrics/delivery7/delivery75_external_unknown_morphology_summary_ce_vim.csv}.
\item Full cubical PH and post-hoc ablation tables: \path{artifacts/metrics/delivery2/}, \path{artifacts/metrics/delivery3/}, and \path{artifacts/metrics/delivery4/}.
\item Full VR summaries: \path{artifacts/metrics/delivery7/vr_diagram_summaries.csv}, \path{artifacts/metrics/delivery7/delivery75_vr_seed_stability_summary.csv}, and \path{artifacts/metrics/delivery7/delivery75_vr_split_stability_summary.csv}.
\item Claim and result freezes for the manuscript: \path{paper/CLAIMS.md}, \path{paper/results_freeze.yaml}, and \path{paper/claim_traceability.csv}.
\end{itemize}
""",
    )
    write(
        SUPPLEMENT / "supplementary_figures.tex",
        r"""
\section{Supplementary Figures}

Supplementary figure candidates are inventoried in \path{artifacts/metrics/delivery7/article_figure_inventory.csv}. Delivery 2 and Delivery 3 figures are retained primarily for the cubical persistent-homology negative ablation. Delivery 4 and Delivery 5 figures support post-hoc scorer and representation-learning interpretation. Delivery 6 figures support multiseed and split robustness. Delivery 7 figures support Vietoris-Rips descriptive topology.
""",
    )


def word_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?", " ", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return len([token for token in text.split() if token])


def write_metadata() -> None:
    counts = []
    for path in sorted(SECTIONS.glob("*.tex")):
        counts.append(
            {"section_file": str(path.relative_to(ROOT)), "word_count_approx": word_count(path)}
        )
    with (PAPER / "section_word_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section_file", "word_count_approx"])
        writer.writeheader()
        writer.writerows(counts)

    figure_rows = [
        (
            "1",
            "paper/figures/figure1_protocol.png",
            "Experimental protocol and taxonomy",
            "redesigned",
        ),
        (
            "2",
            "paper/figures/figure2_internal_performance.png",
            "Internal multiseed OSR performance",
            "redesigned",
        ),
        (
            "3",
            "paper/figures/figure3_arcface_stability.png",
            "ArcFace matched seed deltas",
            "redesigned",
        ),
        (
            "4",
            "paper/figures/figure4_domain_shift.png",
            "Internal-to-external domain shift",
            "redesigned",
        ),
        (
            "5",
            "paper/figures/figure5_external_morphology.png",
            "External morphology-dependent failures",
            "redesigned",
        ),
        (
            "6",
            "paper/figures/figure6_failure_cases.png",
            "Known-class attractor failures",
            "redesigned",
        ),
        (
            "7",
            "paper/figures/figure7_topology.png",
            "Embedding geometry and VR topology",
            "redesigned",
        ),
    ]
    with (PAPER / "main_figure_plan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["figure", "artifact", "purpose", "status"])
        writer.writerows(figure_rows)

    table_rows = [
        ("1", "paper/tables/table1_datasets_protocol.tex", "Datasets and protocol"),
        ("2", "paper/tables/table2_internal_multiseed.tex", "Internal multiseed results"),
        ("3", "paper/tables/table3_external_results.tex", "External results"),
        ("4", "paper/tables/table4_claim_evidence.tex", "Claim-evidence summary"),
    ]
    with (PAPER / "main_table_plan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["table", "artifact", "purpose"])
        writer.writerows(table_rows)


def clean_build_tmp() -> None:
    for pattern in ["*.aux", "*.log", "*.out", "*.toc", "*.bbl", "*.blg", "*.fls", "*.fdb_latexmk"]:
        for path in PAPER.glob(pattern):
            path.unlink(missing_ok=True)
        for path in SUPPLEMENT.glob(pattern):
            path.unlink(missing_ok=True)


def main() -> None:
    mkdirs()
    write_claim_files()
    write_freeze_yaml()
    write_references()
    make_figures()
    write_tables()
    write_main_tex()
    write_sections()
    write_supplement()
    write_metadata()
    clean_build_tmp()
    print(f"Wrote manuscript to {PAPER / 'main.tex'}")


if __name__ == "__main__":
    main()
