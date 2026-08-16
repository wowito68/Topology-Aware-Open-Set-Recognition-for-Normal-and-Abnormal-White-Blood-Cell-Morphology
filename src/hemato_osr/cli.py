"""Command line interface for research workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hemato_osr.data.audit import LeakageError, audit_manifest
from hemato_osr.data.manifest import ManifestOptions, build_manifest, save_manifest
from hemato_osr.data.near_duplicates import (
    NearDuplicateConfig,
    build_candidate_components,
    generate_candidate_pairs,
    write_contact_sheets,
)
from hemato_osr.data.split_v2 import (
    ConservativeSplitConfig,
    build_conservative_split_v2,
    compare_splits,
    write_v2_manifest,
)
from hemato_osr.data.splitting import SplitConfig, assert_no_split_overlap, split_manifest
from hemato_osr.data.summary import write_dataset_summary
from hemato_osr.data.synthetic import SyntheticConfig, generate_synthetic_dataset
from hemato_osr.data.taxonomy import Taxonomy
from hemato_osr.embeddings.extract import EmbeddingExtractConfig, extract_embeddings
from hemato_osr.evaluation.pipeline import OpenSetEvaluationConfig, evaluate_open_set
from hemato_osr.experiments.delivery3 import (
    benchmark_feature_maps,
    benchmark_shape_tda,
    evaluate_delivery3,
    extract_feature_map_tda,
    extract_morphology_tda,
    run_morphology_qc,
    write_predeclared_matrix,
)
from hemato_osr.models.tda_baseline import TDAOnlyExperimentConfig, evaluate_tda_only
from hemato_osr.smoke import SmokeConfig, run_smoke_test
from hemato_osr.topology.extract import TDAExtractConfig, extract_tda_features
from hemato_osr.topology.pipeline import (
    TDADiagramExtractConfig,
    TDAResolutionBenchmarkConfig,
    TDAVectorizeConfig,
    extract_diagram_archive,
    run_resolution_benchmark,
    vectorize_diagram_archive,
)
from hemato_osr.training.fusion_train import (
    FrozenFusionExperimentConfig,
    FusionTrainConfig,
    train_frozen_fusion,
    train_fusion,
)
from hemato_osr.training.train import TrainConfig, train_closed_set


def _taxonomy_from_args(args: argparse.Namespace) -> Taxonomy:
    known = tuple(args.known_classes.split(",")) if getattr(args, "known_classes", "") else None
    unknown = (
        tuple(args.unknown_classes.split(",")) if getattr(args, "unknown_classes", "") else None
    )
    if known is None and unknown is None:
        return Taxonomy()
    return Taxonomy(
        known_classes=known or Taxonomy().known_classes,
        unknown_classes=unknown or Taxonomy().unknown_classes,
    )


def _add_taxonomy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--known-classes", default="", help="Comma-separated known class override.")
    parser.add_argument(
        "--unknown-classes",
        default="",
        help="Comma-separated unknown class override.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser."""

    parser = argparse.ArgumentParser(prog="hemato_osr")
    subcommands = parser.add_subparsers(dest="command", required=True)

    data = subcommands.add_parser("data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    synthetic = data_sub.add_parser("synthetic")
    synthetic.add_argument("output_dir", type=Path)
    synthetic.add_argument("--seed", type=int, default=13)
    synthetic.add_argument("--samples-per-known-class", type=int, default=6)
    synthetic.add_argument("--samples-per-unknown-class", type=int, default=4)
    synthetic.add_argument("--image-size", type=int, default=64)

    index = data_sub.add_parser("index")
    index.add_argument("root", type=Path)
    index.add_argument("--output", type=Path, required=True)
    index.add_argument("--dataset", default="mll23")
    index.add_argument("--group-from-parent-depth", type=int, default=None)
    _add_taxonomy_args(index)

    split = data_sub.add_parser("split")
    split.add_argument("manifest", type=Path)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument(
        "--method",
        choices=["stratified", "group-stratified", "predefined"],
        default="stratified",
    )
    split.add_argument("--seed", type=int, default=13)

    audit = data_sub.add_parser("audit")
    audit.add_argument("manifest", type=Path)
    audit.add_argument("--near-duplicate-hamming", type=int, default=4)
    audit.add_argument("--max-hamming-pairs", type=int, default=2_000_000)
    _add_taxonomy_args(audit)

    summarize = data_sub.add_parser("summarize")
    summarize.add_argument("manifest", type=Path)
    summarize.add_argument("--output-dir", type=Path, required=True)

    near_duplicates = data_sub.add_parser("near-duplicates")
    near_duplicates.add_argument("manifest", type=Path)
    near_duplicates.add_argument("--output-dir", type=Path, required=True)
    near_duplicates.add_argument("--candidate-hamming-max", type=int, default=0)
    near_duplicates.add_argument("--max-pairwise-pairs", type=int, default=2_000_000)
    near_duplicates.add_argument("--top-contact-sheet-pairs", type=int, default=100)

    split_v2 = data_sub.add_parser("split-v2")
    split_v2.add_argument("manifest", type=Path)
    split_v2.add_argument("--candidates", type=Path, required=True)
    split_v2.add_argument("--components", type=Path, required=True)
    split_v2.add_argument("--output", type=Path, required=True)
    split_v2.add_argument("--report", type=Path, required=True)
    split_v2.add_argument("--seed", type=int, default=37)

    train = subcommands.add_parser("train")
    train_sub = train.add_subparsers(dest="train_command", required=True)
    closed = train_sub.add_parser("closed-set")
    closed.add_argument("--manifest", type=Path, required=True)
    closed.add_argument("--output-dir", type=Path, required=True)
    closed.add_argument("--backbone", default="resnet18")
    closed.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    closed.add_argument("--image-size", type=int, default=224)
    closed.add_argument("--batch-size", type=int, default=16)
    closed.add_argument("--epochs", type=int, default=5)
    closed.add_argument("--learning-rate", type=float, default=1e-3)
    closed.add_argument("--weight-decay", type=float, default=1e-4)
    closed.add_argument("--imbalance-strategy", default="none")
    closed.add_argument("--seed", type=int, default=13)
    closed.add_argument("--device", default="auto")
    closed.add_argument("--precision", choices=["fp32", "amp"], default="fp32")
    closed.add_argument("--early-stopping-patience", type=int, default=5)
    closed.add_argument("--num-workers", type=int, default=0)
    closed.add_argument("--log-file", type=Path, default=None)
    _add_taxonomy_args(closed)

    fusion = train_sub.add_parser("fusion")
    fusion.add_argument("--embeddings", type=Path, required=True)
    fusion.add_argument("--tda-features", type=Path, required=True)
    fusion.add_argument("--output", type=Path, required=True)
    fusion.add_argument("--epochs", type=int, default=5)
    fusion.add_argument("--seed", type=int, default=13)

    frozen_fusion = train_sub.add_parser("frozen-fusion")
    frozen_fusion.add_argument("--embeddings", type=Path, required=True)
    frozen_fusion.add_argument("--tda-features", type=Path, required=True)
    frozen_fusion.add_argument("--output-dir", type=Path, required=True)
    frozen_fusion.add_argument("--components", default="")
    frozen_fusion.add_argument("--epochs", type=int, default=50)
    frozen_fusion.add_argument("--batch-size", type=int, default=256)
    frozen_fusion.add_argument("--learning-rate", type=float, default=1e-3)
    frozen_fusion.add_argument("--seed", type=int, default=37)
    frozen_fusion.add_argument("--device", default="auto")

    embeddings = subcommands.add_parser("embeddings")
    embeddings_sub = embeddings.add_subparsers(dest="embeddings_command", required=True)
    extract = embeddings_sub.add_parser("extract")
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--checkpoint", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--batch-size", type=int, default=32)
    extract.add_argument("--image-size", type=int, default=None)
    extract.add_argument("--device", default="auto")

    tda = subcommands.add_parser("tda")
    tda_sub = tda.add_subparsers(dest="tda_command", required=True)
    tda_extract = tda_sub.add_parser("extract")
    tda_extract.add_argument("--manifest", type=Path, required=True)
    tda_extract.add_argument("--output", type=Path, required=True)
    tda_extract.add_argument("--filtration", choices=["sublevel", "superlevel"], default="sublevel")
    tda_extract.add_argument("--image-size", type=int, default=64)
    tda_extract.add_argument(
        "--vectorizer",
        choices=[
            "persistence-entropy",
            "betti-curve",
            "persistence-image",
            "persistence-landscape",
        ],
        default="persistence-entropy",
    )

    tda_benchmark = tda_sub.add_parser("benchmark-resolution")
    tda_benchmark.add_argument("--manifest", type=Path, required=True)
    tda_benchmark.add_argument("--output-dir", type=Path, required=True)
    tda_benchmark.add_argument("--sample-size", type=int, default=1000)
    tda_benchmark.add_argument("--seed", type=int, default=37)

    tda_archive = tda_sub.add_parser("extract-archive")
    tda_archive.add_argument("--manifest", type=Path, required=True)
    tda_archive.add_argument("--output", type=Path, required=True)
    tda_archive.add_argument("--image-size", type=int, required=True)
    tda_archive.add_argument("--workers", type=int, default=2)
    tda_archive.add_argument("--seed", type=int, default=37)

    tda_vectorize = tda_sub.add_parser("vectorize-archive")
    tda_vectorize.add_argument("--diagrams", type=Path, required=True)
    tda_vectorize.add_argument("--output", type=Path, required=True)
    tda_vectorize.add_argument("--filtrations", default="sublevel,superlevel")
    tda_vectorize.add_argument(
        "--vectorizers",
        default="persistence-entropy,betti-curve,persistence-landscape,persistence-image",
    )
    tda_vectorize.add_argument("--seed", type=int, default=37)

    tda_only = tda_sub.add_parser("evaluate-only")
    tda_only.add_argument("--features", type=Path, required=True)
    tda_only.add_argument("--output-dir", type=Path, required=True)
    tda_only.add_argument("--components", default="")
    tda_only.add_argument("--osr-method", default="centroid_distance")
    tda_only.add_argument("--seed", type=int, default=37)

    evaluate = subcommands.add_parser("evaluate")
    eval_sub = evaluate.add_subparsers(dest="evaluate_command", required=True)
    open_set = eval_sub.add_parser("open-set")
    open_set.add_argument("--embeddings", type=Path, required=True)
    open_set.add_argument("--output-dir", type=Path, required=True)
    open_set.add_argument("--method", default="msp")
    open_set.add_argument("--target-known-recall", type=float, default=0.95)

    smoke = subcommands.add_parser("smoke")
    smoke.add_argument("--output-dir", type=Path, default=Path("artifacts/smoke"))
    smoke.add_argument("--seed", type=int, default=13)

    delivery3 = subcommands.add_parser("delivery3")
    d3_sub = delivery3.add_subparsers(dest="delivery3_command", required=True)
    d3_qc = d3_sub.add_parser("morphology-qc")
    d3_qc.add_argument("--manifest", type=Path, required=True)
    d3_qc.add_argument("--output-dir", type=Path, required=True)
    d3_qc.add_argument("--per-class", type=int, default=20)
    d3_qc.add_argument("--seed", type=int, default=37)

    d3_shape = d3_sub.add_parser("benchmark-shape")
    d3_shape.add_argument("--manifest", type=Path, required=True)
    d3_shape.add_argument("--output-dir", type=Path, required=True)
    d3_shape.add_argument("--sample-size", type=int, default=500)
    d3_shape.add_argument("--seed", type=int, default=37)

    d3_fmap = d3_sub.add_parser("benchmark-feature-map")
    d3_fmap.add_argument("--manifest", type=Path, required=True)
    d3_fmap.add_argument("--checkpoint", type=Path, required=True)
    d3_fmap.add_argument("--output-dir", type=Path, required=True)
    d3_fmap.add_argument("--sample-size", type=int, default=500)
    d3_fmap.add_argument("--batch-size", type=int, default=64)
    d3_fmap.add_argument("--seed", type=int, default=37)
    d3_fmap.add_argument("--device", default="auto")

    d3_predeclare = d3_sub.add_parser("predeclare")
    d3_predeclare.add_argument("--output", type=Path, required=True)
    d3_predeclare.add_argument("--selected-layer", required=True)
    d3_predeclare.add_argument("--include-cytoplasm", action="store_true")

    d3_morph_extract = d3_sub.add_parser("extract-morphology")
    d3_morph_extract.add_argument("--manifest", type=Path, required=True)
    d3_morph_extract.add_argument("--diagrams", type=Path, required=True)
    d3_morph_extract.add_argument("--features", type=Path, required=True)
    d3_morph_extract.add_argument("--views", default="cell,nucleus,cytoplasm")
    d3_morph_extract.add_argument("--workers", type=int, default=2)
    d3_morph_extract.add_argument("--seed", type=int, default=37)

    d3_fmap_extract = d3_sub.add_parser("extract-feature-map")
    d3_fmap_extract.add_argument("--manifest", type=Path, required=True)
    d3_fmap_extract.add_argument("--checkpoint", type=Path, required=True)
    d3_fmap_extract.add_argument("--diagrams", type=Path, required=True)
    d3_fmap_extract.add_argument("--features", type=Path, required=True)
    d3_fmap_extract.add_argument("--layer", required=True)
    d3_fmap_extract.add_argument("--batch-size", type=int, default=64)
    d3_fmap_extract.add_argument("--seed", type=int, default=37)
    d3_fmap_extract.add_argument("--device", default="auto")

    d3_eval = d3_sub.add_parser("evaluate")
    d3_eval.add_argument("--embeddings", type=Path, required=True)
    d3_eval.add_argument("--morphology-features", type=Path, required=True)
    d3_eval.add_argument("--feature-map-features", type=Path, required=True)
    d3_eval.add_argument("--raw-tda-predictions", type=Path, required=True)
    d3_eval.add_argument("--matrix", type=Path, required=True)
    d3_eval.add_argument("--output-dir", type=Path, required=True)
    d3_eval.add_argument("--checkpoint", type=Path, required=True)
    d3_eval.add_argument("--seed", type=int, default=37)
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "data" and args.data_command == "synthetic":
        generate_synthetic_dataset(
            SyntheticConfig(
                output_dir=args.output_dir,
                samples_per_known_class=args.samples_per_known_class,
                samples_per_unknown_class=args.samples_per_unknown_class,
                image_size=args.image_size,
                seed=args.seed,
            )
        )
        print(args.output_dir)
        return

    if args.command == "data" and args.data_command == "index":
        frame = build_manifest(
            ManifestOptions(
                root=args.root,
                dataset=args.dataset,
                group_from_parent_depth=args.group_from_parent_depth,
            ),
            _taxonomy_from_args(args),
        )
        save_manifest(frame, args.output)
        print(args.output)
        return

    if args.command == "data" and args.data_command == "split":
        frame = pd.read_csv(args.manifest)
        result = split_manifest(frame, SplitConfig(method=args.method, seed=args.seed))
        assert_no_split_overlap(result)
        save_manifest(result, args.output)
        print(args.output)
        return

    if args.command == "data" and args.data_command == "audit":
        frame = pd.read_csv(args.manifest)
        try:
            report = audit_manifest(
                frame,
                _taxonomy_from_args(args),
                near_duplicate_hamming=args.near_duplicate_hamming,
                max_hamming_pairs=args.max_hamming_pairs,
            )
        except LeakageError as exc:
            print(json.dumps({"passed": False, "errors": str(exc).split("; ")}, indent=2))
            raise SystemExit(2) from exc
        print(json.dumps(report.__dict__ | {"passed": report.passed}, indent=2))
        return

    if args.command == "data" and args.data_command == "summarize":
        frame = pd.read_csv(args.manifest)
        summary = write_dataset_summary(frame, output_dir=args.output_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.command == "data" and args.data_command == "near-duplicates":
        frame = pd.read_csv(args.manifest)
        cfg = NearDuplicateConfig(
            candidate_hamming_max=args.candidate_hamming_max,
            max_pairwise_pairs=args.max_pairwise_pairs,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        candidates = generate_candidate_pairs(frame, cfg)
        candidates_path = args.output_dir / "candidate_pairs.csv"
        candidates.to_csv(candidates_path, index=False)
        components = build_candidate_components(frame, candidates, cfg)
        components_path = args.output_dir / "components.csv"
        components.to_csv(components_path, index=False)
        sheets = write_contact_sheets(
            candidates,
            args.output_dir / "contact_sheets",
            top_n=args.top_contact_sheet_pairs,
        )
        (args.output_dir / "near_duplicate_config.json").write_text(
            json.dumps(cfg.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "candidate_pairs": int(len(candidates)),
                    "components": int(components["component_id"].nunique()),
                    "contact_sheets": [str(path) for path in sheets],
                    "config": cfg.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "data" and args.data_command == "split-v2":
        frame = pd.read_csv(args.manifest)
        candidates = pd.read_csv(args.candidates)
        components = pd.read_csv(args.components)
        result = build_conservative_split_v2(
            frame,
            components,
            ConservativeSplitConfig(seed=args.seed),
        )
        assert_no_split_overlap(result)
        hash_value = write_v2_manifest(result, args.output)
        split_report = compare_splits(frame, result, candidates, components)
        split_report["manifest"] = str(args.output)
        split_report["manifest_hash"] = hash_value
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(split_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(split_report, indent=2, sort_keys=True))
        return

    if args.command == "train" and args.train_command == "closed-set":
        taxonomy = _taxonomy_from_args(args)
        checkpoint = train_closed_set(
            TrainConfig(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                known_classes=taxonomy.known_classes,
                backbone=args.backbone,
                pretrained=args.pretrained,
                image_size=args.image_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                imbalance_strategy=args.imbalance_strategy,
                early_stopping_patience=args.early_stopping_patience,
                seed=args.seed,
                device=args.device,
                precision=args.precision,
                log_path=args.log_file,
            )
        )
        print(checkpoint)
        return

    if args.command == "train" and args.train_command == "fusion":
        output = train_fusion(
            FusionTrainConfig(
                embeddings_path=args.embeddings,
                tda_features_path=args.tda_features,
                output_path=args.output,
                epochs=args.epochs,
                seed=args.seed,
            )
        )
        print(output)
        return

    if args.command == "train" and args.train_command == "frozen-fusion":
        output = train_frozen_fusion(
            FrozenFusionExperimentConfig(
                embeddings_path=args.embeddings,
                tda_features_path=args.tda_features,
                output_dir=args.output_dir,
                feature_components=tuple(filter(None, args.components.split(","))),
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
                device=args.device,
            )
        )
        print(output)
        return

    if args.command == "embeddings" and args.embeddings_command == "extract":
        output = extract_embeddings(
            EmbeddingExtractConfig(
                manifest_path=args.manifest,
                checkpoint_path=args.checkpoint,
                output_path=args.output,
                batch_size=args.batch_size,
                image_size=args.image_size,
                device=args.device,
            )
        )
        print(output)
        return

    if args.command == "tda" and args.tda_command == "extract":
        output = extract_tda_features(
            TDAExtractConfig(
                manifest_path=args.manifest,
                output_path=args.output,
                filtration=args.filtration,
                image_size=args.image_size,
                vectorizer=args.vectorizer,
            )
        )
        print(output)
        return

    if args.command == "tda" and args.tda_command == "benchmark-resolution":
        output = run_resolution_benchmark(
            TDAResolutionBenchmarkConfig(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                sample_size=args.sample_size,
                seed=args.seed,
            )
        )
        print(output)
        return

    if args.command == "tda" and args.tda_command == "extract-archive":
        output = extract_diagram_archive(
            TDADiagramExtractConfig(
                manifest_path=args.manifest,
                output_path=args.output,
                image_size=args.image_size,
                workers=args.workers,
                seed=args.seed,
            )
        )
        print(output)
        return

    if args.command == "tda" and args.tda_command == "vectorize-archive":
        output = vectorize_diagram_archive(
            TDAVectorizeConfig(
                diagram_path=args.diagrams,
                output_path=args.output,
                filtrations=tuple(filter(None, args.filtrations.split(","))),
                vectorizers=tuple(filter(None, args.vectorizers.split(","))),
                seed=args.seed,
            )
        )
        print(output)
        return

    if args.command == "tda" and args.tda_command == "evaluate-only":
        output = evaluate_tda_only(
            TDAOnlyExperimentConfig(
                features_path=args.features,
                output_dir=args.output_dir,
                feature_components=tuple(filter(None, args.components.split(","))),
                osr_method=args.osr_method,
                seed=args.seed,
            )
        )
        print(output)
        return

    if args.command == "evaluate" and args.evaluate_command == "open-set":
        output = evaluate_open_set(
            OpenSetEvaluationConfig(
                embeddings_path=args.embeddings,
                output_dir=args.output_dir,
                open_set_method=args.method,
                target_known_recall=args.target_known_recall,
            )
        )
        print(output)
        return

    if args.command == "smoke":
        outputs = run_smoke_test(SmokeConfig(output_dir=args.output_dir, seed=args.seed))
        print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
        return

    if args.command == "delivery3" and args.delivery3_command == "morphology-qc":
        output = run_morphology_qc(
            args.manifest,
            args.output_dir,
            per_class=args.per_class,
            seed=args.seed,
        )
        print(output)
        return

    if args.command == "delivery3" and args.delivery3_command == "benchmark-shape":
        output = benchmark_shape_tda(
            args.manifest,
            args.output_dir,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        print(output)
        return

    if args.command == "delivery3" and args.delivery3_command == "benchmark-feature-map":
        output = benchmark_feature_maps(
            args.manifest,
            args.checkpoint,
            args.output_dir,
            sample_size=args.sample_size,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )
        print(output)
        return

    if args.command == "delivery3" and args.delivery3_command == "predeclare":
        output = write_predeclared_matrix(
            args.output,
            selected_layer=args.selected_layer,
            include_cytoplasm=args.include_cytoplasm,
        )
        print(output)
        return

    if args.command == "delivery3" and args.delivery3_command == "extract-morphology":
        output = extract_morphology_tda(
            args.manifest,
            args.diagrams,
            args.features,
            views=tuple(filter(None, args.views.split(","))),
            workers=args.workers,
            seed=args.seed,
        )
        print(output)
        return

    if args.command == "delivery3" and args.delivery3_command == "extract-feature-map":
        output = extract_feature_map_tda(
            args.manifest,
            args.checkpoint,
            args.diagrams,
            args.features,
            layer=args.layer,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )
        print(output)
        return

    if args.command == "delivery3" and args.delivery3_command == "evaluate":
        output = evaluate_delivery3(
            args.embeddings,
            args.morphology_features,
            args.feature_map_features,
            args.raw_tda_predictions,
            args.matrix,
            args.output_dir,
            checkpoint_path=args.checkpoint,
            seed=args.seed,
        )
        print(output)
        return

    parser.error("Unhandled command")
