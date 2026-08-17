#!/usr/bin/env python
"""Run Delivery 7 external validation and topology utilities."""

from __future__ import annotations

import argparse

from hemato_osr.experiments.delivery7 import (
    Delivery7Paths,
    build_external_manifest,
    build_vr_sample_manifest,
    export_all_external_embeddings,
    run_external_evaluation,
    run_vr_analysis,
    write_external_audit_update,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "build-external-manifest",
            "export-external-embeddings",
            "evaluate-external",
            "build-vr-sample-manifest",
            "analyze-vr",
        ],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = Delivery7Paths()
    if args.command == "build-external-manifest":
        build_external_manifest(
            data_root=paths.external_data_root,
            annotations_path=paths.external_annotations_path,
            taxonomy_path=paths.external_taxonomy_path,
            output_path=paths.external_manifest_path,
            qc_path=paths.data_audit_dir / "external_qc.csv",
        )
        write_external_audit_update(paths)
        return
    if args.command == "export-external-embeddings":
        export_all_external_embeddings(paths, device=args.device, num_workers=args.num_workers)
        return
    if args.command == "evaluate-external":
        run_external_evaluation(paths)
        return
    if args.command == "build-vr-sample-manifest":
        build_vr_sample_manifest(paths)
        return
    if args.command == "analyze-vr":
        run_vr_analysis(paths)
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
