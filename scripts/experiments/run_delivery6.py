#!/usr/bin/env python
"""Run the predeclared Delivery 6 multiseed confirmation matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from hemato_osr.experiments.delivery6 import Delivery6Paths, run_delivery6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("artifacts/checkpoints/delivery6"),
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=Path("artifacts/embeddings/delivery6"),
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("artifacts/metrics/delivery6"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("artifacts/figures/delivery6"),
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("artifacts/logs/delivery6"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = Delivery6Paths(
        checkpoint_dir=args.checkpoint_dir,
        embedding_dir=args.embedding_dir,
        metrics_dir=args.metrics_dir,
        figures_dir=args.figures_dir,
        logs_dir=args.logs_dir,
    )
    run_delivery6(
        paths,
        device=args.device,
        num_workers=args.num_workers,
        smoke=args.smoke,
        analyze_only=args.analyze_only,
    )


if __name__ == "__main__":
    main()
