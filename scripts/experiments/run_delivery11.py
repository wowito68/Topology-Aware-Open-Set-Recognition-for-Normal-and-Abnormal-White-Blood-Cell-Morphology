"""Delivery 1.1 remote-first utilities.

This script intentionally works on compact signal CSVs and metadata. Real image
loading, checkpoint inference, Raabin extraction, and detector training belong on
EC2 and are orchestrated separately.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from leukocyte_hil.triage.calibration import (
    CalibrationRecord,
    calibrate_operating_points,
    operating_points_to_yaml,
)
from leukocyte_hil.utils.checkpoints import verify_sha256

EXPECTED_CE_SHA256 = "a1ff5939b431805a70eb5bda1e4e09059ff8e3fc0521fad2d41884b7c415deca"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify-checkpoint")
    verify_parser.add_argument("--checkpoint", required=True, type=Path)
    verify_parser.add_argument("--expected-sha256", default=EXPECTED_CE_SHA256)
    verify_parser.add_argument("--output-json", required=True, type=Path)

    calibrate_parser = subparsers.add_parser("calibrate-triage")
    calibrate_parser.add_argument("--signals-csv", required=True, type=Path)
    calibrate_parser.add_argument("--output-yaml", required=True, type=Path)
    calibrate_parser.add_argument(
        "--targets",
        nargs="*",
        type=float,
        default=[0.95, 0.90, 0.80, 0.70, 0.60],
    )

    args = parser.parse_args()
    if args.command == "verify-checkpoint":
        _verify_checkpoint(args.checkpoint, args.expected_sha256, args.output_json)
    elif args.command == "calibrate-triage":
        _calibrate_triage(args.signals_csv, args.output_yaml, args.targets)


def _verify_checkpoint(checkpoint: Path, expected_sha256: str, output_json: Path) -> None:
    result = verify_sha256(checkpoint, expected_sha256)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "checkpoint": str(result.path),
                "expected_sha256": result.expected_sha256,
                "observed_sha256": result.observed_sha256,
                "exact_match": result.exact_match,
                "exists": result.exists,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if not result.exists:
        raise SystemExit("FROZEN CHECKPOINT RECOVERY BLOCKED")
    if not result.exact_match:
        raise SystemExit("FROZEN CHECKPOINT CHECKSUM MISMATCH")


def _calibrate_triage(signals_csv: Path, output_yaml: Path, targets: list[float]) -> None:
    records = _read_signal_records(signals_csv)
    points = calibrate_operating_points(records, targets=targets)
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text(operating_points_to_yaml(points))


def _read_signal_records(path: Path) -> list[CalibrationRecord]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            CalibrationRecord(
                sample_id=row["sample_id"],
                split=row["split"],
                known_or_unknown=row["known_or_unknown"],
                confidence=float(row["confidence"]),
                margin=float(row["margin"]),
                unknown_score=float(row["unknown_score"]),
                quality_pass=_parse_bool(row.get("quality_pass", "true")),
                correct_if_known=_parse_optional_bool(row.get("correct_if_known")),
            )
            for row in reader
        ]


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    return _parse_bool(value)


if __name__ == "__main__":
    main()
