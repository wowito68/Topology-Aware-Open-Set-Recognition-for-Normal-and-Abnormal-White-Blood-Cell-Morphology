#!/usr/bin/env python3
"""Download and verify MLL23 from Zenodo metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ZenodoFile:
    filename: str
    size: int
    checksum: str
    url: str


def fetch_record(record_id: str) -> dict[str, Any]:
    url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_files(record: dict[str, Any]) -> list[ZenodoFile]:
    files = []
    for item in record.get("files", []):
        links = item.get("links", {})
        files.append(
            ZenodoFile(
                filename=str(item["key"]),
                size=int(item.get("size", 0)),
                checksum=str(item.get("checksum", "")),
                url=str(links.get("self") or links.get("download")),
            )
        )
    return files


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(file: ZenodoFile, output_dir: Path) -> dict[str, object]:
    output_path = output_dir / file.filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "8",
            "--retry-delay",
            "10",
            "--connect-timeout",
            "30",
            "-C",
            "-",
            "-o",
            str(output_path),
            file.url,
        ],
        check=True,
    )
    algorithm, expected = file.checksum.split(":", maxsplit=1)
    actual = file_digest(output_path, algorithm)
    status = (
        "verified" if actual == expected and output_path.stat().st_size == file.size else "failed"
    )
    return {
        "filename": file.filename,
        "bytes_expected": file.size,
        "bytes_actual": output_path.stat().st_size,
        "checksum_expected": file.checksum,
        "checksum_actual": f"{algorithm}:{actual}",
        "status": status,
    }


def write_manifest(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "bytes_expected",
        "bytes_actual",
        "checksum_expected",
        "checksum_actual",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: str(row["filename"])))


def safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
    output_root = output_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in sorted(archive.infolist(), key=lambda item: item.filename):
            target = (output_root / member.filename).resolve()
            if not str(target).startswith(str(output_root) + "/") and target != output_root:
                msg = f"Unsafe zip member path: {member.filename}"
                raise ValueError(msg)
            archive.extract(member, output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-id", default="14277609")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/mll23"))
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = fetch_record(args.record_id)
    record_path = args.output_dir / f"zenodo_record_{args.record_id}.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = parse_files(record)

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = [executor.submit(download_file, file, args.output_dir) for file in files]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    manifest_path = args.output_dir / "download_manifest.csv"
    write_manifest(rows, manifest_path)
    failed = [row for row in rows if row["status"] != "verified"]
    if failed:
        msg = f"{len(failed)} files failed checksum verification"
        raise SystemExit(msg)

    if args.extract:
        for file in sorted(files, key=lambda item: item.filename):
            if file.filename.lower().endswith(".zip"):
                safe_extract_zip(args.output_dir / file.filename, args.output_dir)


if __name__ == "__main__":
    main()
