"""Dataset validation summaries."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hemato_osr.data.manifest import manifest_hash


def write_dataset_summary(
    manifest: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, object]:
    """Write dataset summary JSON and class-count CSV artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    class_counts = (
        manifest.groupby(["canonical_label", "known_status", "split"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["known_status", "canonical_label", "split"])
    )
    class_counts.to_csv(output_dir / "mll23_class_counts.csv", index=False)

    summary = {
        "rows": int(len(manifest)),
        "manifest_hash": manifest_hash(manifest),
        "datasets": sorted(set(manifest["dataset"].astype(str))),
        "classes": sorted(set(manifest["canonical_label"].astype(str))),
        "known_status_counts": {
            str(key): int(value)
            for key, value in manifest["known_status"].value_counts().sort_index().items()
        },
        "split_counts": {
            str(key): int(value)
            for key, value in manifest["split"]
            .fillna("")
            .astype(str)
            .value_counts()
            .sort_index()
            .items()
        },
        "image_width": {
            "min": int(manifest["width"].min()) if len(manifest) else 0,
            "max": int(manifest["width"].max()) if len(manifest) else 0,
        },
        "image_height": {
            "min": int(manifest["height"].min()) if len(manifest) else 0,
            "max": int(manifest["height"].max()) if len(manifest) else 0,
        },
        "extensions": sorted(
            {Path(str(path)).suffix.lower() for path in manifest["path"].astype(str)}
        ),
    }
    (output_dir / "mll23_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
