from __future__ import annotations

import pandas as pd

from hemato_osr.data.audit import audit_manifest
from hemato_osr.data.taxonomy import Taxonomy


def test_audit_detects_duplicate_path(synthetic_manifest) -> None:  # type: ignore[no-untyped-def]
    duplicate = synthetic_manifest.iloc[[0]].copy()
    frame = pd.concat([synthetic_manifest, duplicate], ignore_index=True)

    report = audit_manifest(frame, Taxonomy(), abort_on_leakage=False)

    assert not report.passed
    assert report.details["duplicated_paths"] >= 1
