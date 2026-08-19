"""Risk-coverage and open-set review metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from leukocyte_hil.triage.rules import TriageStatus


@dataclass(frozen=True)
class SelectiveRecord:
    """Ground-truthed crop-level triage record."""

    known_or_unknown: str
    triage_status: TriageStatus
    correct_if_known: bool | None


def selective_metrics(records: list[SelectiveRecord]) -> dict[str, float]:
    """Compute coverage, selective risk, KRR, and UAAR from triage records."""

    if not records:
        return {
            "coverage": 0.0,
            "selective_risk": float("nan"),
            "known_review_rate": float("nan"),
            "unknown_auto_accept_rate": float("nan"),
            "unknown_possible_unknown_rate": float("nan"),
        }

    statuses = np.asarray([record.triage_status for record in records], dtype=object)
    known = np.asarray([record.known_or_unknown == "known" for record in records], dtype=bool)
    unknown = np.asarray([record.known_or_unknown == "unknown" for record in records], dtype=bool)
    auto = statuses == TriageStatus.AUTO_ACCEPT
    accepted_known_records = [
        record
        for record in records
        if record.known_or_unknown == "known" and record.triage_status == TriageStatus.AUTO_ACCEPT
    ]
    incorrect_known_auto = [
        record for record in accepted_known_records if record.correct_if_known is False
    ]

    return {
        "coverage": float(np.mean(auto)),
        "selective_risk": _safe_divide(len(incorrect_known_auto), len(accepted_known_records)),
        "known_review_rate": _safe_divide(int(np.sum(known & ~auto)), int(np.sum(known))),
        "unknown_auto_accept_rate": _safe_divide(int(np.sum(unknown & auto)), int(np.sum(unknown))),
        "unknown_possible_unknown_rate": _safe_divide(
            int(np.sum(unknown & (statuses == TriageStatus.POSSIBLE_UNKNOWN))),
            int(np.sum(unknown)),
        ),
    }


def aurc(coverage: np.ndarray, risk: np.ndarray) -> float:
    """Compute area under the risk-coverage curve."""

    coverage = np.asarray(coverage, dtype=float)
    risk = np.asarray(risk, dtype=float)
    finite = np.isfinite(coverage) & np.isfinite(risk)
    if np.sum(finite) < 2:
        return float("nan")
    order = np.argsort(coverage[finite])
    return float(np.trapezoid(risk[finite][order], coverage[finite][order]))


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)
