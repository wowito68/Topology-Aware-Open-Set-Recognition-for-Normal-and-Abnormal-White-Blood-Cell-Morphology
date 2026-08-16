"""Experiment identifiers and reproducibility metadata."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


def git_commit() -> str:
    """Return the current git commit, or ``unknown`` outside git."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def experiment_id(config: dict[str, Any]) -> str:
    """Create a non-overwriting experiment id from date, method, seed, and config."""

    payload = json.dumps(config, sort_keys=True, default=str)
    digest = hashlib.sha1(payload.encode()).hexdigest()[:8]
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    backbone = str(config.get("backbone", config.get("model", "experiment")))
    method = str(config.get("open_set_method", config.get("method", "run")))
    seed = str(config.get("seed", "na"))
    return f"{today}_{backbone}_{method}_seed{seed}_{digest}"


def environment_metadata(manifest_hash: str | None = None) -> dict[str, Any]:
    """Collect runtime metadata needed for reproducibility."""

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else ""
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda": torch.version.cuda,
        "gpu": gpu_name,
        "manifest_hash": manifest_hash,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
