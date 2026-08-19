from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sync_to_ec2_is_non_destructive_and_excludes_large_artifacts() -> None:
    script = (REPO_ROOT / "scripts" / "remote" / "sync_to_ec2.sh").read_text()

    assert "--delete" not in script
    assert "--exclude '*.pem'" in script
    assert "--exclude 'artifacts/checkpoints/'" in script
    assert "--exclude 'artifacts/embeddings/'" in script
    assert "--exclude 'data/raw/'" in script


def test_sync_from_ec2_allows_only_small_result_families() -> None:
    script = (REPO_ROOT / "scripts" / "remote" / "sync_from_ec2.sh").read_text()

    assert "--delete" not in script
    assert "--include 'artifacts/metrics/***'" in script
    assert "--include 'artifacts/remote_inventory/***'" in script
    assert "--exclude '*'" in script
    assert "artifacts/checkpoints/***" not in script
    assert "data/raw/***" not in script


def test_remote_inventory_documents_remote_only_large_artifacts() -> None:
    inventory = (
        REPO_ROOT / "artifacts" / "remote_inventory" / "delivery1_remote_artifacts.csv"
    ).read_text()

    assert "raabin_raw,dataset,/home/ubuntu/datasets/raabin_wbc/,none" in inventory
    assert "ce_checkpoint,checkpoint" in inventory
    assert "metadata_only" in inventory
    assert "Do not copy locally." in inventory
