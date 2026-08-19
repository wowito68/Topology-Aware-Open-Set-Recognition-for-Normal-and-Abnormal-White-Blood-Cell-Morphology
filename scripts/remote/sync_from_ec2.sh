#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${EC2_HOST:-}" ]]; then
  echo "EC2_HOST is required, for example ubuntu@host" >&2
  exit 2
fi

if [[ -z "${REMOTE_REPO:-}" ]]; then
  echo "REMOTE_REPO is required, for example /home/ubuntu/hemato-osr" >&2
  exit 2
fi

LOCAL_REPO="${LOCAL_REPO:-$(pwd)}"

rsync -av \
  --include 'configs/***' \
  --include 'docs/***' \
  --include 'data/manifests/***' \
  --include 'artifacts/data_audit/***' \
  --include 'artifacts/metrics/***' \
  --include 'artifacts/figures/delivery1/***' \
  --include 'artifacts/remote_inventory/***' \
  --include 'artifacts/checkpoint_metadata/***' \
  --include 'artifacts/logs/delivery1/*.summary.log' \
  --exclude '*' \
  "${EC2_HOST}:${REMOTE_REPO%/}/" \
  "${LOCAL_REPO%/}/"
