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
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pem' \
  --exclude '*.key' \
  --exclude 'data/raw/' \
  --exclude 'data/interim/' \
  --exclude 'data/processed/' \
  --exclude 'artifacts/checkpoints/' \
  --exclude 'artifacts/embeddings/' \
  --exclude 'artifacts/crops/' \
  --exclude 'artifacts/cache/' \
  --exclude 'artifacts/topology/' \
  --exclude 'artifacts/predictions/' \
  --exclude 'artifacts/logs/' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  --exclude '*.ckpt' \
  "${LOCAL_REPO%/}/" \
  "${EC2_HOST}:${REMOTE_REPO%/}/"
