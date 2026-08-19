#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${EC2_HOST:-}" ]]; then
  echo "REMOTE EXECUTION BLOCKED: EC2_HOST is required" >&2
  exit 2
fi

REMOTE_REPO="${REMOTE_REPO:-/home/ubuntu/hemato-osr}"

ssh "${EC2_HOST}" "cd '${REMOTE_REPO}' && \
  pwd && \
  git status --short && \
  df -h && \
  free -h && \
  nvidia-smi && \
  .venv/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.version.cuda)
PY
  .venv/bin/ruff check . && \
  .venv/bin/ruff format --check . && \
  .venv/bin/mypy src && \
  .venv/bin/pytest -q"
