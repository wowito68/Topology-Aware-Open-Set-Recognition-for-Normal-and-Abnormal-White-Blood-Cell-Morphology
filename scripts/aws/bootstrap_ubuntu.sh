#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  git \
  python3 \
  python3-pip \
  python3-venv

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA driver detected; not installing or modifying GPU drivers."
  nvidia-smi || true
else
  echo "nvidia-smi not found. Install GPU drivers through the cloud image or AWS tooling."
fi

python3 -m pip install --upgrade pip
echo "Bootstrap complete. Create a venv, then run: python -m pip install -e '.[dev]'"

