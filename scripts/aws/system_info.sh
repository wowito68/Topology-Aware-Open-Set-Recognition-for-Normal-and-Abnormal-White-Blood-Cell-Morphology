#!/usr/bin/env bash
set -euo pipefail

echo "== GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi unavailable"
fi

echo "== CUDA =="
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo "nvcc unavailable"
fi

echo "== RAM =="
free -h || true

echo "== Disk =="
df -h .

echo "== Python =="
python3 --version

echo "== PyTorch CUDA =="
python3 - <<'PY'
try:
    import torch
except ImportError:
    print("torch not installed")
else:
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"torch_cuda={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
PY

