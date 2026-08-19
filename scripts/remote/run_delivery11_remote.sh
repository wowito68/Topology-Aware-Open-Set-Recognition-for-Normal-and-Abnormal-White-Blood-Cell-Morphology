#!/usr/bin/env bash
set -euo pipefail

REMOTE_REPO="${REMOTE_REPO:-/home/ubuntu/hemato-osr}"
EXPECTED_CE_SHA256="a1ff5939b431805a70eb5bda1e4e09059ff8e3fc0521fad2d41884b7c415deca"
CE_CHECKPOINT="${CE_CHECKPOINT:-${REMOTE_REPO}/artifacts/checkpoints/delivery6/d6_v2_ce_seed37_2d9d5aa0027e6f49/best_checkpoint.pt}"
SIGNALS_CSV="${SIGNALS_CSV:-${REMOTE_REPO}/artifacts/metrics/delivery11/oracle_validation_signals.csv}"

cd "${REMOTE_REPO}"

echo "DELIVERY 1.1 REMOTE GATE"
pwd
git status --short
df -h
free -h
nvidia-smi

.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src
.venv/bin/pytest -q

echo "VERIFY FROZEN CE CHECKPOINT"
.venv/bin/python scripts/experiments/run_delivery11.py verify-checkpoint \
  --checkpoint "${CE_CHECKPOINT}" \
  --expected-sha256 "${EXPECTED_CE_SHA256}" \
  --output-json artifacts/checkpoint_metadata/delivery11_ce_seed37_v2_verification.json

if [[ ! -f "${SIGNALS_CSV}" ]]; then
  echo "REMOTE EXECUTION BLOCKED: oracle signal CSV missing: ${SIGNALS_CSV}" >&2
  echo "Generate oracle_validation_signals.csv on EC2 from MLL23 known validation before calibration." >&2
  exit 3
fi

echo "CALIBRATE ORACLE TRIAGE OPERATING POINTS"
.venv/bin/python scripts/experiments/run_delivery11.py calibrate-triage \
  --signals-csv "${SIGNALS_CSV}" \
  --output-yaml configs/triage/triage_delivery11_operating_points.yaml

echo "Delivery 1.1 remote checkpoint verification and oracle calibration completed."
