#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:-data/manifests/mll23_split.csv}"
BACKBONE="${BACKBONE:-resnet18}"

for SEED in 13 37 73 101 137; do
  OUT="artifacts/checkpoints/mll23_closed_set_seed${SEED}"
  python -m hemato_osr train closed-set \
    --manifest "${MANIFEST}" \
    --output-dir "${OUT}" \
    --backbone "${BACKBONE}" \
    --seed "${SEED}"
done

