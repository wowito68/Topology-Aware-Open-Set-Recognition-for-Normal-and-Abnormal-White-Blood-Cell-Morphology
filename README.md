# Topology-Aware Open-Set Recognition for White Blood Cell Morphology

Research code for **Topology-Aware Open-Set Recognition for Normal and Abnormal White Blood Cell Morphology**.

This repository builds a reproducible experimental pipeline for asking:

> Can deep representations plus topological descriptors classify known mature leukocytes while identifying morphologies outside the training taxonomy?

`UNKNOWN` means outside the configured known training taxonomy. It is not a cancer, leukemia, malignancy, or clinical diagnosis label.

## Architecture Overview

The code is organized as importable Python modules under `src/hemato_osr`:

- `data`: taxonomy, manifests, deterministic splitting, audits, image datasets, synthetic fixtures.
- `models`: image backbones, TDA-only baseline, deep+TDA fusion MLP.
- `training`: closed-set training, checkpoints, seeds, fusion training.
- `openset`: MSP, entropy, energy, and Mahalanobis scoring plus threshold calibration.
- `topology`: cubical-complex diagrams, vectorizers, cache metadata, TDA extraction.
- `evaluation`: closed-set/open-set/calibration metrics and evaluation artifacts.
- `embeddings`: export logits and embeddings for downstream analysis.
- `utils`: Hydra/OmegaConf config helpers and environment tracking.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Expected Data Layout

MLL23 is downloaded manually. Do not commit raw images.

```text
data/raw/mll23/
  basophil/
  eosinophil/
  lymphocyte/
  monocyte/
  neutrophil_segmented/
  myeloblast/
  ...
```

Dataset: https://zenodo.org/records/14277609

Paper: https://www.nature.com/articles/s41597-025-06223-x

## Manifest Generation

```bash
python -m hemato_osr data index data/raw/mll23 \
  --dataset mll23 \
  --output data/manifests/mll23_manifest.csv
```

The manifest records sample id, path, dataset, original/canonical labels, known status, split, group id, checksum, width, and height. Code never trains directly from an implicit directory tree.

## Dataset Audit

```bash
python -m hemato_osr data audit data/manifests/mll23_split.csv
```

The audit checks duplicate paths/checksums, cross-split overlap, group leakage, unreadable images, and perceptual near-duplicates. If no reliable patient/group id exists, the limitation is reported rather than invented.

## Splitting

```bash
python -m hemato_osr data split data/manifests/mll23_manifest.csv \
  --method stratified \
  --seed 13 \
  --output data/manifests/mll23_split.csv
```

Unknown classes are not placed in training by default. Group-aware splitting is available when `group_id` is reliable.

## Training

```bash
python -m hemato_osr train closed-set \
  --manifest data/manifests/mll23_split.csv \
  --output-dir artifacts/checkpoints/mll23_closed_set \
  --backbone resnet18 \
  --epochs 30 \
  --precision amp
```

The baseline exposes both `logits` and `embedding`, supports checkpointing, early stopping, gradient clipping, AMP, deterministic seeds, weighted cross entropy, and class-aware sampling.

## Embedding Export

```bash
python -m hemato_osr embeddings extract \
  --manifest data/manifests/mll23_split.csv \
  --checkpoint artifacts/checkpoints/mll23_closed_set/best_checkpoint.pt \
  --output artifacts/embeddings/mll23_embeddings.npz
```

## Open-Set Evaluation

```bash
python -m hemato_osr evaluate open-set \
  --embeddings artifacts/embeddings/mll23_embeddings.npz \
  --method msp \
  --output-dir artifacts/metrics/mll23_open_set_msp
```

Evaluation writes `metrics.json`, `predictions.csv`, confusion matrix files, ROC/PR artifacts, exact config, and environment metadata.

## TDA Extraction

```bash
python -m hemato_osr tda extract \
  --manifest data/manifests/mll23_split.csv \
  --filtration sublevel \
  --vectorizer persistence-entropy \
  --output artifacts/topology/mll23_tda_features.npz
```

Available vectorizers include persistence entropy, Betti curve, persistence image, and persistence landscape. Vectorizers are fit on training diagrams only.

## Deep + TDA Fusion

```bash
python -m hemato_osr train fusion \
  --embeddings artifacts/embeddings/mll23_embeddings.npz \
  --tda-features artifacts/topology/mll23_tda_features.npz \
  --output artifacts/checkpoints/mll23_fusion/fusion.pt
```

TDA features are precomputed and cached; they are not computed inside the GPU forward pass.

## Synthetic Smoke Test

```bash
python -m hemato_osr smoke --output-dir artifacts/smoke
```

The generated images are explicitly marked as `SYNTHETIC TEST DATA - NOT SCIENTIFIC DATA`. Smoke results are only pipeline checks and must not be interpreted scientifically.

## Tests and Quality Gates

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

## Reproducibility

Experiments record seeds, config, git commit, Python, PyTorch/CUDA metadata, and manifest hashes where available. Multiple seeds are supported through configuration and `scripts/experiments/run_multiseed.sh`.

## Medical and Research Disclaimer

This is research software for morphology classification experiments. It is not a clinical product, diagnostic system, triage tool, or medical device. The `UNKNOWN` output only means outside the configured known training taxonomy.
