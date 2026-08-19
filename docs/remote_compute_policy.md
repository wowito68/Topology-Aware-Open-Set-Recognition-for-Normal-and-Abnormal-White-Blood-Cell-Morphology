# Remote-First Compute Policy

This project uses a remote-compute / thin-local-mirror workflow.

The local workstation is the Git source of truth for source code, configs,
tests, documentation, small manifests, compact CSV/JSON metrics, selected
figures, hashes, and experiment summaries.

AWS EC2 is the canonical execution location for computationally significant
work:

- raw and extracted datasets;
- large image collections;
- model checkpoints;
- embeddings;
- crop caches;
- prediction caches;
- dataset-wide hashing and duplicate audits;
- detector training;
- classifier inference over real datasets;
- end-to-end evaluation;
- large figure generation.

If AWS is unavailable, heavy computation must stop. Do not use the local
workstation as fallback compute.

## Canonical Paths

Preferred remote repository path:

`/home/ubuntu/hemato-osr`

Alternative remote project path:

`/home/ubuntu/leukocyte-hil`

Preferred Raabin-WBC raw-data path:

`/home/ubuntu/datasets/raabin_wbc/`

The final path must be recorded in
`artifacts/remote_inventory/delivery1_remote_artifacts.csv`.

## Sync Direction

Local to AWS:

- sync source/config/docs/tests and small metadata only;
- do not use destructive `rsync --delete` against a mixed code/artifact
  directory;
- exclude private keys, local virtual environments, raw data, checkpoints,
  embeddings, crop caches, and bulky logs.

AWS to local:

- sync small reproducibility outputs only;
- copy metrics, summaries, selected figures, compact logs, manifests, hashes,
  checkpoint metadata, and remote inventory;
- do not copy raw datasets, large checkpoints, embeddings, crop corpora, or raw
  prediction tensors.

## Authoritative Gates

The authoritative gates run on EC2:

```bash
ruff check .
ruff format --check .
mypy src
pytest -q
```

Local lightweight checks are optional and must not become a reason to run heavy
dataset or GPU work locally.

## Required Final Footprint Report

Every remote-execution report should include:

- local repository size;
- synchronized local artifact size;
- whether large datasets exist locally;
- whether large checkpoints exist locally;
- remote dataset size;
- remote checkpoint size;
- remote generated artifact size;
- remote disk free.
