# AGENTS.md

This is a research repository for topology-aware open-set recognition of white blood cell morphology. It is not a web app, API, SaaS product, or clinical tool.

## Scientific Goal

Study whether deep image embeddings plus topological features can classify known mature leukocytes and identify morphology classes outside the training taxonomy.

Known classes start as:

- `basophil`
- `eosinophil`
- `lymphocyte`
- `monocyte`
- `neutrophil_segmented`

`UNKNOWN` means outside the known training taxonomy only. Never present it as cancer, leukemia, malignancy, or diagnosis.

## Structure

- `configs/`: Hydra/OmegaConf experiment configs.
- `src/hemato_osr/data`: taxonomy, manifests, audits, splitting, datasets.
- `src/hemato_osr/models`: backbones, TDA baseline, fusion.
- `src/hemato_osr/training`: training loops and checkpoints.
- `src/hemato_osr/openset`: open-set scorers and thresholds.
- `src/hemato_osr/topology`: persistent homology and vectorization.
- `src/hemato_osr/evaluation`: metrics and artifacts.
- `tests/`: small synthetic tests only.

## Commands

- Install: `python -m pip install -e ".[dev]"`
- Lint: `ruff check .`
- Format check: `ruff format --check .`
- Type check: `mypy src`
- Tests: `pytest`
- Smoke: `python -m hemato_osr smoke --output-dir artifacts/smoke`

## Conventions

- Keep scientific logic in importable, testable Python modules.
- Do not put core logic in notebooks.
- Centralize taxonomy changes in config or `hemato_osr.data.taxonomy`.
- Keep augmentations conservative unless a methodological document justifies them.
- Never version raw datasets or large artifacts.

## Reproducibility

- Record seed, exact config, git commit, manifest hash, environment, Python, PyTorch, CUDA, and device metadata.
- Do not overwrite previous experiment results.
- Document any methodological change in `docs/`.
- Never modify predictions, labels, or metrics to make results look better.

## Leakage Rules

Never use test data for model selection, threshold selection, calibration, hyperparameter tuning or preprocessing decisions.

- Run leakage audits before training.
- Abort on unequivocal train/validation/test overlap.
- If patient/group identifiers are absent, document the limitation and do not invent IDs.
- Any vectorizer or scaler that learns parameters must be fit on training data only.
- Unknown development classes and unknown test classes must remain disjoint when the protocol requires it.

## Tests

Tests must run without MLL23. Use synthetic fixtures only. Synthetic images are pipeline fixtures, not scientific data.

