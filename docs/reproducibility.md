# Reproducibility

## Steps

1. Install the package and dev dependencies.
2. Place manually downloaded data under `data/raw/mll23`.
3. Generate a manifest.
4. Split deterministically with a recorded seed.
5. Run the leakage audit.
6. Train the closed-set model.
7. Export embeddings.
8. Extract TDA features.
9. Evaluate open-set metrics.
10. Save exact config and environment metadata.

Delivery 2 extends this sequence with:

1. Reconstruct candidate near-duplicate pairs.
2. Generate contact sheets for human inspection.
3. Build Split V2 without overwriting Split V1.
4. Train only the ResNet18 V2 sensitivity baseline.
5. Run a stratified TDA resolution benchmark before full extraction.
6. Extract and cache full persistent-homology diagrams.
7. Vectorize TDA features with train-only fitting.
8. Train TDA-only and frozen Deep+TDA models.
9. Generate per-unknown-class and paired-bootstrap reports.

Delivery 3 extends the sequence without retraining the primary ResNet18:

1. Freeze and checksum the V1 seed-37 ResNet18 checkpoint and embeddings.
2. Generate a deterministic known-train morphology QC subset.
3. Produce contact sheets and candidate-mask metadata before full extraction.
4. Benchmark morphology distance-map TDA on a stratified subset.
5. Benchmark frozen ResNet18 activation-map TDA layers before final metrics.
6. Persist the final Delivery 3 experiment matrix and decision criteria.
7. Extract morphology-aware TDA diagrams/features.
8. Extract selected feature-map TDA diagrams/features.
9. Fit TDA anomaly statistics on known train only.
10. Calibrate deep/TDA scores with known validation only.
11. Evaluate V1 once for the predeclared matrix.
12. Run V2 only if a V1 method satisfies the progression criteria.

Delivery 4 extends the sequence without retraining and without new TDA development:

1. Verify the Delivery 3 commit is present and the worktree is controlled.
2. Verify the frozen ResNet18 checkpoint SHA256 `7fb2e2a6741a4fcd1b7d9e3a985ab136fa0550741b69dcd62f0f21c28b6ae39d`.
3. Verify the embedding archive manifest hash `4b8da8df49655b6f04b53aa75a912f0efceac776d81761b2b45b68c440c27bae`.
4. Persist `configs/experiment/delivery4_predeclared_matrix.json` before final evaluation.
5. Reproduce historical MSP, energy `T=1`, and historical Mahalanobis baselines within tolerance.
6. Fit all post-hoc method statistics using known-train rows only.
7. Calibrate strict thresholds using known-validation rows only.
8. Evaluate the full predeclared V1 method matrix once.
9. Generate per-unknown-class, subgroup, bootstrap, efficiency, data-usage, and embedding-geometry reports.
10. Run V2 or multiseed confirmation only if a primary method passes the predeclared Delivery 4 progression rule.

Delivery 5 extends the sequence with representation training:

1. Persist `configs/experiment/delivery5_predeclared_matrix.json` before unknown evaluation.
2. Verify the historical CE checkpoint and embedding archive hashes.
3. Run one-epoch GPU smoke training for SupCon and ArcFace on a small known-train subset.
4. Train SupCon and ArcFace sequentially under tmux on AWS, using seed `37` and known-validation checkpoint selection only.
5. Export logits and 512-dimensional embeddings for all samples from each trained representation.
6. Evaluate CE, SupCon, and ArcFace with the fixed scorers MSP, Energy `T=1`, cosine kNN `k=5`, and ViM.
7. Recompute known geometry, unknown geometry, attractor matrices, per-unknown-class metrics, subgroup metrics, and paired bootstrap deltas versus CE+ViM.
8. Run V2 only for a passing representation/scorer pair; otherwise V2 is not run.

Delivery 6 confirms the frozen ArcFace + MSP candidate without changing the method:

1. Persist `configs/experiment/delivery6_predeclared_matrix.json` before new training.
2. Train exactly 20 sequential GPU runs: five seeds, two split protocols, and CE/ArcFace pairs.
3. Fit MSP directly and ViM independently per run using known-train rows only.
4. Summarize matched ArcFace-minus-CE deltas at the seed/run level, not the image level.
5. Compute geometry, class-level stability, subgroup stability, and attractor stability after all runs are frozen.
6. Apply the predeclared criteria A-E without adding losses, scorers, backbones, or split changes.

Delivery 6 result: ArcFace + MSP failed confirmation. V1 mean matched AUROC delta was
`+0.0130` with a seed-level 95% CI crossing zero, and V2 mean matched AUROC delta was
`-0.0199` with only one of five V2 seeds nonnegative. The reproducibility recommendation is
external comparison of CE and ArcFace, rather than further MLL23 method search.

## Required Metadata

Record git commit, seed, timestamp, Python, PyTorch, CUDA, GPU, manifest hash, exact config, threshold protocol, and class taxonomy.

Each experiment artifact should also record, when applicable, git dirty state, config hash, split protocol, hostname, GUDHI version, NumPy version, scikit-learn version, source manifest, preprocessing config, resolution, filtration, homology dimensions, vectorizer component slices, and threshold calibration split.

The AWS DLAMI environment is reused as-is. CUDA and PyTorch are not replaced during Delivery 2. The previously observed `transformer_engine_cu13 2.13.0 is not supported on this platform` warning is documented if present; it is not repaired unless it blocks the pipeline.

Delivery 3 additionally records the frozen checkpoint SHA256, embedding archive SHA256, morphology config hash, activation layer, activation aggregation, activation normalization, feature-map layer benchmark, selected-layer reason, TDA feature config hash, and the persisted pre-test matrix path.

Delivery 4 additionally records checkpoint SHA256, embedding manifest hash, method config hashes, data-usage audit, fit split, calibration split, strict threshold protocol, ReAct clipping percentile/value, ViM PCA component count and explained variance, covariance estimator, paired-bootstrap seed/count, progression decision, and software environment metadata.

Delivery 5 additionally records representation loss, SupCon lambda/temperature, projection-head dimensions, ArcFace scale/margin, sampler policy, CE weighting policy, checkpoint selection rule, training epochs completed, best epoch, wall-clock time, GPU-hours, peak VRAM, checkpoint SHA256, embedding export forward time, and whether UMAP was available or PCA fallback was used.

Delivery 6 additionally records run ids, split hashes, seed-level matched deltas, t-based
confidence intervals over seeds, per-run checkpoint SHA256, per-run training time, per-run
peak VRAM, per-class stability labels, subgroup stability, attractor stability, and the
predeclared confirmation decision.

## Multiple Seeds

Delivery 2 uses only seed `37`. Planned future validation seeds are `13`, `37`, `73`, `101`, and `137`, but multiseed evaluation starts only after a final method is selected.

Delivery 3 also uses only seed `37` for primary exploratory method development. Multiseed confirmation is explicitly deferred until after the predeclared progression criteria are met.

Delivery 4 also uses only seed `37` for exploratory method development on V1. Multiseed or Split V2 confirmation is explicitly deferred until after a primary post-hoc method passes the predeclared progression criteria.

Delivery 5 also uses only seed `37`. If SupCon or ArcFace passes progression criteria, multiseed and external-domain validation are deferred to Delivery 6 rather than run automatically in Delivery 5.

Delivery 6 uses exactly seeds `13`, `37`, `73`, `101`, and `137` over V1 and V2 for matched
CE versus ArcFace stability analysis. These seeds must not be replaced or dropped because of
unfavorable results.

## Artifacts

Delivery 2 artifacts are organized under:

- `artifacts/data_audit/near_duplicates/`
- `artifacts/benchmarks/tda_resolution/`
- `artifacts/topology/diagrams/`
- `artifacts/topology/features/`
- `artifacts/checkpoints/resnet18_v2_seed37/`
- `artifacts/checkpoints/fusion/`
- `artifacts/metrics/delivery2/`
- `artifacts/figures/delivery2/`
- `artifacts/logs/delivery2/`

Delivery 1 artifacts must not be deleted or overwritten.

Delivery 3 artifacts are organized under:

- `artifacts/figures/delivery3/morphology_qc/`
- `artifacts/benchmarks/delivery3/shape_tda/`
- `artifacts/benchmarks/delivery3/feature_map/`
- `artifacts/topology/delivery3/`
- `artifacts/metrics/delivery3/`
- `artifacts/figures/delivery3/`
- `artifacts/logs/delivery3/`
- `configs/experiment/delivery3_predeclared_matrix.json`

Delivery 1 and Delivery 2 artifacts must not be deleted or overwritten.

Delivery 4 artifacts are organized under:

- `configs/experiment/delivery4.yaml`
- `configs/experiment/delivery4_predeclared_matrix.json`
- `artifacts/metrics/delivery4/`
- `artifacts/figures/delivery4/`
- `artifacts/logs/delivery4/`

Delivery 1, Delivery 2, and Delivery 3 artifacts must not be deleted or overwritten.

Delivery 5 artifacts are organized under:

- `configs/experiment/delivery5.yaml`
- `configs/experiment/delivery5_predeclared_matrix.json`
- `artifacts/checkpoints/delivery5/supcon_seed37/`
- `artifacts/checkpoints/delivery5/arcface_seed37/`
- `artifacts/embeddings/delivery5/`
- `artifacts/metrics/delivery5/`
- `artifacts/figures/delivery5/`
- `artifacts/logs/delivery5/`
- `artifacts/benchmarks/delivery5/`

Previous delivery artifacts must not be deleted or overwritten.

Delivery 6 artifacts are organized under:

- `configs/experiment/delivery6_predeclared_matrix.json`
- `artifacts/checkpoints/delivery6/`
- `artifacts/embeddings/delivery6/`
- `artifacts/metrics/delivery6/`
- `artifacts/figures/delivery6/`
- `artifacts/logs/delivery6/`

Previous delivery artifacts must not be deleted or overwritten.
