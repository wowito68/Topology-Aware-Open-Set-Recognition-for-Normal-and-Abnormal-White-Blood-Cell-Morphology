# Open-Set Protocol

## Splits

- `known train`: known classes used for model fitting.
- `known validation`: known classes used for early stopping and strict threshold calibration.
- `known test`: known classes used only for final reporting.
- `unknown development`: optional unknown classes reserved for development calibration.
- `unknown test`: unknown classes reserved for final open-set evaluation.

Delivery 2 uses the strict protocol. No unknown-development split is used for model selection or threshold tuning.

Delivery 3 also uses the strict protocol. Candidate-mask QC, feature-map layer selection, vectorizer fitting, TDA class statistics, empirical score calibration, and threshold calibration are all performed without unknown-test labels or unknown-test score distributions.

Delivery 4 uses the same strict protocol but stops TDA method development. All new methods are post-hoc scores over the frozen ResNet18 seed-37 logits or 512-dimensional embeddings. Fit-time statistics are allowed only on known-train rows. Thresholds are calibrated only on known-validation rows. Unknown-test rows are reserved for final reporting and explanatory geometry.

Delivery 5 keeps the strict protocol while changing the learned representation. SupCon and ArcFace train only on known-train images. Known validation may select checkpoints by macro-F1 and known-only geometry tie-breakers. Unknown-test labels are analysis-only and cannot choose lambda, temperature, margin, checkpoint, OSR method, threshold, ViM parameters, or kNN reference banks.

## Split Protocols

Split V1 is the original seed-37 split used by Delivery 1. Split V2 is a conservative near-duplicate-aware sensitivity split. V2 keeps all unknown samples outside training and moves only samples required by configured high-confidence near-duplicate components.

The V1-versus-V2 ResNet18 comparison changes only the split. Architecture, initialization, loss, seed, batch size, input size, precision, and training policy remain matched to the V1 baseline.

## Strict Open Set

The strict protocol calibrates an unknown threshold with known validation scores only. It does not use unknown samples for threshold selection.

Delivery 3 additionally calibrates heterogeneous deep/TDA anomaly scores by empirical percentile against known-validation scores only:

```text
q(s) = fraction of known-validation anomaly scores <= s
```

This maps each score to `[0,1]` while preserving the convention that larger values are more unknown-like. The transformation is used for transparent late fusion; it is not fit on unknown test samples.

## Development Open Set

The development protocol may use reserved unknown classes for calibration. These unknown development classes must be disjoint from unknown test classes.

## Leakage Prohibitions

Never tune thresholds, preprocessing, model hyperparameters, vectorizers, scalers, or calibration using final test data. If patient or acquisition-group ids exist, groups cannot cross splits.

MLL23 manifests used here do not provide reliable patient or acquisition-group identifiers. This is documented as residual risk. The near-duplicate audit therefore remains a sensitivity analysis rather than proof that all same-cell or same-patient leakage has been eliminated.

## Score Orientation

Open-set metrics use UNKNOWN as the positive class. Larger `unknown_score` values mean more unknown-like for MSP, entropy, energy, Mahalanobis distance, and TDA distance after each method-specific transformation.

For Delivery 3:

- Deep MSP anomaly is `1 - max_softmax_probability`.
- TDA anomaly is nearest known-class centroid distance after train-only scaling.
- Late fusion combines calibrated scores with fixed alpha `0.50` as the primary rule.
- Optional fixed alphas `0.25` and `0.75` are secondary and must be reported as such.
- No trained binary known-vs-unknown classifier is fit on unknown-test labels.

For Delivery 4:

- MSP anomaly is `1 - max_softmax_probability`.
- Predictive entropy is unknown-like when entropy is larger.
- Energy is `-T * logsumexp(logits / T)`, with `T=1` primary and `T=0.5/2` secondary.
- Euclidean NCM is the distance to the nearest known-train class mean.
- Cosine NCM is `1 - max cosine similarity` to normalized known-train prototypes.
- Cosine kNN is the mean cosine distance to known-train neighbors; `k=5` is primary.
- Historical Mahalanobis uses the previous pooled pseudo-inverse covariance recipe as a secondary reproduction baseline.
- Regularized Mahalanobis uses pooled Ledoit-Wolf covariance from known-train residuals.
- Relative Mahalanobis is the minimum class-conditional distance minus the global known-train background distance.
- ReAct clips pre-classifier embeddings at a known-train percentile, recomputes the frozen linear head, and uses energy; the 90th percentile is primary.
- ViM fits known-train PCA statistics and scale, with an ID-only 95% explained-variance rule, and scores `alpha * residual_norm - logsumexp(logits)`.
- DICE is secondary only if implemented rigorously; otherwise it is documented as skipped.
- ODIN is not primary because it would require an input-gradient perturbation pathway outside the fixed post-hoc embedding protocol.

Reported AUROC and AUPR are therefore unknown-positive unless explicitly stated otherwise. FPR@95TPR is the false-positive rate among known samples when true-positive rate for unknown samples is fixed at 95%.

## Per-Unknown-Class Analysis

Each unknown morphology is evaluated against all known test samples. The required table reports AUROC, FPR@95TPR when defined, sample count, and score summaries. These one-vs-known analyses identify unknown classes that are easiest or hardest to reject; they are not used for tuning.

Delivery 3 also reports predefined subgroup aggregates:

- `immature_related_myeloid`
- `lymphoid_related`
- `other`

Special analyses compare `neutrophil_segmented` versus `neutrophil_band` and inspect lymphocyte-related unknowns. These analyses explain behavior; they do not redefine the primary overall metric.

Delivery 4 reports the same predefined subgroup aggregates and additionally writes embedding geometry tables: per-class nearest known centroid, centroid distance, kNN distance summaries, known-train compactness, class-centroid distance matrix, and unknown-to-known attractor frequencies. These geometry outputs are explanatory only and are not used to choose thresholds or primary methods.

Delivery 5 evaluates each representation with the same fixed OSR scorers: MSP, Energy `T=1`, cosine kNN `k=5`, and ViM. ArcFace inference uses normalized cosine classifier logits without target margin, because ground-truth labels are unavailable at inference time. ViM and kNN are independently fitted per representation using known-train embeddings only. All scores retain the convention that larger values are more unknown-like.
