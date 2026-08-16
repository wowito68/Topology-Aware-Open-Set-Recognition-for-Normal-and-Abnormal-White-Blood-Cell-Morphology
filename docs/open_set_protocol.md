# Open-Set Protocol

## Splits

- `known train`: known classes used for model fitting.
- `known validation`: known classes used for early stopping and strict threshold calibration.
- `known test`: known classes used only for final reporting.
- `unknown development`: optional unknown classes reserved for development calibration.
- `unknown test`: unknown classes reserved for final open-set evaluation.

Delivery 2 uses the strict protocol. No unknown-development split is used for model selection or threshold tuning.

Delivery 3 also uses the strict protocol. Candidate-mask QC, feature-map layer selection, vectorizer fitting, TDA class statistics, empirical score calibration, and threshold calibration are all performed without unknown-test labels or unknown-test score distributions.

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

Reported AUROC and AUPR are therefore unknown-positive unless explicitly stated otherwise. FPR@95TPR is the false-positive rate among known samples when true-positive rate for unknown samples is fixed at 95%.

## Per-Unknown-Class Analysis

Each unknown morphology is evaluated against all known test samples. The required table reports AUROC, FPR@95TPR when defined, sample count, and score summaries. These one-vs-known analyses identify unknown classes that are easiest or hardest to reject; they are not used for tuning.

Delivery 3 also reports predefined subgroup aggregates:

- `immature_related_myeloid`
- `lymphoid_related`
- `other`

Special analyses compare `neutrophil_segmented` versus `neutrophil_band` and inspect lymphocyte-related unknowns. These analyses explain behavior; they do not redefine the primary overall metric.
