# Open-Set Protocol

## Splits

- `known train`: known classes used for model fitting.
- `known validation`: known classes used for early stopping and strict threshold calibration.
- `known test`: known classes used only for final reporting.
- `unknown development`: optional unknown classes reserved for development calibration.
- `unknown test`: unknown classes reserved for final open-set evaluation.

Delivery 2 uses the strict protocol. No unknown-development split is used for model selection or threshold tuning.

## Split Protocols

Split V1 is the original seed-37 split used by Delivery 1. Split V2 is a conservative near-duplicate-aware sensitivity split. V2 keeps all unknown samples outside training and moves only samples required by configured high-confidence near-duplicate components.

The V1-versus-V2 ResNet18 comparison changes only the split. Architecture, initialization, loss, seed, batch size, input size, precision, and training policy remain matched to the V1 baseline.

## Strict Open Set

The strict protocol calibrates an unknown threshold with known validation scores only. It does not use unknown samples for threshold selection.

## Development Open Set

The development protocol may use reserved unknown classes for calibration. These unknown development classes must be disjoint from unknown test classes.

## Leakage Prohibitions

Never tune thresholds, preprocessing, model hyperparameters, vectorizers, scalers, or calibration using final test data. If patient or acquisition-group ids exist, groups cannot cross splits.

MLL23 manifests used here do not provide reliable patient or acquisition-group identifiers. This is documented as residual risk. The near-duplicate audit therefore remains a sensitivity analysis rather than proof that all same-cell or same-patient leakage has been eliminated.

## Score Orientation

Open-set metrics use UNKNOWN as the positive class. Larger `unknown_score` values mean more unknown-like for MSP, entropy, energy, Mahalanobis distance, and TDA distance after each method-specific transformation.

Reported AUROC and AUPR are therefore unknown-positive unless explicitly stated otherwise. FPR@95TPR is the false-positive rate among known samples when true-positive rate for unknown samples is fixed at 95%.

## Per-Unknown-Class Analysis

Each unknown morphology is evaluated against all known test samples. The required table reports AUROC, FPR@95TPR when defined, sample count, and score summaries. These one-vs-known analyses identify unknown classes that are easiest or hardest to reject; they are not used for tuning.
