# Safety Scope

This repository contains research software for leukocyte morphology analysis.
It is not a medical device, diagnostic system, clinical decision-support
system, or deployment-ready hematology product.

`UNKNOWN` means morphology outside the known training taxonomy. It does not
mean leukemia, cancer, malignant, pathological, clinically positive, or any
diagnosis.

Permitted output labels are morphology and review-triage labels only:

- known morphology prediction;
- model confidence score;
- MSP unknown score;
- low image quality for model inference;
- human-review triage status.

Machine-readable outputs must not contain fields named `diagnosis`,
`cancer_probability`, `leukemia_probability`, or `malignancy_probability`.
