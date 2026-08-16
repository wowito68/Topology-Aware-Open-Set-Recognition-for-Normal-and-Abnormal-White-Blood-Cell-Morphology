# Research Protocol

## Question

Can deep visual representations and topological image features classify known mature leukocytes while detecting cell morphologies outside the training taxonomy?

Delivery 2 focuses on the following primary research question:

`RQ1`: Do persistent-homology-derived morphological descriptors improve open-set recognition of previously unseen blood-cell morphologies when combined with deep image representations?

Secondary questions:

- `RQ2`: How informative are topological features alone for mature white-blood-cell classification?
- `RQ3`: Are topological descriptors more useful for some unknown morphologies than others?
- `RQ4`: Are conclusions robust to a conservative near-duplicate-aware split?

Delivery 3 adds exploratory method-development questions:

- `RQ5`: Does persistent homology become more informative when computed over morphology-conditioned image representations or internal CNN activation maps instead of raw grayscale images?
- `RQ6`: Can a topology-derived anomaly score improve unknown-cell rejection when used only as late-fusion evidence, without modifying the frozen closed-set classifier?
- `RQ7`: Which hematological morphology families benefit or degrade most from topology-based rejection?

## Hypotheses

- Deep embeddings provide strong closed-set morphology discrimination.
- Topological descriptors may add complementary shape/texture information.
- Fusion may improve known-vs-unknown separation, but this must be tested without leakage.

## Known Classes

- `basophil`
- `eosinophil`
- `lymphocyte`
- `monocyte`
- `neutrophil_segmented`

## Initial Unknown Classes

`myeloblast`, `promyelocyte`, `promyelocyte_atypical`, `myelocyte`, `metamyelocyte`, `hairy_cell`, `lymphocyte_neoplastic`, `lymphocyte_large_granular`, `lymphocyte_reactive`, `plasma_cell`, `smudge_cell`, `normoblast`, and `neutrophil_band`.

## Tasks

Closed-set classification trains and evaluates only known classes. Open-set evaluation asks whether a trained known-class model can identify samples outside that taxonomy.

Delivery 2 compares five representation families with seed `37` only:

- Deep only: existing ImageNet-pretrained ResNet18 embeddings and logits.
- TDA only: persistence-vector features with StandardScaler and LogisticRegression.
- Deep + sublevel TDA: frozen ResNet18 embedding concatenated with sublevel features.
- Deep + superlevel TDA: frozen ResNet18 embedding concatenated with superlevel features.
- Deep + full TDA: frozen ResNet18 embedding concatenated with sublevel and superlevel features.

The frozen-fusion experiment uses a small MLP over precomputed features. It is not an end-to-end CNN fine-tuning experiment.

Delivery 2 result: the best raw-image Deep+TDA fusion variant degraded the primary open-set baseline. Deep MSP on V1/seed `37` achieved AUROC `0.8602`, FPR@95TPR `0.5500`, and OSCR `0.8541`; frozen Deep+raw-TDA MSP fusion achieved AUROC `0.8402`, FPR@95TPR `0.6878`, and OSCR `0.8337`. The paired delta versus Deep MSP was AUROC `-0.0200` with 95% CI `[-0.0235, -0.0160]` and FPR@95TPR `+0.1379` with 95% CI `[+0.1156, +0.1558]`. Therefore naive early concatenation of raw grayscale TDA with deep embeddings is rejected as the primary method, and Delivery 3 must not tune that method post hoc.

Delivery 3 compares only a controlled exploratory matrix:

- Deep MSP baseline.
- Historical raw-image TDA distance baseline from Delivery 2.
- Morphology-aware whole-cell distance-map TDA.
- Morphology-aware nucleus-candidate distance-map TDA.
- Morphology-aware cell+nucleus distance-map TDA.
- Frozen ResNet18 feature-map TDA.
- Deep MSP plus morphology TDA late fusion with fixed alpha `0.50`.
- Deep MSP plus feature-map TDA late fusion with fixed alpha `0.50`.
- Cytoplasm-candidate TDA only if known-train QC indicates the candidate is usable.

## Separation Rules

Training, validation, threshold calibration, and final testing are separate phases. Test data is never used for model selection, threshold selection, calibration, hyperparameter tuning, or preprocessing decisions.

Split V1 is the original deterministic seed-37 manifest. Split V2 is a conservative sensitivity split derived from high-confidence near-duplicate connected components. Split V2 is not allowed to overwrite Split V1, and it is used only to quantify split sensitivity with the same ResNet18 training recipe.

Unknown classes remain excluded from closed-set model fitting. Strict open-set thresholds are calibrated with known validation samples only; unknown test samples are reserved for final reporting.

Delivery 3 additionally forbids selecting segmentation thresholds, activation layers, fusion alphas, score calibrations, or feature combinations using unknown test metrics. Feature-map layer selection is based on pre-test feasibility: spatial resolution, non-degenerate diagrams, runtime, and feature variance. Fusion alpha `0.50` is primary and fixed before test evaluation; optional secondary fixed alphas are `0.25` and `0.75`.

The primary Delivery 3 split is V1 seed `37`, because the historical baselines are available there. Split V2 is reserved only for a method satisfying the predeclared V1 progression criteria.

## Primary Outcomes

Open-set AUROC, FPR@95TPR, and OSCR are primary outcomes. Closed-set macro-F1 and balanced accuracy are secondary outcomes used to detect regressions in known-class recognition. Small closed-set accuracy changes are not treated as the central scientific conclusion.

## Delivery 3 Subgroups

Delivery 3 subgroup analyses are defined before metric generation:

- `immature_related_myeloid`: `myeloblast`, `promyelocyte`, `promyelocyte_atypical`, `myelocyte`, `metamyelocyte`, `neutrophil_band`.
- `lymphoid_related`: `lymphocyte_large_granular`, `lymphocyte_neoplastic`, `lymphocyte_reactive`, `hairy_cell`, `plasma_cell`.
- `other`: `normoblast`, `smudge_cell`.

These groups are morphological analysis strata, not diagnoses.

## Delivery 3 Decision Rule

Delivery 3 is exploratory and does not provide final confirmatory validation. A method becomes a candidate for later multiseed confirmation only if at least one predeclared criterion is met:

- Criterion A: delta overall AUROC versus Deep MSP is at least `+0.01` with a directionally favorable paired-bootstrap confidence interval.
- Criterion B: delta FPR@95TPR is at most `-0.05` with no meaningful overall AUROC degradation.
- Criterion C: a predeclared subgroup has delta AUROC at least `+0.03` without severe overall OSR degradation.

If all defensible variants have delta AUROC `<= 0`, delta FPR@95TPR `>= 0`, and no predeclared subgroup shows a credible complementary effect, the recommendation is to stop TDA method development rather than run an open-ended representation search.

## Near-Duplicate Sensitivity

pHash collisions are treated as candidate near-duplicates, not ground truth. The audit uses a multi-stage evidence score:

- confirmed: exact checksum match.
- probable level 1: very low pHash distance plus very high structural similarity.
- probable level 2: low or moderate pHash distance plus high structural similarity.
- candidate level 3: pHash candidate only.

Only configured high-confidence levels are eligible for connected-component grouping in Split V2. Morphologically similar cells are not automatically declared to be the same cell.
