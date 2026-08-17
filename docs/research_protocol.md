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

Delivery 4 stops TDA method development and asks post-hoc open-set questions over the fixed ResNet18 seed-37 representation:

- `RQ8`: Can stronger post-hoc OSR scores over the frozen 512-dimensional ResNet18 embedding improve unknown-cell rejection versus MSP, entropy, energy, and the historical Mahalanobis baseline?
- `RQ9`: Which unknown morphologies remain hard, and are the failures concentrated in biologically adjacent morphology families?
- `RQ10`: Is poor OSR performance associated with unknown samples lying close to known-class embedding manifolds or nearest known attractors?

Delivery 5 moves from post-hoc scoring to exploratory representation development:

- `RQ11`: Can representation learning explicitly designed to improve intra-class compactness and inter-class separation improve open-set recognition of unseen hematological morphologies?
- `RQ12`: Does supervised contrastive representation learning improve unknown detection without materially degrading closed-set classification?
- `RQ13`: Does an angular-margin representation improve separation of morphologically related known and unknown blood cells?
- `RQ14`: Are improvements in OSR associated with measurable changes in embedding geometry?

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

Delivery 3 result: all defensible topology variants failed to improve the frozen ResNet18 open-set baseline. TDA remains only a historical negative/complementary ablation. Delivery 4 must not add new TDA variants, tune topology features, or reopen topology method development.

Delivery 4 compares post-hoc scoring rules over the frozen ResNet18 checkpoint `artifacts/checkpoints/resnet18_seed37/best_checkpoint.pt`, embedding dimension `512`, checkpoint SHA256 `7fb2e2a6741a4fcd1b7d9e3a985ab136fa0550741b69dcd62f0f21c28b6ae39d`, and embedding manifest hash `4b8da8df49655b6f04b53aa75a912f0efceac776d81761b2b45b68c440c27bae`. No representation learning is allowed: no SupCon, ArcFace, triplet loss, fine-tuning, or new CNN training.

The predeclared Delivery 4 method matrix is persisted before final evaluation at `configs/experiment/delivery4_predeclared_matrix.json`. Primary methods are MSP, energy `T=1`, Euclidean NCM, cosine NCM, cosine kNN `k=5`, Ledoit-Wolf regularized Mahalanobis, relative Mahalanobis, ReAct energy at the 90th known-train activation percentile, and ViM with an ID-only 95% PCA variance rule. Secondary methods include entropy, energy `T=0.5/2`, cosine kNN `k=1/10`, historical Mahalanobis, and ReAct at percentiles `85/95`. DICE is skipped unless a rigorous implementation is available; ODIN is not a primary method.

## Separation Rules

Training, validation, threshold calibration, and final testing are separate phases. Test data is never used for model selection, threshold selection, calibration, hyperparameter tuning, or preprocessing decisions.

Split V1 is the original deterministic seed-37 manifest. Split V2 is a conservative sensitivity split derived from high-confidence near-duplicate connected components. Split V2 is not allowed to overwrite Split V1, and it is used only to quantify split sensitivity with the same ResNet18 training recipe.

Unknown classes remain excluded from closed-set model fitting. Strict open-set thresholds are calibrated with known validation samples only; unknown test samples are reserved for final reporting.

Delivery 3 additionally forbids selecting segmentation thresholds, activation layers, fusion alphas, score calibrations, or feature combinations using unknown test metrics. Feature-map layer selection is based on pre-test feasibility: spatial resolution, non-degenerate diagrams, runtime, and feature variance. Fusion alpha `0.50` is primary and fixed before test evaluation; optional secondary fixed alphas are `0.25` and `0.75`.

Delivery 4 fits all post-hoc statistics on known-train embeddings only. Strict thresholds are calibrated with known-validation scores only. Unknown-test samples are used once for reporting the predeclared matrix, per-unknown-class analysis, subgroup analysis, paired bootstrap deltas, and embedding-geometry interpretation. ViM dimensionality, ReAct clipping, NCM prototypes, kNN banks, and covariance estimates must not use validation, test, or unknown rows.

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

## Delivery 4 Decision Rule

Delivery 4 is exploratory method development, not final confirmatory validation. A post-hoc method progresses only if it satisfies at least one predeclared criterion on V1:

- Criterion A: delta overall AUROC versus MSP is at least `+0.015` with a favorable paired-bootstrap confidence interval.
- Criterion B: delta FPR@95TPR versus MSP is at most `-0.075` while delta AUROC is at least `-0.005`.
- Criterion C: lymphoid-related subgroup delta AUROC is at least `+0.03` without overall AUROC degradation greater than `0.01`.

If no primary method passes these criteria, the scientific recommendation is to proceed to representation learning in a later delivery rather than continue post-hoc score engineering.

## Delivery 5 Representation Matrix

Delivery 5 is exploratory representation development, not independent confirmation, because the MLL23 unknown set has been inspected in previous deliveries. Unknown images and labels remain excluded from training, lambda selection, margin selection, checkpoint selection, threshold fitting, ViM fitting, kNN reference construction, and OSR method selection.

The controlled representation matrix is:

- Historical CE ResNet18 representation.
- ResNet18 trained with `CE + 0.50 * SupCon`, SupCon temperature `0.10`, and a `512 -> 256 -> 128` normalized projection head. The primary OSR representation remains the 512-dimensional backbone embedding.
- ResNet18 trained with ArcFace-style additive angular margin, scale `30`, margin `0.30`, and normalized 512-dimensional embeddings/classifier weights.

No additional losses are introduced in Delivery 5. CosFace is reserved as optional secondary infrastructure only if it does not expand the experiment; it is not part of the minimum primary matrix.

Each representation is evaluated with the fixed OSR scorers `MSP`, `Energy T=1`, `kNN cosine k=5`, and `ViM`. ViM and kNN are refit independently per representation using known train only.

The internal progression baseline is CE representation + ViM from Delivery 4, with AUROC `0.872604` and FPR@95TPR `0.514223`. A representation cannot progress if known-test macro-F1 is below `0.9616`.

Delivery 5 progression criteria:

- Criterion A: best predeclared representation/scorer delta AUROC versus CE+ViM is at least `+0.015` with favorable paired-bootstrap confidence interval.
- Criterion B: delta FPR@95TPR versus CE+ViM is at most `-0.075`, delta AUROC is at least `-0.005`, and the closed-set safeguard passes.
- Criterion C: lymphoid-related subgroup delta AUROC is at least `+0.03` versus the best CE scorer, overall AUROC degradation is at most `0.01`, and the closed-set safeguard passes.

If neither SupCon nor ArcFace passes A/B/C, no new metric-learning loss family is launched automatically; the decision moves to external-domain validation of the strongest existing system.

## Delivery 6 Confirmatory Stability Analysis

Delivery 6 is a confirmatory stability analysis within MLL23 for the frozen Delivery 5
candidate `ArcFace + MSP`. It is not external validation. The method is locked: ResNet18
ImageNet initialization family, 224x224 inputs, AdamW, learning rate `0.0003`, weight
decay `0.0001`, batch size `64`, AMP, weighted CE, max `30` epochs, known-validation
macro-F1 checkpoint selection, ArcFace scale `30`, and ArcFace margin `0.30`.

The predeclared run matrix is exactly 20 training runs: seeds `13`, `37`, `73`, `101`,
and `137`; splits V1 and V2; representations CE and ArcFace. V1 uses manifest hash
`4b8da8df49655b6f04b53aa75a912f0efceac776d81761b2b45b68c440c27bae`; V2 uses
`81ca3db900f788a0446f80ea5d36e8766d8ca050404b755111b3ca898f809026`. These splits must
not be regenerated. Matched comparisons are ArcFace(seed, split) minus CE(seed, split).

The only OSR scores are MSP and ViM. ViM is fit independently per run using known-train
embeddings/logits only. Unknown-test samples are used only after models are frozen for
reporting, class-level stability, geometry, and attractor analysis. Statistical summaries
use seed-level matched deltas as the unit of replication; image-level bootstrap must not be
presented as multiseed evidence.

ArcFace is considered a reproducible primary method only if criteria A, C, D, and E pass.
Criterion A requires V1 mean Delta AUROC `>= +0.015` and a favorable t-based 95% CI over
the five seed-level deltas. Criterion C requires at least four of five V1 seeds to have
positive Delta AUROC. Criterion D requires V2 mean Delta AUROC `>= -0.01` and at least
three of five V2 seeds with nonnegative Delta AUROC. Criterion E requires every ArcFace
run to have macro-F1 `>= 0.9616`, or no ArcFace run to be more than `0.01` absolute below
its matched CE run. Criterion B, mean Delta FPR95 `<= -0.075` with favorable CI, strengthens
the claim but is not mandatory if AUROC evidence is strong.

If ArcFace fails confirmation, no additional margin, loss, backbone, scorer, TDA variant,
or SupCon rerun is launched automatically. The next step becomes external validation of
the strongest stable baseline or an external comparison of CE and ArcFace if the internal
evidence is mixed.

## Near-Duplicate Sensitivity

pHash collisions are treated as candidate near-duplicates, not ground truth. The audit uses a multi-stage evidence score:

- confirmed: exact checksum match.
- probable level 1: very low pHash distance plus very high structural similarity.
- probable level 2: low or moderate pHash distance plus high structural similarity.
- candidate level 3: pHash candidate only.

Only configured high-confidence levels are eligible for connected-component grouping in Split V2. Morphologically similar cells are not automatically declared to be the same cell.
