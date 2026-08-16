# Research Protocol

## Question

Can deep visual representations and topological image features classify known mature leukocytes while detecting cell morphologies outside the training taxonomy?

Delivery 2 focuses on the following primary research question:

`RQ1`: Do persistent-homology-derived morphological descriptors improve open-set recognition of previously unseen blood-cell morphologies when combined with deep image representations?

Secondary questions:

- `RQ2`: How informative are topological features alone for mature white-blood-cell classification?
- `RQ3`: Are topological descriptors more useful for some unknown morphologies than others?
- `RQ4`: Are conclusions robust to a conservative near-duplicate-aware split?

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

## Separation Rules

Training, validation, threshold calibration, and final testing are separate phases. Test data is never used for model selection, threshold selection, calibration, hyperparameter tuning, or preprocessing decisions.

Split V1 is the original deterministic seed-37 manifest. Split V2 is a conservative sensitivity split derived from high-confidence near-duplicate connected components. Split V2 is not allowed to overwrite Split V1, and it is used only to quantify split sensitivity with the same ResNet18 training recipe.

Unknown classes remain excluded from closed-set model fitting. Strict open-set thresholds are calibrated with known validation samples only; unknown test samples are reserved for final reporting.

## Primary Outcomes

Open-set AUROC, FPR@95TPR, and OSCR are primary outcomes. Closed-set macro-F1 and balanced accuracy are secondary outcomes used to detect regressions in known-class recognition. Small closed-set accuracy changes are not treated as the central scientific conclusion.

## Near-Duplicate Sensitivity

pHash collisions are treated as candidate near-duplicates, not ground truth. The audit uses a multi-stage evidence score:

- confirmed: exact checksum match.
- probable level 1: very low pHash distance plus very high structural similarity.
- probable level 2: low or moderate pHash distance plus high structural similarity.
- candidate level 3: pHash candidate only.

Only configured high-confidence levels are eligible for connected-component grouping in Split V2. Morphologically similar cells are not automatically declared to be the same cell.
