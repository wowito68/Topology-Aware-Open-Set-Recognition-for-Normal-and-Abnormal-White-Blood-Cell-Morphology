# TDA Methodology

## Filtration

The initial TDA module supports grayscale sublevel and superlevel filtrations over resized 2D cell images.

The baseline TDA preprocessing is intentionally independent from the CNN preprocessing:

1. Load the original TIFF without modifying the source file.
2. Convert to grayscale.
3. Resize to the benchmark-selected square resolution.
4. Normalize intensities to `[0, 1]`.
5. Build a 2D cubical complex.

The Delivery 2 baseline does not apply CLAHE, adaptive thresholding, learned segmentation, or edge detection. Those choices can be considered later as explicit ablations.

Delivery 3 keeps the raw grayscale pipeline only as a historical baseline and adds two topology sources that are more representation-aware:

1. Morphology-conditioned candidate masks converted to distance maps.
2. Frozen ResNet18 activation-energy maps extracted from internal feature stages.

Neither source is tuned with unknown-test metrics.

For an intensity function `f : pixels -> [0,1]`, the sublevel filtration is `K_a = {x | f(x) <= a}`. The superlevel filtration is implemented by applying the same sublevel machinery to `g(x) = 1 - f(x)`, which is equivalent to sweeping the original image from bright to dark.

## Cubical Complex

Images are represented as cubical complexes. Persistent homology is computed over pixel intensity filtrations with GUDHI when available.

## Homology Dimensions

The first experiments compute `H0` and `H1`. `H2` is not assumed useful for 2D images.

## Persistence Diagram

Each sample produces a persistence diagram keyed by homology dimension. Diagrams must be associated with sample id, dataset, preprocessing config, TDA config, and code/config version.

Infinite bars are preserved in the diagram cache as essential intervals. Vectorizers exclude non-finite intervals rather than replacing infinity with an arbitrary finite endpoint.

The cache stores finite interval arrays and essential-bar counts rather than serialized GUDHI objects. Cache metadata records sample IDs, manifest hash, source checksum availability, preprocessing, resolution, filtrations, homology dimensions, GUDHI version, NumPy version, and a config hash.

## Vectorization

Implemented vectorizers include persistence entropy, Betti curve, persistence image, and persistence landscape. Vectorizers that learn ranges or grids are fit only on training diagrams.

Delivery 2 evaluates exactly these vectorizers:

- Persistence Entropy.
- Betti Curve.
- Persistence Landscape.
- Persistence Image.

Vectorization is performed independently for sublevel and superlevel diagrams over `H0` and `H1`, then concatenated. The feature archive records component slices such as `sublevel_persistence-image` and `superlevel_betti-curve`, allowing ablations without recomputing persistent homology.

The feature archive does not globally standardize features. Downstream models fit scalers on known train only, then transform validation, known test, and unknown test.

## Morphology-Conditioned Distance Maps

MLL23 does not provide validated masks in this pipeline. Automatically generated regions are therefore named `whole-cell candidate`, `nucleus candidate`, and `cytoplasm candidate`; they are not anatomical ground truth.

The Delivery 3 whole-cell candidate is a deterministic foreground estimate based on border-median RGB background color distance plus saturation/value thresholds. It applies small-component removal, morphological opening/closing, hole filling, and a central-component preference because the dataset consists of single-cell crops.

The nucleus candidate is a deterministic chromatin-like estimate inside the whole-cell candidate. It uses relative darkness, saturation, and blue-purple color margin thresholds, followed by the same cleanup. The cytoplasm candidate is defined as:

```text
cytoplasm_candidate = whole_cell_candidate - nucleus_candidate
```

with deterministic cleanup and the nucleus subtracted again after cleanup to keep the regions disjoint.

For a binary candidate mask `M`, the primary scalar field is the normalized inside-mask Euclidean distance transform:

```text
d(x) = distance from x to the nearest background pixel, for x in M
d(x) = 0 otherwise
d_norm(x) = d(x) / max_y d(y)
```

Empty or invalid candidates are encoded as zero distance maps and tracked in metadata. The status labels are `valid`, `warning`, and `invalid`; no Dice, IoU, or segmentation accuracy is reported without ground-truth masks.

Morphology distance-map TDA computes sublevel and superlevel filtrations over `d_norm`, using `H0` and `H1` only. The retained feature sets are `MORPH_CELL`, `MORPH_NUCLEUS`, `MORPH_CYTOPLASM` when QC supports it, and `MORPH_CELL_NUCLEUS`.

## Feature-Map Topology

Delivery 3 extracts activations from the frozen ResNet18 checkpoint without modifying weights or taking gradients. Candidate stages are `layer1`, `layer2`, `layer3`, and `layer4`.

For an activation tensor `A in R^(C x H x W)`, the primary scalar map is RMS activation energy:

```text
E(h,w) = sqrt(mean_c A(c,h,w)^2)
```

Each map is normalized per sample by robust percentiles to `[0,1]`. Persistent homology is then computed for sublevel and superlevel filtrations over `H0` and `H1`.

Layer selection is made before final test evaluation from a benchmark on approximately 500 stratified samples. The criteria are spatial resolution, non-degenerate diagrams, runtime, and feature variance. Unknown-test AUROC is not a layer-selection criterion. Cache metadata records the checkpoint SHA256, layer, activation aggregation, normalization, filtrations, homology dimensions, vectorizer config, manifest hash, and code/config hash so that changing the checkpoint invalidates the cache.

## Fusion

TDA features are precomputed and cached. Fusion concatenates deep embeddings and TDA vectors, then trains an MLP classifier. TDA extraction is not embedded inside the GPU forward pass.

Fusion uses the pre-logit ResNet18 embedding, not logits, as the deep representation. Deep and TDA blocks are standardized separately using known train only before concatenation. The fusion classifier is a small frozen-feature MLP with early stopping on known validation macro-F1.

Delivery 3 does not train a fusion classifier. It produces TDA anomaly scores from nearest known-class centroid distance in vectorized TDA space, with class statistics fit on known train only. Scores are oriented so larger means more unknown-like, calibrated to empirical percentiles using known validation scores only, and combined with calibrated Deep MSP anomaly using fixed late fusion:

```text
S_late = alpha * S_deep_calibrated + (1 - alpha) * S_tda_calibrated
```

The primary alpha is `0.50`; secondary exploratory alphas are `0.25` and `0.75` if reported.

## Resolution Benchmark

Before full extraction, a stratified seed-37 benchmark samples approximately 1,000 images across all 18 classes and compares `64x64`, `96x96`, and `128x128` for both sublevel and superlevel filtrations. Selection balances computational cost and feature stability relative to the maximum benchmarked resolution.
