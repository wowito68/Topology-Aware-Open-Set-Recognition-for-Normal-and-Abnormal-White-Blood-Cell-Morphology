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

## Fusion

TDA features are precomputed and cached. Fusion concatenates deep embeddings and TDA vectors, then trains an MLP classifier. TDA extraction is not embedded inside the GPU forward pass.

Fusion uses the pre-logit ResNet18 embedding, not logits, as the deep representation. Deep and TDA blocks are standardized separately using known train only before concatenation. The fusion classifier is a small frozen-feature MLP with early stopping on known validation macro-F1.

## Resolution Benchmark

Before full extraction, a stratified seed-37 benchmark samples approximately 1,000 images across all 18 classes and compares `64x64`, `96x96`, and `128x128` for both sublevel and superlevel filtrations. Selection balances computational cost and feature stability relative to the maximum benchmarked resolution.
