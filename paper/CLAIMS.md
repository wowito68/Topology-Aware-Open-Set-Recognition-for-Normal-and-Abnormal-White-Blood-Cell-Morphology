# Claim Freeze

The manuscript may use SUPPORTED claims as main conclusions. PARTIALLY_SUPPORTED claims require cautious wording. NOT_SUPPORTED claims must not be stated as conclusions.

## SUPPORTED
- **C1**: Closed-set WBC morphology classification remains high internally but degrades externally.
  - Recommended wording: Closed-set accuracy alone overstates external reliability; macro-F1/balanced accuracy reveal domain degradation.
  - Evidence: Internal D6 closed macro-F1 near 0.97; external closed macro-F1 about 0.71-0.75 depending representation/split.
  - Constraint: External closed accuracy can remain high due to class imbalance.
- **C6**: Open-set reliability is morphology-dependent.
  - Recommended wording: Unknown rejection difficulty is strongly morphology-dependent, with small-class uncertainty acknowledged.
  - Evidence: External per-unknown AUROC varies widely by mapped unknown class; internal D6 also shows class-specific failures.
  - Constraint: Small-n external classes require cautious wording.
- **C7**: External domain shift is a dominant limitation.
  - Recommended wording: External acquisition/domain shift materially changes both closed-set and OSR behavior.
  - Evidence: All major systems lose AUROC and closed macro-F1 externally and increase FPR95.
  - Constraint: Relative system ranking changes by split/method.

## PARTIALLY_SUPPORTED
- **C2**: MSP is limited but externally can outperform ViM in this domain.
  - Recommended wording: No OSR score is uniformly reliable; ViM stability internally does not transfer cleanly to AML-LMU.
  - Evidence: External MSP AUROC exceeds ViM for CE and ArcFace in V1/V2 means.
  - Constraint: Internal D6 favored CE+ViM stability.
- **C8**: Embedding geometry explains OSR failures better than VR alone.
  - Recommended wording: Conventional geometry provides the primary failure explanation; VR is a secondary descriptive lens.
  - Evidence: Attractor and cross-class mixing align with lymphocyte-like and neutrophil-band failures.
  - Constraint: VR captures complementary stability/domain summaries but weak OSR correlations.

## NOT_SUPPORTED
- **C3**: ViM is the externally best open-set method.
  - Recommended wording: CE+ViM is the conservative internal baseline, not the external winner.
  - Evidence: Internal CE+ViM was stable.
  - Constraint: External mean AUROC is lower for ViM than MSP in both CE and ArcFace systems.
- **C4**: Cubical persistent homology improves predictive OSR utility.
  - Recommended wording: Cubical PH is a negative ablation and should not be promoted as a predictive method.
  - Evidence: None in confirmatory artifacts.
  - Constraint: Delivery 3 decision rule stopped TDA method development.
- **C5**: ArcFace is superior to CE.
  - Recommended wording: ArcFace is an informative instability/negative ablation, not a confirmed main method.
  - Evidence: Some external ArcFace+MSP means are favorable.
  - Constraint: Delivery 6 internal multiseed/split confirmation failed; neutrophil-band degradation persisted.
- **C9**: Vietoris-Rips topology causally explains OSR instability.
  - Recommended wording: VR is explanatory/descriptive only; avoid causal or predictive claims.
  - Evidence: VR artifacts show measurable seed/split/domain variation.
  - Constraint: Topology-OSR correlations are weak/inconsistent and exploratory.
- **C10**: The system externally generalizes robustly.
  - Recommended wording: External validation reveals limited OSR generalization under domain shift.
  - Evidence: Some closed accuracy remains high.
  - Constraint: External AUROC is modest and FPR95 high for all main systems.
