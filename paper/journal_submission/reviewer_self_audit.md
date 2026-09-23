# Reviewer Self-Audit

Audit date: 2026-08-21

## Reviewer 1: Computer Vision / Open-Set Recognition

| Criticism | Status | Response |
|---|---|---|
| Novelty may look like standard OOD benchmarking rather than a new method. | PARTIALLY RESOLVED | The manuscript is positioned as an empirical biomedical validation study, not a method paper. The novelty audit rates the contribution as moderate and protocol-centered. |
| Unknown morphologies could leak into training, validation, calibration, or representation fitting. | RESOLVED | The taxonomy freeze is stated in Methods, and MLL23 unknowns plus AML-LMU unknowns are test-only. Fitted quantities use known-training data only. |
| V2 might be mistaken for patient-level validation. | RESOLVED | The manuscript repeatedly states that V2 is a near-duplicate-aware sensitivity split, not patient-level validation. |
| ArcFace could be overclaimed from the favorable V1 MSP result. | RESOLVED | The text reports the V1/V2 reversal, matched seed counts, and concludes that ArcFace is not robustly superior. |
| MSP and ViM comparisons could be unfair if ViM used external PCA fitting. | RESOLVED | The external test-only rule explicitly excludes external PCA fitting, ViM fitting, calibration, method selection, and preprocessing selection. |
| The paper might lack strong algorithmic novelty for a computer-vision reviewer. | UNRESOLVED | This is a real positioning risk. The mitigation is to emphasize protocol rigor, morphology-specific failures, and external-domain evidence. |

Main concern: algorithmic novelty is limited; the manuscript must be evaluated as a rigorous biomedical image-analysis validation study.

## Reviewer 2: Biomedical / Hematology Imaging

| Criticism | Status | Response |
|---|---|---|
| Unknown labels may be read as cancer, leukemia, malignant, or abnormal diagnosis. | RESOLVED | The manuscript defines unknown strictly as outside the known training taxonomy and explicitly rejects diagnostic interpretation. |
| The known and unknown taxonomy may be biologically unclear. | RESOLVED | Known classes and mapped external unknown morphologies are listed, with band, atypical lymphocyte, blast, and immature labels kept outside mature known classes. |
| Morphology-level failure analysis might imply biological causality. | RESOLVED | The manuscript frames attractors as recognition-space patterns, not biological causal relationships. |
| AML-LMU contains AML-related samples, so the work could be mistaken for leukemia detection. | RESOLVED | The external dataset is described as an acquisition/domain stress test, not a diagnostic endpoint. |
| One external dataset may be too narrow for clinical generalization. | PARTIALLY RESOLVED | The limitation is explicit. The paper does not claim clinical deployment or prospective workflow validity. |

Main concern: biomedical readers may overread the AML-LMU experiment unless the taxonomy-only meaning of unknown remains prominent.

## Reviewer 3: Statistics / Reproducibility

| Criticism | Status | Response |
|---|---|---|
| Mean and standard deviation across five seeds could be overinterpreted. | RESOLVED | The manuscript reports mean plus standard deviation and avoids p-values or unsupported significance language. |
| Conclusions may be stronger than evidence. | RESOLVED | The conclusion is conservative: closed-set performance is insufficient, failures are structured, ArcFace is not robustly superior, and topology is not predictive. |
| Numerical claims need frozen-artifact traceability. | RESOLVED | `claim_traceability.csv` indexes 37 quantitative claims, all marked VERIFIED. |
| External test-only status could be compromised by model or threshold selection. | RESOLVED | No external adaptation, calibration, method selection, PCA fitting, ViM fitting, or preprocessing selection is allowed. |
| Five seeds and two splits are useful but not exhaustive. | PARTIALLY RESOLVED | The limitation is stated, and the paper avoids significance claims beyond the frozen summaries. |
| Missing public code URL and author declarations affect submission completeness. | UNRESOLVED | The science is traceable locally, but author confirmation is still required before journal submission. |

Main concern: submission readiness depends on author-supplied declarations and public code/release metadata.

## Editorial Desk-Rejection Audit

| Risk | Rating | Mitigation |
|---|---|---|
| Insufficient novelty | MODERATE | The manuscript avoids method novelty claims and emphasizes a concrete hematology OSR protocol, external test-only validation, and morphology-level failure analysis. |
| Too engineering-oriented | MODERATE | The Results and Discussion foreground biomedical morphology, taxonomy freezes, leakage control, and external validation rather than software chronology. |
| Narrow dataset scope | MODERATE | The paper uses one internal and one external dataset; this is acknowledged as a limitation. |
| Weak medical framing | LOW | The manuscript links the task to leukocyte morphology recognition while avoiding diagnosis claims. |
| Excessive topology digression | LOW | Topology is confined to a secondary negative/explanatory role and supplementary detail. |
| Incomplete literature review | LOW | The related work now covers hematology classification, OSR/OOD, domain shift, medical OOD, and topology. |
| Unclear practical implication | LOW | The paper gives a clear evaluation implication: closed-set metrics, seed/split sensitivity, per-morphology analysis, and external validation are necessary. |
| Language quality | LOW | The manuscript has been rewritten in professional academic English. |
| Scope mismatch with BSPC | MODERATE | Fit is reasonable for biomedical image analysis, but BSPC may prefer a stronger signal-processing or clinical-application framing. |

Overall desk-rejection risk: MODERATE.

Primary mitigation: present the paper as a careful validation and reliability study for biomedical image analysis, not as a new architecture or diagnostic product.
