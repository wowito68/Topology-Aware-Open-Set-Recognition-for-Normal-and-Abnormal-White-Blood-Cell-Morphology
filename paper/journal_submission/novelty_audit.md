# Novelty Audit

Consultation date: 2026-08-21

Focused question: has open-set recognition already been studied specifically for leukocyte morphology with held-out named morphologies and external test-only validation?

## Summary

The focused audit found substantial prior work on automated leukocyte and blood-cell classification, several studies on domain shift or cross-domain blood-cell classification, and extensive general OSR/OOD literature. It did not identify a directly comparable peer-reviewed study that combines all of the following: a fixed mature-leukocyte known taxonomy, named hematological morphologies held out as unknown classes, multiseed open-set scoring, near-duplicate split sensitivity, external test-only hematology validation, and morphology-level unknown failure analysis.

This is not proof that no such work exists. The manuscript should avoid absolute claims such as "first", "unprecedented", or "the first ever". A defensible claim is: "To our knowledge, few hematology morphology studies have evaluated open-set rejection of named unseen morphologies under an explicit taxonomy freeze and external test-only domain shift."

## Audit Table

| Paper | Year | Dataset | Task | Known/unknown protocol | External validation | OSR/OOD method | Morphology-level analysis | Relation to our work |
|---|---:|---|---|---|---|---|---|---|
| Acevedo et al., Recognition of peripheral blood cell images using CNNs | 2019 | Peripheral blood cell dataset | Closed-set cell classification | No explicit open-set protocol | No principal test-only OSR domain | Not OSR | Class-level closed-set reporting | Shows strong WBC classification but not unknown rejection. |
| Matek et al., Human-level recognition of blast cells in AML | 2019 | AML single-cell images | Blast recognition | Diagnostic/task-specific closed-set framing | Not comparable to held-out morphology OSR | Not OSR | Blast-focused | Demonstrates deep hematology morphology performance, not open-set taxonomy reliability. |
| Matek et al., Bone marrow morphology classification | 2021 | Large bone marrow image set | Closed-set morphology classification | No held-out unknown protocol as central task | Not comparable to AML-LMU test-only OSR | Not OSR | Broad morphology classification | Important morphology classifier precedent; different question. |
| Khan et al., WBC classification review | 2021 | Literature review | WBC classification methods | Summarizes closed-set literature | Not a protocol study | Not a primary OSR study | Review-level | Supports that WBC classification literature is broad but mostly closed-set. |
| Deshpande et al., Microscopic blood-cell analysis review | 2021 | Literature review | AI for blood-cell disease analysis | No protocol | No | Not OSR | Review-level | Useful broader review source replacing the legacy temporary citation. |
| Salehi et al., Cross-domain feature extraction for single blood cell classification | 2022 | Single blood-cell datasets | Cross-domain closed-set classification | No named unknown rejection | Yes, domain transfer focus | Not OSR | Limited for unknowns | Closest on domain shift, but not held-out morphology OSR. |
| Tsutsui et al., Benchmarking WBC classification under domain shift | 2023 | WBC image datasets | Domain-shift benchmarking | No open-set unknown rejection as main task | Yes | Not OSR | Closed-set domain-shift emphasis | Very close motivation on WBC domain shift; preprint and closed-set focused. |
| Sadafi et al., Continual learning for cross-domain WBC classification | 2024 | WBC domains | Continual/cross-domain classification | No explicit held-out morphology OSR | Cross-domain | Not OSR | Not central | Supports domain shift relevance in WBC recognition. |
| MLL23 Scientific Data descriptor | 2025 | MLL23 | Dataset release and dataset facts | Provides 18 classes, not an OSR evaluation | No OSR external validation | Not OSR | Class inventory | Enables our fixed known/unknown morphology protocol. |
| General OSR/OOD literature: Scheirer, OpenMax, MSP, Energy, ReAct, ViM, surveys | 2013-2024 | Generic CV/OOD benchmarks | Open-set/OOD detection | Yes, generally | Some cross-dataset settings | Multiple | Usually not hematology-specific | Provides methods and definitions, not hematology-specific evidence. |

## Answers to Requested Novelty Questions

Has open-set recognition already been studied specifically for leukocyte morphology?

- No directly comparable peer-reviewed leukocyte morphology OSR study was identified in the focused search. Some adjacent open-set object-detection or few-shot blood-smear preprints exist, but they do not match this fixed mature-leukocyte taxonomy and held-out morphology protocol.

Has external domain-style validation been studied for this task?

- Domain shift has been studied for WBC/blood-cell classification, including cross-domain feature extraction and benchmarking. The identified work is primarily closed-set, not test-only unknown-morphology OSR.

Have unseen hematological morphologies been explicitly held out as unknown classes?

- This specific protocol was not found in the focused literature audit. MLL23 provides the class inventory that makes such a protocol possible.

Are there comparable hematology studies evaluating OSR/OOD under domain shift?

- Closest comparable work evaluates domain shift in blood-cell classification. Direct OSR/OOD under external hematology domain shift appears limited based on this search.

## Novelty Rating

Assessment: MODERATE.

Reason: the methods are mostly established, but the evaluation design, fixed morphology taxonomy, external test-only validation, and morphology-dependent failure analysis are a meaningful contribution for hematological image analysis. The novelty is empirical and protocol-centered, not architectural.
