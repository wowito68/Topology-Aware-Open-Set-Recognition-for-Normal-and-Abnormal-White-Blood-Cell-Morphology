# BSPC Author Guidelines Audit

Consultation date: 2026-08-21

Official sources checked:

- Elsevier journal guide for Biomedical Signal Processing and Control: https://www.elsevier.com/journals/biomedical-signal-processing-and-control/1746-8094/guide-for-authors
- ScienceDirect journal guide mirror: https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control/publish/guide-for-authors
- Elsevier graphical abstract guidance: https://www.elsevier.com/researcher/author/tools-and-resources/graphical-abstract
- Elsevier author preparation guidance: https://www.elsevier.com/subject/next/guide-for-authors

## Scope Fit

Biomedical Signal Processing and Control publishes work on processing and analysis of biomedical signals and data. The manuscript is a biomedical image-analysis evaluation study involving hematological cell morphology, open-set recognition, domain shift, and reproducibility. Scope fit is reasonable if positioned as rigorous biomedical image-analysis validation, not as a clinical diagnostic product.

## Article Type

Appropriate article type: full paper / original research article.

Rationale: the work reports a complete empirical evaluation, frozen protocols, multiseed results, external validation, and morphology-level analysis. It is not a review, short communication, clinical trial, or methods note.

## Structure

Recommended structure for this manuscript:

- Title
- Abstract
- Keywords
- Introduction
- Related Work
- Materials and Methods
- Results
- Discussion
- Limitations
- Conclusion
- Declarations
- References
- Supplementary material

Elsevier guidance allows clearly defined numbered sections and does not require mechanical template conversion at initial submission. Elsevier recommends `elsarticle.cls` for LaTeX when available; it is not installed in this local TeX environment, so the submission candidate uses a generic article class that compiles locally.

## Abstract

Official guide language found: the abstract should be concise, factual, stand alone, state purpose, principal results, and major conclusions, and avoid references where possible.

Action taken: the abstract was rewritten as a single quantitative summary of the problem, methods, key internal and external results, morphology-dependent failure, and conclusion. The abstract is approximately 250 words.

## Keywords

BSPC guide requirement found in official search snippet: 1 to 7 keywords for indexing.

Action taken: 6 keywords are provided.

## Highlights

BSPC guide requirement found: highlights are required as 3 to 5 bullet points, each with a maximum of 85 characters including spaces.

Action taken: `highlights.txt` contains 5 concise highlights. Each line is under 85 characters.

## Graphical Abstract

BSPC/Elsevier guidance found: a graphical abstract is encouraged at submission, not identified as mandatory in the accessible official source text. Elsevier recommends a separate image that summarizes the article.

Action taken: no graphical abstract file was generated. Figure 1 could be adapted into a graphical abstract, but a final graphical abstract should be author-approved.

## Declarations

Required or expected declarations identified from Elsevier guidance:

- Data availability
- Code availability where applicable
- Author contributions / CRediT confirmation
- Funding
- Declaration of competing interest
- Ethics or institutional-review statement where relevant
- Declaration of generative AI use during manuscript preparation upon submission

Action taken: manuscript declaration sections were added. Funding, competing interest, final CRediT wording, public code URL, and generative-AI disclosure require author confirmation before journal submission.

## References

Official guidance found: references should be correct and complete; the journal reference style is applied after acceptance/proof, while initial submission can use a consistent style.

Action taken: the manuscript uses a numbered BibTeX style and verified metadata where possible. The legacy temporary hematology review citation was replaced with real sources.

## Figures and Tables

Official Elsevier guidance found: figures should be numbered in text order, have readable lettering, and use high-resolution files. Tables should be editable text, numbered in order, and avoid unnecessary duplication.

Action taken: the main manuscript uses 7 high-value figures copied from existing reviewed paper figures and 3 compact editable LaTeX tables. Existing PNG figure dimensions are sufficient for review PDF readability, though a final production pass could generate vector artwork.

## Supplementary Material

Elsevier guidance allows supplementary material to be submitted with concise descriptive captions.

Action taken: `supplement/main_supplement.tex` was created with protocol details, full result context, and reproducibility notes.
