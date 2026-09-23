# Spanish Translation Audit

Source manuscript: `paper/journal_submission/main.tex`

Spanish manuscript: `paper/journal_submission/es/main_es.tex`

## Scope

- Full Spanish academic translation created from the existing English manuscript.
- Original English LaTeX files and `references.bib` were not modified.
- Scientific claims, metrics, datasets, model names, seeds, splits, figures, tables, equations, limitations, declarations, and cautious interpretation were preserved.
- Bibliographic titles, journals, proceedings, dataset names, DOI strings, and canonical morphology/taxonomy labels were kept in their original language where appropriate.

## Equivalence Checks

- Quantitative claims preserved: YES
- Figures preserved: YES; seven original figures are referenced from `../figures/`.
- Tables preserved: YES; three tables were translated and embedded in `main_es.tex`.
- Equations preserved: YES; taxonomy, ArcFace, classifier prediction, and Vietoris-Rips equations are retained.
- Citations preserved: YES; all 29 citation keys used in the English source are used in the Spanish source.
- Conclusions preserved: YES; no scientific result was strengthened, weakened, or reinterpreted.
- Clinical-safety wording preserved: YES; `unknown` remains a protocol status outside the known training taxonomy and is not described as malignancy, leukemia, diagnosis, or patient-level clinical endpoint.
- Topology interpretation preserved: YES; persistent homology remains negative or explanatory, not predictive or causal.

## Editorial Decisions

- `references_es.bib` is an audited copy for the Spanish manuscript only. The English `references.bib` was left intact.
- Two DOI/metadata corrections found during audit were applied only to `references_es.bib`: Tsutsui et al. 2023 and Wang et al. 2022 ViM.
- The Spanish PDF uses `naturemagdoi.bst` because the local `unsrt` BibTeX style does not support DOI/URL fields. DOI entries in `references_es.bib` also include visible `note = {DOI: ...}` fields so DOI strings appear in both the `.bib` and the compiled PDF.

## Human Review Required

- Funding statement remains unavailable and needs author confirmation.
- Competing-interest statement remains unavailable and needs author confirmation.
- CRediT author-contribution statement should be confirmed by G. Álvarez Sánchez.
- Public repository URL and software release identifier should be supplied before journal submission.
- Institution-specific ethics wording should be confirmed before submission.
