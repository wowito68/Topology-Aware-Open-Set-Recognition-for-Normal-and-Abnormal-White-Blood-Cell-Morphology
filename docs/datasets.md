# Datasets

## MLL23

- Dataset: https://zenodo.org/records/14277609
- Paper: https://www.nature.com/articles/s41597-025-06223-x
- DOI: TODO confirm from dataset metadata before citation.
- License: TODO confirm before redistribution or publication.
- Acquisition domain: single-cell peripheral blood images.

## Download

MLL23 is downloaded manually. No command in this repository downloads large datasets automatically.

## Class Mapping

Raw folder names are canonicalized through `hemato_osr.data.taxonomy`. The initial known taxonomy is limited to mature leukocyte classes. Other configured MLL23 morphologies are treated as unknown for open-set evaluation.

## Limitations

If reliable patient or slide identifiers are absent, patient-level leakage cannot be excluded. Do not invent identifiers. Document this limitation in any experiment report.

