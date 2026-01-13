# Label Processing

This repo assumes labels are defined by `code2idx_3lvl.json`, mapping 3-level AIS codes to indices.

Examples of 3-level codes:
- `1/2/03`
- `6/4/06`

## Normalization

For prefix-based metrics we normalize codes by removing `/`:
- `1/2/03` -> `1203`

Prefix lengths:
- 1: region
- 2: system
- 4: 4-digit code (region+system+2-digit subcode)

## Ground truth parsing

The evaluation script extracts ground-truth codes from `completion`.
If your ground truth is stored as 7-digit numeric AIS codes, add a project-specific conversion step that maps
7-digit codes to the 3-level space used by `code2idx_3lvl.json`.
