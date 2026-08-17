# Section 5 Figures: QA Report

## Verification summary

- `nature-figure` static preflight: 14 pass, 0 warning, 0 failure.
- Backend: Python/matplotlib in the repository `r2026-fsa` environment.
- Source integrity: Figure 6 reads the canonical Main-I and Efficiency CSV files; Figure 7 reads the canonical author-corrected transfer CSV. No rows are sampled or excluded after the figure-specific system and reporting-scope filters defined in the contract.
- Source-data bundle: `source_data/figure6_accuracy_system_cost.csv` and `source_data/figure7_decoder_transfer.csv` reproduce every plotted value and derived percentage.
- Exports: both figures are available as editable SVG, vector PDF, 600 dpi LZW-compressed TIFF and review PNG.

## Visual inspection

### Figure 6

- Direct labels, model counts and total logged training hours are legible at the target single-column width.
- No bubble or label is clipped; the parameter and MSE axes include all three systems without a broken scale.
- Bubble area encodes training time, while the exact values remain printed in the point labels.
- The lower-left direction marker is visually subordinate to the data.
- Reviewer-risk check: the figure does not imply lower training time for ISCF-BSCA and is captioned as a trade-off rather than uniform efficiency.

### Figure 7

- Panel titles, legend, dataset labels and percentage annotations remain legible at double-column width.
- Both panels share a zero-based MSE axis, preventing exaggerated paired-bar differences.
- Original Decoder and ISCF-BSCA use a neutral gray versus deep teal distinction that remains interpretable in grayscale by ordering and labels.
- No error bars are shown because the canonical corrected source contains aggregate point estimates rather than repeated-run uncertainty; this is disclosed in the caption.
- Reviewer-risk check: the plot contains the complete author-refined Weather/ETTm1/ETTm2 reporting scope for both backbones and does not imply universal transfer.

## Decision

`Decision=section5_figures_ready_for_author_review`.
