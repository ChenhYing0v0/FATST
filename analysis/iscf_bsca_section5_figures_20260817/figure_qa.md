# Section 5 Figures: QA Report

## Verification summary

- `nature-figure` static preflight: 14 pass, 0 warning, 0 failure.
- Backend: Python 3.11.7 (`/opt/anaconda3`) with matplotlib 3.8.0; all rendering and QA stayed on the saved Python backend.
- Source integrity: Figure 6 reads the canonical nine-system Efficiency macro CSV, whose accuracy columns are generated from Main-I. It applies the author-requested exact filter `system != SimpleTM`, preserving 8/9 rows; Figure 7 reads the canonical author-corrected transfer CSV. No other rows are sampled or excluded.
- Source-data bundle: `source_data/figure6_accuracy_system_cost.csv` and `source_data/figure7_decoder_transfer.csv` reproduce every plotted value and derived percentage.
- Exports: both figures are available as editable SVG, vector PDF, 600 dpi LZW-compressed TIFF and review PNG. Figure 6 raster dimensions are 4,269 × 2,259 px at 600 dpi (TIFF) and 2,134 × 1,129 px at 300 dpi (PNG); its PDF is one page at approximately double-column width.

## Visual inspection

### Figure 6

- Direct labels and exact storage/peak-memory values are legible at the target double-column width.
- No bubble, label, size legend or footnote is clipped; all eight included systems are visible.
- The storage axis is explicitly labeled as logarithmic. Bubble area is directly proportional to peak memory, with 25/100/225 MiB reference circles.
- The $\dagger$ marker separates four official-configuration architecture-equivalent rows from actual trained-checkpoint resource rows.
- Reviewer-risk check: SimpleTM is excluded only from the figure at the author's request and remains in complete Table 3; DLinear and QDF preserve the negative resource boundary, so the figure is captioned as a trade-off rather than uniform efficiency.
- Statistics boundary: all marks are deterministic seven-dataset macro point estimates from the frozen table; no seed-level interval is available or implied, so no error bars are drawn.

### Figure 7

- Panel titles, legend, dataset labels and percentage annotations remain legible at double-column width.
- Both panels share a zero-based MSE axis, preventing exaggerated paired-bar differences.
- Original Decoder and ISCF-BSCA use a neutral gray versus deep teal distinction that remains interpretable in grayscale by ordering and labels.
- No error bars are shown because the canonical corrected source contains aggregate point estimates rather than repeated-run uncertainty; this is disclosed in the caption.
- Reviewer-risk check: the plot contains the complete author-refined Weather/ETTm1/ETTm2 reporting scope for both backbones and does not imply universal transfer.

## Decision

`Decision=section5_figures_ready_for_author_review`.
