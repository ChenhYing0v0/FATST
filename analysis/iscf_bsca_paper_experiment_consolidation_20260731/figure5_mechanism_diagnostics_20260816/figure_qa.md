# Figure 5 QA

## Contract and integrity

- Core conclusion and five-panel evidence map are frozen in `design_and_prelaunch_gate.md`.
- Backend：Python/matplotlib only；no cross-backend rendering。
- Split：validation only；seed=2021。
- Aggregate `n`：each dataset uses every sequential validation series row in its frozen NPZ；the exact row count is retained in source CSVs。
- Center statistic：dataset-level arithmetic mean followed by equal-weight dataset macro mean。
- Variability/inferential test：none；panels are descriptive diagnostics and do not display confidence intervals or p-values。
- Comparator：jointly trained `Fixed Scope (s=144)` checkpoint for the qualitative panel。
- No missing-value filtering, smoothing, manual row exclusion or post-selection panel substitution。

## Export QA

- Static source preflight：13 PASS、0 FAIL；the single width WARN is a parser artifact caused by reading `180 / 25.4` as a literal width product。
- PDF：1 page，`509.76 × 452.88 pt`，corresponding to the frozen `180 × 160 mm` canvas within rounding。
- Editable text：SVG uses `svg.fonttype=none`；PDF uses TrueType font embedding。
- Raster export：TIFF at 600 dpi with LZW compression；PNG at 300 dpi。
- Visual inspection：panel labels, heatmap annotations, colorbars, horizon markers and legend are readable at the final two-column size；no clipping or overlap observed。

## Reviewer-risk audit

- Panel b/c visibly show near-uniform probabilities; caption/result text must not call this strong specialization。
- Panel d measures scope-arm error heterogeneity, not successful routing。
- Panel e is explicitly performance-selected from 1,280 rows and must not be described as representative or typical。
- Exact CHPC is a structural/numerical property, not an accuracy improvement。
