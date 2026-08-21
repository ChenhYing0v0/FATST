# Figure 5 v2 QA

## Decision

- Render status: `pass_author_review_draft`.
- Canonical-paper status: `not_promoted`.
- The previous Figure 5 is treated as rejected for the main-text redesign; its evidence package remains unchanged.
- New implementation, training, remote execution and formal test: `0/0/0/0`.

## Data integrity

- Panel a contains `200/200` planned dataset--region--scope aggregates: 5 datasets × 8 future regions × 5 scopes.
- Every Panel-a value uses all available validation origin--variable rows for its dataset; no sample, region, scope or dataset was selected out.
- Panel-a arrays are finite. The maximum displayed excess MSE is `14.120345%`.
- Every one of the 40 dataset--region columns has exactly one outlined lowest-MSE scope. Winner counts are $s=1$: `1`, $s=48$: `3`, $s=144$: `2`, $s=360$: `17`, and $s=720$: `17`.
- Panels b and c each contain all five evaluated datasets plus the macro aggregate. All five dataset-level MSE directions are positive for both controls.
- Panel-b gains are `1.4409%`, `2.3166%`, `1.4528%`, `1.2698%` and `1.3825%`; the current author-corrected macro gate is `1.6129%`.
- Panel-c gains are `3.1161%`, `1.9380%`, `6.2212%`, `2.5078%` and `1.8349%`; the current author-corrected macro gate is `3.4810%`.
- Panels b/c use the current author-corrected aggregate table. The corrected per-horizon checkpoint provenance is not synchronized, so this draft must not be promoted as final auditable paper evidence until that boundary is accepted or resolved.

## Statistical contract

- Panel a split: validation.
- Panel a metric: MSE averaged over each future region, then averaged over all dataset-specific origin--variable rows; excess is measured relative to the lowest mean MSE among the five scopes in the same dataset and region.
- Panels b/c split: official-test aggregate supplied by the author-corrected rerun record.
- Panels b/c aggregation: four-horizon dataset mean; macro value from the current five-dataset aggregate gate.
- Seeds: one recorded seed for the ablation aggregate; no confidence interval or error bar is available.
- Tests and multiple-comparison correction: none; the figure reports descriptive effect sizes.

## Claim boundary

- Supported: scope-conditioned forecasts show region-dependent relative error within the frozen validation diagnostic.
- Supported: removing Target-Adaptive Allocation or BSCA worsens the current aggregate MSE results on all five displayed datasets.
- Not supported by this figure alone: reliable selection of the region-best scope, sparse or semantic specialization, universal scope preferences, or a unique causal explanation for the ablation gains.
- The near-uniform probability and `8/40` allocation-to-lowest-error alignment diagnostics are preserved in the earlier evidence bundle. Their omission from this author-review layout is a presentation decision, not a contradictory-evidence deletion.

## Visual and export QA

- Backend: Python-only `matplotlib` generation and preview.
- Canvas: `180 × 120 mm` double-column layout.
- Hierarchy: Panel a is the hero heatmap; Panels b/c are aligned subordinate effect-size plots with a shared x-range.
- Palette: blue-cyan sequential scale for regional error, violet for Target-Adaptive Allocation, orange for BSCA and regional-minimum outlines; no rainbow mapping.
- Panel labels, titles, region labels, dataset labels, point annotations and colorbar were inspected at final size; no overlaps or clipping were observed.
- Static preflight: `14 pass, 0 warn, 0 fail`.
- SVG contains editable text and LF line endings with no trailing whitespace.
- PDF: one page at `509.76 × 339.84 pt` (180 × 120 mm).
- PNG: `2124 × 1416 px` at 300 dpi.
- TIFF: `4248 × 2832 px` at 600 dpi with LZW compression.

## Delivered source data

- `source_data/panel_a_scope_competence.csv`: 200 rows.
- `source_data/panel_b_allocation_gain.csv`: 6 rows.
- `source_data/panel_c_bsca_gain.csv`: 6 rows.
