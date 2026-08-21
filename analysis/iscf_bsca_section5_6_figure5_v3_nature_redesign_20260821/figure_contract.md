# Figure 5 v3 Nature-oriented redesign contract

## Status

- Role: `author_review_nature_redesign`.
- Canonical paper asset: `no`.
- Reuse level: `build anew`; v2 supplies the evidence contract and source data but not the visual structure.
- New training, remote execution and formal test: `0`.

## Core conclusion and archetype

- Core conclusion: The sharing scope with the lowest regional validation error varies across datasets and future regions, while Target-Adaptive Allocation and BSCA both improve aggregate forecasting accuracy in the corresponding controlled ablations.
- Archetype: asymmetric quantitative composite with one dominant evidence panel and two subordinate effect-size panels.
- Backend: Python (`matplotlib`) exclusively.
- Final size: 180 × 110 mm; editable SVG/PDF, 300-dpi PNG and 600-dpi LZW TIFF.

## Panel map

- Panel a — preferred-scope map: one marker for every dataset--future-region cell. The number and restrained scope-family colour identify the lowest-MSE scope; marker area encodes the best-to-worst regional MSE gap. This compresses the complete 5 × 8 × 5 error surface into 40 non-selected summaries without changing the statistic.
- Panel b — Target-Adaptive Allocation: full-model MSE reduction relative to equal scope fusion for all five datasets and the current macro aggregate.
- Panel c — BSCA: full-model MSE reduction relative to prefix-only training for all five datasets and the current macro aggregate.

## Visual strategy

- Replace the dense 25 × 8 heatmap and repeated orange rectangles with a 5 × 8 preference map.
- Use scope numbers as an explicit print-safe encoding; colour is supportive rather than the only category channel.
- Use marker area for the error-separation magnitude and provide a three-point size key.
- Place Panels b/c on one aligned lower row with identical x limits and bullet-plot tracks.
- Use sentence-case titles, small panel labels, restrained low-saturation colours, minimal spines and whitespace-based grouping.

## Evidence and reviewer boundaries

- Panel a uses all five datasets, eight future regions and five scopes. No dataset, region or scope is filtered.
- The preferred scope is the minimum of the dataset-level mean regional MSE; the gap is the maximum excess MSE above that minimum within the same dataset--region cell.
- Panels b/c use the current author-corrected ablation aggregates and contain every evaluated dataset.
- The figure does not establish that learned Scope Probabilities select the preferred scope, nor does it establish sparse specialization or a unique causal mechanism.
- Earlier internal allocation-health diagnostics remain preserved outside this positive-evidence author-review layout.
