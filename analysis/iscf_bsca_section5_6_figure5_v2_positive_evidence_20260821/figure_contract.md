# Figure 5 v2 positive-evidence contract

## Status

- Role: `author_review_positive_evidence_draft`.
- Canonical paper asset: `no`.
- Replaces the rejected Figure 5 only after author review and a final claim-boundary audit.
- New training, remote execution and formal test: `0`.

## Five-point contract

- Core conclusion: Scope-conditioned forecasts exhibit region-dependent relative competence, while Target-Adaptive Allocation and BSCA improve aggregate forecasting accuracy in the corresponding controlled ablations.
- Figure archetype: asymmetric quantitative grid with one hero heatmap and two compact matched-comparison panels.
- Target output: two-column journal figure, 180 mm wide, editable SVG/PDF plus 300-dpi PNG and 600-dpi TIFF.
- Backend: Python (`matplotlib`) only.
- Evidence hierarchy: Panel a is an aggregate validation diagnostic computed from every available validation row; Panels b and c use the current author-corrected official-test ablation aggregates.

## Panel map

- Panel a — `Region-wise scope competence`: for every dataset, future region and sharing scope, report the percentage excess MSE above the lowest-MSE scope in that dataset-region; orange outlines identify the regional minimum. This panel uses all five datasets, all eight preregistered future regions and all five scopes.
- Panel b — `Target-Adaptive Allocation`: report the four-horizon mean MSE reduction of Full ISCF-BSCA relative to `w/o Target-Adaptive Allocation` for every dataset and the current macro aggregate.
- Panel c — `Balanced Scope Co-Adaptation`: report the four-horizon mean MSE reduction of Full ISCF-BSCA relative to `w/o BSCA` for every dataset and the current macro aggregate.

## Evidence and claim boundaries

- Panel a supports region-dependent scope error heterogeneity within the frozen ISCF-BSCA validation diagnostic. It does not show that learned Scope Probabilities reliably select the lowest-error scope.
- Panels b and c support aggregate component utility within the evaluated ablation protocol. They do not by themselves establish a unique causal mechanism.
- The author-corrected allocation and BSCA aggregates are the current paper-facing values, but their corrected aggregate provenance must remain disclosed in the internal QA record until checkpoint-level synchronization is available.
- Existing near-uniform probability and allocation-alignment diagnostics are preserved in the rejected Figure 5 evidence bundle. They are omitted from this positive-evidence author-review layout, not invalidated or deleted.

## Reviewer-risk checks

- No dataset, region, scope or requested metric is selected out of Panel a.
- Panels b and c show all five evaluated datasets; no negative dataset-level cell is hidden.
- No uncertainty bars are shown because the frozen author-corrected table provides one seed-level four-horizon aggregate per dataset.
- The figure must not be described as evidence of reliable region-best routing, sparse specialization or universal scope semantics.

## Source data

- Panel a: frozen Full ISCF-BSCA validation diagnostic arrays `arm_row_bin_mse`.
- Panels b/c: current author-corrected dataset means and aggregate gates from the controlled ablation package.
