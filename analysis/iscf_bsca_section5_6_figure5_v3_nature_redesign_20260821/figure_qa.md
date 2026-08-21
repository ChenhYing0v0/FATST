# Figure 5 v3 QA and evidence boundary

## Decision

- Status: `panel_a_main_text_candidate_panel_bc_redundant_with_ablation`.
- Canonical paper asset: `no`; author approval remains required before replacing the current `paper-figures/` asset or drafting visible Section 5.6 prose.
- Reuse level: `build anew`. The v2 evidence and source values are preserved, but its dense heatmap layout is not reused.
- New method implementation, training and formal test: `0/0/0`.

## Data integrity

- Panel a retains the complete $5\times8\times5=200$ dataset--region--scope validation aggregates and deterministically summarizes them into 40 dataset--region cells.
- The displayed scope is the unique minimum-mean-MSE scope in each cell. Marker area uses the maximum excess MSE above that cell minimum; no dataset, region or scope is filtered.
- Preferred-scope counts are $s=1/48/144/360/720\rightarrow1/3/2/17/17$. Best-to-worst excess MSE ranges from 0.391% to 14.120%.
- Panel b retains all five dataset-level comparisons and their macro average. Full-model MSE reductions are 1.441%, 2.317%, 1.453%, 1.270% and 1.382%; macro=1.613%.
- Panel c retains all five dataset-level comparisons and their macro average. Full-model MSE reductions are 3.116%, 1.938%, 6.221%, 2.508% and 1.835%; macro=3.481%.
- Panels b/c reuse the current author-corrected aggregate ablation values. Their per-horizon raw scorecards and checkpoint hashes remain unsynchronized, so the figure supports aggregate component utility rather than a newly reconstructed checkpoint-level audit.

## Claim boundary

- Supported: preferred sharing scope varies across the displayed future regions and datasets; Target-Adaptive Allocation and BSCA improve aggregate MSE relative to their corresponding controls.
- Not supported: learned Scope Probabilities reliably identify the lowest-error scope, sparse expert specialization, an oracle allocation interpretation, or a unique causal mechanism.
- The earlier near-uniform probability profile and 8/40 utilization--error alignment remain preserved in internal records. Their omission from this author-review composition is a presentation decision, not an evidence reversal.

## Visual and export QA

- Final canvas: 180 × 110 mm, asymmetric three-panel layout.
- Panel a uses number, colour and marker area as redundant encodings; colour is not the sole carrier of scope identity.
- Panels b/c share the same 0--7% axis and show every dataset plus the macro average.
- White background, sentence-case titles, restrained low-saturation palette, minimal spines and whitespace-based grouping were checked at final size.
- Static preflight: 14 pass, 0 warn, 0 fail.
- Editable outputs: SVG and PDF. Raster outputs: 2124 × 1299 px PNG at 300 dpi and 4248 × 2598 px LZW TIFF at 600 dpi.
- SVG contains editable text, LF line endings and no trailing whitespace.

## Cross-figure redundancy audit

- Section 3 Figure 3 and current Panel a use different predictor families, scope sets and aggregation levels.
- Their main-text conclusion is nevertheless the same: the lowest-error sharing extent changes across future regions. Both also visualize regional winners and error separation.
- The overlap is therefore low at the raw-experiment level but high at the narrative and visual-task levels.
- Current Panel a remains a main-text candidate only when framed as an aggregate diagnostic of the jointly trained ISCF scope field. Its current generic title must be revised to distinguish this role from Section 3 Figure 3.
- A CHPC visualization is not recommended because zero CHPD follows from the inference graph and would serve only as implementation verification.
- Panels b/c pass visual QA, and additional upper y-axis padding separates their comparison subtitles from the first dataset row. They nevertheless fail the Section 5.6 evidence-role audit because they repeat two controls already reported in Table 3.

## Section 5.5 versus 5.6 audit

- Section 5.5 uses matched interventions and official-test MSE/MAE to establish component utility.
- Section 5.6 uses the unchanged Full model and aggregate validation diagnostics to characterize internal scope competence and allocation behavior.
- Repeating Full-versus-control effect sizes in Section 5.6 would answer the Section 5.5 question a second time.
- Recommended minimal Figure 5 is therefore the full-width scope-competence map. Probability range, normalized entropy and 8/40 alignment may be reported in the prose without an additional panel. If allocation is visualized later, the panel must encode learned probabilities or their relation to scope competence.
