# Section 5.6 / Figure 5 v5.1: sample-specific scope and allocation behavior

## 1. Argument before visualization

Section 5.6 analyzes the internal behavior of the unchanged Full ISCF-BSCA model and remains separate from the matched component-utility tests in Section 5.5. It addresses three bounded questions:

1. Does the Target-Adaptive Allocation Path assign the same dominant scope to every future region of a sample?
2. Are the Scope-conditioned Forecasts duplicated trajectories, or do they retain distinct prediction signals?
3. Does the implementation satisfy CHPC numerically, in addition to the architectural guarantee?

CHPC is reported in prose because it follows from the inference graph and all 20 numerical checks have maximum absolute CHPD equal to zero. A zero-valued visualization would not add evidence. Figure 5 is therefore reserved for the two sample-dependent behaviors that require empirical inspection.

## 2. Figure contract

- **Core conclusion**: one disclosed validation example illustrates that the five Scope-conditioned Forecasts remain non-identical and that the dominant component of the soft allocation changes across future regions.
- **Archetype**: `asymmetric mixed-modality quantitative figure`.
- **Backend**: Python / matplotlib only.
- **Final size**: 183 mm × 142 mm.
- **Panel a**: the actual fused forecast. Segment colour and the lower strip identify the scope with the highest Scope Probability after averaging within each displayed future region. The curve is not assembled by hard-selecting scope forecasts.
- **Panel b**: five aligned small multiples of $\mathcal F_{\tau}^{(s)}-\widehat y_{\tau}$ on one shared scale. The residual view makes non-identical scope signals visible without overplotting five similar raw trajectories.
- **Panel c**: the complete $5\times720$ matrix of actual per-step Scope Probabilities for the same probe. Region boundaries are shown without aggregation or smoothing.

## 3. Selection rule and evidence

All three panels use ETTm1 validation probe 113. The probe was selected from the complete pool of 5 datasets × 256 sequential probes = 1,280 candidates using one deterministic lexicographic rule:

1. maximize the number of distinct region-dominant scopes, where the dominant scope is the argmax of the mean soft-allocation probability within each future region;
2. break ties by maximizing mean pairwise absolute disagreement among the five Scope-conditioned Forecasts.

The selected probe has the regional dominant-scope sequence $[1,48,720,144,360,144,48,720]$, covering all five scopes. Its mean pairwise absolute scope-forecast disagreement is 0.081196. The per-step Scope Probabilities range from 0.188866 to 0.212594, and their regional means range from 0.194320 to 0.205943.

The narrow probability range is retained in the source data and visible on the Panel c colour bar. The figure supports region-dependent soft reweighting in this example, not sparse routing or strong expert separation.

## 4. Evidence and claim boundaries

| Evidence | Supports | Does not support |
| --- | --- | --- |
| Panel a | the dominant component of the soft allocation changes across the eight regions of the selected probe | hard routing, oracle selection, typicality or prevalence |
| Panel b | the five Scope-conditioned Forecasts are not duplicated trajectories in the selected probe | causal specialization, component utility or population-level diversity |
| Panel c | per-step probability profiles vary within and across future regions for the selected probe | reliable identification of the lowest-error scope or sparse allocation |
| CHPC audit in prose | the frozen implementation matches the construction property in 20/20 dataset--horizon checks | improved forecast accuracy |

The historical allocation--competence alignment audit and near-uniform aggregate probability behavior remain in the canonical internal evidence records. They are not used as positive main-text evidence and are not deleted or overwritten by this selected-example figure.

## 5. Source-data integrity

- source artifacts: five frozen Full ISCF-BSCA validation diagnostic objects;
- datasets in selection pool: ETTm1, ETTm2, ETTh1, ETTh2 and Weather;
- scopes: $\{1,48,144,360,720\}$;
- future regions: 8, with boundaries $[0,48,96,144,192,288,336,512,720]$;
- probes inspected: 1,280 / 1,280;
- selected steps displayed: 720 / 720;
- missing-value filtering: 0;
- smoothing, interpolation, clipping or manual replacement: 0;
- new implementation, remote training or formal test: 0 / 0 / 0.

Generated source data:

- `source_data/selected_fused_and_scope_forecasts.csv`;
- `source_data/selected_scope_probabilities.csv`;
- `source_data/selected_regional_allocation.csv`;
- `source_data/sample_selection_audit.csv`;
- `figure_summary.json`.

## 6. Visual and export QA

- Nature Figure static preflight: 14 PASS, 0 WARN, 0 FAIL;
- output family: editable SVG/PDF, 300-dpi PNG and 600-dpi LZW TIFF;
- PDF page size: 518.4 × 402.48 pt, corresponding to 183 × 142 mm;
- palette: the five scopes retain the low-saturation blue--violet identity used in the method figure; the probability heatmap uses one sequential violet scale because colour encodes a scalar probability;
- data transparency: the probability colour bar shows the plotted range explicitly rather than implying a 0--1 spread;
- Panel c contrast refinement: the sequential violet palette now spans a larger lightness range, and the colour limits are rounded outward to `0.188--0.213` from the unchanged observed range `0.188866--0.212594`; no probability transformation, smoothing or clipping is applied;
- layout: Panel a is the hero panel; Panel b answers forecast diversity; Panel c answers probability variation;
- reviewer risk: the caption explicitly distinguishes the actual soft-fused output from hard scope selection and discloses the post-hoc validation-example selection rule.

## 7. Decision

Decision=`section5_6_v5p1_probability_contrast_author_review_candidate`.

Figure 5 v5.1 replaces v5 as the current author-review candidate. The probability data, panel semantics, sample selection and manuscript claim remain unchanged; only the Panel c visual mapping is revised. The v4 bundle remains a historical audit artifact. Sections 1--4 and all frozen experiment results are unchanged.
