# Section 5.6 narrative-first reassessment

## Decision

The previous recommendation to replace Panel a with a numerical CHPC visualization is withdrawn. CHPC follows from the future-step-indexed inference graph and is therefore an architectural property rather than a result that requires a main-text figure. A zero-CHPD plot would function only as implementation verification.

Current Panel a remains a useful main-text candidate if its role is reframed from generic sharing-demand heterogeneity to **region-wise competence within the jointly trained ISCF scope field**. The visual should support the paragraph argument below rather than restate the Section 3 motivation.

## Recommended subsection title

`5.6 Regional scope behavior and adaptive integration`

This title removes an unnecessary promise of empirical CHPC analysis and focuses the subsection on the behavior and utility that require evidence.

## Paragraph contract

### Paragraph 1 — analysis question

CHPC is acknowledged as a construction fact and is not promoted as empirical evidence. The paragraph moves directly to the remaining question: whether jointly trained ISCF scope-conditioned forecasts preserve heterogeneous regional competence and whether the mechanisms that integrate and optimize this field improve accuracy.

### Paragraph 2 — scope-field competence

Define the dataset--region mean MSE of each Scope-conditioned Forecast using all aligned validation origin--variable rows. Report the lowest-error scope and the best-to-worst excess MSE for each cell. Figure 5a supports the observation that no scope uniformly dominates all datasets and future regions after joint training. Its role differs from Section 3 Figure 3: Figure 3 establishes the problem with separately trained capacity-matched single-extent predictors and one selected ETTm2 origin, whereas Figure 5a diagnoses the internal scope field of Full ISCF-BSCA over five complete validation sets.

### Paragraph 3 — adaptive integration

State explicitly that the lowest-error scope is a diagnostic label, not a decision made by the Allocation MLP. Use Figure 5b to show that learned Target-Adaptive Allocation improves aggregate MSE relative to equal fusion on all five datasets, with a macro reduction of 1.613%. Retain the internal-health boundary: Scope Probabilities remain distributed, and the highest-weight scope agrees with the lowest-error arm in 8/40 dataset--region cells. The permitted conclusion is useful target-conditioned soft integration, not hard selection or oracle recovery.

### Paragraph 4 — balanced co-adaptation and synthesis

Use Figure 5c to show that Full ISCF-BSCA improves aggregate MSE relative to prefix-only training on all five datasets, with a macro reduction of 3.481%. Conclude that the jointly trained scope field carries region-dependent competence, while Target-Adaptive Allocation and BSCA improve how that field is integrated and optimized. Do not claim sparse specialization or a unique causal explanation.

## Proposed manuscript-facing draft

### 5.6 Regional scope behavior and adaptive integration

CHPC follows directly from the future-step-indexed inference graph: every horizon request returns a prefix of the same predicted trajectory. We therefore do not treat zero cross-horizon disagreement as a separate empirical contribution. Instead, we examine whether the jointly trained ISCF forecast field retains the future-region-dependent scope behavior that motivated the decoder, and whether adaptive integration and BSCA improve the resulting forecasts.

For each dataset, future region and sharing scope, we average the validation MSE of the corresponding Scope-conditioned Forecast over all aligned forecast origins and variables. Figure 5a identifies the lowest-error scope in each dataset--region cell and uses marker area to encode the error separation between the best and worst scopes. The lowest-error scope changes across both datasets and future regions: scopes $s=1$, $48$, $144$, $360$ and $720$ attain the lowest error in 1, 3, 2, 17 and 17 of the 40 cells, respectively, while the best-to-worst excess MSE ranges from 0.39% to 14.12%. Unlike the selected single-origin diagnostic in Figure 3, this analysis uses the scope-conditioned forecasts jointly produced by Full ISCF-BSCA over all available validation origins. The result indicates that joint training does not reduce the scope-indexed forecast field to one uniformly dominant sharing extent.

These region-wise winners characterize the error surface of the Scope-conditioned Forecasts rather than hard decisions made by the Allocation MLP. We therefore evaluate Target-Adaptive Allocation through its matched predictive control. As shown in Figure 5b, replacing learned Scope Probabilities with equal fusion degrades MSE on all five datasets, with a mean reduction of 1.61% for the full model. The learned probabilities remain distributed, and their highest-weight scope coincides with the lowest-error scope in 8 of the 40 dataset--region cells. We consequently interpret the ablation gain as evidence that target-conditioned soft integration is useful, rather than that the allocator recovers a region-wise oracle scope.

Figure 5c further shows that Full ISCF-BSCA outperforms prefix-only training on every dataset, reducing mean MSE by 3.48%. Direct scope supervision and allocation balancing therefore improve the joint optimization of the scope field and its integration. Together, these analyses show that ISCF constructs scope-conditioned forecasts with region-dependent relative competence, while Target-Adaptive Allocation and BSCA improve how these forecasts are combined and trained.

## Figure consequence

- Keep Panel a, but change its title to `Region-wise competence within the jointly trained scope field`.
- Change the subtitle to `Full ISCF-BSCA; number = lowest-MSE scope; area = best-to-worst MSE gap`.
- Keep Panels b/c and their current effect-size design.
- Do not add a CHPC panel.
- Preserve the current preferred-scope map as a main-text candidate only under the method-internal aggregate framing above.
