# Section 5.6 narrative-first reassessment

## Decision

The previous recommendation to replace Panel a with a numerical CHPC visualization is withdrawn. CHPC follows from the future-step-indexed inference graph and is therefore an architectural property rather than a result that requires a main-text figure. A zero-CHPD plot would function only as implementation verification.

Current Panel a remains a useful main-text candidate if its role is reframed from generic sharing-demand heterogeneity to **region-wise competence within the jointly trained ISCF scope field**. The current Panels b/c repeat two controls already reported in Section 5.5 and should not be used to make the same component-utility argument again in Section 5.6.

## Recommended subsection title

`5.6 Regional scope competence and allocation behavior`

This title removes an unnecessary promise of empirical CHPC analysis and separates internal behavior from the component-utility question answered by Section 5.5.

## Boundary with Section 5.5

| Section | Scientific question | Evidence operation | Evidence layer | Permitted conclusion |
| --- | --- | --- | --- | --- |
| 5.5 Component and training-objective ablations | Does removing each component reduce forecasting accuracy? | Retrain matched model variants and compare official-test MSE/MAE | Matched component/objective ablation | The evaluated component contributes to aggregate accuracy within the tested design family |
| 5.6 Regional scope competence and allocation behavior | What internal structure does Full ISCF-BSCA learn, and how does its allocator use that structure? | Inspect scope-conditioned errors and Scope Probabilities without changing the trained Full model | Aggregate validation diagnostic | Scope forecasts retain region-dependent competence and allocation remains distributed rather than collapsing to one scope |

Section 5.5 establishes **utility** through intervention. Section 5.6 characterizes **behavior** through observation. An ablation gain does not by itself explain the internal allocation mechanism, while an active or varying internal signal does not by itself establish accuracy utility.

## Paragraph contract

### Paragraph 1 — analysis question

CHPC is acknowledged as a construction fact and is not promoted as empirical evidence. The paragraph moves directly to the remaining question: whether jointly trained ISCF scope-conditioned forecasts preserve heterogeneous regional competence and how the learned Scope Probabilities use this field.

### Paragraph 2 — scope-field competence

Define the dataset--region mean MSE of each Scope-conditioned Forecast using all aligned validation origin--variable rows. Report the lowest-error scope and the best-to-worst excess MSE for each cell. Figure 5a supports the observation that no scope uniformly dominates all datasets and future regions after joint training. Its role differs from Section 3 Figure 3: Figure 3 establishes the problem with separately trained capacity-matched single-extent predictors and one selected ETTm2 origin, whereas Figure 5a diagnoses the internal scope field of Full ISCF-BSCA over five complete validation sets.

### Paragraph 3 — allocation behavior

State explicitly that the lowest-error scope is a diagnostic label, not a decision made by the Allocation MLP. Aggregate the learned Scope Probabilities over the same validation origin--variable rows. Their dataset--region means range from 0.18258 to 0.21479, with normalized entropy above 0.9989, so all scopes remain active and the allocation is close to uniform rather than winner-take-all. The highest-weight scope changes across regions in four of the five datasets but agrees with the lowest-error arm in only 8/40 dataset--region cells. The permitted conclusion is distributed target-conditioned reweighting, not hard selection or oracle recovery.

### Paragraph 4 — relation to the ablation evidence

Connect the behavioral evidence back to Section 5.5 without repeating its numerical comparisons. Table 3 establishes that replacing learned allocation with equal fusion and removing BSCA both reduce aggregate accuracy. Section 5.6 adds the narrower mechanism interpretation: the gain does not arise from a hard router that recovers the region-wise lowest-error scope; it is associated with small, target-dependent deviations within a non-collapsed multi-scope mixture. Current diagnostics do not separately attribute the observed probability profile to BSCA, so the subsection should not claim that the balance regularizer causes a particular routing pattern.

## Proposed manuscript-facing draft

### 5.6 Regional scope competence and allocation behavior

CHPC follows directly from the future-step-indexed inference graph: every horizon request returns a prefix of the same predicted trajectory. We therefore do not treat zero cross-horizon disagreement as a separate empirical contribution. Instead, we examine whether the jointly trained ISCF forecast field retains the future-region-dependent scope behavior that motivated the decoder and how the learned allocation uses this field.

For each dataset, future region and sharing scope, we average the validation MSE of the corresponding Scope-conditioned Forecast over all aligned forecast origins and variables. Figure 5a identifies the lowest-error scope in each dataset--region cell and uses marker area to encode the error separation between the best and worst scopes. The lowest-error scope changes across both datasets and future regions: scopes $s=1$, $48$, $144$, $360$ and $720$ attain the lowest error in 1, 3, 2, 17 and 17 of the 40 cells, respectively, while the best-to-worst excess MSE ranges from 0.39% to 14.12%. Unlike the selected single-origin diagnostic in Figure 3, this analysis uses the scope-conditioned forecasts jointly produced by Full ISCF-BSCA over all available validation origins. The result indicates that joint training does not reduce the scope-indexed forecast field to one uniformly dominant sharing extent.

These region-wise winners characterize the error surface of the Scope-conditioned Forecasts rather than hard decisions made by the Allocation MLP. We therefore inspect the learned Scope Probabilities over the same validation regions. Their dataset--region means range from 0.18258 to 0.21479, and the normalized allocation entropy remains above 0.9989. All five scopes consequently remain active, with the allocator applying relatively small deviations from equal fusion rather than selecting one dominant scope. Although the highest-weight scope changes across future regions in four of the five datasets, it coincides with the lowest-error scope in only 8 of the 40 dataset--region cells. The observed allocation should therefore be interpreted as distributed target-conditioned reweighting, not as recovery of a region-wise oracle scope.

This behavioral analysis complements rather than repeats the ablation study. Table 3 shows that learned allocation improves aggregate forecasting accuracy over equal fusion and that BSCA improves upon prefix-only training. The present analysis explains the internal regime in which those gains are obtained: the scope-conditioned forecasts retain region-dependent relative competence, while the allocator combines them through a non-collapsed soft mixture rather than hard regional routing. Because the highest-weight and lowest-error scopes are only partially aligned, the results support adaptive integration but not oracle selection or semantic scope specialization.

## Figure consequence

- Keep Panel a, but change its title to `Region-wise competence within the jointly trained scope field`.
- Change the subtitle to `Full ISCF-BSCA; number = lowest-MSE scope; area = best-to-worst MSE gap`.
- Do not use the current Panels b/c as Section 5.6 evidence: they replot two comparisons already reported in Table 3.
- Preferred minimal main-text design: retain one full-width Panel a and report the probability-range, entropy and 8/40 alignment diagnostics in the prose and caption without creating additional panels.
- If a visual allocation panel is later judged necessary, it must show the learned probability distribution or allocation--competence relation rather than another ablation effect size.
- Do not add a CHPC panel.
- Preserve the current preferred-scope map as a main-text candidate only under the method-internal aggregate framing above.
