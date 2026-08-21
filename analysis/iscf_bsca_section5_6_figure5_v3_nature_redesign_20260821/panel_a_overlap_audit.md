# Panel a overlap audit against Section 3 Figure 3

## Decision

Current Panel a should not be promoted to the main text in its present role. It is not a duplicate experiment, but it substantially repeats the reader-facing conclusion and visual task of Section 3 Figure 3. Panels b/c remain suitable after the subtitle-spacing refinement.

## Evidence comparison

| Dimension | Section 3 Figure 3 | Current Figure 5 Panel a |
| --- | --- | --- |
| Manuscript role | Establish the problem and motivate adaptive output-side sharing | Diagnose the trained ISCF-BSCA scope field |
| Predictor family | Capacity-matched single-extent predictors | Scope-conditioned arms within Full ISCF-BSCA |
| Scope set | $\{1,8,32,128,720\}$ | $\{1,48,144,360,720\}$ |
| Data aggregation | One validation-selected ETTm2 origin | All validation origin--variable rows in five datasets |
| Future partition | 12 contiguous 60-step regions | Eight preregistered regions |
| Displayed statistic | Excess MSE above the regional minimum and oracle headroom over a fixed extent | Lowest-MSE scope and best-to-worst excess MSE gap |
| Reader-facing conclusion | Preferred sharing extent varies across future regions | Preferred sharing scope varies across future regions |

The source model, scope grid and aggregation level differ. These distinctions make Panel a useful as an expanded diagnostic, but they do not create a sufficiently different main-text message. Both figures ask the reader to inspect which scope wins in each future region and how strongly alternatives underperform.

## Reviewer risk

1. The repeated conclusion can be read as using a second figure to re-establish motivation rather than validate the proposed method.
2. The preferred-scope map does not display learned Scope Probabilities or their relation to the fused forecast.
3. Because utilization agrees with the lowest-error scope in only 8/40 dataset--region cells, a winner-only main-text panel can invite an unsupported routing-success interpretation.

## Recommended routing

- Retain the current preferred-scope map as an Appendix diagnostic showing that the multi-scope field preserves heterogeneous regional competence across five datasets.
- Replace main-text Panel a with numerical CHPC fulfillment: one nested-prefix forecast trajectory, accompanied by an aggregate verification summary reporting 20/20 comparisons with maximum absolute CHPD equal to zero across the five evaluated datasets and four paper horizons.
- Keep Panels b/c as the accuracy-level controls for Target-Adaptive Allocation and BSCA.

This replacement would create the progression `Section 3 problem evidence -> Section 4 architectural solution -> Section 5 contract verification and component utility` without promoting allocation specialization beyond the available evidence.
