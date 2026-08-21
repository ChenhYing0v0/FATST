# Panel a overlap audit against Section 3 Figure 3

## Decision

The initial recommendation to replace Panel a with CHPC fulfillment evidence is withdrawn after a narrative-first reassessment. CHPC is guaranteed by the inference graph and does not warrant a main-text result panel. Current Panel a remains a defensible main-text candidate if it is explicitly framed as an aggregate diagnostic of the jointly trained ISCF scope field rather than another demonstration of the Section 3 problem.

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

- Retain Panel a in the main-text candidate, but rename it around `region-wise competence within the jointly trained scope field` and identify Full ISCF-BSCA plus all-validation aggregation in the subtitle and caption.
- Use the body text to distinguish Section 3 Figure 3's problem-level, selected single-origin control from Figure 5a's method-internal, five-dataset aggregate analysis.
- Keep Panels b/c as the accuracy-level controls for Target-Adaptive Allocation and BSCA.
- Do not call the lowest-error scope an allocation decision. It is a diagnostic property of the scope-conditioned forecast field.
- Do not add a CHPC visualization; retain numerical CHPD checks as implementation QA or Appendix material if needed.

This framing creates the progression `Section 3 problem evidence -> Section 4 architectural response -> Section 5 method-internal scope behavior and component utility` without promoting allocation specialization beyond the available evidence.
