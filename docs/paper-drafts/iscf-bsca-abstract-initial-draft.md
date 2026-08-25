# ISCF-BSCA Abstract and Keywords

## Draft status

| Field | Content |
| --- | --- |
| `document_role` | Clean manuscript-facing initial draft of the Abstract and Keywords |
| `version` | `v0.1-pending-author-review` |
| `date` | `2026-08-25` |
| `review_status` | `pending_author_review` |
| `upstream_dependency` | Temporarily frozen Sections 1--7 and Appendices A--C |
| `evidence_scope` | CHPC construction; completed experiments in Sections 5.2--5.7 |
| `experiment_change` | None; no new implementation, training or formal test |
| `claim_boundary` | Validation-only motivation and selected internal diagnostics are not promoted to population-level or final-effectiveness claims |

The status table above is editorial metadata and is not part of the manuscript body submitted for review.

## Abstract

Multi-horizon forecasting is typically implemented with one predictor per horizon, increasing system cost while leaving shared future steps unconstrained. We formulate varied-horizon forecasting as one horizon-agnostic trajectory whose nested prefixes serve different endpoints, making cross-horizon prefix consistency (CHPC) a structural requirement. Because future regions may prefer different extents of latent-state sharing, we propose Independent Scope-Conditioned Forecasting with Balanced Scope Co-Adaptation (ISCF-BSCA). ISCF generates region-wise forecasts at multiple sharing scopes and integrates them using target-conditioned probabilities; BSCA balances joint optimization across scopes. The model satisfies CHPC by construction and serves all supported horizons with one model. Across seven multivariate benchmarks and four horizons, ISCF-BSCA achieves the best result in 13 of 14 dataset--metric comparisons with horizon-specific baselines. Relative to TimeAlign, it reduces average MSE and MAE by 4.94% and 2.54%, checkpoint storage by 81.48%, and peak inference memory by 64.42%. Under a one-model-all-horizons protocol, it ranks first in all 14 comparisons and reduces average MSE and MAE over TimeAlign by 6.45% and 3.72%. Ablations support the evaluated components, while DLinear and PatchTST replacements support compatibility across two Encoder families. These results establish output-side sharing granularity as an effective basis for unified, prefix-consistent forecasting.

## Keywords

Time series forecasting; varied-horizon forecasting; unified forecasting; cross-horizon prefix consistency; multi-scope forecasting.

## Claim--evidence map

| Abstract claim | Evidence | Status |
| --- | --- | --- |
| Horizon-specific predictors can disagree on shared future steps | CHPC formulation and ETTh2 DLinear diagnostic in Section 3.2 | Supported as an architectural/protocol possibility and audited diagnostic; not used as an accuracy claim |
| ISCF-BSCA satisfies CHPC by construction | Future-step-indexed prediction and prefix-restricted inference in Sections 3.1 and 4.4 | Supported by architecture |
| Best in 13 of 14 horizon-specific dataset--metric comparisons | Table 1 over seven datasets and two metrics | Supported |
| 4.94% MSE and 2.54% MAE improvement over TimeAlign | Section 5.2 seven-dataset averages | Supported |
| 81.48% checkpoint-storage and 64.42% peak-memory reductions over TimeAlign | Figure 5 and Section 5.4 | Supported within the reported service-cost protocol |
| First in all 14 one-model-all-horizons comparisons, with 6.45%/3.72% gains | Table 2 and Section 5.3 | Supported |
| Component contribution and compatibility with two Encoder families | Table 3 and Figure 7 | Supported within the evaluated ablation family and DLinear/PatchTST replacement study |
