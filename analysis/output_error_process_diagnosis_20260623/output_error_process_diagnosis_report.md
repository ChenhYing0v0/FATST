# Output/Error-Process Decoder Problem Diagnosis

## Purpose

[Fact] This report uses completed H720 step-wise artifacts to decide whether a future fallback direction should target the decoder output/error process rather than latent future-state alignment.

## Main Finding

[Strong Evidence] The current target-set/future-state line does not only have an average-MSE problem; it has a step-region error-process problem. R.3 often improves early steps but can degrade middle or late H720 regions, and Phase2-A changes this pattern in a dataset-dependent way.

## Segment-Level H720 Evidence

| Dataset | Segment | R.3 vs FixedHead | Phase2-A vs R.3 | Interpretation |
| --- | --- | ---: | ---: | --- |
| ETTh2 | 1-96 | -1.80% | +6.42% | R.3 helps, Phase2-A erases gain |
| ETTh2 | 97-192 | -0.89% | +5.41% | R.3 helps, Phase2-A erases gain |
| ETTh2 | 193-336 | +4.07% | +2.54% | both stages worsen this region |
| ETTh2 | 337-720 | +0.47% | +3.29% | both stages worsen this region |
| ETTm1 | 1-96 | -2.72% | -0.72% | both stages help this region |
| ETTm1 | 97-192 | -1.76% | -0.62% | both stages help this region |
| ETTm1 | 193-336 | -1.20% | -1.50% | both stages help this region |
| ETTm1 | 337-720 | +3.03% | -1.73% | R.3 weak region, Phase2-A repairs |
| Weather | 1-96 | -4.04% | +0.52% | R.3 helps, Phase2-A erases gain |
| Weather | 97-192 | -1.91% | -0.50% | both stages help this region |
| Weather | 193-336 | -1.16% | -0.38% | both stages help this region |
| Weather | 337-720 | -0.05% | -0.30% | both stages help this region |

## Curve Summary

| Dataset | Comparison | Mean | Early 1-96 | Late 337-720 | Improved steps | Degraded steps | Sign changes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | r3_vs_fixed | +0.85% | -2.69% | +0.95% | 318 | 402 | 153 |
| ETTh2 | phase2_vs_r3 | +3.67% | +5.56% | +3.16% | 63 | 657 | 85 |
| ETTm1 | r3_vs_fixed | +0.67% | -3.35% | +2.99% | 313 | 407 | 85 |
| ETTm1 | phase2_vs_r3 | -1.36% | -0.60% | -1.70% | 579 | 141 | 161 |
| Weather | r3_vs_fixed | -1.01% | -3.90% | -0.05% | 483 | 237 | 109 |
| Weather | phase2_vs_r3 | -0.25% | +0.45% | -0.31% | 430 | 290 | 199 |

## Decision Impact

[Inference] If Phase2-R.1 fails, the next problem should not be another stronger future teacher. A more defensible decoder problem is: how should a one-model decoder model the non-uniform error growth and residual process across forecast steps?

[Candidate Problem] Current decoder states produce point segments independently after conditioning on target queries. They do not explicitly model that error is an output process with step-region structure, sign-changing gains, and late-horizon growth. A next architecture should treat the prediction trajectory or residual trajectory as the object being decoded, not just each segment's mean state.

[Candidate Mechanism Direction] Output/error-process decoder: generate a base forecast plus a structured residual process over future steps, with constraints or parameterization that can express monotone error growth, covariance between adjacent steps, and segment-specific residual corrections. This should be evaluated before adding MoE if Phase2-R.1 fails.

## Figures

- `ETTh2_h720_step_relative_mse.png`
- `ETTm1_h720_step_relative_mse.png`
- `Weather_h720_step_relative_mse.png`
