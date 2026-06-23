# Phase2-R.1 Confidence-Weighted Future Alignment Gate Report

## Decision

[Decision] `PatchEncoderFutureStateAlignmentConfWeighted` fails the Phase2-R.1 repair gate.

[Inference] The repair does not yet prove future-state alignment is a paper-core mechanism. If leakage and prefix checks pass but MSE/MAE do not, the likely issue is semantic mismatch between teacher state and forecasting objective rather than implementation safety.

## Main Metrics vs R.3

- MSE wins vs `PatchEncoderPrefixRiskWeighted`: `7/12`.
- MAE wins vs `PatchEncoderPrefixRiskWeighted`: `7/12`.
- Mean relative MSE vs R.3: `+1.28%`.
- ETTh2 mean relative MSE vs R.3: `+5.08%`.
- ETTm1 mean relative MSE vs R.3: `-1.28%`.
- Weather mean relative MSE vs R.3: `+0.04%`.

## Main Metrics vs FixedHead

- MSE wins vs horizon-specific `PatchEncoderFixedHead`: `6/12`.
- MAE wins vs horizon-specific `PatchEncoderFixedHead`: `6/12`.
- Mean relative MSE vs FixedHead: `+0.83%`.

## Gate

- prediction leakage <= `1e-7`: `True` (`0`).
- prefix mismatch numerical zero: `True` (`4.7318994e-14`).
- ETTh2 conflict repaired <= `+0.3%`: `False` (`+5.08%`).
- ETTm1 signal preserved: `True` (`-1.28%`).
- Weather signal preserved: `False` (`+0.04%`).

## Confidence Diagnostics

- mean teacher/student cosine: `0.471067`.
- mean alignment confidence: `0.710082`.
- mean normalized reconstruction loss: `0.396736`.
- mean raw reconstruction loss: `215.801663`.

| Dataset | Horizon | Relative MSE vs R.3 | Repair MSE | R.3 MSE |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | +7.22% | 0.326801 | 0.304796 |
| ETTh2 | 192 | +6.74% | 0.393912 | 0.369043 |
| ETTh2 | 336 | +2.63% | 0.392993 | 0.382910 |
| ETTh2 | 720 | +3.75% | 0.425848 | 0.410473 |
| ETTm1 | 96 | -1.21% | 0.295086 | 0.298685 |
| ETTm1 | 192 | -1.10% | 0.326045 | 0.329662 |
| ETTm1 | 336 | -1.37% | 0.355802 | 0.360729 |
| ETTm1 | 720 | -1.45% | 0.411228 | 0.417293 |
| Weather | 96 | +0.57% | 0.148875 | 0.148026 |
| Weather | 192 | -0.08% | 0.192253 | 0.192409 |
| Weather | 336 | -0.12% | 0.244504 | 0.244793 |
| Weather | 720 | -0.22% | 0.320157 | 0.320847 |

## Alignment Rows

| Dataset | Horizon | Cosine | Confidence | Norm Recon | Raw Recon | Leakage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | 0.2948 | 0.4567 | 0.6640 | 1.0932 | 0 |
| ETTh2 | 192 | 0.2734 | 0.4392 | 0.6860 | 1.2799 | 0 |
| ETTh2 | 336 | 0.2736 | 0.4046 | 0.7348 | 1.5910 | 0 |
| ETTh2 | 720 | 0.2412 | 0.4238 | 0.6231 | 2.1288 | 0 |
| ETTm1 | 96 | 0.8345 | 0.9525 | 0.0473 | 0.0969 | 0 |
| ETTm1 | 192 | 0.8329 | 0.9474 | 0.0549 | 0.1320 | 0 |
| ETTm1 | 336 | 0.8283 | 0.9442 | 0.0637 | 0.1983 | 0 |
| ETTm1 | 720 | 0.8175 | 0.9402 | 0.0736 | 0.3196 | 0 |
| Weather | 96 | 0.3630 | 0.7591 | 0.3813 | 384.1598 | 0 |
| Weather | 192 | 0.3296 | 0.7532 | 0.4249 | 485.1795 | 0 |
| Weather | 336 | 0.2968 | 0.7483 | 0.4703 | 593.5752 | 0 |
| Weather | 720 | 0.2673 | 0.7516 | 0.5371 | 1119.8659 | 0 |

## Rollback Rule

If this report fails only on performance while leakage/prefix pass, return to loop step 2-3 and reconsider whether future-state alignment is the right decoder problem. Do not stack MoE on this state before that rollback assessment.
