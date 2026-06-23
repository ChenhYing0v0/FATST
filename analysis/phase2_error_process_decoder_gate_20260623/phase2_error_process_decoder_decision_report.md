# Phase2-B Error-Process Decoder Gate Report

## Decision

[Decision] `PatchEncoderErrorProcessDecoder` fails the Phase2-B error-process gate.

[Inference] The error-process mechanism is not yet a paper-core candidate. If residual diagnostics are active but MSE does not improve, the rollback should target objective design or base architecture rather than simply increasing residual capacity.

## Main Metrics vs R.3

- MSE wins vs `PatchEncoderPrefixRiskWeighted`: `4/12`.
- MAE wins vs `PatchEncoderPrefixRiskWeighted`: `4/12`.
- Mean relative MSE vs R.3: `+1.12%`.
- ETTh2 mean relative MSE vs R.3: `+4.15%`.
- ETTm1 mean relative MSE vs R.3: `-1.24%`.
- Weather mean relative MSE vs R.3: `+0.44%`.

## Main Metrics vs FixedHead

- MSE wins vs horizon-specific `PatchEncoderFixedHead`: `6/12`.
- MAE wins vs horizon-specific `PatchEncoderFixedHead`: `8/12`.
- Mean relative MSE vs FixedHead: `+0.67%`.

## Gate

- mean MSE vs R.3 improves: `False`
- MSE wins vs R.3 >= 7/12: `False`
- no dataset degrades over +0.3%: `False`
- prefix mismatch numerical zero: `True`
- focus H720 regions win >= 2/4: `False`
- residual controlled: `True`
- base + residual decomposition: `True`

## Error-Process Diagnostics

- mean residual/base MAE ratio: `0.00305975`.
- mean residual energy: `4.86398e-05`.
- mean residual gain MSE: `+0.02%`.
- max decomposition abs: `9.53674e-07`.

## Per-Horizon Metrics vs R.3

| Dataset | Horizon | Relative MSE vs R.3 | Candidate MSE | R.3 MSE |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | +4.21% | 0.317628 | 0.304796 |
| ETTh2 | 192 | +6.85% | 0.394318 | 0.369043 |
| ETTh2 | 336 | +2.56% | 0.392699 | 0.382910 |
| ETTh2 | 720 | +2.97% | 0.422684 | 0.410473 |
| ETTm1 | 96 | -0.37% | 0.297573 | 0.298685 |
| ETTm1 | 192 | -1.17% | 0.325808 | 0.329662 |
| ETTm1 | 336 | -1.41% | 0.355629 | 0.360729 |
| ETTm1 | 720 | -1.99% | 0.408968 | 0.417293 |
| Weather | 96 | +0.78% | 0.149183 | 0.148026 |
| Weather | 192 | +0.46% | 0.193286 | 0.192409 |
| Weather | 336 | +0.31% | 0.245552 | 0.244793 |
| Weather | 720 | +0.22% | 0.321552 | 0.320847 |

## H720 Focus Regions

| Dataset | Segment | Focus | Relative MSE vs R.3 | Candidate MSE | R.3 MSE |
| --- | --- | --- | ---: | ---: | ---: |
| ETTh2 | 1-96 | False | +2.55% | 0.256116 | 0.249752 |
| ETTh2 | 97-192 | False | +6.80% | 0.361099 | 0.338119 |
| ETTh2 | 193-336 | True | +3.12% | 0.381197 | 0.369671 |
| ETTh2 | 337-720 | True | +2.32% | 0.495280 | 0.484043 |
| ETTm1 | 1-96 | False | -0.80% | 0.281889 | 0.284174 |
| ETTm1 | 97-192 | False | -1.75% | 0.345873 | 0.352049 |
| ETTm1 | 193-336 | False | -2.14% | 0.396967 | 0.405651 |
| ETTm1 | 337-720 | True | -2.17% | 0.461012 | 0.471249 |
| Weather | 1-96 | True | +0.74% | 0.148549 | 0.147463 |
| Weather | 97-192 | False | +0.22% | 0.233200 | 0.232687 |
| Weather | 193-336 | False | +0.01% | 0.313123 | 0.313093 |
| Weather | 337-720 | False | +0.23% | 0.390052 | 0.389141 |

## Residual Rows

| Dataset | Horizon | Scope | Residual/Base MAE | Residual Gain MSE | Decomp Max Abs |
| --- | ---: | --- | ---: | ---: | ---: |
| ETTh2 | 96 | all | 0.000787525 | +0.04% | 9.54e-07 |
| ETTh2 | 192 | all | 0.00118789 | +0.08% | 4.77e-07 |
| ETTh2 | 336 | all | 0.00141982 | +0.06% | 9.54e-07 |
| ETTh2 | 720 | all | 0.00172593 | +0.07% | 9.54e-07 |
| ETTm1 | 96 | all | 0.00314262 | -0.01% | 9.54e-07 |
| ETTm1 | 192 | all | 0.00495959 | -0.01% | 9.54e-07 |
| ETTm1 | 336 | all | 0.00596514 | -0.02% | 9.54e-07 |
| ETTm1 | 720 | all | 0.00593049 | -0.01% | 9.54e-07 |
| Weather | 96 | all | 0.00253308 | +0.00% | 9.54e-07 |
| Weather | 192 | all | 0.00296718 | -0.00% | 9.54e-07 |
| Weather | 336 | all | 0.00303525 | -0.00% | 9.54e-07 |
| Weather | 720 | all | 0.00306253 | -0.00% | 4.77e-07 |

## Rollback Rule

If this report fails while residual activity is controlled, return to loop step 2-3 and consider objective-level modeling such as step covariance weighting. Do not add MoE to this residual state unless this gate passes or a specific failure mode is repaired.
