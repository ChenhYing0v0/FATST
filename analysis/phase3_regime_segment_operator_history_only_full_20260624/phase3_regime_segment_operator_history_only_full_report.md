# Phase3-C Regime/Segment Operator Gate Report

## Decision

[Decision] performance gate does not pass as a paper-story candidate.

[Fact] This run used `TARGET_HORIZONS=96,192,336,720`.
[Fact] Horizon-set confound vs R.3: `False`.
[Fact] This run used `window_index_norm`: `False`.
[Decision] The result can be considered clean only when window-position and horizon-set controls pass.

## Summary

- MSE wins vs R.3: `1/12`.
- Mean relative MSE vs R.3: `+2.00%`.
- Observed aggregate-gap wins: `0/2`.
- Observed H720 segment-gap wins: `1/3`.
- Non-gap mean relative MSE vs R.3: `+1.95%`.
- Max prefix mismatch MSE: `5.319e-14`.
- Mean operator abs scale: `0.080888`.
- Mean operator abs shift: `0.022848`.

## Main Metrics vs R.3

| Dataset | Horizon | Candidate MSE | R.3 MSE | Rel MSE | Win | Observed gap |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| ETTh2 | 96 | 0.309985 | 0.304796 | +1.70% | False | False |
| ETTh2 | 192 | 0.381503 | 0.369043 | +3.38% | False | False |
| ETTh2 | 336 | 0.391810 | 0.382910 | +2.32% | False | False |
| ETTh2 | 720 | 0.423572 | 0.410473 | +3.19% | False | False |
| ETTm1 | 96 | 0.300644 | 0.298685 | +0.66% | False | True |
| ETTm1 | 192 | 0.330623 | 0.329662 | +0.29% | False | False |
| ETTm1 | 336 | 0.360733 | 0.360729 | +0.00% | False | False |
| ETTm1 | 720 | 0.415235 | 0.417293 | -0.49% | True | False |
| Weather | 96 | 0.153616 | 0.148026 | +3.78% | False | True |
| Weather | 192 | 0.198602 | 0.192409 | +3.22% | False | False |
| Weather | 336 | 0.252028 | 0.244793 | +2.96% | False | False |
| Weather | 720 | 0.330409 | 0.320847 | +2.98% | False | False |

## H720 Segment Metrics vs R.3

| Dataset | Segment | Candidate MSE | R.3 MSE | Rel MSE | Win | Observed gap |
| --- | --- | ---: | ---: | ---: | --- | --- |
| ETTh2 | 1-96 | 0.255456 | 0.249752 | +2.28% | False | False |
| ETTh2 | 97-192 | 0.354104 | 0.338119 | +4.73% | False | False |
| ETTh2 | 193-336 | 0.388129 | 0.369671 | +4.99% | False | True |
| ETTh2 | 337-720 | 0.496259 | 0.484043 | +2.52% | False | True |
| ETTm1 | 1-96 | 0.283804 | 0.284174 | -0.13% | True | False |
| ETTm1 | 97-192 | 0.350983 | 0.352049 | -0.30% | True | False |
| ETTm1 | 193-336 | 0.404167 | 0.405651 | -0.37% | True | False |
| ETTm1 | 337-720 | 0.468305 | 0.471249 | -0.62% | True | True |
| Weather | 1-96 | 0.152270 | 0.147463 | +3.26% | False | False |
| Weather | 97-192 | 0.238531 | 0.232687 | +2.51% | False | False |
| Weather | 193-336 | 0.320585 | 0.313093 | +2.39% | False | False |
| Weather | 337-720 | 0.401596 | 0.389141 | +3.20% | False | False |

## Prefix Consistency

| Dataset | Prefix MSE | Prefix MAE |
| --- | ---: | ---: |
| ETTh2 | 1.558e-14 | 6.688e-08 |
| ETTh2 | 1.377e-14 | 6.284e-08 |
| ETTh2 | 9.355e-15 | 4.648e-08 |
| ETTm1 | 5.319e-14 | 1.289e-07 |
| ETTm1 | 5.040e-14 | 1.255e-07 |
| ETTm1 | 3.374e-14 | 9.521e-08 |
| Weather | 9.387e-15 | 5.336e-08 |
| Weather | 0.000e+00 | 0.000e+00 |
| Weather | 0.000e+00 | 0.000e+00 |

## Operator Magnitude

| Dataset | Horizon | Mean abs scale | Mean abs shift | Max abs scale | Max abs shift |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | 0.063933 | 0.016873 | 0.099190 | 0.072045 |
| ETTh2 | 192 | 0.067368 | 0.017761 | 0.099219 | 0.072045 |
| ETTh2 | 336 | 0.070075 | 0.018679 | 0.099219 | 0.072045 |
| ETTh2 | 720 | 0.071044 | 0.018822 | 0.099219 | 0.072045 |
| ETTm1 | 96 | 0.084059 | 0.021877 | 0.099941 | 0.095230 |
| ETTm1 | 192 | 0.087588 | 0.023627 | 0.099970 | 0.096647 |
| ETTm1 | 336 | 0.089838 | 0.024785 | 0.099977 | 0.096976 |
| ETTm1 | 720 | 0.090321 | 0.023723 | 0.099977 | 0.096976 |
| Weather | 96 | 0.084232 | 0.028962 | 0.099998 | 0.095475 |
| Weather | 192 | 0.086042 | 0.027391 | 0.099999 | 0.095794 |
| Weather | 336 | 0.087681 | 0.026331 | 0.099999 | 0.095813 |
| Weather | 720 | 0.088469 | 0.025347 | 0.099999 | 0.095813 |

## Training Log

| Dataset | Epoch rows | Best val mean MSE | Last val mean MSE | Target horizons | Window index |
| --- | ---: | ---: | ---: | --- | --- |
| ETTh2 | 12 | 0.364984 | 0.435295 | 96,192,336,720 | False |
| ETTm1 | 11 | 0.606500 | 0.763917 | 96,192,336,720 | False |
| Weather | 12 | 0.527275 | 0.622039 | 96,192,336,720 | False |

## Window-Index Concern

[Concern] `window_index_norm` is prediction-before, but it is not a robust causal or calendar variable.
It is normalized inside each split, so it can encode train/val/test split position rather than a deployable regime.

[Decision] Before claiming a mechanism, Phase3-C needs controls:

1. same architecture without `window_index_norm`, using history-only regime features;
2. same target horizon set as R.3, using `96,192,336,720`.

[Next] If the no-window-position control keeps most gains, continue the regime-token route.
If gains disappear, treat the current result as split-position shortcut evidence and rollback to Step 4-6.
