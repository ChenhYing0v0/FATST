# Phase3-C Regime/Segment Operator Gate Report

## Decision

[Decision] performance gate passes numerically, but the mechanism claim is blocked by controls.

[Fact] This run used `TARGET_HORIZONS=96,720`.
[Fact] Horizon-set confound vs R.3: `True`.
[Fact] This run used `window_index_norm`: `True`.
[Decision] The result can be considered clean only when window-position and horizon-set controls pass.

## Summary

- MSE wins vs R.3: `5/6`.
- Mean relative MSE vs R.3: `-0.39%`.
- Observed aggregate-gap wins: `1/2`.
- Observed H720 segment-gap wins: `2/3`.
- Non-gap mean relative MSE vs R.3: `-0.66%`.
- Max prefix mismatch MSE: `4.925e-14`.
- Mean operator abs scale: `0.079033`.
- Mean operator abs shift: `0.019258`.

## Main Metrics vs R.3

| Dataset | Horizon | Candidate MSE | R.3 MSE | Rel MSE | Win | Observed gap |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| ETTh2 | 96 | 0.301044 | 0.304796 | -1.23% | True | False |
| ETTh2 | 720 | 0.410215 | 0.410473 | -0.06% | True | False |
| ETTm1 | 96 | 0.298320 | 0.298685 | -0.12% | True | True |
| ETTm1 | 720 | 0.414320 | 0.417293 | -0.71% | True | False |
| Weather | 96 | 0.148630 | 0.148026 | +0.41% | False | True |
| Weather | 720 | 0.318859 | 0.320847 | -0.62% | True | False |

## H720 Segment Metrics vs R.3

| Dataset | Segment | Candidate MSE | R.3 MSE | Rel MSE | Win | Observed gap |
| --- | --- | ---: | ---: | ---: | --- | --- |
| ETTh2 | 1-96 | 0.252092 | 0.249752 | +0.94% | False | False |
| ETTh2 | 97-192 | 0.336289 | 0.338119 | -0.54% | True | False |
| ETTh2 | 193-336 | 0.373090 | 0.369671 | +0.92% | False | True |
| ETTh2 | 337-720 | 0.482149 | 0.484043 | -0.39% | True | True |
| ETTm1 | 1-96 | 0.282870 | 0.284174 | -0.46% | True | False |
| ETTm1 | 97-192 | 0.349470 | 0.352049 | -0.73% | True | False |
| ETTm1 | 193-336 | 0.403402 | 0.405651 | -0.55% | True | False |
| ETTm1 | 337-720 | 0.467490 | 0.471249 | -0.80% | True | True |
| Weather | 1-96 | 0.147914 | 0.147463 | +0.31% | False | False |
| Weather | 97-192 | 0.232491 | 0.232687 | -0.08% | True | False |
| Weather | 193-336 | 0.311457 | 0.313093 | -0.52% | True | False |
| Weather | 337-720 | 0.385963 | 0.389141 | -0.82% | True | False |

## Prefix Consistency

| Dataset | Prefix MSE | Prefix MAE |
| --- | ---: | ---: |
| ETTh2 | 1.470e-14 | 6.498e-08 |
| ETTm1 | 4.925e-14 | 1.254e-07 |
| Weather | 9.475e-15 | 5.439e-08 |

## Operator Magnitude

| Dataset | Horizon | Mean abs scale | Mean abs shift | Max abs scale | Max abs shift |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | 0.054291 | 0.012933 | 0.097419 | 0.058878 |
| ETTh2 | 720 | 0.063334 | 0.018561 | 0.097434 | 0.058837 |
| ETTm1 | 96 | 0.083050 | 0.020607 | 0.099867 | 0.076681 |
| ETTm1 | 720 | 0.091303 | 0.018487 | 0.099929 | 0.079620 |
| Weather | 96 | 0.088249 | 0.023229 | 0.099983 | 0.084942 |
| Weather | 720 | 0.093967 | 0.021727 | 0.099988 | 0.087329 |

## Training Log

| Dataset | Epoch rows | Best val mean MSE | Last val mean MSE | Target horizons | Window index |
| --- | ---: | ---: | ---: | --- | --- |
| ETTh2 | 13 | 0.406327 | 0.490868 | 96,720 | True |
| ETTm1 | 12 | 0.655505 | 0.845025 | 96,720 | True |
| Weather | 12 | 0.535838 | 0.592073 | 96,720 | True |

## Window-Index Concern

[Concern] `window_index_norm` is prediction-before, but it is not a robust causal or calendar variable.
It is normalized inside each split, so it can encode train/val/test split position rather than a deployable regime.

[Decision] Before claiming a mechanism, Phase3-C needs controls:

1. same architecture without `window_index_norm`, using history-only regime features;
2. same target horizon set as R.3, using `96,192,336,720`.

[Next] If the no-window-position control keeps most gains, continue the regime-token route.
If gains disappear, treat the current result as split-position shortcut evidence and rollback to Step 4-6.
