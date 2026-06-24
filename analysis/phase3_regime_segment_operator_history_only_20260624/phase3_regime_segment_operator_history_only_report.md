# Phase3-C Regime/Segment Operator Gate Report

## Decision

[Decision] history-only performance gate passes numerically; next control must align horizon set.

[Fact] This run used `TARGET_HORIZONS=96,720`.
[Fact] Horizon-set confound vs R.3: `True`.
[Fact] This run used `window_index_norm`: `False`.
[Decision] The result can be considered clean only when window-position and horizon-set controls pass.

## Summary

- MSE wins vs R.3: `5/6`.
- Mean relative MSE vs R.3: `-0.51%`.
- Observed aggregate-gap wins: `1/2`.
- Observed H720 segment-gap wins: `2/3`.
- Non-gap mean relative MSE vs R.3: `-0.64%`.
- Max prefix mismatch MSE: `4.861e-14`.
- Mean operator abs scale: `0.079619`.
- Mean operator abs shift: `0.019816`.

## Main Metrics vs R.3

| Dataset | Horizon | Candidate MSE | R.3 MSE | Rel MSE | Win | Observed gap |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| ETTh2 | 96 | 0.300827 | 0.304796 | -1.30% | True | False |
| ETTh2 | 720 | 0.410184 | 0.410473 | -0.07% | True | False |
| ETTm1 | 96 | 0.296043 | 0.298685 | -0.88% | True | True |
| ETTm1 | 720 | 0.414905 | 0.417293 | -0.57% | True | False |
| Weather | 96 | 0.148619 | 0.148026 | +0.40% | False | True |
| Weather | 720 | 0.318870 | 0.320847 | -0.62% | True | False |

## H720 Segment Metrics vs R.3

| Dataset | Segment | Candidate MSE | R.3 MSE | Rel MSE | Win | Observed gap |
| --- | --- | ---: | ---: | ---: | --- | --- |
| ETTh2 | 1-96 | 0.251893 | 0.249752 | +0.86% | False | False |
| ETTh2 | 97-192 | 0.336213 | 0.338119 | -0.56% | True | False |
| ETTh2 | 193-336 | 0.373160 | 0.369671 | +0.94% | False | True |
| ETTh2 | 337-720 | 0.482133 | 0.484043 | -0.39% | True | True |
| ETTm1 | 1-96 | 0.280989 | 0.284174 | -1.12% | True | False |
| ETTm1 | 97-192 | 0.349349 | 0.352049 | -0.77% | True | False |
| ETTm1 | 193-336 | 0.404449 | 0.405651 | -0.30% | True | False |
| ETTm1 | 337-720 | 0.468694 | 0.471249 | -0.54% | True | True |
| Weather | 1-96 | 0.147928 | 0.147463 | +0.32% | False | False |
| Weather | 97-192 | 0.232434 | 0.232687 | -0.11% | True | False |
| Weather | 193-336 | 0.311469 | 0.313093 | -0.52% | True | False |
| Weather | 337-720 | 0.385991 | 0.389141 | -0.81% | True | False |

## Prefix Consistency

| Dataset | Prefix MSE | Prefix MAE |
| --- | ---: | ---: |
| ETTh2 | 1.464e-14 | 6.473e-08 |
| ETTm1 | 4.861e-14 | 1.246e-07 |
| Weather | 9.415e-15 | 5.421e-08 |

## Operator Magnitude

| Dataset | Horizon | Mean abs scale | Mean abs shift | Max abs scale | Max abs shift |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | 0.054000 | 0.012129 | 0.097025 | 0.055291 |
| ETTh2 | 720 | 0.064161 | 0.017488 | 0.097326 | 0.056467 |
| ETTm1 | 96 | 0.085784 | 0.020731 | 0.099884 | 0.079729 |
| ETTm1 | 720 | 0.092356 | 0.023914 | 0.099941 | 0.084213 |
| Weather | 96 | 0.087624 | 0.023698 | 0.099984 | 0.086028 |
| Weather | 720 | 0.093790 | 0.020938 | 0.099989 | 0.086342 |

## Training Log

| Dataset | Epoch rows | Best val mean MSE | Last val mean MSE | Target horizons | Window index |
| --- | ---: | ---: | ---: | --- | --- |
| ETTh2 | 13 | 0.406327 | 0.486887 | 96,720 | False |
| ETTm1 | 12 | 0.653990 | 0.843691 | 96,720 | False |
| Weather | 12 | 0.535711 | 0.587025 | 96,720 | False |

## Window-Index Concern

[Concern] `window_index_norm` is prediction-before, but it is not a robust causal or calendar variable.
It is normalized inside each split, so it can encode train/val/test split position rather than a deployable regime.

[Decision] Before claiming a mechanism, Phase3-C needs controls:

1. same architecture without `window_index_norm`, using history-only regime features;
2. same target horizon set as R.3, using `96,192,336,720`.

[Next] If the no-window-position control keeps most gains, continue the regime-token route.
If gains disappear, treat the current result as split-position shortcut evidence and rollback to Step 4-6.
