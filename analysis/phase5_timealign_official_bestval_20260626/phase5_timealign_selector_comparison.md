# Phase5 TimeAlign Selector Comparison

## Decision

[Decision] `dataset_dependent_unified_behavior_not_selector_artifact`.

## Selector Summary

| dataset | settings | official_last_wins | best_val_wins | official_last_mean_relative_mse_pct | best_val_mean_relative_mse_pct | mean_gap_delta_pct |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 4 | 3 | 3 | -8.01 | -8.38 | -0.37 |
| ETTm2 | 4 | 0 | 0 | 3.72 | 3.68 | -0.04 |
| Weather | 4 | 0 | 0 | 1.05 | 1.15 | 0.10 |
| ALL | 12 | 3 | 3 | -1.08 | -1.18 | -0.10 |

## Per-Horizon Gap Comparison

| dataset | target_horizon | official_last_relative_mse_pct | best_val_relative_mse_pct | gap_delta_pct | official_last_unified_win | best_val_unified_win |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | -7.64 | -7.31 | 0.33 | True | True |
| ETTh2 | 192 | -11.55 | -14.14 | -2.60 | True | True |
| ETTh2 | 336 | -12.84 | -12.06 | 0.79 | True | True |
| ETTh2 | 720 | 0.00 | 0.00 | 0.00 | False | False |
| ETTm2 | 96 | 8.19 | 9.50 | 1.31 | False | False |
| ETTm2 | 192 | 4.41 | 3.02 | -1.39 | False | False |
| ETTm2 | 336 | 2.28 | 2.19 | -0.09 | False | False |
| ETTm2 | 720 | 0.00 | 0.00 | 0.00 | False | False |
| Weather | 96 | 1.95 | 1.55 | -0.41 | False | False |
| Weather | 192 | 1.44 | 1.69 | 0.24 | False | False |
| Weather | 336 | 0.80 | 0.43 | -0.37 | False | False |
| Weather | 720 | 0.00 | 0.92 | 0.92 | False | False |

## Reading

[Fact] `best-val` does not change the winner pattern: ETTh2 remains `3/4` unified wins, while ETTm2 and Weather remain `0/4` unified wins.

[Strong Evidence] The unified/fixed split is not mainly caused by the author-intended last-epoch selector. It is a dataset-dependent behavior of the TimeAlign unified setting.

[Decision] Do not claim a global unified multi-horizon degradation. The next research step should either define a dataset/state-dependent TimeAlign-HSS problem or first run a look-back horizon sweep to align with the paper reproduction protocol.
