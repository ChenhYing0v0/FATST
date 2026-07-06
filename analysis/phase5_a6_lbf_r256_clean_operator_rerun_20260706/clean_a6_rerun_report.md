# Phase5 Clean A6-LBF-r256 Rerun Report

## Decision

[Decision] `clean_a6_validated`.
[Current Step] StageA clean validation after removing the A6 future reconstruction/alignment branch.
[Gate] This report only decides whether the clean A6-LBF-r256 operator remains a valid Contribution 1 evidence base.
[Rollback] If this gate fails, roll back to StageA Step 9/10 and inspect the code cut or retrain variance before opening StageB methods.

## Summary

- vs fixed-horizon TimeAlign: `-4.13%, 9/12 MSE wins`.
- vs official unified TimeAlign: `-1.75%, 11/12 MSE wins`.
- vs historical A6-LBF-r256: `+0.20%, 6/12 MSE wins`.

## Clean Run Training Check

| dataset | status | epochs_ran | best_epoch | best_val_mean_mse | last_val_mean_mse | effective_w_recon | effective_w_align | readout_mode | basis_rank | pred_loss_mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | ok | 10 | 1 | 0.410330 | 0.475124 | 0.000000 | 0.000000 | learned-basis-forecast-operator | 256 | multi-prefix |
| ETTm1 | ok | 10 | 9 | 0.599628 | 0.599933 | 0.000000 | 0.000000 | learned-basis-forecast-operator | 256 | multi-prefix |
| Weather | ok | 10 | 8 | 0.488468 | 0.489485 | 0.000000 | 0.000000 | learned-basis-forecast-operator | 256 | multi-prefix |

## vs Fixed-Horizon TimeAlign

| dataset | settings | mse_wins | mean_relative_mse_pct | mean_relative_mae_pct |
| --- | --- | --- | --- | --- |
| ETTh2 | 4 | 4 | -10.531103 | -5.266192 |
| ETTm1 | 4 | 3 | -1.636748 | -0.249383 |
| Weather | 4 | 2 | -0.221626 | 0.761090 |
| ALL | 12 | 9 | -4.129826 | -1.584828 |

## vs Official Unified TimeAlign

| dataset | settings | mse_wins | mean_relative_mse_pct | mean_relative_mae_pct |
| --- | --- | --- | --- | --- |
| ETTh2 | 4 | 4 | -2.778588 | -1.491957 |
| ETTm1 | 4 | 3 | -1.199131 | -0.778916 |
| Weather | 4 | 4 | -1.257825 | -1.086542 |
| ALL | 12 | 11 | -1.745181 | -1.119138 |

## vs Historical A6-LBF-r256

| dataset | settings | mse_wins | mean_relative_mse_pct | mean_relative_mae_pct |
| --- | --- | --- | --- | --- |
| ETTh2 | 4 | 0 | 0.691545 | 0.245375 |
| ETTm1 | 4 | 3 | -0.035431 | 0.097328 |
| Weather | 4 | 3 | -0.044682 | 0.090000 |
| ALL | 12 | 6 | 0.203811 | 0.144235 |

## Reading

- [Fact] A clean pass validates the pure learned-basis forecast operator as the current paper-core method.
- [Fact] This gate does not revive B6-PLO or any StageB objective candidate.
- [Inference] If clean A6 remains close to historical A6 while preserving fixed/unified wins, the future-recon branch removal improves narrative clarity without weakening the empirical base.
