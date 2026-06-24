# Phase3-B Regime/Segment Mechanism Diagnostic

## Decision

[Decision] pre-input regime/segment signals exist; continue to conditioned target-operator design, not output residual correction.

## Gate

- short_regime_pre_input_signal: `True`
- late_segment_pre_input_signal: `True`
- no_output_residual_mechanism_used: `True`
- supports_conditioned_target_operator_design: `True`

## Known Short-Gap Feature Signals

| Dataset | Horizon | Feature | AUC | SMD | Extra MSE | Aligned MSE |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| ETTm1 | 96 | history_mean | 0.997619 | -3.221514 | 0.549860 | 0.284174 |
| Weather | 96 | window_index_norm | 1.000000 | 2.599896 | 0.156898 | 0.147463 |
| ETTm1 | 96 | window_index_norm | 1.000000 | 2.586690 | 0.549860 | 0.284174 |
| Weather | 96 | history_std | 0.979425 | 2.494574 | 0.156898 | 0.147463 |
| Weather | 96 | history_abs_mean | 0.969142 | 2.235503 | 0.156898 | 0.147463 |
| Weather | 96 | history_mean | 0.940804 | -2.211765 | 0.156898 | 0.147463 |
| Weather | 96 | history_recent_mean | 0.928097 | -1.971744 | 0.156898 | 0.147463 |
| Weather | 96 | history_last_abs_mean | 0.800067 | 1.299200 | 0.156898 | 0.147463 |

## Known H720 Late-Gap Feature Signals

| Dataset | Segment | Feature | AUC | SMD | High-error MSE | Other MSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ETTh2 | 337-720 | window_index_norm | 0.845886 | 1.543815 | 0.826450 | 0.369696 |
| ETTh2 | 337-720 | history_slope_abs_mean | 0.828835 | 1.262857 | 0.826450 | 0.369696 |
| ETTm1 | 337-720 | window_index_norm | 0.786843 | 1.219849 | 0.814483 | 0.356796 |
| ETTh2 | 193-336 | window_index_norm | 0.777454 | 1.165940 | 0.753092 | 0.241627 |
| ETTh2 | 337-720 | history_std | 0.747348 | 0.987736 | 0.826450 | 0.369696 |
| ETTh2 | 193-336 | history_abs_mean | 0.754357 | -0.755957 | 0.753092 | 0.241627 |
| ETTh2 | 193-336 | history_mean | 0.781932 | 0.692256 | 0.753092 | 0.241627 |
| ETTh2 | 193-336 | history_recent_mean | 0.610338 | 0.601395 | 0.753092 | 0.241627 |

## Interpretation

[Fact] Features in this diagnostic are computed only from historical input windows and window position. Prediction errors are used only as labels for analysis.

[Decision Rule] A future model candidate may use these signals to condition target states or segment operators before output generation. It should not add a free output residual correction head.
