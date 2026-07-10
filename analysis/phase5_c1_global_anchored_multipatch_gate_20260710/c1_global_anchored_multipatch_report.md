# Phase5 C1 Global-Anchored Multi-Patch Gate Report

## Decision

`c1_carrier_normalization_gate_failed`

C1 只评估统一 carrier/interface，不构成 StageB 创新点。

## Protocol audit

| arm | dataset | effective_learning_rate | source_preset_learning_rate | source_learning_rate_match |
| --- | --- | --- | --- | --- |
| a6_clean | ETTh2 | 0.0001 | 0.0005 | 0 |
| a6_clean | ETTm1 | 0.0001 | 0.0001 | 1 |
| a6_clean | Weather | 0.0001 | 0.0001 | 1 |
| gamp_p16s8 | ETTh2 | 0.0001 | 0.0005 | 0 |
| gamp_p16s8 | ETTm1 | 0.0001 | 0.0001 | 1 |
| gamp_p16s8 | Weather | 0.0001 | 0.0001 | 1 |
| gamp_p48s24 | ETTh2 | 0.0001 | 0.0005 | 0 |
| gamp_p48s24 | ETTm1 | 0.0001 | 0.0001 | 1 |
| gamp_p48s24 | Weather | 0.0001 | 0.0001 | 1 |

Runner 对全部 arms 显式使用 `learning_rate=1e-4`。ETTh2 source preset 为 `5e-4`，所以 ETTh2 A6 不是 source-faithful reproduction；同一 dataset 内的 C1/A6 controlled comparison仍使用相同 learning rate。ETTm1 与 Weather 无此偏差。

## Gate summary

| arm | selector | overall_mean_mse_vs_a6_pct | max_horizon_mse_vs_a6_pct | ETTh2_mean_mse_vs_a6_pct | ETTm1_mean_mse_vs_a6_pct | Weather_mean_mse_vs_a6_pct | overall_mean_mse_vs_fixed_pct | wins_vs_fixed | gate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gamp_p16s8 | last | 5.3704 | 12.2613 | 8.2317 | 1.0977 | 6.7817 | 0.2484 | 5 | 0 |
| gamp_p16s8 | best_val | 3.7526 | 6.1828 | 2.7971 | 2.9744 | 5.4864 | 0.0272 | 4 | 0 |
| gamp_p48s24 | last | 5.8672 | 12.2005 | 4.5326 | 2.0465 | 11.0225 | 0.9621 | 4 | 0 |
| gamp_p48s24 | best_val | 4.7326 | 9.6322 | 3.9605 | 1.9257 | 8.3114 | 0.9820 | 4 | 0 |
| validation_selected | last | 5.8672 | 12.2005 | 4.5326 | 2.0465 | 11.0225 | 0.9621 | 4 | 0 |
| validation_selected | best_val | 4.7326 | 9.6322 | 3.9605 | 1.9257 | 8.3114 | 0.9820 | 4 | 0 |

## Validation-only scale selection

| dataset | selected_arm | p16s8_best_val_mean_mse | p48s24_best_val_mean_mse |
| --- | --- | --- | --- |
| ETTh2 | gamp_p48s24 | 0.3898 | 0.3859 |
| ETTm1 | gamp_p48s24 | 0.6157 | 0.6086 |
| Weather | gamp_p48s24 | 0.4984 | 0.4979 |

Scale selection只读取 training log 的 minimum validation mean MSE，不读取 test metrics。

## Next action

- shared/selected gate通过：追加 seeds，并执行 local-token use control；
- near miss且 training-validation gap支持 dropout问题：只追加一个 dropout policy；
- 其余情况：恢复 accepted A6 + exact HPM，关闭 C1。
