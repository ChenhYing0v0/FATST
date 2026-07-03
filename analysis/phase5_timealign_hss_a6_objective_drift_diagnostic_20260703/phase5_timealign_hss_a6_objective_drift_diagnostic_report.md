# Phase5-A6OD Objective Drift Diagnostic Report

本文档分析 ETTh2-only A6 objective drift diagnostic。该实验为 diagnostic-only，所有 run 使用 `official-last` / without early stop。

## Conclusion

[Strong Evidence] objective switch 没有修复 A6 的 ETTh2 gap。最佳 variant `lbf_r256_stochastic_p1` 相对 best stage control 平均 `1.79%`，wins `0/4`。

[Fact] `full` 与 stochastic/continuous prefix 目标没有消除 official-last drift：最佳 variant 的 last-vs-best validation drift 仍为 `6.25%`。

## Variant Summary

| Variant | Family | Objective | mean MSE | vs best control | wins | last-vs-best val | best epoch |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `der_continuous_p4` | `der` | `continuous_p4` | 0.3095 | +2.71% | 0/4 | +12.84% | 1 |
| `der_full` | `der` | `full` | 0.3188 | +5.81% | 0/4 | +20.80% | 1 |
| `der_stochastic_p1` | `der` | `stochastic_p1` | 0.3080 | +2.08% | 0/4 | +6.16% | 1 |
| `lbf_r256_continuous_p4` | `lbf_r256` | `continuous_p4` | 0.3087 | +2.47% | 0/4 | +10.96% | 3 |
| `lbf_r256_full` | `lbf_r256` | `full` | 0.3182 | +5.61% | 0/4 | +13.44% | 1 |
| `lbf_r256_stochastic_p1` | `lbf_r256` | `stochastic_p1` | 0.3069 | +1.79% | 0/4 | +6.25% | 3 |

## Interpretation

[Strong Evidence] 仅调整 prefix supervision sampling 不足以解释或修复 A6-LBF 的 ETTh2 partial-pass：LBF variants 的 best-control gap 范围是 `1.79%` 到 `5.61%`，DER variants 范围是 `2.08%` 到 `5.81%`。

[Decision] A6OD 不通过 repair gate。下一步不应继续 objective-sampling sweep；应回 Step 4/5 设计 explicit stability path，例如 official-last-compatible regularization、teacher/nested stability control，或重新评估 best controls 的 regularization advantage。

## Statistic Definitions

`relative_mse_vs_best_stage_control_pct` 来自每个 target horizon 的 final test MSE 与 A6 comparison 中 ETTh2 best stage control MSE 的比值。`last_vs_best_val_mse_pct` 来自该 run 的 `training_log.csv`，计算 `last_val_mean_mse / best_val_mean_mse - 1`，用于衡量 official-last drift。

## Artifacts

- `phase5_timealign_hss_a6od_comparison.csv`
- `phase5_timealign_hss_a6od_summary.csv`
- `phase5_timealign_hss_a6od_analysis_config.json`
- ignored raw metrics/logs under `raw/`
