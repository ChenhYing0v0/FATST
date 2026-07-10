# Phase5 StageB C0 ETTm1 Encoder Control Report

## Scope

本报告只审计 Encoder/carrier 与 training protocol，不构成 StageB 创新点。

## Summary

| comparison | selector | status | mse_wins | mean_relative_mse_pct | max_relative_mse_pct | preliminary_gate_pass |
| --- | --- | --- | --- | --- | --- | --- |
| global_width_d09 | official-last | ok | 0 | 1.3441 | 2.3073 | 0 |
| global_width_d09 | best-val | ok | 0 | 1.4413 | 2.5294 | 0 |
| patch_low_capacity_d09 | official-last | ok | 0 | 4.5497 | 6.8394 | 0 |
| patch_low_capacity_d09 | best-val | ok | 0 | 4.5851 | 6.9494 | 0 |
| patch_matched_d09 | official-last | ok | 0 | 4.2163 | 4.9825 | 0 |
| patch_matched_d09 | best-val | ok | 0 | 4.1706 | 5.0060 | 0 |
| patch_matched_d02 | official-last | ok | 0 | 1.9213 | 2.1700 | 0 |
| patch_matched_d02 | best-val | ok | 0 | 2.4998 | 3.1068 | 0 |

## Gate decision

`patch_num_performance_defect_not_supported`

Gate pass 只授权 multi-seed confirmation；不授权将 patching 或 mixer 写为 StageB method。

## Statistic definitions

- `relative_mse_pct = (candidate_mse / baseline_mse - 1) * 100`；负值更好。
- `mse_wins` 是四个 target horizons 中 candidate MSE 更低的数量。
- `preliminary_gate_pass` 要求 mean delta <= -0.5%、至少 3/4 wins、最大 regression <= +1.0%。
- `selector` 的 last 与 best-val 来自同一次 training trajectory。
