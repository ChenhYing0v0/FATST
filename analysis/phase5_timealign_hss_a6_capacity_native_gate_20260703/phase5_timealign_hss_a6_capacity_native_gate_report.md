# Phase5-A6 Capacity-Native Unified Head Gate Report

本文档分析 A6 capacity-native remote gate。该实验验证 `A6-DER` capacity ceiling 与 `A6-LBF-r256/r512` learned-basis forecast operator。

## 结论摘要

[Strong Evidence] A6 成功修复了 A5-Q/A5-B 暴露出的主要 capacity collapse：`A6-DER` 相对 `best_stage_control` 平均仅差 `0.91%`，且相对 A5-B-r128 平均改善 `-11.27%`。

[Strong Evidence] `A6-LBF-r256` 已基本贴住 dense-equivalent ceiling：相对 `A6-DER` 平均 `-0.03%`，相对 `best_stage_control` 平均 `0.88%`。`r512` 没有带来稳定收益。

[Fact] A6 仍未形成明确 paper-core pass：按单 arm 统计，`A6-LBF-r256/r512` 对 `best_stage_control` 的 wins 均为 `0/12`，`A6-DER` 仅在 Weather 两个 setting 上略胜。

[Decision] A6-LBF 应标记为 `partial_pass_capacity_recovered_not_yet_core`。它证明 learned-basis dense-capacity path 有效，但还需要 best-val/early-stopping 或 objective-level 诊断来判断 ETTh2 与 long horizon 的剩余差距。

## Arm Summary

| Arm | mean MSE | vs best control | wins | vs official unified | vs A5-B-r128 | vs A6-DER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `a6_der` | 0.2858 | +0.91% | 2/12 | -1.91% | -11.27% | +0.00% |
| `a6_lbf_r256` | 0.2857 | +0.88% | 0/12 | -1.94% | -11.30% | -0.03% |
| `a6_lbf_r512` | 0.2859 | +0.96% | 0/12 | -1.86% | -11.22% | +0.05% |

## Best A6 Per Setting

Best-of-A6 oracle 平均仍相对 best stage control `0.79%`，wins `2/12`。

| Dataset | Horizon | Best A6 | MSE | Best control | Gap |
| --- | ---: | --- | ---: | --- | ---: |
| ETTh2 | 96 | `a6_lbf_r512` | 0.2415 | `a3e_best` | +0.42% |
| ETTh2 | 192 | `a6_lbf_r512` | 0.2828 | `h1_target_set` | +2.00% |
| ETTh2 | 336 | `a6_der` | 0.3121 | `a3d_w03` | +3.15% |
| ETTh2 | 720 | `a6_der` | 0.3936 | `a3d_w03` | +2.41% |
| ETTm1 | 96 | `a6_lbf_r256` | 0.2731 | `a3c_warm` | +0.92% |
| ETTm1 | 192 | `a6_lbf_r256` | 0.3094 | `a3c_warm` | +0.38% |
| ETTm1 | 336 | `a6_lbf_r256` | 0.3472 | `a3c_warm` | +0.14% |
| ETTm1 | 720 | `a6_der` | 0.4069 | `official_unified` | +0.05% |
| Weather | 96 | `a6_der` | 0.1413 | `a2_nested` | -0.00% |
| Weather | 192 | `a6_der` | 0.1824 | `a3d_w03` | +0.06% |
| Weather | 336 | `a6_der` | 0.2316 | `a2_nested` | -0.05% |
| Weather | 720 | `a6_lbf_r256` | 0.3034 | `a2_nested` | +0.02% |

## Mechanism Interpretation

### A6-DER ceiling

A6-DER 的结果支持一个重要机制判断：head operator capacity 是 A5 collapse 的主因之一。只要保留 dense-equivalent row capacity 并改成 prefix-native invocation，性能就从 A5-B/A5-Q 的明显失败恢复到 best controls 附近。

### A6-LBF learned basis

A6-LBF-r256 与 A6-DER 的平均差距接近 0，说明 fixed Fourier/polynomial basis 是 A5-B 的关键瓶颈，而 learned temporal basis 可以在较低 rank 下近似 dense-equivalent capacity。r512 没有稳定优于 r256，第一轮不支持继续简单扩大 rank。

### Remaining gap

A6 尚未超过 best stage controls。ETTh2 的 training summary 显示 A6-DER 在 epoch 1 已达到 best val，随后 last-val 变差；这提示 official-last checkpoint policy 可能低估 A6 on ETTh2。但在未跑 best-val 对照前，不能把 A6-LBF 升级为 paper-core pass。

## Decision

- `A6-DER_prefix_native_dense_equivalent_row_bank`: `control_passed_as_capacity_ceiling`。
- `A6-LBF_learned_basis_forecast_operator`: `partial_pass_capacity_recovered_not_yet_core`。
- 不建议继续 rank-only sweep；下一步优先做 best-val / early-stopping diagnostic，并同步检查 whether A6-LBF 的 learned basis 具备可解释 low-rank structure。

## Artifacts

- `phase5_timealign_hss_a6_metrics.csv`
- `phase5_timealign_hss_a6_comparison.csv`
- `phase5_timealign_hss_a6_summary.csv`
- `phase5_timealign_hss_a6_best_per_setting.csv`
- `phase5_timealign_hss_a6_training_summary.csv`
- raw metrics/logs under ignored `raw/` directory
