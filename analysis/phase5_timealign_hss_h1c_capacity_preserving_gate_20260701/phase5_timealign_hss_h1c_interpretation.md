# Phase5 H1C Capacity-Preserving Prefix Decoder Interpretation

## 结论

[Decision] H1C 是 `partial evidence`，但不通过 paper-core gate。当前最佳 arm 是
`row_gated_dense_head_multiprefix`，它证明 capacity-preserving readout modulation 比直接
替换 dense head 更稳，但没有显著超过 H1 `target_set_decoder_multiprefix`。

## Gate Summary

| Arm | ALL vs H1 target-set | ALL vs fixed | H1 wins | fixed wins | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `dense_prefix_residual_adapter_multiprefix` | `+5.04%` | `+0.94%` | `0/12` | `3/12` | fail |
| `row_gated_dense_head_multiprefix` | `+0.43%` | `-3.29%` | `5/12` | `7/12` | partial evidence |
| `prefix_adapter_shared_dense_multiprefix` | `+10.89%` | `+6.30%` | `0/12` | `0/12` | fail |

## Dataset-Level Reading

- ETTh2: `row_gated_dense_head_multiprefix` 保留 unified benefit，相对 fixed 为 `-11.42%`，
  但仍低于 H1 target-set `+1.44%`。
- ETTm2: `row_gated_dense_head_multiprefix` 相对 H1 target-set 为 `-0.17%`，但相对 fixed
  仍为 `+1.65%`，只比 H1 的 `+1.81%` 有轻微改善。
- Weather: `row_gated_dense_head_multiprefix` 基本追平 H1 target-set，均值为 `+0.01%`，
  相对 fixed 为 `-0.11%`。

## Mechanism Judgment

[Fact] `dense_prefix_residual_adapter_multiprefix` 和 `prefix_adapter_shared_dense_multiprefix`
都失败，说明在 dense 720 base 上增加可学习 residual 或 hidden adapter 很容易破坏 TimeAlign
已有 readout/representation balance。

[Strong Evidence] `row_gated_dense_head_multiprefix` 是唯一稳定 arm。它不改 hidden，也不增加
new output residual，只用 `[step/720, target_prefix/720]` 对 dense rows 做小幅 multiplicative
calibration，因此最不容易破坏 base capacity。

[Limit] 但 `row_gated_dense_head_multiprefix` 只是稳定 control，不构成新的 paper-core decoder。
它没有在 ALL 上超过 H1 target-set，也没有把 ETTm2 fixed gap 从 `+1.81%` 明显压低。

## Decision

H1C decision: `capacity_preserving_readout_partial_fail_row_gate_control`。

不继续扩大 readout/head sweep。下一步 rollback 到 11-step 的 Step 2/3/6：重新确认当前
TimeAlign-HSS 的问题支点是否应从 prediction interface 转向 future supervision reliability。
具体下一步是 Stage B / D1：在 H1 target-set 或 row-gated control 上诊断 future units 的
reconstruction difficulty、alignment consistency、residual volatility 与 segment-level unified
gap 的关系。
