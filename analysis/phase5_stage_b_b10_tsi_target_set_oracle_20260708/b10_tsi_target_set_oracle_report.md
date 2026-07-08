# Phase5 StageB B10-TSI-C Target-Set Oracle Control

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `B10-TCO` |
| `diagnostic_id` | `B10-TSI-C` |
| `current_step` | Step 3：target-set oracle/control diagnostic |
| `problem` | 检查 target-set-specific coefficient readout 的 headroom 是否超过 no-target-set capacity control |
| `scope` | Frozen A6 encoder/basis；offline ridge oracle；不训练新 forecasting model |
| `decision` | 见文末；本诊断只能评价 frozen-coeff linear readout，不能做方向级拒绝 |

## Readout Definition

本诊断固定 clean A6 的 encoder、`coeff` 与 `learned_temporal_basis`，只在 normalized basis-coeff
interface 内拟合 coefficient delta oracle：

```text
A6:        y_s = basis_s @ coeff
TS-aware:  y_s = basis_s @ (coeff + Linear_s(coeff))
Control:   y_s = basis_s @ (coeff + Linear_shared(coeff))
Pooled-4H: y_s = basis_s @ (coeff + mean_j Linear_pooled_j(coeff))
```

`TS-aware` 为每个 target segment 使用不同 readout；`Pooled-4H` 有 4 个 pooled heads，但不按 target set
选择 head，是主要 no-target-set capacity control。

## Summary

| dataset | train_rows | val_rows | test_rows | shared_alpha | pooled_multihead_alpha | target_set_alpha | base_mse | shared_control_mse | pooled_multihead_control_mse | target_set_aware_mse | target_vs_shared_reduction_pct | target_vs_pooled_multihead_reduction_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 20000 | 10000 | 10000 | 10000000.000000 | 10000000.000000 | 10000000.000000 | 2.943057 | 5.223757 | 5.494364 | 15.688144 | -200.322989 | -185.531564 |
| ETTm1 | 20000 | 10000 | 10000 | 1000.000000 | 100.000000 | 1000.000000 | 0.696880 | 0.689380 | 0.689790 | 0.687850 | 0.221997 | 0.281211 |
| Weather | 20000 | 10000 | 10000 | 10.000000 | 10.000000 | 10.000000 | 3.831468 | 3.859691 | 3.773654 | 4.776249 | -23.746947 | -26.568276 |

## Segment Detail

| dataset | segment | base_mse | shared_control_mse | pooled_multihead_control_mse | target_set_aware_mse | target_vs_shared_reduction_pct | target_vs_pooled_multihead_reduction_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | early_0_96 | 1.428422 | 1.477538 | 1.477269 | 1.493019 | -1.047762 | -1.066137 |
| ETTh2 | mid_96_192 | 1.871106 | 2.268370 | 2.293764 | 2.534086 | -11.713994 | -10.477197 |
| ETTh2 | late_192_336 | 2.490717 | 4.022284 | 4.100615 | 10.830132 | -169.253300 | -164.109964 |
| ETTh2 | tail_336_720 | 3.759331 | 7.349711 | 7.821444 | 24.347193 | -231.267344 | -211.287716 |
| ETTm1 | early_0_96 | 0.623478 | 0.620080 | 0.620149 | 0.619309 | 0.124235 | 0.135422 |
| ETTm1 | mid_96_192 | 0.764570 | 0.756956 | 0.758390 | 0.750279 | 0.882203 | 1.069516 |
| ETTm1 | late_192_336 | 0.745676 | 0.740970 | 0.742415 | 0.738775 | 0.296221 | 0.490212 |
| ETTm1 | tail_336_720 | 0.680010 | 0.670465 | 0.670315 | 0.670281 | 0.027495 | 0.005155 |
| Weather | early_0_96 | 1.158788 | 1.280444 | 1.278291 | 1.457231 | -13.806720 | -13.998382 |
| Weather | mid_96_192 | 2.794644 | 2.778411 | 2.704606 | 3.194491 | -14.975470 | -18.113011 |
| Weather | late_192_336 | 5.054692 | 4.958924 | 4.870452 | 5.108758 | -3.021494 | -4.892876 |
| Weather | tail_336_720 | 4.300135 | 4.362610 | 4.253458 | 5.876753 | -34.707279 | -38.164114 |

## Decision

[Fact] Target-set-aware readout vs shared control 的平均额外 reduction 为 `-74.6160%`。
[Fact] Target-set-aware readout vs pooled 4-head no-target control 的平均额外 reduction 为 `-70.6062%`。

[Failure Attribution] `B10-TSI-C` 暴露了 readout/head 设计和数值稳定性问题：target-set-aware readout
没有稳定超过 pooled 4-head no-target capacity control，且 ETTh2/Weather 退化幅度异常。

[Decision] 该结果只能阻断 `frozen coeff -> Linear_s(coeff)` 这个 late linear readout。它不能否定
target-set-aware architecture 方向，也不能阻断 `target query -> history memory -> coeff/state` 的更原生
机制。下一步应做 B10-TSI-D failure attribution，分离 target-set 信息价值、intervention point 和
readout/head 稳定性。
