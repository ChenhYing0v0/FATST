# Phase5 StageB B10-TSI-A Basis Geometry Diagnostic

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `B10-TCO` |
| `diagnostic_id` | `B10-TSI-A` |
| `current_step` | Step 3：problem-existence diagnostic |
| `problem` | A6-LBF 的 requested target set 不进入 computation graph；需要先判断 learned basis 是否已经足以承载 stage/target-set 差异 |
| `scope` | checkpoint-only basis geometry audit；不训练新模型，不读取 test labels，不评估 prediction MSE |
| `decision` | `partial_support_continue_tsi`; basis 不是 stage-blind，但 target set 仍缺席于 history-to-coeff/state path |

## 诊断定义

本诊断读取 clean A6 checkpoint 中的 `learned_temporal_basis: [720, 256]`，从三个角度检查 basis 的 stage 结构：

- `segment_effective_rank`: 每个 future segment 的 basis 子矩阵需要多少 rank 才能覆盖自身能量；
- `atom_stage_share`: 每个 temporal atom 的能量是否集中在单一 stage，还是横跨多个 stage；
- `row_space_overlap`: 不同 stage 在 coefficient 维度上的 row subspace 是否高度重叠。

如果 basis 已经强烈 stage-specialized，那么继续把 stage 信息注入 `coeff` 可能容易退化为 extra capacity。
如果 basis 的主要 atoms 和 row spaces 横跨多个 stage，则更支持 B10 的问题定义：target set 应该进入
`history -> target state -> basis-coeff coupling`，而不是只在 output prefix 上 slicing。

## Dataset Summary

| dataset | global_effective_rank_95 | tail_effective_rank_95 | short_mean_effective_rank_95 | top64_atom_entropy_mean | top64_atom_max_stage_share_mean | top64_atom_stage_specialized_rate_0p70 | rank32_pair_overlap_mean | rank64_pair_overlap_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 202 | 170 | 86.6667 | 0.8108 | 0.5777 | 0.0156 | 0.1324 | 0.2649 |
| ETTm1 | 200 | 176 | 87.0000 | 0.8764 | 0.5129 | 0.0000 | 0.1510 | 0.2862 |
| Weather | 204 | 178 | 88.3333 | 0.8658 | 0.5254 | 0.0000 | 0.1368 | 0.2712 |

## Global Basis Rank

| dataset | horizon | rank | effective_rank_90 | effective_rank_95 | effective_rank_99 | top32_energy_pct | top64_energy_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 720 | 256 | 173 | 202 | 239 | 32.9690 | 52.4180 |
| ETTm1 | 720 | 256 | 172 | 200 | 236 | 29.1067 | 49.9757 |
| Weather | 720 | 256 | 176 | 204 | 240 | 27.8336 | 48.7287 |

## Segment Geometry

| dataset | segment | length | segment_energy_share_pct | effective_rank_95 | top32_energy_pct | top64_energy_pct |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | early_0_96 | 96 | 11.6543 | 79 | 57.7383 | 87.5781 |
| ETTh2 | mid_96_192 | 96 | 12.7371 | 77 | 61.2080 | 89.0228 |
| ETTh2 | late_192_336 | 144 | 19.6684 | 104 | 53.7919 | 79.4848 |
| ETTh2 | tail_336_720 | 384 | 55.9402 | 170 | 43.0026 | 63.4872 |
| ETTm1 | early_0_96 | 96 | 14.1912 | 77 | 59.7760 | 88.6562 |
| ETTm1 | mid_96_192 | 96 | 14.0115 | 78 | 60.1275 | 88.5559 |
| ETTm1 | late_192_336 | 144 | 20.1627 | 106 | 50.0238 | 77.5878 |
| ETTm1 | tail_336_720 | 384 | 51.6346 | 176 | 35.0247 | 58.2371 |
| Weather | early_0_96 | 96 | 13.5065 | 79 | 58.4084 | 87.6429 |
| Weather | mid_96_192 | 96 | 13.7880 | 79 | 58.0456 | 87.8137 |
| Weather | late_192_336 | 144 | 20.1264 | 107 | 49.6997 | 77.3371 |
| Weather | tail_336_720 | 384 | 52.5791 | 178 | 34.0847 | 57.1891 |

## Rank-32 Stage Row-Space Overlap

| dataset | left_segment | right_segment | mean_squared_canonical_corr | min_canonical_corr |
| --- | --- | --- | --- | --- |
| ETTh2 | early_0_96 | mid_96_192 | 0.1165 | 0.0053 |
| ETTh2 | early_0_96 | late_192_336 | 0.1212 | 0.0127 |
| ETTh2 | early_0_96 | tail_336_720 | 0.1266 | 0.0024 |
| ETTh2 | mid_96_192 | late_192_336 | 0.1419 | 0.0062 |
| ETTh2 | mid_96_192 | tail_336_720 | 0.1360 | 0.0112 |
| ETTh2 | late_192_336 | tail_336_720 | 0.1519 | 0.0147 |
| ETTm1 | early_0_96 | mid_96_192 | 0.1297 | 0.0024 |
| ETTm1 | early_0_96 | late_192_336 | 0.1494 | 0.0067 |
| ETTm1 | early_0_96 | tail_336_720 | 0.1596 | 0.0198 |
| ETTm1 | mid_96_192 | late_192_336 | 0.1462 | 0.0038 |
| ETTm1 | mid_96_192 | tail_336_720 | 0.1572 | 0.0090 |
| ETTm1 | late_192_336 | tail_336_720 | 0.1637 | 0.0095 |
| Weather | early_0_96 | mid_96_192 | 0.1313 | 0.0077 |
| Weather | early_0_96 | late_192_336 | 0.1305 | 0.0011 |
| Weather | early_0_96 | tail_336_720 | 0.1438 | 0.0043 |
| Weather | mid_96_192 | late_192_336 | 0.1286 | 0.0062 |
| Weather | mid_96_192 | tail_336_720 | 0.1436 | 0.0031 |
| Weather | late_192_336 | tail_336_720 | 0.1431 | 0.0075 |

## Observed Decision

[Fact] 三个数据集 top64 atom 的 mean normalized entropy 为 `0.8510`，
`max_stage_share >= 0.70` 的 mean rate 只有 `0.0052`。
这说明高能 temporal atoms 并没有强烈局部化到单一 future stage。

[Fact] 但 stage row-space overlap 不高：rank32 mean overlap 为 `0.1400`，
rank64 mean overlap 为 `0.2741`。这说明不同 future segments 在 coefficient 维度上读取
的是明显不同的 row subspaces。

[Interpretation] 因此 B10 不能再用“basis 不包含 stage 信息”作为问题叙事。更严谨的说法是：
A6 的 `learned_temporal_basis` 已经形成 stage-differentiated coefficient geometry，但
`learned_basis_coeff(hidden)` 仍然只产生一个 target-set-blind coefficient vector。requested target set
没有进入 `history -> coeff/state` 生成路径，模型只能让同一个 coefficient 同时服务多个 stage row subspaces。

[Decision] B10-TSI-A 支持继续做 B10-TSI-B，但只支持更收窄的问题：
`target-set-conditioned history readout / coefficient state`，而不是继续向现有 coefficient 后面加
stage modulation。下一步必须检查真实 forward batch 中 coefficient 能量如何被不同 stage subspaces 使用，
并加入 no-target-set capacity control。

## Interpretation Rule

- [Supports B10 problem] high `top64_atom_entropy_mean` and high `rank32_pair_overlap_mean`: basis atoms/subspaces are shared across stages, so requested target set is not natively resolved by basis alone.
- [Weakens B10 problem] high `top64_atom_stage_specialized_rate_0p70` and low row-space overlap: basis already creates stage-specific coefficient axes; B10 must then show a stronger history-target readout problem.
- [Boundary] This diagnostic does not compare B10 with no-target-set controls. Passing this audit only allows B10 to proceed to target-set interface diagnostic, not to implementation.
