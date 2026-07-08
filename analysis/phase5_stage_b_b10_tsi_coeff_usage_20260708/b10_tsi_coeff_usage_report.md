# Phase5 StageB B10-TSI-B Coefficient Usage Diagnostic

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `B10-TCO` |
| `diagnostic_id` | `B10-TSI-B` |
| `current_step` | Step 3：target-set interface diagnostic |
| `problem` | A6 basis 已有 stage-differentiated row-space geometry，但 `learned_basis_coeff(hidden)` 仍生成 target-set-blind coefficient/state |
| `scope` | 读取真实 forward batch 的 `coeff`，分析其在 stage row subspaces 中的使用方式；不训练新模型 |
| `decision` | 见文末；本诊断不能单独升级为 method result |

## 诊断定义

对 clean A6 checkpoint，在 test split 上取若干 batch，执行 A6 forward 到：

```text
coeff = learned_basis_coeff(hidden)              # [B, C, 256]
basis_s = learned_temporal_basis[start:end]      # [L_s, 256]
Q_s = row_space(basis_s)                         # [256, rank]
projection_share_s = ||coeff @ Q_s||^2 / ||coeff||^2
```

同时计算同一个 `coeff` 通过四个 segment basis 生成的 normalized output energy share。

## Rank64 Summary

| dataset | rank | batches | projection_share_mean | projection_share_min_mean | projection_share_max_mean | projection_pair_cosine_mean | output_energy_entropy_mean | output_energy_max_stage_share_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 64 | 8 | 0.3882 | 0.3287 | 0.5000 | 0.3759 | 0.7969 | 0.5564 |
| ETTm1 | 64 | 8 | 0.4950 | 0.4379 | 0.5519 | 0.4702 | 0.8958 | 0.4895 |
| Weather | 64 | 8 | 0.2764 | 0.1764 | 0.4245 | 0.1639 | 0.9042 | 0.4416 |

## Rank64 Segment Detail

| dataset | projection_share_early_0_96 | projection_share_mid_96_192 | projection_share_late_192_336 | projection_share_tail_336_720 | output_energy_share_early_0_96 | output_energy_share_mid_96_192 | output_energy_share_late_192_336 | output_energy_share_tail_336_720 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 0.5000 | 0.3287 | 0.3492 | 0.3749 | 0.1434 | 0.1092 | 0.1914 | 0.5560 |
| ETTm1 | 0.5083 | 0.4481 | 0.4717 | 0.5519 | 0.1628 | 0.1488 | 0.1993 | 0.4891 |
| Weather | 0.4245 | 0.3111 | 0.1927 | 0.1772 | 0.2523 | 0.1882 | 0.1680 | 0.3915 |

## Decision

[Fact] Rank64 下，平均 `projection_share_mean` 为 `0.3865`，说明每个 `coeff` 在多个 stage row subspaces 上都有可观投影。
[Fact] `projection_pair_cosine_mean` 为 `0.3367`，说明这些 stage projections 不是同一方向的简单重复。
[Fact] `output_energy_entropy_mean` 为 `0.8656`，`output_energy_max_stage_share_mean` 为 `0.4958`。

[Decision] `B10-TSI-B` 支持继续：同一个 target-set-blind `coeff` 同时激活多个 stage row subspaces，且 output energy 不是单 stage 主导。这与 B10 的收窄问题一致：requested target set 应进入 `history -> coeff/state` 路径。

[Boundary] 这仍不是方法通过。下一步必须做 target-set oracle/control，证明 target-set-aware readout 不能被 no-target-set capacity control 解释。
