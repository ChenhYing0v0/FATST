# Phase5 StageB B10-TSI-B Coefficient Usage Diagnostic Code Explanation

## 诊断位置

| 字段 | 内容 |
| --- | --- |
| `script` | `scripts/analyze_phase5_stage_b_b10_tsi_coeff_usage.py` |
| `candidate_id` | `B10-TCO` |
| `diagnostic_id` | `B10-TSI-B` |
| `current_step` | StageB Step 3：target-set interface diagnostic |
| `input` | clean A6 checkpoint、test split batch、`learned_temporal_basis` 与真实 forward 中的 `coeff` |
| `output` | `analysis/phase5_stage_b_b10_tsi_coeff_usage_20260708/` |
| `scope` | 不训练新模型；不拟合 residual；只分析 target-set-blind `coeff` 如何被不同 stage row subspaces 使用 |

## Reader Path

B10-TSI-A 已经说明：A6 的 basis 不是 stage-blind，不同 future stage 在 coefficient 维度上的 row spaces
有明显差异。

B10-TSI-B 接着问：

> 真实 forward 产生的同一个 `coeff[b,c]`，是否同时在多个 stage row subspaces 上有强投影？

如果答案是肯定的，说明 A6 当前确实让一个 target-set-blind coefficient/state 同时服务多个不同的
stage geometry；这支持继续做 target-set interface diagnostic。

## Tensor Flow

脚本复用 clean A6 的实际 forward path：

```text
batch_x                              # [B, 720, C]
hidden = encoder(batch_x)            # [B, C, R]
coeff = learned_basis_coeff(hidden)  # [B, C, 256]
basis = learned_temporal_basis       # [720, 256]
```

对每个 segment：

```text
basis_s = basis[start:end]           # [L_s, 256]
Q_s = row_space(basis_s)             # [256, rank]
```

其中 `Q_s` 来自 `basis_s` 的 SVD right singular vectors，表示该 stage 在 coefficient 维度上读取的主要
subspace。

## Statistic Definitions

### `projection_share_s`

对每个 `coeff` 向量：

```text
projection_share_s = ||coeff @ Q_s||^2 / ||coeff||^2
```

它表示同一个 coefficient 有多少能量落在 stage `s` 的 row subspace 里。若多个 stage 都有高
`projection_share_s`，说明同一个 coefficient 同时服务多个 stage-specific coefficient axes。

### `projection_pair_cosine`

脚本把 `coeff` 分别投影回不同 stage row subspaces：

```text
P_s coeff = (coeff @ Q_s) @ Q_s^T
```

再计算不同 `P_s coeff` 的 cosine。该值低，表示不同 stage 从同一个 coefficient 中读取的不是同一方向的简单重复。

### `output_energy_entropy`

脚本计算同一个 `coeff` 通过四段 basis 生成的 normalized output energy：

```text
energy_s = ||basis_s @ coeff||^2
share_s = energy_s / sum_s energy_s
```

`output_energy_entropy` 是四个 `share_s` 的归一化 entropy。高 entropy 说明输出能量分布在多个 stage，
不是单一 stage 主导。

## Current Result

Rank64 summary：

| Dataset | projection share | projection cosine | output entropy | max stage share |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | `0.3882` | `0.3759` | `0.7969` | `0.5564` |
| ETTm1 | `0.4950` | `0.4702` | `0.8958` | `0.4895` |
| Weather | `0.2764` | `0.1639` | `0.9042` | `0.4416` |

[Fact] 三个数据集的 rank64 mean projection share 为 `0.3865`，说明真实 `coeff` 在多个 stage row
subspaces 上都有可观投影。

[Fact] mean projection cosine 为 `0.3367`，说明这些 projections 不是同一方向的重复。

[Fact] output entropy 为 `0.8656`，max stage share 为 `0.4958`，说明同一个 coefficient 生成的 output
energy 不是单 stage 主导。

## Code-Theory Consistency

[Intended theory] B10 的收窄问题是：basis 已有 stage geometry，但 `history -> coeff/state` 仍是
target-set-blind。

[Code realization] 脚本直接读取真实 A6 forward 中的 `coeff`，并用 checkpoint 中实际学到的 basis row spaces
衡量这个 `coeff` 如何被各 stage 使用。

[Observed support] 结果支持继续 B10-TSI：同一个 `coeff` 同时激活多个低同向性的 stage subspaces。

[Boundary] 这不是 method evidence。它不能证明 target-set-conditioned architecture 会提升 MSE，也不能排除
“只是增加 readout capacity 就能解释”的可能性。

## Follow-Up Result

后续 `B10-TSI-C target_set_oracle_control` 已完成：

1. 比较了 target-set-aware coefficient readout 与 no-target-set capacity controls；
2. 加入了参数量更接近的 `Pooled-4H` control；
3. 结果显示 frozen-coeff linear readout 未稳定超过 controls，且出现 ETTh2/Weather 病态；
4. 因此当前 readout/head 设计被阻断，但不能据此否定 target-set-aware architecture 方向。
