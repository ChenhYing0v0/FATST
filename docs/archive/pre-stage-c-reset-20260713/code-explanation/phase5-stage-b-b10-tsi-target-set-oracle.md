# Phase5 StageB B10-TSI-C Target-Set Oracle Control Code Explanation

## 诊断位置

| 字段 | 内容 |
| --- | --- |
| `script` | `scripts/analyze_phase5_stage_b_b10_tsi_target_set_oracle.py` |
| `candidate_id` | `B10-TCO` |
| `diagnostic_id` | `B10-TSI-C` |
| `current_step` | StageB Step 3：target-set oracle/control diagnostic |
| `input` | clean A6 checkpoint、train/val/test split、frozen `coeff` 与 `learned_temporal_basis` |
| `output` | `analysis/phase5_stage_b_b10_tsi_target_set_oracle_20260708/` |
| `scope` | Frozen-A6 offline ridge oracle；不训练 forecasting model；只判断当前 late linear readout 是否稳定，不做方向级拒绝 |

## Reader Path

B10-TSI-A/B 已经说明：

- A6 basis 不是 stage-blind；
- 真实 `coeff` 会同时激活多个 stage row subspaces；
- 因而 target-set-blind `history -> coeff/state` 路径值得做最后的 oracle/control 检查。

B10-TSI-C 问的是更严格的问题：

> target-set-aware coefficient readout 的 headroom 是否超过 no-target-set capacity control？

如果不能超过 control，只能说明当前 readout/head 设计不足。若同时出现明显数值病态，不能据此否定
target-set-aware architecture 方向。

## Readout Design

脚本固定 clean A6 的 encoder、`coeff` 和 `learned_temporal_basis`。对每个 segment：

```text
A6:        y_s = basis_s @ coeff
TS-aware:  y_s = basis_s @ (coeff + Linear_s(coeff))
Control:   y_s = basis_s @ (coeff + Linear_shared(coeff))
Pooled-4H: y_s = basis_s @ (coeff + mean_j Linear_pooled_j(coeff))
```

其中：

- `TS-aware`: 四个 segments 各有一个 `Linear_s`，代表 target-set-aware readout；
- `Control`: 一个全局 `Linear_shared` 服务所有 segments；
- `Pooled-4H`: 四个 pooled heads，参数量接近 `TS-aware`，但所有 heads 都在 pooled segments 上训练，预测时不按 target set 选择，只取平均；这是主要 no-target-set capacity control。

## Coefficient Target Construction

诊断不直接拟合 time-domain residual，而是先把 normalized residual 投影回 A6 learned basis coefficient space。

对 segment residual：

```text
residual_s = target_norm_s - pred_norm_s          # [N, L_s]
basis_s                                      # [L_s, 256]
delta_s* = argmin_delta ||basis_s @ delta - residual_s||^2 + lambda ||delta||^2
```

闭式解实现为：

```text
delta_s* = residual_s @ basis_s @ inv(basis_s.T @ basis_s + lambda I)
```

随后用 ridge regression 学习：

```text
delta_s* ~= Linear_s(coeff)
```

## Validation Protocol

为了避免单个 alpha 造成误判，脚本使用：

```text
train fit -> val select alpha -> test report
```

候选 `readout_ridge_alpha` 默认为：

```text
10, 100, 1000, 10000, 100000, 1000000, 10000000
```

`shared_control`、`pooled_multihead_control` 和 `target_set_aware` 各自按 validation MSE 选择 alpha，再在 test split 报告。

## Current Result

| Dataset | target vs shared | target vs pooled-4H | Decision |
| --- | ---: | ---: | --- |
| ETTh2 | `-200.3230%` | `-185.5316%` | target-set-aware 明显更差 |
| ETTm1 | `+0.2220%` | `+0.2812%` | 只有很小正向 |
| Weather | `-23.7469%` | `-26.5683%` | target-set-aware 明显更差 |

[Fact] 三数据集平均 `target_set_aware` 相对 `shared_control` 的额外 reduction 为 `-74.6160%`。

[Fact] 三数据集平均 `target_set_aware` 相对 `pooled_4head` 的额外 reduction 为 `-70.6062%`。

## Code-Theory Consistency

[Intended theory] 如果 requested target set 本身有价值，那么 target-set-specific coefficient readout 应该稳定超过 no-target-set capacity control。

[Code realization] 脚本将 comparison 限制在 frozen A6 basis-coeff interface 内，并加入参数量更接近的 `Pooled-4H` control。

[Observed boundary] 结果不支持当前 `frozen coeff -> Linear_s(coeff)` readout 进入 method design。
ETTm1 有小正向，但 ETTh2 和 Weather 明显负向，且 pooled no-target control 更稳。该结果更像
`readout_or_head_design_wrong` / `optimization_or_numeric_pathology`，不能作为方向拒绝依据。

## Decision

`B10-TSI-C` 对当前 late linear readout 未通过。

后续不应实现该 frozen-coeff linear readout。下一步应做 B10-TSI-D failure attribution：分离
target-set 信息是否有用、信息介入预测的位置是否正确、readout/head 设计是否造成不稳定。更合理的诊断应
把 target query 前移到 history patch memory readout，而不是修补已经生成好的 `coeff`。
