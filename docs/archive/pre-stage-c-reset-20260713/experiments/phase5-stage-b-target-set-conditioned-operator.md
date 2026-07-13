# Phase5 StageB B10 Target-Set Conditioned Operator

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `B10-TCO` |
| `current_step` | Superseded by B11-ESA after user reframed StageB away from explicit stage/target conditioning |
| `problem` | A6-LBF-r256 是 prefix-compatible 720-step trajectory operator，但 requested horizon / target set 没有进入 computation graph；短 horizon 只是从同一条 720 trajectory 上 prefix slicing |
| `existence_evidence` | A6 统一模型成立但 multi-horizon 原生性不足；B7 显示 multi-prefix supervision 的 tail weakness；B9-SCF 显示单纯 stage-token coefficient modulation 被 no-stage control 解释；B10-TSI-A/B 支持问题收窄；B10-TSI-C/D 暴露 frozen/offline readout 病态，不能作为方向拒绝依据 |
| `idea` | 将 requested target set $J$ 原生输入 basis-coeff operator，使模型按 $J$ 生成预测，同时保持 prefix consistency |
| `theory_check` | B10-TSI-D 说明 frozen ridge memory-level readout 仍不稳定；后续若继续，必须转向 native trainable target-query memory readout，而不是继续 offline oracle |
| `design` | 候选方向是 target-set conditioned basis-coeff interface，不是 full-720 prediction 后 slicing，也不是 residual correction |
| `narrative_gate` | `pending`; only a native target-query memory readout may enter Step 4-6, frozen/offline readout is blocked |
| `effectiveness_gate` | `not_evaluated` |
| `artifacts` | 本文档；`analysis/phase5_stage_b_b10_tsi_basis_geometry_20260708/b10_tsi_basis_geometry_report.md`; `analysis/phase5_stage_b_b10_tsi_coeff_usage_20260708/b10_tsi_coeff_usage_report.md`; `analysis/phase5_stage_b_b10_tsi_target_set_oracle_20260708/b10_tsi_target_set_oracle_report.md`; `analysis/phase5_stage_b_b10_tsi_failure_attribution_20260708/b10_tsi_failure_attribution_report.md`; `analysis/phase5_stage_b_b10_tsi_failure_attribution_rank16_20260708/b10_tsi_failure_attribution_report.md` |
| `decision` | `offline_readout_route_blocked_but_direction_not_rejected`; no longer active as next route because StageB has moved to B11 emergent basis-subspace utilization |

## Motivation

[Fact] 当前 clean A6-LBF-r256 的 prediction head 是：

```text
hidden = encoder(history)                  # [B, C, R]
coeff = learned_basis_coeff(hidden)        # [B, C, K]
y[t,c] = learned_temporal_basis[t] @ coeff[b,c] + bias[t]
```

其中 `K=256`。这可以解释为：

```text
prediction_curve[b,c,:] = sum_k coeff[b,c,k] * temporal_atom_k[:]
```

因此 A6 的强点是把 dense future-step projection 分解为：

- shared learned temporal coordinate system / temporal atoms；
- sample-wise and channel-wise coefficients；
- prefix-native slicing of the same temporal coordinate system。

[Caution] 但这不等于强意义上的 native multi-horizon architecture。当前 requested horizon `H` 不进入
representation、coefficient generation 或 basis-coeff coupling；模型先形成同一条最长 `720` future trajectory，
再返回 prefix：

```text
y_96  = y_720[:, :96]
y_192 = y_720[:, :192]
y_336 = y_720[:, :336]
y_720 = y_720
```

这使 A6 更准确地说是 `prefix-compatible unified trajectory operator`，不是完整的
`target-set-conditioned multi-horizon operator`。

## Literature Evidence

本轮方向落地不能只参考本地 notes。外部网络与本地 notes 共同支持“target set / future target positions
应进入 computation graph”：

- TimePerceiver: 使用 target timestamp queries 作为 decoder queries，从 encoded input representations 中取信息；
- ElasTST: 使用 future placeholders 与 structured masks 支持 varied-horizon / horizon-invariant forecasting；
- MQ-RNN: 使用 horizon-specific contexts；
- Temporal Fusion Transformer: 明确处理 known future inputs 和 multi-horizon architecture；
- 本地 note `Papers/srp-step-specific-representation.md`: 指出 shared representation 服务多步预测存在
  step-specific representation bottleneck。

[Boundary] B10 不复制这些完整架构。B10 的边界是 A6 learned-basis operator 的 `target set -> basis-coeff
interface`，不是 generalized forecasting full framework，也不是 future value placeholder encoder。

## Problem Statement

StageA 的 A6-LBF-r256 已证明：

> 一个统一的 learned-basis temporal-coordinate operator 可以在多 horizon benchmark 上整体优于
> fixed-horizon TimeAlign specialist。

StageB 现在要补的不是“再从 720 输出上切 prefix”，而是：

> requested target set $J$ 如何原生进入 prediction operator，使模型按 $J$ 生成预测，同时保持 prefix
> consistency？

形式上，当前 A6 是：

```text
f_A6(history) -> y_{1:720}
return y_J = y_{1:720}[J]
```

B10 想研究的是：

```text
f_B10(history, J) -> y_J
```

其中 $J$ 可以是：

```text
{1..96}, {1..192}, {1..336}, {1..720}
```

或更一般的 target index set。

## Prefix-Invariant vs Horizon-Conditioned

B10 必须先选论文立场。

### Option A: Prefix-Invariant Target-Set Operator

同一个 future time 在不同 requested horizon 下应有一致输出：

```text
f(history, {1..96}) == f(history, {1..720})[:, :96]
```

优点：

- 与 unified prediction 主线一致；
- 可避免“同一未来点因请求长度不同而预测不同”的解释负担；
- 与 ElasTST 的 horizon-invariant motivation 对齐。

### Option B: Horizon-Conditioned Target-Set Operator

同一个 future time 可因 requested horizon 不同而改变：

```text
f(history, {1..96}) != f(history, {1..720})[:, :96]
```

优点是更接近 horizon specialist amortization；风险是论文必须解释为什么同一 future timestamp 可以有不同
answers。当前项目更适合 Option A。

[Decision] B10 默认采用 `prefix-invariant target-set computation`：requested target set 进入 forward graph，
但后续 target queries 不允许改写已有 prefix outputs。

## Candidate Tensor Path

### Per-Target Query Form

```text
hidden = encoder(history)                         # [B, C, R]
memory = MemoryProject(hidden)                    # [B, C, M, D]
q_J = TargetQuery(J)                              # [|J|, D]
target_state = CrossRead(q_J, memory)             # [B, C, |J|, D]
coeff_t = CoeffHead(target_state)                 # [B, C, |J|, K_small]
basis_t = BasisRow(q_t) or learned_basis[t]        # [|J|, K_small]
y_t = basis_t @ coeff_t
```

调用 `H=96` 时只生成 96 个 target queries；调用 `H=720` 时生成 720 个 target queries。区别是 requested
target set 决定 forward graph，而不是先计算 720 再裁剪。

### Prefix Consistency Constraint

为了不破坏 unified prediction：

- target query `q_t` 可以 cross-read history memory；
- `q_t` 不允许 attend 到 `q_{t'>t}`；
- 最小版本可以完全禁止 target-query self-attention，只允许 independent target queries read history。

这样 `H=720` 的后续 target positions 不会改写 `H=96` 的 prefix outputs。

### Segment Target-Set Form

更贴合当前 horizons 的轻量版本：

```text
segment_queries = q_{0:96}, q_{96:192}, q_{192:336}, q_{336:720}
segment_state_s = CrossRead(q_s, history_memory)
coeff_s = CoeffHead(segment_state_s)
y[t in segment_s] = learned_basis[t] @ coeff_s
```

这比 per-step query 成本更低，也与 B9-SGC 的 stage-gradient conflict 对齐。但必须避免退化成 B9-SCF：
stage query 不应只调制已有 `coeff_base`，而应决定 target set 如何读取 history memory。

## Difference From B9-SCF

B9-SCF 是：

```text
coeff_s = coeff_base * (1 + gate_s * delta_s)
y[t in stage_s] = basis[t] @ coeff_s
```

small gate 已证明该做法被 no-stage control 阻断。原因是 `basis[t]` 已经携带 future position/stage 信息，
再给 `coeff` 加 stage token 很容易只是 extra capacity。

B10-TCO 必须改变问题位置：

```text
target set J -> target query/state -> read history -> basis-coeff coupling -> y_J
```

也就是说，future stage 不只是调制已有 coefficient，而是决定“请求的 target set 如何从 history 中形成预测
state”。这是更原生的 multi-horizon architecture 问题。

## B10-TSI-A Basis Geometry Result

[Fact] `B10-TSI-A` 已读取三个 clean A6 checkpoint 的 `learned_temporal_basis: [720, 256]`，检查
segment rank、atom stage share 和 stage row-space overlap。

| Dataset | top64 atom entropy | stage-specialized rate | rank32 row-space overlap |
| --- | ---: | ---: | ---: |
| ETTh2 | `0.8108` | `0.0156` | `0.1324` |
| ETTm1 | `0.8764` | `0.0000` | `0.1510` |
| Weather | `0.8658` | `0.0000` | `0.1368` |

[Interpretation] 这个结果不支持“basis 没有 future-stage 信息”的简单叙事。高能 atoms 并不强烈局部化到
单一 stage，但不同 stage 在 coefficient 维度上的 row subspace overlap 很低，说明 A6 basis 已经形成
stage-differentiated coefficient geometry。

[Decision] B10 的问题因此必须收窄为：

> requested target set 没有进入 `history -> coeff/state` 生成路径；单一 target-set-blind coefficient
> vector 必须同时服务多个 stage row subspaces。

后续不得退回 B9 式 “在既有 `coeff_base` 后加 stage modulation”。下一步是 B10-TSI-B：在真实 forward batch
中检查 `coeff` 如何被不同 stage row subspaces 使用，并加入 no-target-set capacity control。

## B10-TSI-B Coefficient Usage Result

[Fact] `B10-TSI-B` 在 clean A6 checkpoint 的 test split 上读取真实 forward 中的
`coeff: [B, C, 256]`，计算其在四个 stage row subspaces 上的 projection share，并检查同一个 `coeff`
生成的 segment output energy 分布。

Rank64 summary：

| Dataset | projection share | projection cosine | output entropy | max stage share |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | `0.3882` | `0.3759` | `0.7969` | `0.5564` |
| ETTm1 | `0.4950` | `0.4702` | `0.8958` | `0.4895` |
| Weather | `0.2764` | `0.1639` | `0.9042` | `0.4416` |

[Interpretation] 同一个 target-set-blind `coeff` 同时在多个 stage row subspaces 上有可观投影，而且这些
投影不是同一方向的简单重复；output energy 也不是单一 stage 主导。这支持 B10 的收窄问题：requested
target set 应进入 `history -> coeff/state` 路径。

[Boundary] B10-TSI-B 仍不是 method evidence。它只支持进入 `B10-TSI-C target_set_oracle_control`。
若 no-target-set capacity control 能解释 target-set-aware readout 的 headroom，则 B10 不得进入 Step 4-6。

## B10-TSI-C Target-Set Oracle/Control Result

[Fact] `B10-TSI-C` 固定 A6 encoder、`coeff` 与 `learned_temporal_basis`，在 normalized basis-coeff
interface 内拟合 coefficient delta oracle：

```text
A6:        y_s = basis_s @ coeff
TS-aware:  y_s = basis_s @ (coeff + Linear_s(coeff))
Control:   y_s = basis_s @ (coeff + Linear_shared(coeff))
Pooled-4H: y_s = basis_s @ (coeff + mean_j Linear_pooled_j(coeff))
```

其中 `Pooled-4H` 有 4 个 pooled heads，但不按 target set 选择 head，是主要 no-target-set capacity control。
脚本使用 `train fit -> val select alpha -> test report`。

| Dataset | target vs shared | target vs pooled-4H |
| --- | ---: | ---: |
| ETTh2 | `-200.3230%` | `-185.5316%` |
| ETTm1 | `+0.2220%` | `+0.2812%` |
| Weather | `-23.7469%` | `-26.5683%` |

[Failure Attribution] B10-TSI-C 不能否定 target-set-aware 方向。它只说明当前
`frozen coeff -> Linear_s(coeff)` readout/head 设计存在明显缺陷：信息介入太晚、readout 过线性，且
ETTh2/Weather 出现数值/泛化病态。该诊断只能阻断这个 readout 设计，不能阻断 `target query -> history
memory -> coeff/state` 这类更原生的 target-set-aware architecture。

## Problem Evidence From Existing Artifacts

### A6 Is Strong But Still Prefix-Slicing

Clean A6 vs fixed-horizon per-horizon TimeAlign:

| Dataset | MSE wins | Mean MSE vs fixed |
| --- | ---: | ---: |
| ETTh2 | `4/4` | `-10.53%` |
| ETTm1 | `3/4` | `-1.64%` |
| Weather | `2/4` | `-0.22%` |
| Overall | `9/12` | `-4.13%` |

[Inference] A6 的 learned-basis operator 是有效的，但 Weather 和 ETTm1-720 仍显示 unified prefix trajectory
不是全局压倒 fixed-horizon specialist。

### Tail Region Remains Weak Under Current Multi-Prefix Training

B7-UPO 诊断显示：

| Region | Mean relative MSE vs fixed |
| --- | ---: |
| early `0-96` | `-3.57%` |
| mid `96-192` | `-4.81%` |
| late `192-336` | `-4.32%` |
| tail `336-720` | `-0.16%` |

[Inference] 当前 prefix slicing + multi-prefix objective 仍对不同 target regions 产生不均衡优化。B7 作为 objective
小贡献候选可保留，但 B10 关注的是 architecture：requested target set 应进入 computation graph。

### B9-SCF Blocked By No-Stage Control

B9 small gate:

| Comparison | Overall MSE wins | Mean relative MSE |
| --- | ---: | ---: |
| `b9_fsn_scf` vs `a6_clean` | `12/12` | `-0.13%` |
| `b9_no_stage` vs `a6_clean` | `12/12` | `-0.13%` |
| `b9_fsn_scf` vs `b9_no_stage` | `2/12` | `+0.0036%` |

[Decision] 这说明“把 stage token 塞进 coefficient”不够。B10 必须让 target set 决定 history readout 或
basis-coeff coupling，而不是只提供 coefficient-space extra capacity。

## B10-TSI Diagnostic Plan

在实现任何 B10 method 前，先做 `B10-TSI`: Target-Set Interface diagnostic。

需要回答：

1. A6 的 error / gradient 是否主要来自 target-set blind coupling，而不是单纯 horizon-distance？
2. target-set query 若只作为 no-stage capacity control 是否已经解释收益？
3. prefix consistency 能否在 target-set-native computation 中保持？

### Proposed Offline Diagnostics

1. `basis_stage_subspace_audit`
   - 对 `learned_temporal_basis` 做 stage-wise Gram/cosine/subspace overlap；
   - 判断不同 future regions 是否已有自然 subspace separation；
   - `B10-TSI-A` 已完成：basis 不是 stage-blind，但 stage 信息主要在 basis row-space geometry 中，未进入
     target-set-conditioned history readout。

2. `coeff_usage_by_stage`
   - 对 A6 checkpoint 的 prediction contribution `basis[t,k] * coeff[b,c,k]` 做 stage-wise energy 分解；
   - 判断同一批 coefficient dimensions 是否被所有 stages 共同使用，还是已经存在 stage-specialized dimensions；
   - `B10-TSI-B` 已完成：同一个 `coeff` 同时激活多个低同向性 stage subspaces，支持继续进入 oracle/control。

3. `target_set_oracle_control`
   - 用 frozen A6 hidden 和 basis，比较 target-set-specific readout 的 oracle headroom 与 no-target-set capacity control；
   - 不能拟合 output residual correction；只能在 basis-coeff coupling 内做诊断。
   - `B10-TSI-C` 已完成：target-set-aware readout 未能超过 no-target-set controls，但 ETTh2/Weather
     出现 pathology；只能阻断 frozen-coeff linear readout，不能阻断 B10 方向。

4. `failure_attribution_memory_readout`
   - 比较 `coeff_late`、`memory_pool`、`memory_plus_coeff` 三个 feature sources；
   - 用 rank-truncated basis row-space target 避免 full coefficient inverse；
   - 加入 `shared_control`、`pooled_multihead_control`、`wrong_target_control` 和 shrinkage target-set readout；
   - `B10-TSI-D` 已完成：rank64 和 rank16 均显示 offline target-set readout 仍输给 pooled control，
     且 memory-level route 仍有 pathology。结论是 frozen/offline readout route 被阻断，不能作为 method，
     也不能方向级拒绝 native trainable target-query architecture。

5. `prefix_consistency_contract`
   - 任何候选 forward graph 都必须报告：

```text
max_abs(pred_H96 - pred_H720[:, :96])
MSE(pred_H96, pred_H720[:, :96])
```

## Next Decision

B10 当前不进入 implementation，但不能从 B10-TSI-C 推出 direction rejection。

`B10-TSI-D` 已完成。结论不是“target-set-aware 方向失败”，而是：

1. frozen/offline ridge readout 不是可靠的 method path；
2. full independent target head 与 shrinkage target head 都没有稳定超过 pooled no-target control；
3. memory-level pooling 仍不是原生 target-query memory readout，只能说明该 diagnostic path 被阻断。

该文档原本的下一步是二选一：

1. 写 Step 4-6 native trainable target-query memory readout 的 narrative/method gate，明确它如何不同于
   frozen ridge readout，并保留 no-target query implementation control；
2. 若该 narrative gate 不成立，则 B10 回到 StageB Step 2/3，不再用 offline readout oracle 继续消耗。

[Superseded] 用户随后明确不希望主路线依赖显式 stage/target-set encoding。StageB 当前 active route
已转向 `B11-ESA`: 利用 A6 basis 自发形成的 continuous subspace geometry，而不是把 stage/target-set
作为硬条件注入。
