# Phase5 StageB B10 Target-Set Conditioned Operator

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `B10-TCO` |
| `current_step` | Step 2/3：target-set-native multi-horizon problem redefinition |
| `problem` | A6-LBF-r256 是 prefix-compatible 720-step trajectory operator，但 requested horizon / target set 没有进入 computation graph；短 horizon 只是从同一条 720 trajectory 上 prefix slicing |
| `existence_evidence` | A6 统一模型成立但 multi-horizon 原生性不足；B7 显示 multi-prefix supervision 的 tail weakness；B9-SCF 显示单纯 stage-token coefficient modulation 被 no-stage control 解释 |
| `idea` | 将 requested target set $J$ 原生输入 basis-coeff operator，使模型按 $J$ 生成预测，同时保持 prefix consistency |
| `theory_check` | 尚未完成；当前只建立问题边界与诊断计划 |
| `design` | 候选方向是 target-set conditioned basis-coeff interface，不是 full-720 prediction 后 slicing，也不是 residual correction |
| `narrative_gate` | `not_evaluated` |
| `effectiveness_gate` | `not_evaluated` |
| `artifacts` | 本文档；后续需要 B10-TSI diagnostic |
| `decision` | `problem_redefinition_ready`; 下一步做 B10-TSI problem diagnostic |

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
   - 判断不同 future regions 是否已有自然 subspace separation。

2. `coeff_usage_by_stage`
   - 对 A6 checkpoint 的 prediction contribution `basis[t,k] * coeff[b,c,k]` 做 stage-wise energy 分解；
   - 判断同一批 coefficient dimensions 是否被所有 stages 共同使用，还是已经存在 stage-specialized dimensions。

3. `target_set_oracle_control`
   - 用 frozen A6 hidden 和 basis，比较 target-set-specific readout 的 oracle headroom 与 no-target-set capacity control；
   - 不能拟合 output residual correction；只能在 basis-coeff coupling 内做诊断。

4. `prefix_consistency_contract`
   - 任何候选 forward graph 都必须报告：

```text
max_abs(pred_H96 - pred_H720[:, :96])
MSE(pred_H96, pred_H720[:, :96])
```

## Next Decision

B10 当前不进入 implementation。下一步必须先写并运行 `B10-TSI` diagnostic。若 diagnostic 证明 target-set
interface 的问题存在且不能被 no-target-set capacity control 解释，再进入 Step 4-6 method/narrative gate。

若 `B10-TSI` 也被 no-target-set control 阻断，则 StageB 应回到 Step 2/3 继续寻找第二主创新点，或将
B7 objective optimization 降级为小贡献继续处理。
