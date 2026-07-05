# Phase5 Post-A8TAG Candidate Backtracking

本文档执行 A8TAG 失败后的 stage-ledger 回溯规则。目标不是立即启动下一轮 remote gate，而是先
检查未执行候选是否仍有 SCI narrative gate，并明确下一步 rollback point。

## 11-Step State

| Field | Content |
| --- | --- |
| `current_step` | Step 4/5：A8TAG failed 后回溯未执行候选并重评理论可行性 |
| `problem` | A6-LBF 恢复了 dense-level capacity，但仍未超过 best controls；A6ST/A7DG 证明 selective stability 有局部价值，A8TAG 证明 supervised teacher advantage 不是可靠 gate |
| `existence_evidence` | A8TAG best 相对 A6-LBF `+0.03%`、相对 A7DG 变差约 `+0.47%`；A7DG 的 ETTh2 gain 来自 high-disagreement stability，而非 teacher 当前 supervised risk 更低 |
| `idea` | 停止 self-teacher gate sweep，重新回到 architecture-side capacity/native target-query 机制；优先评估 `A6-QBR_query_bilinear_readout` |
| `theory_check` | 若 A6-LBF 的 learned basis 已接近 dense row dictionary，但仍缺 best-control wins，则下一步应测试 query-conditioned row interaction，而不是再改 teacher loss |
| `design` | 本文只做 triage；下一文档应单独写 A6-QBR design/gate，包括 tensor contract、prefix-invariance check、parameter budget 和 controls |
| `narrative_gate` | A6-QBR conditional pass；A5-S/A5-I/A5-M 不进入下一轮 primary implementation |
| `effectiveness_gate` | pending；A6-QBR 若进入 remote gate，必须与 A6-LBF-r256、A6-DER、A7DG best 同口径比较 |
| `artifacts` | A5 candidate proposal、A6 capacity-native mechanism doc、A6 partial-pass diagnostic、A8TAG interpretation |
| `decision` | 激活 `A6-QBR_query_bilinear_readout` 为下一步 Step 4/5->Step 6 候选；不启动 teacher/self-distillation 新变体 |

## Evidence Synthesis

[Fact] A6-LBF-r256 在 A6 capacity gate 中已接近 dense-equivalent control A6-DER，说明 A5-B 的主要
问题不是 prefix contract，而是 fixed/operator capacity。

[Fact] A6 partial-pass diagnostic 显示 ETTh2 的 official-last drift 是主要剩余缺口；ETTm1/Weather
的 last-vs-best drift 很小。

[Fact] A7DG 能把 uniform A6ST 的跨数据集损伤降下来，并在 ETTh2 保留部分 stability gain；但它仍是
loss-side gate，且相对 best controls 只有 `2/12` wins。

[Fact] A8TAG falsify 了一个自然解释：teacher 在当前 supervised prefix 上更接近 label 并不等价于
teacher trajectory 是有价值的 consistency target。Binary gate 的高激活反而伤害 ETTm1/Weather。

[Inference] 当前路线的瓶颈不应继续表述为“如何选择 teacher”。更合理的 Step 4 问题是：
在 prefix-native dense-capacity 已基本恢复后，如何让 requested target positions 以更强的方式进入
row/operator generation，同时不退化为 fixed dense row table。

## Candidate Backtracking

| Candidate | Status After Backtracking | Reason |
| --- | --- | --- |
| `A5-S_step_specific_hypernetwork_head` | `control_deferred_keep_deferred` | 它最容易被审稿人理解为 generated dense rows；在 A6-LBF 已接近 dense row dictionary 后，A5-S 的 contribution boundary 更弱 |
| `A5-I_cumulative_innovation_process_decoder` | `diagnostic_deferred` | trajectory-process 叙事仍有价值，但当前 ETTh2 的核心问题就是 late drift；cumulative operator 有放大 bias 的结构风险，不适合作为下一轮 primary |
| `A5-M_masked_future_placeholder_head` | `backlog_diagnostic` | prefix-native 叙事强，但与 ElasTST 机制相近且实现成本高；更适合在需要 mask/invariance diagnostic 时使用 |
| `A6-QBR_query_bilinear_readout` | `selected_for_narrative_design` | 它继承 A5-Q 的 target-query 语义，同时用 A6-LBF/A6-DER 已验证的 dense-equivalent bilinear capacity path 避免 query decoder collapse |

## Selected Candidate: A6-QBR

[Idea] 将 future coordinate/query 只用于生成 row key，而不是让 query decoder 独自承担 forecasting：

```text
hidden: [B, C, R]
feature = P(hidden): [B, C, K]
query_t = q(t / 720): [Dq]
row_key_t = G(query_t): [K]
y_hat_t = dot(row_key_t, feature) + b_t
```

对 requested prefix `H`，只生成 `t=1..H` 的 `row_key_t`，直接输出 `[B,H,C]`。如果
`G(query_t)` 是 learnable coordinate-to-row-key function，则它保留 target-position semantics；如果
`K` 足够大，它接近 A6-LBF/dense row-bank 的 capacity。

## Narrative Gate

| Item | Assessment |
| --- | --- |
| problem motivation | strong：A5-Q 语义强但 capacity collapse，A6-LBF capacity 恢复但 target-query semantics 弱 |
| mechanism novelty | medium-strong：target-query semantic indexing + dense-equivalent bilinear readout |
| tensor/gradient path | strong：每个 requested future position 的 row key 直接参与 supervised loss |
| capacity preservation | medium-high：由 `K` 与 row-key generator 决定，可与 A6-LBF 同 rank 对照 |
| contribution boundary | conditional pass：必须证明不是 A6-LBF 的改名，也不是 dense row table reparameterization |

## Required Design Before Implementation

下一步 A6-QBR design doc 必须先回答：

1. `row_key_t` 是独立 learnable embedding、coordinate MLP，还是两者组合；若使用独立 embedding，
   容易退化为 dense rows。
2. `K` 与 A6-LBF-r256/r512 如何对齐，避免参数量 confounder。
3. `decode(96)` 与 `decode(720)[:, :96]` 的 prefix-invariance smoke 如何实现。
4. 是否需要 nonlinear interaction；若需要，如何避免变成 A5-Q decoder 的低容量 attention path。
5. remote gate 的 minimal arms：至少包含 A6-QBR-r256、A6-QBR-r512、A6-LBF-r256 control，不包含
   teacher/self-distillation 变体。

## Decision

[Decision] 下一步进入 `A6-QBR_query_bilinear_readout` 的 Step 6 design/code-theory gate。未通过设计
gate 前不实现，不启动 remote。若 A6-QBR narrative/design gate 失败，则回 Step 2/3 重审 Stage A
是否应继续作为 paper-core architecture，而不是继续扩展 head variants。

