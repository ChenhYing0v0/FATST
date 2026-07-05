# Phase5 Stage A Architecture Exhaustion Audit

本文档记录 A6-QBR 失败后的 Step 2/3 rollback audit。目标是判断 Stage A 的
`unified prediction architecture / prefix-native head` 路线是否仍适合作为 paper-core contribution，
以及下一步是否应转向新的问题定义。

## 11-Step State

| Field | Content |
| --- | --- |
| `current_step` | Step 2/3：重新判断 Stage A architecture problem 是否仍值得作为 paper-core |
| `problem` | 多轮 prefix-native / capacity-native / stability-gated head 均未超过 best controls；继续堆 head 机制可能只产生 engineering variants |
| `existence_evidence` | A5-Q/A5-B capacity collapse，A6-LBF 恢复 capacity 但 0/12 wins，A7DG partial positive 仍弱于 best controls，A8TAG/QBR 失败 |
| `idea` | 将 Stage A 从 paper-core architecture candidate 降级为 problem evidence / control scaffold，重新定义 paper-core |
| `theory_check` | 若 dense-equivalent capacity path 与 selective stability path 均不能跨数据集超过 best controls，则“一个更好的 unified head”不是当前最有证据的核心贡献 |
| `design` | 汇总候选证据，给出 route-level decision 和下一步问题定义约束 |
| `narrative_gate` | Stage A standalone architecture route failed as paper-core；保留为 paper-story motivation 和 control evidence |
| `effectiveness_gate` | failed：没有 active head candidate 同时满足 cross-dataset performance 和 SCI narrative boundary |
| `artifacts` | A5/A6/A7/A8TAG/A6QBR reports and ledgers |
| `decision` | Stage A architecture route 暂停；rollback 到 Step 2/3 重新定义 paper-core，不能继续实现 A5-S/A5-I/A5-M 或 QBR repair |

## Evidence Table

| Candidate | Status | Core Evidence | Interpretation |
| --- | --- | --- | --- |
| A5-Q target-query decoder | `failed_as_core_candidate` | best still `+42.41%/+55.62%` vs best controls in sync gate；diagnostic repair后仍 `0` wins | target-query contract 成立，但 attention decoder capacity/optimization 不足 |
| A5-B fixed/continuous basis | `failed_as_core_candidate` | best `a5b_r128` still `+14.19%`, `0/12` wins | fixed deterministic basis under-capacity |
| A6-DER dense-equivalent row bank | `control_passed_as_capacity_ceiling` | 恢复到接近 dense capacity，但仍无法超过 best controls | capacity 是必要条件，但不是充分 paper-core |
| A6-LBF learned-basis operator | `partial_pass_capacity_recovered_not_core` | r256 接近 A6-DER；相对 best controls 仍 `0/12` wins | learned-basis 是最干净的 architecture evidence，但 performance 不足 |
| A6ST self-teacher | `failed_as_universal_method` | ETTh2 positive，ETTm1/Weather 负向 | stability 有数据集条件性，不能作为 universal head |
| A7DG disagreement gate | `partial_positive_not_paper_core` | vs uniform A6ST `-0.40%`, `11/12` wins；vs best controls `+0.46%`, `2/12` wins | selective stability 是真实现象，但 threshold-gated loss route 不足 |
| A8TAG teacher-advantage gate | `failed_as_core_candidate` | best vs A6-LBF `+0.03%`，弱于 A7DG | supervised teacher advantage 不解释 useful self-teacher |
| A6-QBR query-bilinear readout | `failed_as_core_candidate` | best vs A6-LBF `+35.69%`, `0/12` wins；r512 不改善 | coordinate-generated row-key path 无法承接 dense row dictionary |

## Route-Level Diagnosis

[Strong Evidence] Stage A 的关键发现不是“某个 unified head 成功”，而是：

1. prefix-native contract 本身不够，A5-Q/A5-B 证明没有 dense-level capacity 会 collapse；
2. dense-equivalent capacity 也不够，A6-DER/A6-LBF 证明恢复 capacity 后仍难超过 best controls；
3. official-last stability 是真实因素，但 self-teacher / selective gate 只产生 partial evidence；
4. 将 target-query semantics 重新接入 dense capacity path 的 QBR route 失败，说明当前 target-position
   semantics 不能自然转化为更强 forecasting function class。

[Inference] 当前证据更支持“Stage A 是 paper problem evidence / design constraint”，而不是 paper-core
method 本身。继续沿 head 结构扩展会把贡献变成一串失败/局部修复，不利于 SCI narrative。

## Decision

[Decision] Stage A architecture route 暂停，不再启动新的 paper-core head candidate，除非先有新的 Step 2/3
problem definition 和 Step 4/5 narrative gate。

[Decision] A5-S、A5-I、A5-M 保持 deferred/backlog，不因 QBR 失败而自动升级：

- A5-S 叙事弱，容易变成 generated dense rows；
- A5-I 与当前 observed late drift 风险冲突；
- A5-M 与 ElasTST 接近且实现成本高，不适合作为无新问题定义的下一步。

## Next Problem Definition

下一步应回 Step 2/3，重新定义 paper-core。当前可辩护的方向只有两个：

1. **Stage B / reliability-aware future supervision**：利用 Stage A 证明的 interface/capacity/stability
   constraints，把主问题转为“何时、如何选择或塑造 future-aware supervision / computation path”。这必须避免
   manual routing，并重新过 SCI narrative gate。
2. **Official-last stability/capacity conflict as core problem**：不再承诺一个更强 head，而是研究为什么
   dense-capacity prefix-native head 在 official-last 协议下出现 trajectory drift，以及如何构造 architecture-level
   stability。该方向必须区别于 generic EMA / early stopping。

[Decision] 推荐下一步优先做 `Stage_B_problem_redefinition_and_narrative_gate`。理由是 Stage A 已提供足够的
negative/partial evidence 证明 unified head 不是独立突破口，而 future-aware reliability / conditional computation
仍与原论文目标一致。

## Rules For Next Step

- 不启动 remote experiment，直到新的 Step 2/3 problem definition 和 Step 4/5 narrative gate 写清楚；
- 不把 A7DG/A8TAG/QBR 混合成新 gate；
- 不用 validation/early-stop 作为 paper protocol；
- 不把 Stage A failed heads 写成主贡献，只作为 motivation、constraints 和 ablation evidence；
- 若转向 Stage B，必须先回溯 Stage B ledger/backlog，检查是否有未完成的 candidate 和 diagnostic rules。

