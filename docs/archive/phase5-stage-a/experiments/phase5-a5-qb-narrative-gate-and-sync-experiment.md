# Phase5-A5: A5-Q / A5-B Narrative Gate 与同步实验计划

## 当前步骤

| Field | Content |
| --- | --- |
| `current_step` | Step 4/5/6：完成 A5-Q 与 A5-B 的 narrative gate，并设计最小同步实验 |
| `stage` | Phase5-A5 |
| `active_carrier` | official-source TimeAlign |
| `source_candidates` | `docs/experiments/phase5-a5-first-principles-unified-head-candidates.md` |
| `decision_date` | 2026-07-02 |
| `status` | `narrative_gate_passed_for_A5-Q_A5-B` |

## 背景问题

[Fact] A5 的直接触发点不是某个单点 metric 失败，而是 A2/A3/A4 系列共同暴露的
interface mismatch：fixed 720-step dense head 的短 horizon 输出只是 crop，不能说明 unified
multi-horizon 模型具备公平的 prefix-native prediction architecture。

[Fact] PCF 已被撤回为 `narrative_rejected_after_review`。主要原因是它依赖 trained dense rows
作为 anchor，并以 correction/residual 形态修补旧路径，不满足“重新设计 unified head”的贡献边界。

[Decision] 本轮只允许 first-principles unified head 进入 paper-core 候选。通过 gate 的候选必须满足：

1. requested prefix 进入主生成路径；
2. 输出是 direct `[B, H, C]`，不是先生成 720 再 crop；
3. prefix consistency 来自 architecture 或主生成 contract；
4. capacity preservation 来自 head/operator class，而不是 pretrained dense anchor；
5. 机制能被写成清晰 SCI contribution，而不是旧模块混合。

## Candidate A5-Q narrative gate

### Idea

[Idea] `A5-Q_elastic_causal_target_query_decoder` 将 future positions 或 small future segments
作为 target queries。queries 从 TimeAlign history tokens 中 cross-attend，再通过 prefix-causal
self-attention 建模 target-target dependency，最后直接生成 requested prefix。

### Theory check

[Strong Evidence] 该候选与 `TIMEPERCEIVER` 的 target-query 思路一致：预测目标不应只由固定
dense rows 表示，而应由 requested target coordinates 显式定义。

[Strong Evidence] 该候选与 `ElasTST` 的 horizon-invariance / structured mask 思路一致：
不同 requested prefixes 的重叠 prefix 应共享同一组 absolute target coordinates 与可见历史信息。

[Hypothesis] 若 query embedding 只依赖 absolute future coordinate `t / max_pred_len`，且 query
self-attention 使用 causal mask，则 `decode(96)` 与 `decode(720)[:, :96]` 在 eval/no-dropout 下
应接近数值一致。这使 prefix consistency 成为 architecture-level invariant。

[Self-Critique] A5-Q 的主要风险是 capacity collapse。若 decoder 太小，它会重复 H1B 的失败：
prefix request 进入了 head，但 target graph 表达力不足。为避免该问题，实验需要同时验证 small
与 wider query decoder，且必须做 prefix-invariance smoke。

### Narrative gate

| Item | Judgment | Reason |
| --- | --- | --- |
| problem motivation | pass | 直接针对 fixed full-head crop 的 interface mismatch |
| novelty boundary | pass | 以 target-query graph 替代 dense row table，不依赖旧 anchor/residual |
| tensor/gradient path | pass | requested prefix 通过 query set 与 causal mask 进入主生成路径 |
| capacity story | conditional pass | 容量来自 query decoder width/depth，需要 small/wide 对照 |
| relation to old candidates | pass | 不是 A2/A3D/A3E 组合，也不是 existing-path selector |
| SCI contribution potential | pass | 可表述为 prefix-elastic target-query decoder |

[Decision] A5-Q 通过 narrative gate，进入 Step 6/7。它是本轮同步实验的 primary architecture
candidate。

## Candidate A5-B narrative gate

### Idea

[Idea] `A5-B_continuous_forecast_basis_operator` 将 unified head 写成 continuous forecast
function/operator。TimeAlign hidden 生成 basis coefficients；requested prefix 只决定 future
coordinate grid；输出由同一 basis/operator 在这些 coordinates 上求值得到。

### Theory check

[Strong Evidence] 该候选把 evaluation horizons 从离散 head rows 中移除，符合当前论文主线的
horizon-agnostic 约束。

[Hypothesis] 若 basis 使用 fixed multi-resolution coordinate features，且 coefficients 由 TimeAlign
hidden 生成，则不同 requested prefixes 的重叠 future coordinates 会得到同一函数值。这天然满足
prefix consistency。

[Self-Critique] A5-B 的风险是 basis bottleneck。若 rank 太低，h96 high-frequency detail 会受损；
若 rank 太高，又可能被审稿人视作 disguised dense head。因此本轮只做 two-rank gate，并记录
basis rank / parameter count / per-prefix behavior。

### Narrative gate

| Item | Judgment | Reason |
| --- | --- | --- |
| problem motivation | pass | 把 full-head row table 改成 forecast function/operator |
| novelty boundary | pass | 不需要 pretrained dense rows、teacher 或 residual correction |
| tensor/gradient path | pass | requested prefix 通过 future coordinate grid 进入主生成路径 |
| capacity story | conditional pass | 容量由 basis rank 与 multi-resolution features 控制 |
| relation to old candidates | pass | 不是旧 nested/row-gate 路线，也不是 horizon-id conditioning |
| SCI contribution potential | pass | 可表述为 continuous prefix-consistent forecast operator |

[Decision] A5-B 通过 narrative gate，进入 Step 6/7。它是本轮同步实验的 lightweight architecture
candidate，也是 A5-Q 的效率/结构对照。

## 本轮不进入同步实验的候选

| Candidate | Status | Reason |
| --- | --- | --- |
| `A5-S_step_specific_hypernetwork_head` | `control_deferred` | 容易被解释为 generated dense rows，贡献边界弱于 A5-B；待 A5-B 结果后再决定是否作为 capacity control |
| `A5-I_cumulative_innovation_process_decoder` | `control_deferred` | 与 output-process evidence 对齐，但 cumulative drift 风险较大；适合作为后续 trajectory-process control |
| `A5-M_masked_future_placeholder_head` | `backlog_diagnostic` | 与 ElasTST implementation motif 较近，且改动重；不适合作为当前最快 gate |

## 同步实验设计

### 最小实验矩阵

[Decision] 远程算力充足时，本轮同时启动 A5-Q 与 A5-B，但矩阵仍保持 gate 规模。目的不是完整
paper table，而是快速筛出哪个 head family 有继续深挖价值。

| Arm | Role | Key setting | Gate purpose |
| --- | --- | --- | --- |
| `A5-B-r64` | lightweight candidate | continuous basis rank 64 | 检查 basis/operator 是否已足够替代 dense head |
| `A5-B-r128` | capacity control | continuous basis rank 128 | 区分 basis bottleneck 与机制失败 |
| `A5-Q-seg48-small` | primary candidate | segment query length 48, shallow decoder | 检查 target-query graph 的最小可用性 |
| `A5-Q-seg24-wide` | capacity control | segment query length 24, wider decoder | 检查 A5-Q 是否受 decoder capacity 限制 |

### Dataset universe

[Decision] 首轮仍使用 Phase5 当前 universe：

- `ETTh2`
- `ETTm1`
- `Weather`

原因：

- `ETTm1` 与 `Weather` 能覆盖较难/较慢数据集；
- `ETTh2` 保留与前序 Phase5 gate 的可比性；
- 3 datasets × 4 horizons 足以判断是否存在全局潜力，不把 gate 扩成完整论文实验。

### Effectiveness gate

本轮 effectiveness gate 在 Step 9/10 执行，暂定为：

1. 任一 A5 arm 的 ALL mean MSE 相对 H1/H1C/A3D strong controls 不劣于 `+0.3%`，或至少明显优于 A2/A3C/A3E；
2. 至少 `6/12` MSE wins vs 当前 strongest available control；
3. `decode(96)` 与 `decode(720)[:, :96]` 的 eval prefix mismatch 接近 numerical zero；
4. h96 不系统性损伤，h720 late segment 不出现明显 drift；
5. 若 A5-B-r128 明显强于 A5-B-r64，说明 basis bottleneck 真实存在；若二者都弱，A5-B 回 Step 4/5；
6. 若 A5-Q-wide 明显强于 A5-Q-small，说明 target-query family 仍值得扩容；若二者都弱，A5-Q 回 Step 4/5。

## Rollback 规则

- 若 A5-Q 与 A5-B 均未达到最低 effectiveness gate，但 prefix-invariance smoke 成立，回 Step 4/5 评估 A5-S/A5-I 是否值得进入第二批；
- 若 prefix-invariance smoke 不成立，回 Step 6 修正 architecture contract，不允许远程结果被解释为方法结果；
- 若只有 A5-B 通过，论文 head 方向倾向 `continuous forecast operator`；
- 若只有 A5-Q 通过，论文 head 方向倾向 `prefix-elastic target-query decoder`；
- 若两者都通过，优先选择叙事边界更清晰、参数更稳、诊断更可解释的一支作为 paper-core，另一支作为 ablation/control。

## 当前决策

[Decision] A5-Q 与 A5-B 均通过 narrative gate，可进入实现、本地 smoke 与远程同步 gate。

[Decision] A5-S/A5-I/A5-M 暂不进入本轮远程同步实验，避免把 candidate queue 扩成无边界 sweep。

[Next Action] 实现 A5-B 与 A5-Q 的最小 head modes，完成 shape/prefix-invariance smoke 后，提交并推送代码，再按远程 policy 做 GPU preflight 与同步启动。
