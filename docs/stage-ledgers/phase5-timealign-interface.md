# Phase5 TimeAlign Interface Stage Ledger

本文档是 Phase5 中 `TimeAlign + unified multi-horizon interface` 的阶段执行账本。它只记录阶段内
candidate queue、实验决策和未完成任务；完整分析报告保存在 `analysis/`。

## Stage Scope

| Field | Content |
| --- | --- |
| `stage_id` | `phase5-timealign-interface` |
| `paper_mainline_role` | 支撑论文贡献 1：`Capacity-Preserving Prefix-Aware Interface`，并为后续 reliability-aware future supervision routing 提供 carrier |
| `active_question` | 如何设计 SCI 级 unified prediction interface，使 unified multi-horizon 不只是训练 720 再 crop，而是在 head/interface 层面对 multi-prefix evaluation 有结构一致性 |
| `active_carrier` | official-source TimeAlign |
| `entry_evidence` | TimeAlign 是强 baseline；fixed/unified 对比存在 unified decrease；H1/H1C/A2/A3B 表明 head/interface 是当前主要 confounder |
| `stage_exit_condition` | 找到一个通过 narrative gate 且 effectiveness 接近或超过 H1/H1C controls 的 prefix-aware primary interface；或证明该 interface family 不成立并重构论文贡献 |
| `stage_rollback_condition` | candidate queue 中 primary-interface 候选均失败或被 narrative gate 拒绝后，回 Step 2/3 重审 interface problem 是否应作为论文贡献 |

## Decision Cursor

| Field | Content |
| --- | --- |
| `current_11_step` | A3D 处于 Step 6/7/8：已完成 narrative triage，进入 teacher-preserved nested implementation / remote gate |
| `current_candidate` | `A3D_teacher_preserved_nested_primary` |
| `latest_decision` | A3C 不通过 paper-core gate：相对 A2 `+0.07%`，相对 H1 `+0.68%`，相对 H1C `+0.25%`；只否定 warm-start row-slice 分支 |
| `next_required_action` | 启动 A3D teacher-preserved nested gate，检验 teacher consistency 是否能保留 H1 learned function |
| `rollback_point` | A3D 失败时仍不重审 Stage A；先评估 A3E target-conditioned nested primary 的 narrative gate |

## Candidate Queue

| ID | Status | Hypothesis | Narrative Gate | Effectiveness Gate | Blocking Or Next Action | Artifacts |
| --- | --- | --- | --- | --- | --- | --- |
| `A3C_warm_started_nested_primary` | `failed_as_core_candidate` | A2 nested 的主要瓶颈是缺少 learned capacity；从 H1 learned full head warm-start 后，primary nested interface 应显著优于 A2/A3B | passed：保留 primary nested interface，且 learned capacity 来自 trained checkpoint | 未通过：相对 A2 基本持平，未超过 H1/H1C | 保留为 negative evidence：row-slice warm-start 不足以 preserve learned function | `analysis/phase5_timealign_hss_a3c_warm_started_nested_gate_20260701/` |
| `A3D_teacher_preserved_nested_primary` | `narrative_ready` | full head/teacher 保留 dense prediction capacity，nested primary head 学习 prefix-consistent decomposition，可避免直接替换 head 带来的 capacity loss | passed：它直接修复 A3C 暴露的 function-preservation gap，且 nested 仍是 primary interface | 至少优于 A3C/A2；paper-core gate 要求接近或超过 H1/H1C，并证明不是单纯 teacher imitation | 已实现，下一步启动 remote gate | `scripts/remote/run_phase5_timealign_hss_a3d_teacher_preserved_nested_gate.sh` |
| `A3E_target_conditioned_nested_primary` | `proposed` | requested target set/prefix condition 应进入 decoder/head 本身，而不是 condition before 720-step projection 后再 crop | 待评估；叙事潜力强，因为它直接解决 multi-prefix evaluation 与 unified head 不一致 | 应优于 H1 target-set 和 A2 nested，或至少显著降低 unified/fixed gap | 若 A3C 失败，与 A3D 并列做 narrative-gate triage | pending |
| `A3F_teacher_preserved_target_conditioned_nested` | `deferred` | teacher preservation 解决 capacity，target conditioning 解决 requested-prefix specialization，二者组合可能形成最终 paper-core interface | 仅在 A3D/A3E 各自通过 narrative gate 后评估，避免未证实机制堆叠 | 必须超过单机制候选，且不能只靠参数量提升 | 等 A3D/A3E 至少一个 partial/pass 后再考虑 | pending |
| `A3B_nested_residual_gate` | `failed_as_core_candidate` | nested structure 作为 residual path 可修复 dense head 的 prefix behavior | failed：nested 变成 dense head 附属补丁，削弱 primary interface 叙事 | 0/12 win，不能作为 paper-core | 仅保留为 negative evidence/control | `analysis/phase5_timealign_hss_a3b_nested_residual_gate_20260701/` |
| `A3A_dense_initialized_nested_segment` | `failed_as_core_candidate` | 随机 dense row-copy 可作为 capacity-preserving repair | failed：随机初始化复制不等于 learned capacity preservation | 不通过 | 标记为设计错误，不再沿用 | `analysis/phase5_timealign_hss_a3_interface_repair_20260701/` |
| `A2_nested_segment_primary` | `partial_pass` | nested segment primary interface 可能比 full dense head 更适配 multi-prefix evaluation | partial：有结构叙事，但 capacity 不足 | 有正向信号但不足以 paper-core | 作为 A3D/A3E 的机制来源 | `analysis/phase5_timealign_hss_a2_interface_gate_20260630/` |

## Experiment Ledger

| Experiment | Candidate | Role | Result Summary | Decision | Full Report |
| --- | --- | --- | --- | --- | --- |
| H1/H1C interface controls | H1/H1C | control | target-set / row-gated 思路未形成稳定 paper-core interface，但保留强 control | control baseline | `analysis/phase5_timealign_hss_h1c_capacity_preserving_gate_20260701_partial/` |
| A2 interface gate | `A2_nested_segment_primary` | method candidate | nested primary 有局部正向信号，但 capacity/稳定性不足 | `partial_pass` | `analysis/phase5_timealign_hss_a2_interface_gate_20260630/` |
| A3A interface repair | `A3A_dense_initialized_nested_segment` | method candidate | 随机初始化 row-copy 不能证明 capacity preservation | `failed_as_core_candidate` | `analysis/phase5_timealign_hss_a3_interface_repair_20260701/` |
| A3B nested residual | `A3B_nested_residual_gate` | diagnostic/control | residual path 破坏 primary nested 叙事且效果差 | `failed_as_core_candidate` | `analysis/phase5_timealign_hss_a3b_nested_residual_gate_20260701/` |
| A3C warm-started nested | `A3C_warm_started_nested_primary` | method candidate | 相对 A2 `+0.07%`，相对 A3B `-4.06%`，相对 H1 `+0.68%`，相对 H1C `+0.25%` | `failed_as_core_candidate` | `analysis/phase5_timealign_hss_a3c_warm_started_nested_gate_20260701/phase5_timealign_hss_a3c_interpretation.md` |
| A3D teacher-preserved nested | `A3D_teacher_preserved_nested_primary` | method candidate | 待 remote gate | `narrative_ready` | pending |

## Pending Tasks

| Task | Owner | Trigger | Status | Next Action |
| --- | --- | --- | --- | --- |
| 分析 A3C 结果 | Codex | 用户通知远程完成 | `completed` | 已生成 A3C interpretation，并将 A3C 降级为 failed core candidate |
| A3C 失败后的 candidate triage | Codex | A3C effectiveness gate 不通过 | `completed` | A3D 通过 narrative gate，优先启动；A3E 保留 proposed |
| 启动 A3D remote gate | Codex | A3D implementation verification 通过 | `pending` | commit/push 后远程 git pull，预检 GPU 并启动 |
| paper-mainline 同步检查 | Codex | A3C 或 A3D/A3E 产生 pass/fail_as_family 结论 | `pending` | 只有影响贡献边界或主实验安排时修改 `docs/paper-mainline.md` |

## Paper Mainline Sync Log

| Date | Trigger | Paper Section | Change Type | Decision |
| --- | --- | --- | --- | --- |
| 2026-07-01 | A3B 失败后纠正 A3C rollback 表述 | `转向规则` | 转向规则修正 | A3C 失败不等于 Stage A 失败；必须先 triage A3D/A3E |
| 2026-07-01 | 建立研究路径保存体系 | `当前状态` / 文档入口 | 管理机制修正 | paper-mainline 继续管论文总纲，阶段内候选由本 ledger 管理 |
| 2026-07-01 | A3C 不通过 paper-core gate | `当前状态` | 当前阶段状态更新 | active candidate 从 A3C 切换为 A3D，不改变论文贡献边界 |

## Notes For Next Continuation

- A3C 已失败为 paper-core，但只否定 `warm-started primary nested`，不能代表 teacher-preserved 或 target-conditioned 候选。
- 当前优先 A3D；若 A3D 失败，下一步不是“重新审稿 Stage A”，而是评估 A3E target-conditioned nested primary。
- 不允许再把 residual patch 或 shallow initialization 当作 paper-core interface 候选。
- 详细 metric 和诊断报告不要写入本文件，只写 conclusion summary 和 artifact path。
