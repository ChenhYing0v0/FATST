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
| `current_11_step` | Stage A contribution re-evaluation 已完成 Step 2/3；Stage A standalone method route 暂停 |
| `current_candidate` | `Stage_A_contribution_reevaluation` |
| `latest_decision` | 当前证据不足以支撑 standalone interface method；interface 保留为 paper-level problem evidence 和 carrier/control constraint；主方法转入 Stage B future-supervision gradient routing |
| `next_required_action` | 新建/切换 Stage B ledger，先做 `Reliability-Aware Future Supervision Routing` 的 Step 2/3/4 diagnostic plan，再决定是否实现 |
| `rollback_point` | 回 Step 2/3：论文贡献边界重构为 `Problem Formulation and Interface-Controlled Evaluation` + `Reliability-Aware Future Supervision Routing`；若 Stage B 发现收益依赖 head choice，再回 Stage A 重新设计 first-principles interface |

## Candidate Queue

| ID | Status | Hypothesis | Narrative Gate | Effectiveness Gate | Blocking Or Next Action | Artifacts |
| --- | --- | --- | --- | --- | --- | --- |
| `A3C_warm_started_nested_primary` | `failed_as_core_candidate` | A2 nested 的主要瓶颈是缺少 learned capacity；从 H1 learned full head warm-start 后，primary nested interface 应显著优于 A2/A3B | passed：保留 primary nested interface，且 learned capacity 来自 trained checkpoint | 未通过：相对 A2 基本持平，未超过 H1/H1C | 保留为 negative evidence：row-slice warm-start 不足以 preserve learned function | `analysis/phase5_timealign_hss_a3c_warm_started_nested_gate_20260701/` |
| `A3D_teacher_preserved_nested_primary` | `partial_pass` | full head/teacher 保留 dense prediction capacity，nested primary head 学习 prefix-consistent decomposition，可避免直接替换 head 带来的 capacity loss | passed：它直接修复 A3C 暴露的 function-preservation gap，且 nested 仍是 primary interface | 部分通过：`w03` overall 接近/略超 H1/H1C，但 ETTm2 仍失败 | 保留为 partial evidence，不直接作为 paper-core | `analysis/phase5_timealign_hss_a3d_teacher_preserved_nested_gate_20260701/` |
| `A3E_target_conditioned_nested_primary` | `failed_as_core_candidate` | requested target set/prefix condition 应进入 decoder/head 本身，而不是 condition before 720-step projection 后再 crop | passed：它直接解决 multi-prefix evaluation 与 unified head 不一致；warm-start 只作为与 A3C 对齐的 initialization control，不作为机制贡献 | 未通过：ALL 相对 A3C 只有约 `-0.25%`，相对 A3D/H1 仍弱；ETTm1 上不如 A3C | 保留为 negative evidence；下一步做 reliability diagnostic，不直接进入 A3F | `analysis/phase5_timealign_hss_a3e_ettm1_replacement_gate_20260701/phase5_timealign_hss_a3e_ettm1_deep_dive.md` |
| `A3F_teacher_preserved_target_conditioned_nested` | `deferred` | teacher preservation 解决 capacity，target conditioning 解决 requested-prefix specialization，二者组合可能形成最终 paper-core interface | 仅在 A3D/A3E 各自通过 narrative gate 后评估，避免未证实机制堆叠 | 必须超过单机制候选，且不能只靠参数量提升 | 等 A3D/A3E 至少一个 partial/pass 后再考虑 | pending |
| `A4_interface_reliability_diagnostic` | `diagnostic_only` | A3D/A3E/A3C/A2/H1/H1C 的相对表现可能说明 capacity-preserving path 的可靠性随 future context 变化，而不是存在一个 universal head | not_required：launch 前定义为 diagnostic，不可直接升级为 paper-core | 已完成：best path 分散，oracle 上限存在但较小；只证明 reliability 差异存在，不证明可部署 routing | 进入 A4R signal diagnostic；禁止把 dataset/horizon 手工选择当作方法 | `analysis/phase5_timealign_hss_a4_interface_reliability_diagnostic_20260701/phase5_timealign_hss_a4_interface_reliability_diagnostic.md` |
| `A4R_reliability_signal_diagnostic` | `diagnostic_only` | 若可观测 signals 能预测 path reliability 或 gap-to-best，则 Stage A 可重构为 `Reliability-Aware Capacity-Preserving Interface`，避免弱化为 manual routing | not_required：使用现有日志的 diagnostic-only | 未通过：ALL-level 最强 signal Spearman `0.321`，且 dataset 内方向不稳定 | 不进入 routing；下一步 A4S 设计更明确的 validation-prefix signal export | `analysis/phase5_timealign_hss_a4r_reliability_signal_diagnostic_20260701/phase5_timealign_hss_a4r_reliability_signal_diagnostic.md` |
| `A4S_validation_prefix_signal_export` | `diagnostic_only` | 当前日志过粗，缺少 prefix-wise validation behavior；若 prefix validation residual / teacher-student disagreement 能解释 path reliability，才有资格进入 routing method | not_required：diagnostic-only；若后续作为 paper-core signal，必须重新过 narrative gate | 未通过：ALL strongest Spearman `0.388`；ETTh2/ETTm1/Weather 的 top signals 方向不一致 | 触发 Step 2/3 rollback；不进入 learned routing | `analysis/phase5_timealign_hss_a4s_validation_prefix_signal_export_20260702/phase5_timealign_hss_a4s_validation_prefix_signal_export.md` |
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
| A3D teacher-preserved nested | `A3D_teacher_preserved_nested_primary` | method candidate | `w03` 相对 A3C `-0.73%`，相对 H1 `-0.06%`，相对 H1C `-0.48%`，但 ETTm2 仍负；deep dive 判断 teacher preservation 有效但缺 target-prefix specialization | `partial_pass` | `analysis/phase5_timealign_hss_a3d_teacher_preserved_nested_gate_20260701/phase5_timealign_hss_a3d_deep_dive.md` |
| A3E target-conditioned nested | `A3E_target_conditioned_nested_primary` | method candidate | warm/scratch ALL 相对 A3C `-0.25/-0.26%`，但相对 A3D/H1 仍弱；ETTm1 上 A3C 仍最强 | `failed_as_core_candidate` | `analysis/phase5_timealign_hss_a3e_ettm1_replacement_gate_20260701/phase5_timealign_hss_a3e_ettm1_deep_dive.md` |
| A4 interface reliability diagnostic | `A4_interface_reliability_diagnostic` | diagnostic | best path map 分散；ALL best static 为 A3D，oracle 相对 best static `-0.431%`，说明 reliability 差异真实但手工 routing 叙事弱 | `diagnostic_only_completed` | `analysis/phase5_timealign_hss_a4_interface_reliability_diagnostic_20260701/phase5_timealign_hss_a4_interface_reliability_diagnostic.md` |
| A4R existing-log signal diagnostic | `A4R_reliability_signal_diagnostic` | diagnostic | 现有 training-log signals 解释力不足：ALL 最强 Spearman `0.321`，dataset 内方向不稳定 | `diagnostic_only_failed` | `analysis/phase5_timealign_hss_a4r_reliability_signal_diagnostic_20260701/phase5_timealign_hss_a4r_reliability_signal_diagnostic.md` |
| A4S validation-prefix signal export | `A4S_validation_prefix_signal_export` | diagnostic | prefix-wise validation signals 仍不足：ALL 最强 teacher-student MAE Spearman `0.388`，dataset-level 方向不一致 | `diagnostic_only_failed` | `analysis/phase5_timealign_hss_a4s_validation_prefix_signal_export_20260702/phase5_timealign_hss_a4s_deep_dive.md` |
| Stage A contribution re-evaluation | `Stage_A_contribution_reevaluation` | reviewer-style decision | Stage A 不能作为 standalone interface method contribution；interface 问题必须保留为 problem evidence 与 carrier/control constraint；主方法转入 Stage B future-supervision gradient routing | `stage_a_method_route_paused` | `analysis/phase5_stage_a_contribution_reevaluation_20260702/stage_a_contribution_reevaluation.md` |

## Pending Tasks

| Task | Owner | Trigger | Status | Next Action |
| --- | --- | --- | --- | --- |
| 分析 A3C 结果 | Codex | 用户通知远程完成 | `completed` | 已生成 A3C interpretation，并将 A3C 降级为 failed core candidate |
| A3C 失败后的 candidate triage | Codex | A3C effectiveness gate 不通过 | `completed` | A3D 通过 narrative gate，优先启动；A3E 保留 proposed |
| 启动 A3D remote gate | Codex | A3D implementation verification 通过 | `completed` | 已在 3090 启动，等待用户通知完成后同步分析 |
| 分析 A3D 结果 | Codex | 用户通知远程完成 | `completed` | A3D 标为 partial_pass；下一步进入 A3E |
| 启动 A3E ETTm1 replacement remote gate | Codex | A3E implementation verification 通过且 ETTm1 presets/sync 已补齐 | `completed` | 已在 3090 启动，等待用户通知完成后同步分析 |
| 分析 A3E ETTm1 replacement gate | Codex | 用户通知远程完成 | `completed` | A3E 标为 failed_as_core_candidate；下一步回 Step 2/3/4 做 reliability diagnostic |
| A4 interface reliability diagnostic | Codex | A3E 失败后 rollback 到 Step 2/3/4 | `completed` | 已生成 A4 diagnostic；下一步 A4R signal diagnostic |
| A4R reliability signal diagnostic 设计 | Codex | A4 证明 best-path reliability 差异存在但不能直接手工 routing | `completed` | 现有日志信号不足，不启动 routing |
| A4S validation-prefix signal export 设计 | Codex | A4R 证明现有 logs 太粗 | `completed` | 已实现 exporter/wrapper/sync/analyzer |
| 启动 A4S remote diagnostic-only run | Codex | A4S 本地验证通过 | `completed` | 21/21 diagnostics 已完成并同步 |
| Stage A contribution 重评估 | Codex | A4S signal-existence gate 未通过 | `completed` | 已完成 reviewer-style re-evaluation；下一步建立 Stage B diagnostic plan |
| Stage B diagnostic plan | Codex | Stage A standalone method route 暂停，论文主方法转入 future-supervision gradient routing | `pending` | 新建或切换 Stage B ledger，定义 problem/evidence/idea/theory check/narrative gate |
| paper-mainline 同步检查 | Codex | A4 将 Stage A 从 universal head 改为 reliability-aware interface 诊断 | `completed` | 已同步当前状态与贡献边界，不改变 working title |

## Paper Mainline Sync Log

| Date | Trigger | Paper Section | Change Type | Decision |
| --- | --- | --- | --- | --- |
| 2026-07-01 | A3B 失败后纠正 A3C rollback 表述 | `转向规则` | 转向规则修正 | A3C 失败不等于 Stage A 失败；必须先 triage A3D/A3E |
| 2026-07-01 | 建立研究路径保存体系 | `当前状态` / 文档入口 | 管理机制修正 | paper-mainline 继续管论文总纲，阶段内候选由本 ledger 管理 |
| 2026-07-01 | A3C 不通过 paper-core gate | `当前状态` | 当前阶段状态更新 | active candidate 从 A3C 切换为 A3D，不改变论文贡献边界 |
| 2026-07-01 | A3D partial pass | `当前状态` | 当前阶段状态更新 | active candidate 从 A3D 切换为 A3E，不改变论文贡献边界 |
| 2026-07-01 | A3E ETTm1 replacement gate 失败 | `当前状态` | 当前阶段状态更新 | 不直接进入 A3F；先回 Step 2/3/4 做 interface reliability diagnostic |
| 2026-07-01 | A4 reliability diagnostic 完成 | `当前状态` / `修订后的论文主线` | 贡献边界微调 | Stage A 不写成 universal head，也不写成手工 routing；下一步先验证可观测 reliability signals |
| 2026-07-01 | A4R existing-log signal diagnostic 完成 | `当前状态` | 转向规则细化 | 现有 logs 不足以解释 path reliability；下一步只做 validation-prefix signal export，不进入 routing |
| 2026-07-02 | A4S validation-prefix signal diagnostic 完成 | `当前状态` / `贡献边界` | rollback 触发 | prefix-wise validation signals 未通过；Stage A 回 Step 2/3 重审贡献 1 |
| 2026-07-02 | Stage A contribution re-evaluation 完成 | `当前状态` / `预期贡献` / `方法边界` | 贡献边界重构 | Stage A standalone interface method route 暂停；interface 保留为 problem evidence 和 carrier/control constraint；主方法转入 Stage B |

## Remote Launch Log

| Date | Candidate | Commit | GPU Preflight | Remote PID | Output Path | Launcher Log |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-07-01 | `A3D_teacher_preserved_nested_primary` | `354e895` | GPU 0/1/2 all free: `18 MiB used`, `24107 MiB free` each | `3848377` | `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_a3d_teacher_preserved_nested_gate` | `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_a3d_teacher_preserved_nested_gate/_launcher/a3d_launcher.log` |
| 2026-07-01 | `A3E_target_conditioned_nested_primary_ettm1_replacement` | `0a59296` | GPU 0/1/2 all free: `18 MiB used`, `24107 MiB free` each | `3942988` | `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_a3e_target_conditioned_nested_gate` plus ETTm1 reference roots | `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_a3e_ettm1_replacement_gate/_launcher/a3e_ettm1_launcher.log` |
| 2026-07-02 | `A4S_validation_prefix_signal_export` | `9c86588` | GPU 0/1/2 all free before launch: `18 MiB used`, `24107 MiB free` each | `1255624` | `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_a4s_validation_prefix_signal_export` | `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_a4s_validation_prefix_signal_export/_launcher/a4s_launcher.log` |

## Notes For Next Continuation

- A3C 已失败为 paper-core，但只否定 `warm-started primary nested`，不能代表 teacher-preserved 或 target-conditioned 候选。
- A3D 是 partial pass，说明 function preservation 有效但不足以解决 ETTm2 和 target-prefix specialization。
- 当前优先 A3E；A3E 的 warm arm 只能解释为对齐 A3C 的 initialization control，不能把 warm-start 当作机制贡献。
- 2026-07-01 artifact audit 未发现远端 A3E 结果目录或 launcher log；A3E 仍是 pending remote gate，不应被当作已完成实验分析。
- 用户要求加入 ETTm1 替换 ETTm2；A3E analysis universe 改为 `ETTh2 + ETTm1 + Weather`。ETTm1 过去没有 references，必须先补跑 fixed/H1/H1C/A2/A3C/A3D。
- A3E 已失败为 paper-core：target conditioning 进入 primary nested head 的增量太小，且 ETTm1 上 A3C 仍是最强候选。
- 不要直接进入 A3F `teacher_preserved + target_conditioned`；两个组成机制在 ETTm1 上没有同时正向，叠加会违反 narrative gate。
- A4 诊断显示 best path 分散，但 oracle routing 上限较小：ALL 相对 best static A3D 只有 `-0.431%`。这支持 reliability 问题真实存在，但不支持把 dataset/horizon 手工选择路径写成最终方法。
- A4R 使用现有 training logs 后发现 signals 太粗：ALL 最强 Spearman 只有 `0.321`。下一步必须新增 prefix-wise validation diagnostic export，而不是直接实现 routing。
- A4S 使用 prefix-wise validation signals 后仍失败：ALL 最强 Spearman 只有 `0.388`，且 dataset-level top signals 方向不一致。不能继续 existing-path routing。
- Stage A Step 2/3 重评估已完成：standalone interface method route 暂停；interface 不删除，而是保留为 problem evidence 与 HSS carrier/control constraint。
- 下一步必须建立 Stage B `Reliability-Aware Future Supervision Routing` ledger 和 diagnostic plan；先验证 future supervision 的 gradient-path conflict/reliability 问题是否真实存在，再实现方法。
- 不允许再把 residual patch 或 shallow initialization 当作 paper-core interface 候选。
- 详细 metric 和诊断报告不要写入本文件，只写 conclusion summary 和 artifact path。
