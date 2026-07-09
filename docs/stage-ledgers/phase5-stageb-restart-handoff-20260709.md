# Phase5 StageB Restart Handoff 2026-07-09

本文档用于新对话重启研究。它不是完整实验报告，而是当前 StageA/StageB 的最小上下文、已裁决路线、禁止误读点和下一步研究入口。

## Minimal Reading Order

新对话建议按以下顺序读取：

1. `docs/stage-ledgers/phase5-stageb-restart-handoff-20260709.md`
2. `docs/paper-mainline.md`
3. `docs/research-roadmap.md`
4. `docs/stage-ledgers/phase5-timealign-interface.md`

需要复查细节时再读：

- `analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706/clean_a6_rerun_report.md`
- `analysis/phase5_stage_b_b12_stbo_rank_diagnostic_20260708/b12_stbo_rank_diagnostic_report.md`
- `analysis/phase5_stage_b_b12_stbo_rank_diagnostic_20260708/b12_stbo_rank_deep_analysis.md`
- `docs/code-explanation/phase5-clean-timealign-a6-lbf.md`
- `docs/code-explanation/phase5-stage-b-b12-stbo-rank-diagnostic.md`

## Current State

| Field | Content |
| --- | --- |
| `current_stage` | Phase5 StageA accepted; StageB rolled back after B12-STBO rank/capacity diagnostic |
| `current_11_step` | StageB Step 11: rollback to Step 2/3 architecture search |
| `active_carrier` | `A6-LBF-r256` pure learned-basis forecast operator |
| `accepted_paper_core` | `A6-LBF-r256` only |
| `open_problem` | Find a second architecture-level contribution for unified multi-horizon forecasting |
| `do_not_launch_next` | Do not launch another B12-STBO rank sweep or full matrix without a new operator hypothesis |

## StageA Fixed Claim

[Fact] StageA 当前已收敛到 clean `A6-LBF-r256`。它移除了 future reconstruction/alignment branch，使用：

- `readout_mode=learned-basis-forecast-operator`
- `basis_rank=256`
- `pred_loss_mode=multi-prefix`
- `w_recon=0.0`
- `w_align=0.0`

[Strong Evidence] Clean A6 相对 fixed-horizon per-horizon TimeAlign official-last 的整体 mean MSE 为 `-4.13%`，`9/12` MSE wins；相对 official unified TimeAlign 的整体 mean MSE 为 `-1.75%`，`11/12` MSE wins。

[Decision] 论文 Contribution 1 可写为 learned-basis unified forecast operator。它是 prefix-compatible 720-step trajectory operator：模型生成同一条 future trajectory，再按 requested horizon 返回 prefix。它不是 target-set-conditioned forecaster，也不是 teacher、EMA、future-recon auxiliary loss 或 residual repair。

## StageB Current Decision

[Decision] StageB 尚未形成第二个 accepted paper-core method。B12-STBO rank/capacity diagnostic 后，StageB 必须回到 Step 2/3 architecture search。

当前回滚不是因为“native multi-horizon architecture 不可能”，而是因为已测试的 STBO family 未能证明比 A6 full-basis operator 更有效、更清晰：

- [Fact] 第一轮 B12 small gate 中，`stbo_shared` vs A6 为 `+1.59%` mean MSE、`0/12` wins；`stbo_bank4` vs A6 为 `+1.98%`，只有 Weather 上 3 个很小 wins。
- [Fact] learned STBO 没有超过 fixed local DCT；`stbo_bank4` entropy 约 `0.999`，没有形成 bank specialization。
- [Fact] rank diagnostic 证明低 rank 是部分瓶颈：`L360-R256:stbo_independent` 接近 A6，mean MSE `+0.014%`。
- [Counter-Evidence] paper-relevant 的 learned shared/bank 仍未超过 A6：最佳 `L360-R256:stbo_shared` 为 `+0.33%` mean MSE，`4/12` wins；`stbo_bank4` entropy 仍接近最大值 `0.9997-0.99999`。

[Conclusion] Capacity can recover performance, but current shared/bank STBO mechanism is unsupported. B12-STBO 不能进入 full matrix，也不能作为 Contribution 2。

## Candidate Verdicts

| Candidate | Status | Main Reason |
| --- | --- | --- |
| `B1-RED` | `partial_pass_distance_confounded` | future-unit difficulty 存在，但几乎被 forecast step/distance 解释 |
| `B2-RAS` | `rejected_by_narrative_gate` | reliability weighting 会退化为 horizon-distance weighting |
| `B3-DSR` | `partial_pass_not_method_ready` | seasonal residual proxy 不够 robust |
| `B4-TDA` | `completed_diagnostic` | 支持 A6 head/operator attribution，但不支持 urgent align innovation |
| `B5-BAFA` | `deferred_by_diagnostic` | no-align/no-recon 不伤 A6，basis-aware align 缺少必要性 |
| `B6-PLO` | `rejected_by_diagnostic` | learned basis objective 容易被 DCT/low-frequency control 解释 |
| `B7-UPO` | `deferred_small_contribution_candidate` | objective optimization 可作为小贡献，不能撑主创新点 |
| `B8-FQA` | `rejected_by_ocd_control` | segment correction headroom 被 DCT control 解释 |
| `B9-FSN-SCF` | `blocked_by_no_stage_control` | no-stage control 解释了相对 A6 的微弱收益 |
| `B10-TCO/TSI` | `offline_readout_route_blocked` | frozen/offline readout 病态；不能否定方向，但不继续显式 target/stage conditioning 主线 |
| `B11-ESA/BCF` | `blocked_by_required_controls` | no-basis/constant-slot controls 解释了 BCF 收益 |
| `B12-STBO` | `blocked_by_rank_diagnostic` | rank helps, but shared/bank STBO mechanism still unsupported |

## Failure Attribution For B12

- `hypothesis_false`: not proven. B12 不能否定所有 native multi-horizon operator redesign。
- `intervention_point_wrong`: possible. 当前 STBO 直接替换 full step basis，可能破坏了 A6 的 full-trajectory expressiveness。
- `readout_or_head_design_wrong`: possible. shared/bank local basis readout 未证明能承接 A6 的 coefficient-basis interface。
- `optimization_or_numeric_pathology`: no catastrophic divergence, but `stbo_bank4` gate entropy near uniform indicates inactive specialization.
- `capacity_control_explains`: partially yes. `L360-R256:independent` 接近 A6，说明 capacity/rank 是重要 confounder。

[Decision] 只能拒绝 current STBO family as paper-core method。不能把该结果写成 unified prediction architecture 方向失败。

## Next Research Entry

下一步从 11-step loop 的 Step 2/3 重新开始：

> 在不走 residual correction、不走 hard stage/horizon coding、不走 channel-correlation route 的前提下，如何从 A6 的 learned-basis unified operator 出发，提出一个真正 architecture-level 的 unified multi-horizon problem？

新的候选必须先回答：

1. 它解决的问题是否不是 fixed-horizon artifact？
2. 它是否比 A6 的 prefix slicing 更 native 地支持 multi-horizon？
3. 它是否改变 primary forecast path，而不是附加 residual、auxiliary loss 或 late readout patch？
4. 它是否能区分于 TransDF/FreDF 等 auxiliary loss / frequency decomposition work？
5. 它是否有清晰 tensor path、机制 novelty、必要 controls 和 rollback point？

## Guardrails For Next Conversation

- 不要把 failed diagnostic design 上升为 failed research direction；必须使用 failure attribution rule。
- 不要再把 residual-style repair 作为 StageB paper-core route。
- 不要把 explicit `stage_id` / `horizon_id` hard coding 作为主叙事，除非新证据证明它不会破坏 unified model 的连贯性。
- 暂不考虑通道间相关性建模；当前主线是 unified prediction / architecture。
- 深度调研不能只看 Zotero；必须同时查外部网络，优先 arXiv、OpenReview、conference proceedings、official code repositories。
- 远程实验前必须 `nvidia-smi` preflight，并按 workload-aware GPU scheduling，避免把 Weather 等长任务堆在同一块 GPU。
- 远程 `/home/yingch/exp_outputs/r-2026-fatst` 存储紧张时，可以清理旧 `exp_outputs`，但不要删除正在运行或尚未同步分析的实验。

## Suggested First Question

新对话可从这个问题启动：

> A6-LBF-r256 已证明 learned-basis unified forecast operator 有效，但它仍是 full-trajectory prefix slicing。StageB 的第二主创新点应如何在 primary architecture path 上更 native 地建模 requested horizon / future trajectory structure，同时保留 A6 的 full-basis expressiveness，并避免 residual、hard stage coding 和 auxiliary-loss 化？

