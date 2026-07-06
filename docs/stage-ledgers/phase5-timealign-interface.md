# Phase5 TimeAlign Interface Stage Ledger

本文档是 Phase5 当前主线账本。旧 StageA 候选细节已归档到
`docs/archive/phase5-stage-a/`；详细实验结果仍保存在 `analysis/`。

## Decision Cursor

| Field | Content |
| --- | --- |
| `stage_id` | `phase5-timealign-interface` |
| `current_11_step` | StageA fixed；StageB B6 Step 2/3 diagnostic returned negative |
| `active_carrier` | `A6-LBF-r256` pure learned-basis forecast operator on official-source TimeAlign encoder |
| `active_question` | A6-LBF-r256 是否需要一个与 learned-basis forecast operator 匹配的 prefix-native label/basis objective |
| `latest_decision` | B6-PLO diagnostic completed：label/residual 结构主要由 DCT low-frequency control 解释，A6 learned basis top32 弱于 DCT；status `diagnostic_not_enough_pause_b6` |
| `next_required_action` | 先重跑 active clean A6-LBF-r256 main matrix，确认移除 future branch 后主结果；不实现 B6 objective |
| `rollback_point` | StageB 暂停；若继续找 Contribution 2，必须重新回到 Step 2/3 定义一个非 generic-frequency、非 distance-confounded 的问题 |

## StageA Fixed Result

| Item | Decision |
| --- | --- |
| Accepted variant | `A6-LBF-r256` / `learned-basis-forecast-operator` |
| Role | 论文重要组成部分；StageB carrier |
| Main claim | 一个 unified model 在当前三数据集上整体优于 fixed-horizon per-horizon TimeAlign baseline |
| Code status | 主代码保留 `official` baseline；A6 `learned-basis-forecast-operator` 已移除 future reconstruction/alignment branch |
| Archived variants | A2/A3/A4/A5/A6-DER/A6-QBR/A6S/A6ST/A7DG/A8TAG/B0 旧文档与脚本均不再作为 active route |

### Key Evidence

相对 `TimeAlignOfficialFixedH{96,192,336,720}_official-last`：

| Dataset | A6-LBF MSE wins | Mean MSE vs fixed |
| --- | ---: | ---: |
| ETTh2 | 4/4 | `-10.89%` |
| ETTm1 | 3/4 | `-1.46%` |
| Weather | 2/4 | `-0.36%` |
| Overall | 9/12 | `-4.82%` |

相对 `TimeAlignOfficialUnified720_official-last`：

| Dataset | A6-LBF MSE wins | Mean MSE vs official unified |
| --- | ---: | ---: |
| ETTh2 | 4/4 | `-3.39%` |
| ETTm1 | 3/4 | `-1.01%` |
| Weather | 4/4 | `-1.19%` |
| Overall | 11/12 | `-1.92%` |

## StageB Entry Rules

- StageB 必须以 A6-LBF-r256 为 carrier；
- StageB 需要重新写 Step 2/3 problem definition 与 Step 4-6 narrative gate；
- 不再复用旧 StageA 变体作为 active candidate；
- 不把 old B0 `w_recon/w_align` ablation 直接升级为 StageB method；
- 后续新增 StageB 代码前，必须先写 `docs/experiments/phase5-stage-b-*.md` 的 design/narrative gate；
- 远程实验前必须 commit/push，并按 `AGENTS.md` 做 GPU preflight。

## Candidate Queue

| ID | Status | Hypothesis | Narrative Gate | Effectiveness Gate | Blocking Or Next Action | Artifacts |
| --- | --- | --- | --- | --- | --- | --- |
| `B1-RED` | `partial_pass` | A6-LBF-r256 的剩余 instability 来自 horizon-free future-unit reliability heterogeneity，且至少部分可由 train-only proxy 捕捉 | partial but distance-confounded：unit heterogeneity 和 volatility 成立；非 step-index reliability signal 不足 | 不看 MSE improvement；full B1 decision 为 `partial_pass_distance_confounded` | 不能推动 B2；只能作为 StageB rollback evidence | `docs/experiments/phase5-stage-b-reliability-aware-supervision-redesign.md`; `analysis/phase5_stage_b_reliability_diagnostic_20260706/stage_b_a6_lbf_reliability_report.md` |
| `B2-RAS` | `rejected_by_narrative_gate` | 若 B1 证明 learnable-hard 与 noisy-hard future units 可区分，则 reliability-aware supervision allocation 可提升 unified stability | failed：B1 未证明稳定的非 distance-confounded train-only proxy；直接实现会退化为 horizon/step-distance weighting | not evaluated | 不实现；若继续 StageB，必须先提出新的 Step 2/3 problem 或 stronger diagnostic proxy | none |
| `B3-DSR` | `partial_pass_not_method_ready` | A6-LBF-r256 的 residual difficulty 可拆成 forecast-distance trend 与 train-only seasonal residual 可解释的 structural residual | partial：linear residual 上跨数据集为正，但 rank/prefix residual 与 bootstrap 稳定性不足；不能支持 method | not evaluated | 不实现 loss weighting；下一步只能增强 proxy 或明确关闭 StageB | `analysis/phase5_stage_b_step23_redefinition_20260706/stage_b_step23_redefinition_report.md`; `docs/experiments/phase5-stage-b-distance-normalized-seasonal-residual-diagnostic.md`; `analysis/phase5_stage_b_distance_normalized_seasonal_residual_20260706/stage_b_b3_report.md` |
| `B4-TDA` | `completed_diagnostic` | A6-LBF same-align gains support head/operator contribution, but inherited TimeAlign align may still explain part of full architecture advantage | passed for Contribution 1 attribution：pure no-align/no-recon arm remains competitive | not a new method；diagnostic only | Use as evidence against urgent B5 align innovation | `analysis/phase5_stage_b_timealign_dependency_audit_20260706/stage_b_timealign_dependency_report.md`; `analysis/phase5_stage_b_timealign_dependency_ablation_20260706/stage_b_dependency_ablation_report.md`; `docs/experiments/phase5-stage-b-timealign-dependency-and-basis-align-diagnostic.md` |
| `B5-BAFA` | `deferred_by_diagnostic` | Replace generic TimeAlign alignment with A6-LBF basis-aware future alignment | failed current priority gate：removing inherited align/recon does not materially hurt A6-LBF | not evaluated | Do not implement unless later B6 or basis-space diagnostics show a stronger alignment-specific failure mode | none |
| `B6-PLO` | `rejected_by_diagnostic` | A6-LBF 的 architecture 已在 learned-basis space 中建模，但当前 supervision 仍主要是 time-domain point loss；可能缺少 prefix-native label/basis objective | failed：train-label/residual structure is largely generic DCT low-frequency; A6 learned basis has no top32 advantage | not evaluated | Do not implement B6 objective; pause StageB or redefine Step 2/3 | `docs/experiments/phase5-stage-b-prefix-native-label-objective-diagnostic.md`; `analysis/phase5_stage_b_prefix_native_objective_diagnostic_20260706/stage_b_b6_report.md` |

## Experiment Ledger

| Experiment | Candidate | Role | Result Summary | Decision | Full Report |
| --- | --- | --- | --- | --- | --- |
| B1 available-artifact audit | `B1-RED` | problem-existence diagnostic | 96-step segment heterogeneity is large on all three datasets, but `Spearman(step, MSE)=1.00`; current evidence may be forecast-distance difficulty rather than reliability-aware signal | partial evidence; full B1 still required | `analysis/phase5_stage_b_reliability_diagnostic_20260706/stage_b_a6_lbf_reliability_report.md` |
| B1 full reliability diagnostic | `B1-RED` | problem-existence diagnostic | 48-step unit heterogeneity is material, but `Spearman(step, MSE)` remains `0.99/0.99/1.00`; detrended proxy signals are inconsistent across datasets | `partial_pass_distance_confounded`; B2 rejected before implementation | `analysis/phase5_stage_b_reliability_diagnostic_20260706/stage_b_a6_lbf_reliability_report.md` |
| StageB Step 2/3 redefinition audit | `B3-DSR` | problem redefinition | Raw reliability and B2 are rejected; `seasonal_residual` is the only current proxy with positive detrended-MSE alignment across all three datasets (`0.35/0.59/0.81`) | B3 proposed as diagnostic-only candidate | `analysis/phase5_stage_b_step23_redefinition_20260706/stage_b_step23_redefinition_report.md` |
| B3 distance-normalized seasonal residual diagnostic | `B3-DSR` | problem-existence diagnostic | Linear residual signal is positive on all datasets/block sizes, but stricter rank/prefix residual and bootstrap checks are unstable | `partial_pass_needs_stronger_proxy_or_method_boundary`; not method-ready | `analysis/phase5_stage_b_distance_normalized_seasonal_residual_20260706/stage_b_b3_report.md` |
| TimeAlign dependency audit | `B4-TDA` | dependency diagnostic | Same-align A6-LBF beats official unified `11/12`, mean `-1.94%`; inherited alignment share remains non-trivial and causal ablations are missing | `partial_dependency_risk_confirmed` | `analysis/phase5_stage_b_timealign_dependency_audit_20260706/stage_b_timealign_dependency_report.md` |
| TimeAlign dependency ablation | `B4-TDA` | causal dependency diagnostic | `no_align_no_recon` mean MSE only `+0.07%` vs current and wins `7/12`; `align_no_recon` is slightly better on mean MSE (`-0.04%`) but effect is tiny | `dependency_ablation_pass_for_head_contribution_but_not_for_b5`; B5 not prioritized | `analysis/phase5_stage_b_timealign_dependency_ablation_20260706/stage_b_dependency_ablation_report.md` |
| A6 clean-operator code cut | `B4-TDA` | code cleanup from diagnostic evidence | A6-LBF no longer instantiates future reconstruction/alignment branch; official baseline remains unchanged | accepted as new clean research start; local smoke passed | `baselines/timealign_official/models/TimeAlign.py`; `baselines/timealign_official/train_repo.py`; `docs/code-explanation/phase5-clean-timealign-a6-lbf.md` |
| B6 prefix-native objective diagnostic | `B6-PLO` | problem-existence diagnostic | PCA/DCT top32 nearly identical; A6 learned basis top32 is weaker than DCT on label and residual coverage | `diagnostic_not_enough_pause_b6`; not method-ready | `analysis/phase5_stage_b_prefix_native_objective_diagnostic_20260706/stage_b_b6_report.md` |

## Pending Tasks

| Task | Owner | Trigger | Status | Next Action |
| --- | --- | --- | --- | --- |
| Complete full B1 post-hoc reliability diagnostic | Codex | Available-artifact audit found partial evidence but no train-only proxy | `completed` | Done; decision is `partial_pass_distance_confounded` |
| Decide whether B2-RAS may enter implementation design | Codex | B1 report complete | `completed` | B2 rejected by narrative gate; do not touch model or runner code |
| Redefine StageB after distance-confounded B1 | Codex | B1 full diagnostic did not justify B2 | `completed` | B3-DSR proposed as stronger diagnostic-only problem candidate |
| Run B3 distance-normalized seasonal residual diagnostic | Codex | Step 2/3 redefinition found a non-distance candidate | `completed` | B3 partial; no method implementation |
| Decide StageB route after B3 partial | Codex | B3 is not method-ready | `completed` | Pivot to TimeAlign dependency / basis-aware align route |
| Run TimeAlign dependency ablation | Codex | Artifact audit confirmed unresolved dependency risk | `completed` | Done; B4 supports Contribution 1 attribution and blocks urgent B5 implementation |
| Define B6 prefix-native objective diagnostic | Codex | B4 made align innovation low-priority, while B1/B3 blocked reliability weighting | `completed` | Protocol written in `docs/experiments/phase5-stage-b-prefix-native-label-objective-diagnostic.md` |
| Run B6 offline diagnostic | Codex | B6 protocol ready | `completed` | Done; decision is `diagnostic_not_enough_pause_b6` |
| Revalidate clean A6 smoke/main run | Codex | A6 future branch removed from code | `completed_local_smoke` | A6 smoke shows `w_recon=w_align=0.0` and zero recon/align logs; official smoke keeps inherited terms. Remote main matrix rerun is optional before B6 training. |
| Rerun clean A6 main matrix | Codex | Active A6 code removed unused future branch and changed initialization order | `pending` | Commit/push B6 diagnostic, then launch clean A6 main runner on 529_Lab-3090 with workload-aware GPU assignment |

## Paper Mainline Sync Log

| Date | Trigger | Paper Section | Change Type | Decision |
| --- | --- | --- | --- | --- |
| 2026-07-06 | StageB B1 diagnostic entry created | none | no paper-mainline change | Not synced because B1 is diagnostic-only and has no new paper claim yet |
| 2026-07-06 | StageB Step 2/3 redefinition found B3-DSR | none | no paper-mainline change | Not synced because B3 is diagnostic-only and has no accepted method or paper claim yet |
| 2026-07-06 | B3 diagnostic returned partial/not method-ready | none | no paper-mainline change | Not synced because the result blocks method implementation and adds no accepted paper claim |
| 2026-07-06 | TimeAlign dependency audit confirms attribution risk | Contribution 2 candidate | update needed | Paper mainline now treats basis-aware alignment as candidate, not accepted claim |
| 2026-07-06 | Dependency ablation launched on 529_Lab-3090 | none | no paper-mainline change | Remote launch only; no returned effectiveness evidence yet |
| 2026-07-06 | Dependency ablation returned | Contribution boundary and StageB plan | updated | Contribution 1 independence strengthened; B5 basis-aware alignment deferred; StageB rolls to B6 prefix-native objective diagnostic |
| 2026-07-06 | B6 diagnostic returned negative | Contribution 2 candidate | updated | Do not claim prefix-native objective as contribution; StageB pauses pending clean A6 rerun and possible new Step 2/3 |

## Active Artifacts

| Artifact | Purpose |
| --- | --- |
| `baselines/timealign_official/models/TimeAlign.py` | clean official + A6-LBF model |
| `baselines/timealign_official/train_repo.py` | clean official/A6-LBF training adapter |
| `docs/code-explanation/phase5-clean-timealign-a6-lbf.md` | code-facing explanation |
| `docs/code-explanation/phase5-stage-b-reliability-diagnostic.md` | B1 full reliability diagnostic script explanation |
| `docs/code-explanation/phase5-stage-b-b3-dsr-diagnostic.md` | B3 diagnostic script explanation |
| `docs/code-explanation/phase5-stage-b-timealign-dependency-audit.md` | TimeAlign dependency audit and ablation code explanation |
| `docs/code-explanation/phase5-stage-b-b6-prefix-objective-diagnostic.md` | B6 diagnostic analyzer explanation |
| `docs/experiments/phase5-stage-b-reliability-aware-supervision-redesign.md` | StageB problem definition and B1/B2 candidate boundary |
| `docs/experiments/phase5-stage-b-distance-normalized-seasonal-residual-diagnostic.md` | B3 diagnostic protocol |
| `docs/experiments/phase5-stage-b-timealign-dependency-and-basis-align-diagnostic.md` | B4 dependency and B5 basis-align protocol |
| `docs/experiments/phase5-stage-b-prefix-native-label-objective-diagnostic.md` | B6 prefix-native label/basis objective diagnostic protocol |
| `scripts/analyze_phase5_stage_b_reliability_diagnostic.py` | B1 full reliability diagnostic generator |
| `scripts/analyze_phase5_stage_b_b3_dsr_diagnostic.py` | B3 distance-normalized seasonal residual diagnostic generator |
| `scripts/analyze_phase5_stage_b_timealign_dependency_audit.py` | TimeAlign dependency artifact audit |
| `scripts/analyze_phase5_stage_b_timealign_dependency_ablation.py` | returned no-align/no-recon ablation analyzer |
| `scripts/analyze_phase5_stage_b_b6_prefix_objective_diagnostic.py` | B6 prefix-native objective offline diagnostic analyzer |
| `scripts/remote/run_phase5_stage_b_timealign_dependency_ablation.sh` | completed remote no-align/no-recon ablation runner |
| `scripts/remote/run_phase5_a6_lbf_r256_main.sh` | clean A6-LBF-r256 remote runner |
| `analysis/phase5_timealign_hss_a6_capacity_native_gate_20260703/` | StageA accepted evidence |
| `analysis/phase5_stage_b_reliability_diagnostic_20260706/` | B1 full diagnostic; decision `partial_pass_distance_confounded` |
| `analysis/phase5_stage_b_step23_redefinition_20260706/` | StageB problem redefinition audit; B3-DSR proposed |
| `analysis/phase5_stage_b_distance_normalized_seasonal_residual_20260706/` | B3 diagnostic; decision `partial_pass_needs_stronger_proxy_or_method_boundary` |
| `analysis/phase5_stage_b_timealign_dependency_audit_20260706/` | TimeAlign dependency audit; decision `partial_dependency_risk_confirmed` |
| `analysis/phase5_stage_b_timealign_dependency_ablation_20260706/` | B4 no-align/no-recon ablation; decision `dependency_ablation_pass_for_head_contribution_but_not_for_b5` |
| `analysis/phase5_stage_b_prefix_native_objective_diagnostic_20260706/` | B6 diagnostic; decision `diagnostic_not_enough_pause_b6` |

## Archived Evidence

旧 StageA route 的详细候选记录不再放在 active ledger 中。若需要审计历史，可读：

- `docs/archive/phase5-stage-a/experiments/`
- `docs/archive/phase5-stage-a/code-explanation/`
- `analysis/phase5_stage_a_architecture_exhaustion_audit_20260705/`
- `analysis/phase5_timealign_hss_a6_capacity_native_gate_20260703/`
