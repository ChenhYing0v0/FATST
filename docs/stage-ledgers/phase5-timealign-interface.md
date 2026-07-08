# Phase5 TimeAlign Interface Stage Ledger

本文档是 Phase5 当前主线账本。旧 StageA 候选细节已归档到
`docs/archive/phase5-stage-a/`；详细实验结果仍保存在 `analysis/`。

## Decision Cursor

| Field | Content |
| --- | --- |
| `stage_id` | `phase5-timealign-interface` |
| `current_11_step` | StageB Step 3 B10-TSI target-set interface diagnostic |
| `active_carrier` | `A6-LBF-r256` pure learned-basis forecast operator on official-source TimeAlign encoder |
| `active_question` | Can requested target set enter the A6 basis-coeff operator natively, instead of generating a 720-step trajectory and slicing prefixes |
| `latest_decision` | B10-TSI-B supports continuing: real A6 coeff activates multiple low-aligned stage row subspaces; this supports target-set-blind history-to-coeff/state pressure |
| `next_required_action` | Run B10-TSI-C target-set oracle/control; if no-target-set capacity explains headroom, do not enter method design |
| `rollback_point` | If B10-TSI is explained by no-target-set capacity control, return to StageB Step 2/3 or keep B7 only as small objective contribution |

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
| ETTh2 | 4/4 | `-10.53%` |
| ETTm1 | 3/4 | `-1.64%` |
| Weather | 2/4 | `-0.22%` |
| Overall | 9/12 | `-4.13%` |

相对 `TimeAlignOfficialUnified720_official-last`：

| Dataset | A6-LBF MSE wins | Mean MSE vs official unified |
| --- | ---: | ---: |
| ETTh2 | 4/4 | `-2.78%` |
| ETTm1 | 3/4 | `-1.20%` |
| Weather | 4/4 | `-1.26%` |
| Overall | 11/12 | `-1.75%` |

Clean rerun after code cleanup:

| Check | Result |
| --- | --- |
| Effective losses | `w_recon=0.0`, `w_align=0.0` |
| Readout | `learned-basis-forecast-operator`, `basis_rank=256`, `pred_loss_mode=multi-prefix` |
| vs historical A6-LBF-r256 | overall mean MSE `+0.20%`, `6/12` MSE wins |
| Decision | `clean_a6_validated` |

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
| `B7-UPO` | `deferred_small_contribution_candidate` | A6-LBF 已有 unified operator，但 nested multi-prefix training 会过度加权短步，并可能弱化 long-tail forecast regions | partial：可作为 objective refinement，但不适合作为第二主创新点 | not evaluated | 暂缓，待 architecture candidate 判定后再作为小贡献候选处理 | `docs/experiments/phase5-stage-b-unified-prefix-optimization-diagnostic.md`; `analysis/phase5_stage_b_unified_prefix_optimization_20260707/stage_b_b7_unified_prefix_optimization_report.md` |
| `B8-FQA` | `rejected_by_ocd_control` | A6-LBF 的 coefficient 是 sample-specific 但 future-position-invariant；future queries 可在 basis prediction 前将 history representation 对齐到 target positions | failed：learned segment-specific correction has headroom, but DCT control has stronger absolute residual reduction | not evaluated | Do not implement B8; return StageB to Step 2/3 | `docs/experiments/phase5-stage-b-future-query-aligned-basis-architecture.md`; `analysis/phase5_stage_b_future_query_aligned_architecture_research_20260707/stage_b_architecture_direction_report.md`; `analysis/phase5_stage_b_b8_ocd_coefficient_oracle_20260707/b8_ocd_report.md` |
| `B9-FSN-SCF` | `blocked_by_no_stage_control` | A6-LBF 的 single coefficient state 同时服务多个 future stages；若 stage losses 对该 coefficient 的梯度方向不一致，则需要 native future-stage-aware coefficient field | passed Step 4-7, but failed effectiveness mechanism gate：B9 cannot beat same-parameter no-stage control | failed：B9 vs A6 `-0.13%`, no-stage vs A6 `-0.13%`, B9 vs no-stage `+0.0036%` and `2/12` wins | Do not launch full matrix; rollback to Step 4 redesign or Step 2/3 | `docs/experiments/phase5-stage-b-native-future-stage-operator.md`; `docs/code-explanation/phase5-stage-b-b9-fsn-scf.md`; `analysis/phase5_stage_b_b9_fsn_scf_small_gate_20260707/b9_fsn_scf_small_gate_report.md` |
| `B10-TCO` | `supports_continue_to_oracle_control` | A6-LBF 是 prefix-compatible 720-step trajectory operator；requested target set 没有进入 computation graph，短 horizon 是 prefix slicing | B10-TSI-A/B support problem narrowing：basis 已有 stage geometry，真实 `coeff` 同时激活多个低同向性 stage subspaces | not evaluated | Run B10-TSI-C target-set oracle/control; require no-target-set capacity control before Step 4-6 | `docs/experiments/phase5-stage-b-target-set-conditioned-operator.md`; `analysis/phase5_stage_b_b10_tsi_basis_geometry_20260708/b10_tsi_basis_geometry_report.md`; `analysis/phase5_stage_b_b10_tsi_coeff_usage_20260708/b10_tsi_coeff_usage_report.md` |

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
| Clean A6-LBF-r256 main rerun | StageA validation | clean carrier validation | Active pure A6 operator beats fixed TimeAlign by `-4.13%` mean MSE with `9/12` wins and official unified by `-1.75%` with `11/12` wins; vs historical A6 mean MSE changes only `+0.20%` | `clean_a6_validated`; StageB remains paused | `analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706/clean_a6_rerun_report.md` |
| B7 unified prefix optimization diagnostic | `B7-UPO` | problem-existence diagnostic | Current multi-prefix loss gives `0-96` steps `14.39x` tail supervision weight; segment-level A6 gains vs fixed shrink from early `-3.57%` to tail `-0.16%` overall, with ETTh2/ETTm1 supporting and Weather countering | `prefix_imbalance_problem_candidate`; not method-ready | `analysis/phase5_stage_b_unified_prefix_optimization_20260707/stage_b_b7_unified_prefix_optimization_report.md` |
| B8 future-query aligned architecture research | `B8-FQA` | architecture direction research | 外部网络调研与 code-theory review 表明，A6 在 basis prediction 前缺少 target-position-aware sample-specific representation；future-query coefficient modulation 曾是优先 architecture route | `superseded_by_b8_ocd`；后续诊断未通过 | `analysis/phase5_stage_b_future_query_aligned_architecture_research_20260707/stage_b_architecture_direction_report.md` |
| B8 coefficient-space oracle diagnostic | `B8-FQA` | problem-existence diagnostic | Rank64 learned segment reduction is ETTh2 `79.05%`, ETTm1 `72.77%`, Weather `61.91%`, but DCT control is stronger at `87.61%/91.85%/91.18%`; residual headroom is generic low-frequency confounded | `rejected_by_ocd_control`; do not implement B8 | `analysis/phase5_stage_b_b8_ocd_coefficient_oracle_20260707/b8_ocd_report.md` |
| B9 stage-gradient diagnostic | `B9-FSN` | native architecture problem diagnostic | Four future stage losses have low cosine gradients on the same A6 `coeff`: mean pairwise cosine ETTh2 `0.072`, ETTm1 `0.171`, Weather `0.048`; early-tail cosine `0.041/0.112/0.014` | `problem_candidate_passed`; Step 4-6 required before implementation | `analysis/phase5_stage_b_b9_stage_gradient_diagnostic_20260707/b9_stage_gradient_report.md` |
| B9-FSN-SCF Step 4-6 design gate | `B9-FSN-SCF` | narrative/method gate | Concrete design extends A6 `coeff` from `[B,C,K]` to stage-indexed `[B,C,S,K]` before basis projection; zero-gated multiplicative coefficient modulation preserves A6 at initialization | `method_candidate_ready_for_small_gate`; no full matrix yet | `docs/experiments/phase5-stage-b-native-future-stage-operator.md` |
| B9-FSN-SCF local implementation smoke | `B9-FSN-SCF` | implementation verification | Added `stage-native-coefficient-field` and `stage-native-coefficient-field-no-stage`; A6 fallback and prefix consistency max abs are all `0.0`; ETTh2 1-batch CPU smoke passed for B9 and no-stage | `local_implementation_smoke_passed`; launch remote small gate after commit/push | `baselines/timealign_official/models/TimeAlign.py`; `baselines/timealign_official/train_repo.py`; `docs/code-explanation/phase5-stage-b-b9-fsn-scf.md` |
| B9-FSN-SCF remote small gate | `B9-FSN-SCF` | effectiveness and mechanism control | B9 beats A6 by `-0.13%` mean MSE with `12/12` wins, but no-stage also beats A6 by `-0.13%`; B9 vs no-stage is `+0.0036%` mean MSE with only `2/12` wins | `blocked_by_no_stage_control`; do not claim stage-token mechanism | `analysis/phase5_stage_b_b9_fsn_scf_small_gate_20260707/b9_fsn_scf_small_gate_report.md` |
| B10 target-set-conditioned operator redefinition | `B10-TCO` | problem redefinition | A6 is better framed as a prefix-compatible learned-basis trajectory operator, not a target-set-native multi-horizon architecture; B10 asks whether requested target set $J$ should enter basis-coeff operator | `problem_redefinition_ready`; run B10-TSI diagnostic | `docs/experiments/phase5-stage-b-target-set-conditioned-operator.md` |
| B10-TSI-A basis geometry diagnostic | `B10-TCO` | checkpoint-only problem diagnostic | Top64 atoms are cross-stage (`entropy=0.8108/0.8764/0.8658`, stage-specialized rate `0.0156/0/0`), but rank32 stage row-space overlap is low (`0.1324/0.1510/0.1368`) | `partial_support_continue_tsi`; B10 problem narrows to target-set-blind history-to-coeff/state path | `analysis/phase5_stage_b_b10_tsi_basis_geometry_20260708/b10_tsi_basis_geometry_report.md`; `docs/code-explanation/phase5-stage-b-b10-tsi-basis-geometry.md` |
| B10-TSI-B coefficient usage diagnostic | `B10-TCO` | forward-path problem diagnostic | Rank64 projection share `0.3882/0.4950/0.2764`, projection cosine `0.3759/0.4702/0.1639`, output entropy `0.7969/0.8958/0.9042`; real `coeff` activates multiple low-aligned stage subspaces | `supports_continue_to_oracle_control`; still not method-ready | `analysis/phase5_stage_b_b10_tsi_coeff_usage_20260708/b10_tsi_coeff_usage_report.md`; `docs/code-explanation/phase5-stage-b-b10-tsi-coeff-usage.md` |

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
| Revalidate clean A6 smoke/main run | Codex | A6 future branch removed from code | `completed` | Local smoke and remote main rerun both passed; active clean A6 is validated. |
| Rerun clean A6 main matrix | Codex | Active A6 code removed unused future branch and changed initialization order | `completed` | Done; decision `clean_a6_validated`; paper evidence now uses clean rerun metrics |
| Run B7 unified prefix optimization diagnostic | Codex | User requested StageB direction within unified prediction, excluding channel modeling | `completed` | Done; B7 is a problem candidate only |
| Run B7-GTD gradient/task diagnostic | Codex | B7-UPO segment evidence is partial and Weather is a counterexample | `deferred` | Keep as objective small-contribution diagnostic after architecture path is resolved |
| Run B8-OCD coefficient-space oracle diagnostic | Codex | B8-FQA is proposed but needs evidence that future-segment-specific coefficients can reduce A6 residuals | `completed` | Done; decision `rejected_by_ocd_control`, do not implement B8 |
| Redefine StageB architecture-level problem after B8-OCD | Codex | B8 narrative was coherent but failed DCT control gate | `completed` | B9-FSN defined as native future-stage-aware route, excluding residual correction |
| Run B9-SGC native future-stage gradient diagnostic | Codex | User rejected residual architecture and requested native future-stage-aware route | `completed` | Done; decision `problem_candidate_passed` |
| Design B9-FSN Step 4-6 narrative/method gate | Codex | B9-SGC supports problem existence | `completed` | Done; `B9-FSN-SCF` may enter minimal implementation and small gate |
| Implement B9-FSN-SCF minimal gate | Codex | Step 4-6 narrative/method gate passed | `completed` | Done; local fallback/prefix checks and smoke passed |
| Launch B9-FSN-SCF remote small gate | Codex | Local implementation smoke passed | `completed` | Done; decision `blocked_by_no_stage_control` |
| Redesign or rollback after B9-FSN-SCF no-stage block | Codex | B9-SCF cannot beat no-stage control | `completed` | Rolled back to B10 target-set-native multi-horizon problem |
| Run B10-TSI target-set interface diagnostic | Codex | B10-TCO problem redefinition is ready | `in_progress` | B10-TSI-A/B done; next B10-TSI-C target-set oracle/control with no-target-set capacity control |

## Paper Mainline Sync Log

| Date | Trigger | Paper Section | Change Type | Decision |
| --- | --- | --- | --- | --- |
| 2026-07-06 | StageB B1 diagnostic entry created | none | no paper-mainline change | Not synced because B1 is diagnostic-only and has no new paper claim yet |
| 2026-07-06 | StageB Step 2/3 redefinition found B3-DSR | none | no paper-mainline change | Not synced because B3 is diagnostic-only and has no accepted method or paper claim yet |
| 2026-07-06 | B3 diagnostic returned partial/not method-ready | none | no paper-mainline change | Not synced because the result blocks method implementation and adds no accepted paper claim |
| 2026-07-06 | TimeAlign dependency audit confirms attribution risk | Contribution 2 candidate | update needed | Paper mainline now treats basis-aware alignment as candidate, not accepted claim |
| 2026-07-06 | Dependency ablation launched on 529_Lab-3090 | none | no paper-mainline change | Remote launch only; no returned effectiveness evidence yet |
| 2026-07-06 | Dependency ablation returned | Contribution boundary and StageB plan | updated | Contribution 1 independence strengthened; B5 basis-aware alignment deferred; StageB rolls to B6 prefix-native objective diagnostic |
| 2026-07-06 | B6 diagnostic returned negative | Contribution 2 candidate | updated | Do not claim prefix-native objective as contribution; StageB pauses until a possible new Step 2/3 problem is found |
| 2026-07-06 | Clean A6 rerun returned | StageA evidence and contribution boundary | updated | Active pure A6 operator validated; StageB remains paused |
| 2026-07-07 | B7-UPO diagnostic returned partial evidence | Contribution 2 candidate | no accepted paper claim | Reopen StageB Step 2/3 around unified prefix optimization; require gradient diagnostic before implementation |
| 2026-07-07 | User requested architecture-level StageB main innovation | Contribution 2 candidate | no accepted paper claim | B7 deferred as small objective candidate; B8-FQA proposed as architecture candidate |
| 2026-07-07 | B8-OCD returned negative control | Contribution 2 candidate | no accepted paper claim | Do not implement B8-FQA; StageB returns to Step 2/3 architecture problem search |
| 2026-07-07 | B9-SGC returned positive stage-gradient evidence | Contribution 2 candidate | no accepted paper claim | B9-FSN becomes active problem candidate; Step 4-6 design required before implementation |
| 2026-07-07 | B9-FSN-SCF Step 4-6 design gate passed | Contribution 2 candidate | no accepted paper claim | B9-FSN-SCF may enter minimal implementation and small gate; no full matrix claim yet |
| 2026-07-07 | B9-FSN-SCF local implementation smoke passed | Contribution 2 candidate | no accepted paper claim | Launch remote small gate after commit/push and GPU preflight |
| 2026-07-07 | B9-FSN-SCF remote small gate launched | Contribution 2 candidate | no accepted paper claim | Wait for artifacts, then run sync/analyzer |
| 2026-07-07 | B9-FSN-SCF small gate blocked by no-stage | Contribution 2 candidate | no accepted paper claim | Do not launch full matrix; rollback to Step 4/2 |
| 2026-07-08 | B10 target-set-native problem redefinition | Contribution 2 candidate | no accepted paper claim | Run B10-TSI diagnostic before implementation |
| 2026-07-08 | B10-TSI-A basis geometry returned | Contribution 2 candidate | no accepted paper claim | Continue B10-TSI-B; problem is target-set-blind history-to-coeff/state path, not basis stage-blindness |
| 2026-07-08 | B10-TSI-B coefficient usage returned | Contribution 2 candidate | no accepted paper claim | Continue to B10-TSI-C oracle/control; do not implement method yet |

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
| `docs/code-explanation/phase5-clean-a6-rerun-analysis.md` | clean A6 validation analyzer explanation |
| `docs/code-explanation/phase5-stage-b-b7-unified-prefix-optimization.md` | B7 unified prefix optimization analyzer explanation |
| `docs/code-explanation/phase5-stage-b-b8-ocd-coefficient-oracle.md` | B8-OCD coefficient oracle analyzer explanation |
| `docs/code-explanation/phase5-stage-b-b9-stage-gradient-diagnostic.md` | B9-SGC stage gradient analyzer explanation |
| `docs/code-explanation/phase5-stage-b-b9-fsn-scf.md` | B9-FSN-SCF model implementation explanation |
| `docs/code-explanation/phase5-stage-b-b10-tsi-basis-geometry.md` | B10-TSI-A basis geometry analyzer explanation |
| `docs/code-explanation/phase5-stage-b-b10-tsi-coeff-usage.md` | B10-TSI-B coefficient usage analyzer explanation |
| `docs/experiments/phase5-stage-b-reliability-aware-supervision-redesign.md` | StageB problem definition and B1/B2 candidate boundary |
| `docs/experiments/phase5-stage-b-distance-normalized-seasonal-residual-diagnostic.md` | B3 diagnostic protocol |
| `docs/experiments/phase5-stage-b-timealign-dependency-and-basis-align-diagnostic.md` | B4 dependency and B5 basis-align protocol |
| `docs/experiments/phase5-stage-b-prefix-native-label-objective-diagnostic.md` | B6 prefix-native label/basis objective diagnostic protocol |
| `docs/experiments/phase5-stage-b-unified-prefix-optimization-diagnostic.md` | B7 unified prefix optimization diagnostic protocol |
| `docs/experiments/phase5-stage-b-future-query-aligned-basis-architecture.md` | B8 future-query aligned basis architecture protocol |
| `docs/experiments/phase5-stage-b-native-future-stage-operator.md` | B9 native future-stage operator protocol |
| `docs/experiments/phase5-stage-b-target-set-conditioned-operator.md` | B10 target-set-conditioned operator protocol |
| `scripts/remote/run_phase5_stage_b_b9_fsn_scf_small_gate.sh` | B9-FSN-SCF small gate remote runner |
| `scripts/analyze_phase5_stage_b_b9_fsn_scf_small_gate.py` | B9-FSN-SCF small gate analyzer |
| `scripts/sync_phase5_stage_b_b9_fsn_scf_small_gate_results.sh` | B9-FSN-SCF small gate sync/analyze wrapper |
| `scripts/analyze_phase5_stage_b_reliability_diagnostic.py` | B1 full reliability diagnostic generator |
| `scripts/analyze_phase5_stage_b_b3_dsr_diagnostic.py` | B3 distance-normalized seasonal residual diagnostic generator |
| `scripts/analyze_phase5_stage_b_timealign_dependency_audit.py` | TimeAlign dependency artifact audit |
| `scripts/analyze_phase5_stage_b_timealign_dependency_ablation.py` | returned no-align/no-recon ablation analyzer |
| `scripts/analyze_phase5_stage_b_b6_prefix_objective_diagnostic.py` | B6 prefix-native objective offline diagnostic analyzer |
| `scripts/analyze_phase5_a6_clean_operator_rerun.py` | clean A6 rerun validation analyzer |
| `scripts/analyze_phase5_stage_b_unified_prefix_optimization.py` | B7 unified prefix optimization diagnostic analyzer |
| `scripts/analyze_phase5_stage_b_b8_ocd_coefficient_oracle.py` | B8-OCD coefficient oracle diagnostic analyzer |
| `scripts/analyze_phase5_stage_b_b9_stage_gradient_diagnostic.py` | B9-SGC stage gradient diagnostic analyzer |
| `scripts/analyze_phase5_stage_b_b10_tsi_basis_geometry.py` | B10-TSI-A basis geometry diagnostic analyzer |
| `scripts/analyze_phase5_stage_b_b10_tsi_coeff_usage.py` | B10-TSI-B coefficient usage diagnostic analyzer |
| `scripts/remote/run_phase5_stage_b_timealign_dependency_ablation.sh` | completed remote no-align/no-recon ablation runner |
| `scripts/remote/run_phase5_a6_lbf_r256_main.sh` | clean A6-LBF-r256 remote runner |
| `analysis/phase5_timealign_hss_a6_capacity_native_gate_20260703/` | StageA accepted evidence |
| `analysis/phase5_stage_b_reliability_diagnostic_20260706/` | B1 full diagnostic; decision `partial_pass_distance_confounded` |
| `analysis/phase5_stage_b_step23_redefinition_20260706/` | StageB problem redefinition audit; B3-DSR proposed |
| `analysis/phase5_stage_b_distance_normalized_seasonal_residual_20260706/` | B3 diagnostic; decision `partial_pass_needs_stronger_proxy_or_method_boundary` |
| `analysis/phase5_stage_b_timealign_dependency_audit_20260706/` | TimeAlign dependency audit; decision `partial_dependency_risk_confirmed` |
| `analysis/phase5_stage_b_timealign_dependency_ablation_20260706/` | B4 no-align/no-recon ablation; decision `dependency_ablation_pass_for_head_contribution_but_not_for_b5` |
| `analysis/phase5_stage_b_prefix_native_objective_diagnostic_20260706/` | B6 diagnostic; decision `diagnostic_not_enough_pause_b6` |
| `analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706/` | Clean A6 rerun; decision `clean_a6_validated` |
| `analysis/phase5_stage_b_unified_prefix_optimization_20260707/` | B7 diagnostic; decision `prefix_imbalance_problem_candidate` |
| `analysis/phase5_stage_b_future_query_aligned_architecture_research_20260707/` | B8 architecture direction research; superseded by B8-OCD |
| `analysis/phase5_stage_b_b8_ocd_coefficient_oracle_20260707/` | B8-OCD diagnostic; decision `rejected_by_ocd_control` |
| `analysis/phase5_stage_b_b9_stage_gradient_diagnostic_20260707/` | B9-SGC diagnostic; decision `problem_candidate_passed` |
| `analysis/phase5_stage_b_b9_fsn_scf_small_gate_20260707/launch_record.md` | B9-FSN-SCF remote launch record |
| `analysis/phase5_stage_b_b9_fsn_scf_small_gate_20260707/b9_fsn_scf_small_gate_report.md` | B9-FSN-SCF small gate decision |
| `analysis/phase5_stage_b_b10_tsi_basis_geometry_20260708/b10_tsi_basis_geometry_report.md` | B10-TSI-A basis geometry decision |
| `analysis/phase5_stage_b_b10_tsi_coeff_usage_20260708/b10_tsi_coeff_usage_report.md` | B10-TSI-B coefficient usage decision |

## Archived Evidence

旧 StageA route 的详细候选记录不再放在 active ledger 中。若需要审计历史，可读：

- `docs/archive/phase5-stage-a/experiments/`
- `docs/archive/phase5-stage-a/code-explanation/`
- `analysis/phase5_stage_a_architecture_exhaustion_audit_20260705/`
- `analysis/phase5_timealign_hss_a6_capacity_native_gate_20260703/`
