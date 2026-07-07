# Research Roadmap

本文档是当前可重启的主研究路径。旧阶段细节已归档或保存在 `analysis/`，本文件只保留会影响后续
决策的结论。

## Current State

| Field | Content |
| --- | --- |
| `paper_target` | 高水平 SCI 期刊时间序列预测论文 |
| `working_title` | Horizon-Agnostic Supervision Scheduling for Unified Multi-Horizon Forecasting |
| `current_stage` | Phase5：A6-LBF-r256 clean operator validated；StageB architecture search returned to Step 2/3 after B8-OCD |
| `current_11_step` | StageB Step 2/3: redefine architecture-level second contribution after B8 negative control |
| `active_carrier` | `A6-LBF-r256` pure learned-basis forecast operator |
| `active_ledger` | `docs/stage-ledgers/phase5-timealign-interface.md` |

## Long Research Loop Rule

每个新 StageB 候选必须记录：

1. Research and analyze existing work.
2. Propose the specific problem.
3. Evaluate whether the problem is real.
4. Propose the core idea.
5. Evaluate theoretical feasibility.
6. Design method and experiment plan.
7. Implement.
8. Run remote training when needed.
9. Evaluate artifacts.
10. Decide paper-story and performance pass/fail.
11. If failed, choose explicit rollback point.

Narrative gate 属于 Step 4-6；effectiveness gate 属于 Step 9-10。Diagnostic-only 实验不能因为
metric 正向就直接升级为 method。

## StageA Final Decision

[Decision] StageA 结果已固定：`A6-LBF-r256` 是当前论文的重要创新点与后续 StageB 起点。

[Claim] A6-LBF-r256 用 learned temporal basis + hidden coefficients 形成 prefix-native unified forecast
operator。在当前实验集合上，一个 unified model 整体优于 fixed-horizon per-horizon TimeAlign。

### Evidence

相对 fixed-horizon per-horizon TimeAlign official-last（clean A6 rerun）：

| Dataset | MSE wins | Mean MSE change |
| --- | ---: | ---: |
| ETTh2 | 4/4 | `-10.53%` |
| ETTm1 | 3/4 | `-1.64%` |
| Weather | 2/4 | `-0.22%` |
| Overall | 9/12 | `-4.13%` |

相对 official unified TimeAlign（clean A6 rerun）：

| Dataset | MSE wins | Mean MSE change |
| --- | ---: | ---: |
| ETTh2 | 4/4 | `-2.78%` |
| ETTm1 | 3/4 | `-1.20%` |
| Weather | 4/4 | `-1.26%` |
| Overall | 11/12 | `-1.75%` |

[Validation] Clean rerun after removing the A6 future reconstruction/alignment branch is at
`analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706/`. It preserves the accepted result and is close to the
historical A6 artifact: overall mean MSE change `+0.20%`, `6/12` MSE wins. Effective clean A6 training uses
`w_recon=0.0`, `w_align=0.0`, `readout_mode=learned-basis-forecast-operator`, `basis_rank=256`, and
`pred_loss_mode=multi-prefix`.

### Contribution Boundary

A6-LBF-r256 可以作为：

- unified multi-horizon forecasting 的核心 architecture contribution；
- 后续 StageB 的 clean carrier；
- 与 fixed-horizon TimeAlign 的主对照证据。

不能写成：

- teacher/self-teacher 或 EMA 稳定化方法；
- target-query / QBR / nested / residual adapter 的混合路线；
- 依赖 best-val/early-stop 的 protocol trick。

## StageB Redesign Entry

[Decision] StageB 从 A6-LBF-r256 出发重新设计，不沿用 pre-cleanup B0 作为正式路线。

StageB 应回答的新问题：

> 在 A6-LBF-r256 这个已成立的 unified carrier 上，future-aware supervision / reliability-aware
> allocation 是否能进一步提升 unified model 的跨数据集稳定性和论文机制深度？

StageB 设计前必须先写新的 experiment note，至少包含：

- `current_step`;
- `problem`;
- `existence_evidence`;
- `idea`;
- `theory_check`;
- `design`;
- `narrative_gate`;
- `effectiveness_gate`;
- `artifacts`;
- `decision`;
- rollback point。

### B1 Reliability Diagnostic Decision

[Decision] B1-RED full diagnostic completed at
`analysis/phase5_stage_b_reliability_diagnostic_20260706/`.

[Strong Evidence] A6-LBF-r256 的 future units 存在 material heterogeneity：
48-step unit max/min MSE gap 为 ETTh2 `195.16%`、ETTm1 `99.09%`、Weather `248.48%`。

[Counter-Evidence] 该 heterogeneity 与 forecast step/distance 高度绑定：
`Spearman(step, MSE)` 为 ETTh2 `0.99`、ETTm1 `0.99`、Weather `1.00`。train-only proxies
对 raw MSE 的相关性不能直接作为 reliability evidence，因为它可能只是共同随 future distance 增长。

[Decision] B1-RED 状态为 `partial_pass_distance_confounded`。B2 reliability-aware supervision
allocation 不通过 narrative gate，不能进入 implementation。StageB 回到 Step 2/3：若继续，必须重新定义
一个能越过 step-distance confounder 的 reliability problem；若不能，则保持 A6-LBF-r256 作为独立
unified forecasting contribution。

### B3 Problem Redefinition Decision

[Decision] StageB Step 2/3 redefinition completed at
`analysis/phase5_stage_b_step23_redefinition_20260706/`.

[Rejected] Raw future-unit reliability 和 B2-RAS 不能继续。原因是 B1 已证明 raw hard/easy units 与
forecast distance 几乎同序，直接加权会退化为 horizon-distance weighting。

[Proposed] 更强的问题候选是 `B3-DSR`: distance-normalized seasonal residual reliability。它不再预测
raw unit MSE，而是先移除 step-distance trend，再测试 train-only `seasonal_residual` 是否解释
structural residual difficulty。

[Evidence] B1 中 `seasonal_residual` 对 detrended MSE 的 Spearman 相关在 ETTh2/ETTm1/Weather
分别为 `0.35/0.59/0.81`，而对 raw MSE 的相关为 `-0.74/-0.79/-0.10`。这说明该 proxy 不只是
随 step 增长的 distance proxy。

[Decision] B3 只通过 problem-candidate gate，尚未通过 Step 4-6 narrative gate。下一步只能做
diagnostic robustness：detrending 形式、block size、bootstrap stability 和 leakage-free train-only
path。若 B3 失败，StageB 应关闭或转入更大的 label-autocorrelation objective problem。

### B3 Diagnostic Decision

[Decision] B3 distance-normalized seasonal residual diagnostic completed at
`analysis/phase5_stage_b_distance_normalized_seasonal_residual_20260706/`.

[Moderate Evidence] `seasonal_residual` 对 `linear_step_residual` 的 Spearman 在所有 dataset/block
size 上均为正：ETTh2 `0.31-0.38`、ETTm1 `0.43-0.62`、Weather `0.81-0.83`。

[Counter-Evidence] 该信号没有通过更严格的 robustness gate。ETTh2/ETTm1 的 `rank_step_residual`
出现负相关；Weather 的 `prefix_normalized_residual` 在 block `24/96` 为负；bootstrap sign stability
在多个 rank/prefix label 上低于阈值。`nan` rank residual 表示对应 block size 下 unit MSE rank 被
monotonic step trend 完全解释，不能作为正证据。

[Decision] B3 状态为 `partial_pass_needs_stronger_proxy_or_method_boundary`。当前不得实现
reliability-aware loss weighting。下一步只能二选一：继续寻找更强 train-only structural proxy，或关闭
StageB 并转向更大的 label-autocorrelation objective problem。

### TimeAlign Dependency Diagnostic Decision

[Decision] StageB dependency audit completed at
`analysis/phase5_stage_b_timealign_dependency_audit_20260706/`.

[Evidence] Under the same inherited TimeAlign align/recon setting, A6-LBF-r256 beats official unified TimeAlign
on `11/12` settings with mean MSE change `-1.94%`. This supports an A6-LBF head/operator contribution.

[Risk] The current objective still uses inherited `w_recon * recon_loss + w_align * align_loss`. Last-epoch weighted
alignment share is ETTh2 `0.19`, ETTm1 `0.08`, Weather `0.12`; therefore full architecture independence is not
established.

[Decision] This audit produced `partial_dependency_risk_confirmed` and required a causal no-align/no-recon ablation
before any B5 basis-aware alignment design. That ablation has now returned and supersedes the audit-level decision
below.

### TimeAlign Dependency Ablation Decision

[Decision] B4 no-align/no-recon dependency ablation completed at
`analysis/phase5_stage_b_timealign_dependency_ablation_20260706/`.

[Fact] The returned 12-run matrix compares four A6-LBF-r256 arms:
`current_align_recon`, `no_align_recon`, `align_no_recon`, and `no_align_no_recon` on ETTh2/ETTm1/Weather with
horizons 96/192/336/720.

[Strong Evidence] Removing both inherited auxiliary losses does not collapse A6-LBF-r256. `no_align_no_recon`
changes mean MSE by only `+0.07%` versus current and wins `7/12` horizon settings. `align_no_recon` is slightly
better on mean MSE (`-0.04%`) and wins `8/12`, but the effect size is too small to justify a new align method by
itself.

[Mechanism Note] `no_align_recon` and `no_align_no_recon` are metric-identical in the returned artifacts. This is
consistent with the current code path: when `w_align=0`, reconstruction alone mostly trains the future branch rather
than the history-derived forecast operator.

[Decision] B4 status is `dependency_ablation_pass_for_head_contribution_but_not_for_b5`. The result strengthens the
paper boundary for Contribution 1: A6-LBF is not merely an inherited TimeAlign alignment artifact. It simultaneously
weakens B5 basis-aware future alignment as the next paper-core method, because the diagnostic did not show a material
dependence on inherited alignment.

[Code Decision] The active A6-LBF implementation now removes the future reconstruction/alignment branch and sets
`w_recon=w_align=0.0` for `readout_mode=learned-basis-forecast-operator`. Official TimeAlign keeps its inherited
future branch for baseline reproduction. This makes A6-LBF a cleaner research carrier: the only active mechanism is
history encoder -> learned coefficients -> prefix-native temporal basis.

### B6 Prefix-Native Objective Entry

[Decision] StageB rolls back to Step 2/3 and opens `B6-PLO`: prefix-native label/basis objective diagnostic.

[Problem] A6-LBF-r256 already changes the forecast operator into learned-basis coefficient space, and its active code
path now removes inherited generic TimeAlign auxiliary terms. The next credible StageB question is whether the
remaining time-domain prefix point objective should explicitly match the prefix-native label autocorrelation /
learned-basis structure.

[Updated After Code Cleanup] After removing inherited auxiliary terms from A6-LBF, the B6 question becomes sharper:
the architecture is now cleanly basis-native, while the objective is still time-domain prefix L1. B6 asks whether the
target labels/residuals have a stable basis-space structure that should guide supervision.

[Required Diagnostic Before Implementation] B6 must first test train-only label autocorrelation, learned-basis
projection coverage, and coefficient-space residual structure. It may use `current_align_recon` and
`no_align_no_recon` as controls, but no new objective or alignment method should be implemented until Step 4-6
narrative gate is written and passed.

### B6 Prefix-Native Objective Diagnostic Decision

[Decision] B6 offline diagnostic completed at
`analysis/phase5_stage_b_prefix_native_objective_diagnostic_20260706/`.

[Fact] Train labels are compressible, but mostly by generic low-frequency structure. PCA top32 versus DCT top32 is:
ETTh2 `0.917/0.889`, ETTm1 `0.939/0.930`, Weather `0.832/0.831`.

[Counter-Evidence] A6 learned temporal basis does not provide top32 advantage over DCT. Label coverage is ETTh2
`0.675`, ETTm1 `0.690`, Weather `0.251`; residual coverage is ETTh2 `0.287`, ETTm1 `0.110`, Weather `0.081`.
All are weaker than or close to DCT at top32.

[Decision] Status is `diagnostic_not_enough_pause_b6`. Do not implement a prefix-native label/basis objective now.
The result would not provide a clean distinction from generic frequency-domain auxiliary losses such as FreDF/TransDF.

### Clean A6 Rerun Decision

[Decision] Clean A6-LBF-r256 rerun completed at
`analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706/` with status `clean_a6_validated`.

[Fact] The active implementation has no A6 future reconstruction/alignment branch and trains with
`effective_w_recon=0.0`, `effective_w_align=0.0`.

[Strong Evidence] The clean rerun still beats fixed-horizon TimeAlign by overall mean MSE `-4.13%` with `9/12`
MSE wins, and official unified TimeAlign by `-1.75%` with `11/12` MSE wins. Relative to historical A6-LBF-r256 it
changes mean MSE by only `+0.20%`.

[Decision] StageA clean carrier is validated. StageB remains paused; do not implement B5 or B6. If research
continues beyond Contribution 1, return to Step 2/3 and define a problem that is neither generic frequency
supervision nor step-distance-confounded reliability weighting.

### B7 Unified Prefix Optimization Diagnostic

[Decision] B7-UPO diagnostic completed at
`analysis/phase5_stage_b_unified_prefix_optimization_20260707/`.

[Problem] A6-LBF-r256 已经提供 unified forecast operator，但当前 `multi-prefix` objective 把
`h96/h192/h336/h720` prefix losses 简单平均。由于每个 prefix loss 又对 `1..H` 求均值，short future
steps 被多个 prefix 重复覆盖，long-tail steps 只被 long horizon 覆盖。

[Fact] 当前 horizons `[96,192,336,720]` 下，`0-96` segment 的平均 scalar supervision weight 是
`336-720` tail segment 的 `14.39x`；`96-192` 是 `6.89x`；`192-336` 是 `3.14x`。

[Moderate Evidence] Segment-level comparison shows A6 gains vs fixed TimeAlign shrink in the under-weighted tail:
overall early `0-96` relative MSE is `-3.57%`, while tail `336-720` is only `-0.16%`. ETTh2 and ETTm1 support this
reading; Weather is a counterexample.

[Decision] B7-UPO status is `prefix_imbalance_problem_candidate`, not method-ready. It is stronger than reviving B6
because it directly deepens unified prediction training mechanics and does not become a generic frequency auxiliary
loss. However, it must pass a stronger gradient/task diagnostic before any implementation.

[Next Required Action] Run `B7-GTD`: compute per-prefix gradient cosine similarities, gradient norms, and conflict
pairs on A6 shared parameters for small train batches. If conflict/imbalance is stable and aligns with segment-tail
weakness, then enter Step 4-6 method design. If not, pause StageB again.

### B8 Future-Query Aligned Architecture Direction

[Decision] `B7-UPO` 有价值，但当前降级为 small objective contribution candidate。随后 StageB 曾提出
`B8-FQA` 作为 architecture-level candidate。

[Motivation] StageA 主要改变 decoder/head。第二个主创新点应改变 feeding A6 learned-basis forecast operator
的 representation interface。

[Problem] A6-LBF-r256 为每个 channel 计算一个 sample-specific coefficient vector：

```text
coeff = learned_basis_coeff(hidden)
y[t, c] = learned_temporal_basis[t] @ coeff[c] + bias[t]
```

因此 `coeff[c]` 对 future position 不变。Future positions 由全局 basis rows 区分，但没有
sample-specific target-position representation。

[Idea] 引入 future-position query/placeholder tokens，使其 attend 到 history tokens，并在 learned-basis
operator 前生成 future-segment-specific coefficient modulation。modulation gate 零初始化，使初始 forward
与 clean A6-LBF-r256 完全等价。

[Literature] 本轮不是只参考 Zotero/本地 notes；已外部核验 TimeAlign、ElasTST、TimePerceiver 的 arXiv 或官方
repository 资料。TimeAlign 提供 future-aligned representation 的问题动机；ElasTST 证明 future
placeholders 与 structured masks 可服务 horizon-invariant varied-horizon forecasting；TimePerceiver 使用
target-position-aware decoder queries。B8 只把这些作为机制证据，并将其改写到 A6 的 basis-coefficient
interface。

[Diagnostic] `B8-OCD` 已完成，见
`analysis/phase5_stage_b_b8_ocd_coefficient_oracle_20260707/b8_ocd_report.md`。诊断固定 clean A6
prediction 与 checkpoint 中的 `learned_temporal_basis`，比较 learned basis 与 DCT control 的 global/segment
residual correction。

[Fact] learned basis 的 segment-specific correction 确实相对 global correction 有明显 headroom。Rank 64 下
segment-minus-global reduction 为 ETTh2 `16.85%`、ETTm1 `28.19%`、Weather `22.01%`。

[Counter-Evidence] DCT control 的绝对 residual reduction 更强。Rank 64 segment reduction 中，learned basis
为 ETTh2 `79.05%`、ETTm1 `72.77%`、Weather `61.91%`，DCT control 为 ETTh2 `87.61%`、ETTm1
`91.85%`、Weather `91.18%`。

[Decision] `B8-FQA` 状态改为 `rejected_by_ocd_control`。叙事逻辑仍然通顺，但当前 evidence 不能证明它是
A6 learned-basis coefficient interface 特有的强 architecture problem；不要实现 B8。

[Rollback] StageB 回到 Step 2/3，继续寻找 architecture-level 第二主创新问题。B7-UPO 仍仅作为 small
objective contribution candidate 保留。

## Active Implementation

| File | Role |
| --- | --- |
| `baselines/timealign_official/models/TimeAlign.py` | clean official + A6-LBF model |
| `baselines/timealign_official/train_repo.py` | clean training/evaluation adapter |
| `scripts/remote/run_phase5_a6_lbf_r256_main.sh` | clean A6-LBF-r256 remote runner |
| `docs/code-explanation/phase5-clean-timealign-a6-lbf.md` | implementation explanation |
| `scripts/analyze_phase5_stage_b_reliability_diagnostic.py` | StageB B1 diagnostic analyzer |
| `scripts/analyze_phase5_stage_b_b3_dsr_diagnostic.py` | StageB B3 diagnostic analyzer |
| `scripts/analyze_phase5_stage_b_timealign_dependency_audit.py` | StageB TimeAlign dependency audit analyzer |
| `scripts/analyze_phase5_stage_b_timealign_dependency_ablation.py` | returned no-align/no-recon dependency ablation analyzer |
| `scripts/analyze_phase5_stage_b_b6_prefix_objective_diagnostic.py` | StageB B6 prefix-native objective diagnostic analyzer |
| `scripts/analyze_phase5_a6_clean_operator_rerun.py` | clean A6 validation analyzer |
| `scripts/analyze_phase5_stage_b_unified_prefix_optimization.py` | B7 unified prefix optimization diagnostic analyzer |
| `scripts/analyze_phase5_stage_b_b8_ocd_coefficient_oracle.py` | B8-OCD coefficient-space oracle diagnostic analyzer |
| `scripts/remote/run_phase5_stage_b_timealign_dependency_ablation.sh` | completed no-align/no-recon ablation runner |
| `docs/experiments/phase5-stage-b-distance-normalized-seasonal-residual-diagnostic.md` | StageB B3 diagnostic protocol |
| `docs/experiments/phase5-stage-b-timealign-dependency-and-basis-align-diagnostic.md` | StageB dependency/basis-align protocol |
| `docs/experiments/phase5-stage-b-prefix-native-label-objective-diagnostic.md` | StageB B6 prefix-native objective diagnostic protocol |
| `docs/experiments/phase5-stage-b-unified-prefix-optimization-diagnostic.md` | StageB B7 unified prefix optimization diagnostic protocol |
| `docs/experiments/phase5-stage-b-future-query-aligned-basis-architecture.md` | StageB B8 future-query aligned basis architecture protocol |
| `docs/code-explanation/phase5-stage-b-b6-prefix-objective-diagnostic.md` | B6 diagnostic analyzer explanation |
| `docs/code-explanation/phase5-clean-a6-rerun-analysis.md` | clean A6 validation analyzer explanation |
| `docs/code-explanation/phase5-stage-b-b7-unified-prefix-optimization.md` | B7 diagnostic analyzer explanation |
| `docs/code-explanation/phase5-stage-b-b8-ocd-coefficient-oracle.md` | B8-OCD analyzer explanation |

## Archive Map

| Path | Content |
| --- | --- |
| `docs/archive/phase5-stage-a/` | old StageA design/code-explanation docs |
| `analysis/phase5_timealign_hss_a6_capacity_native_gate_20260703/` | accepted A6-LBF evidence |
| `analysis/phase5_stage_b_reliability_diagnostic_20260706/` | B1 distance-confounded reliability diagnostic |
| `analysis/phase5_stage_b_step23_redefinition_20260706/` | B3 problem redefinition audit |
| `analysis/phase5_stage_b_distance_normalized_seasonal_residual_20260706/` | B3 partial/not method-ready diagnostic |
| `analysis/phase5_stage_b_timealign_dependency_audit_20260706/` | TimeAlign dependency risk audit |
| `analysis/phase5_stage_b_timealign_dependency_ablation_20260706/` | B4 dependency ablation result |
| `analysis/phase5_stage_b_prefix_native_objective_diagnostic_20260706/` | B6 negative diagnostic |
| `analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706/` | clean A6 validation result |
| `analysis/phase5_stage_b_unified_prefix_optimization_20260707/` | B7 unified prefix optimization diagnostic |
| `analysis/phase5_stage_b_future_query_aligned_architecture_research_20260707/` | B8 future-query aligned architecture direction research |
| `analysis/phase5_stage_b_b8_ocd_coefficient_oracle_20260707/` | B8-OCD negative coefficient oracle diagnostic |
| `analysis/phase5_stage_a_architecture_exhaustion_audit_20260705/` | old route-level audit before A6-LBF was promoted |

## Current Prohibitions

- 不再启动 A5/A6-QBR/A6S/A6ST/A7DG/A8TAG 旧路线；
- 不把 old B0 ablation 直接作为 StageB method；
- 不恢复 teacher/self-teacher/EMA/diagnostic export 旧代码，除非新的 StageB narrative gate 明确需要；
- 不在 remote 端手工修改代码；先 commit/push，再 remote `git pull`。
