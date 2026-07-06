# Research Roadmap

本文档是当前可重启的主研究路径。旧阶段细节已归档或保存在 `analysis/`，本文件只保留会影响后续
决策的结论。

## Current State

| Field | Content |
| --- | --- |
| `paper_target` | 高水平 SCI 期刊时间序列预测论文 |
| `working_title` | Horizon-Agnostic Supervision Scheduling for Unified Multi-Horizon Forecasting |
| `current_stage` | Phase5：A6-LBF-r256 fixed as StageA result；StageB TimeAlign dependency diagnostic |
| `current_11_step` | StageB Step 2/3：TimeAlign dependency ablation and basis-align precondition |
| `active_carrier` | `A6-LBF-r256` |
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

相对 fixed-horizon per-horizon TimeAlign official-last：

| Dataset | MSE wins | Mean MSE change |
| --- | ---: | ---: |
| ETTh2 | 4/4 | `-10.89%` |
| ETTm1 | 3/4 | `-1.46%` |
| Weather | 2/4 | `-0.36%` |
| Overall | 9/12 | `-4.82%` |

相对 official unified TimeAlign：

| Dataset | MSE wins | Mean MSE change |
| --- | ---: | ---: |
| ETTh2 | 4/4 | `-3.39%` |
| ETTm1 | 3/4 | `-1.01%` |
| Weather | 4/4 | `-1.19%` |
| Overall | 11/12 | `-1.92%` |

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

[Decision] Status is `partial_dependency_risk_confirmed`. Reliability-aware B2/B3 should not continue to method
implementation. The next valid StageB route is TimeAlign dependency ablation followed by basis-aware future alignment
only if the diagnostics pass.

[Next Required Action] Remote ablation matrix has been launched after commit/push and GPU preflight. Wait for the
12-run matrix to return, then analyze before any B5 basis-aware alignment design:
`/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_timealign_dependency_ablation`.

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
| `scripts/remote/run_phase5_stage_b_timealign_dependency_ablation.sh` | pending remote no-align/no-recon ablation runner |
| `docs/experiments/phase5-stage-b-distance-normalized-seasonal-residual-diagnostic.md` | StageB B3 diagnostic protocol |
| `docs/experiments/phase5-stage-b-timealign-dependency-and-basis-align-diagnostic.md` | StageB dependency/basis-align protocol |

## Archive Map

| Path | Content |
| --- | --- |
| `docs/archive/phase5-stage-a/` | old StageA design/code-explanation docs |
| `analysis/phase5_timealign_hss_a6_capacity_native_gate_20260703/` | accepted A6-LBF evidence |
| `analysis/phase5_stage_b_reliability_diagnostic_20260706/` | B1 distance-confounded reliability diagnostic |
| `analysis/phase5_stage_b_step23_redefinition_20260706/` | B3 problem redefinition audit |
| `analysis/phase5_stage_b_distance_normalized_seasonal_residual_20260706/` | B3 partial/not method-ready diagnostic |
| `analysis/phase5_stage_b_timealign_dependency_audit_20260706/` | TimeAlign dependency risk audit |
| `analysis/phase5_stage_b_timealign_dependency_ablation_20260706/` | running remote ablation launch record |
| `analysis/phase5_stage_a_architecture_exhaustion_audit_20260705/` | old route-level audit before A6-LBF was promoted |

## Current Prohibitions

- 不再启动 A5/A6-QBR/A6S/A6ST/A7DG/A8TAG 旧路线；
- 不把 old B0 ablation 直接作为 StageB method；
- 不恢复 teacher/self-teacher/EMA/diagnostic export 旧代码，除非新的 StageB narrative gate 明确需要；
- 不在 remote 端手工修改代码；先 commit/push，再 remote `git pull`。
