# Research Roadmap

本文档是当前可重启的主研究路径。旧阶段细节已归档或保存在 `analysis/`，本文件只保留会影响后续
决策的结论。

## Current State

| Field | Content |
| --- | --- |
| `paper_target` | 高水平 SCI 期刊时间序列预测论文 |
| `working_title` | Horizon-Agnostic Supervision Scheduling for Unified Multi-Horizon Forecasting |
| `current_stage` | Phase5：A6-LBF-r256 fixed as StageA result；StageB pending redesign |
| `current_11_step` | StageB Step 2/3：重新定义基于 A6-LBF 的 future-aware reliability problem |
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

## Active Implementation

| File | Role |
| --- | --- |
| `baselines/timealign_official/models/TimeAlign.py` | clean official + A6-LBF model |
| `baselines/timealign_official/train_repo.py` | clean training/evaluation adapter |
| `scripts/remote/run_phase5_a6_lbf_r256_main.sh` | clean A6-LBF-r256 remote runner |
| `docs/code-explanation/phase5-clean-timealign-a6-lbf.md` | implementation explanation |

## Archive Map

| Path | Content |
| --- | --- |
| `docs/archive/phase5-stage-a/` | old StageA design/code-explanation docs |
| `analysis/phase5_timealign_hss_a6_capacity_native_gate_20260703/` | accepted A6-LBF evidence |
| `analysis/phase5_stage_a_architecture_exhaustion_audit_20260705/` | old route-level audit before A6-LBF was promoted |

## Current Prohibitions

- 不再启动 A5/A6-QBR/A6S/A6ST/A7DG/A8TAG 旧路线；
- 不把 old B0 ablation 直接作为 StageB method；
- 不恢复 teacher/self-teacher/EMA/diagnostic export 旧代码，除非新的 StageB narrative gate 明确需要；
- 不在 remote 端手工修改代码；先 commit/push，再 remote `git pull`。
