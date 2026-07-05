# Phase5 TimeAlign Interface Stage Ledger

本文档是 Phase5 当前主线账本。旧 StageA 候选细节已归档到
`docs/archive/phase5-stage-a/`；详细实验结果仍保存在 `analysis/`。

## Decision Cursor

| Field | Content |
| --- | --- |
| `stage_id` | `phase5-timealign-interface` |
| `current_11_step` | StageA fixed；StageB 回到 Step 2/3 problem redefinition |
| `active_carrier` | `A6-LBF-r256` on official-source TimeAlign |
| `active_question` | 如何基于 A6-LBF-r256 这个 unified carrier 继续研究 future-aware / reliability-aware StageB mechanism |
| `latest_decision` | A6-LBF-r256 已从 StageA partial evidence 上调为论文重要创新点与 StageB 起点 |
| `next_required_action` | StageB 重新设计；不得沿用 pre-cleanup B0 diagnostic 作为正式 method |
| `rollback_point` | 若 StageB 新问题无法形成 narrative gate，则回到 A6-LBF-r256 作为独立 unified forecasting contribution |

## StageA Fixed Result

| Item | Decision |
| --- | --- |
| Accepted variant | `A6-LBF-r256` / `learned-basis-forecast-operator` |
| Role | 论文重要组成部分；StageB carrier |
| Main claim | 一个 unified model 在当前三数据集上整体优于 fixed-horizon per-horizon TimeAlign baseline |
| Code status | 主代码仅保留 `official` 与 `learned-basis-forecast-operator` |
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

## Active Artifacts

| Artifact | Purpose |
| --- | --- |
| `baselines/timealign_official/models/TimeAlign.py` | clean official + A6-LBF model |
| `baselines/timealign_official/train_repo.py` | clean official/A6-LBF training adapter |
| `docs/code-explanation/phase5-clean-timealign-a6-lbf.md` | code-facing explanation |
| `scripts/remote/run_phase5_a6_lbf_r256_main.sh` | clean A6-LBF-r256 remote runner |
| `analysis/phase5_timealign_hss_a6_capacity_native_gate_20260703/` | StageA accepted evidence |

## Archived Evidence

旧 StageA route 的详细候选记录不再放在 active ledger 中。若需要审计历史，可读：

- `docs/archive/phase5-stage-a/experiments/`
- `docs/archive/phase5-stage-a/code-explanation/`
- `analysis/phase5_stage_a_architecture_exhaustion_audit_20260705/`
- `analysis/phase5_timealign_hss_a6_capacity_native_gate_20260703/`
