# Paper Mainline

本文档记录当前论文主线。旧 StageA 候选、诊断和中间 rollback 细节已从主线文档移出，保存在
`docs/archive/phase5-stage-a/` 与 `analysis/`。

## Current Position

| Field | Content |
| --- | --- |
| `paper_target` | 高水平 SCI 期刊时间序列预测论文 |
| `working_title` | Horizon-Agnostic Supervision Scheduling for Unified Multi-Horizon Forecasting |
| `current_stage` | Phase5 StageA fixed；StageB pending redesign |
| `active_carrier` | `A6-LBF-r256` |
| `active_stage_ledger` | `docs/stage-ledgers/phase5-timealign-interface.md` |
| `current_11_step` | StageB Step 2/3 problem redefinition |
| `paper_core_status` | A6-LBF-r256 已成为论文重要创新点；后续 StageB 以它为起点 |

## Core Claim

[Claim] A6-LBF-r256 将 TimeAlign 的 final prediction head 改写为 prefix-native learned-basis forecast
operator。它用一个 unified 720-step model 覆盖 96/192/336/720 多个 prediction horizons，并在当前
实验集合上整体优于 fixed-horizon per-horizon TimeAlign。

## Main Contribution Draft

### Contribution 1: Learned-Basis Unified Forecast Operator

A6-LBF-r256 的机制：

- history encoder 输出 `hidden: [B, C, R]`;
- `learned_basis_coeff(hidden)` 生成 per-channel forecast coefficients；
- `learned_temporal_basis[:H]` 根据 requested horizon `H` 选择 prefix-native temporal basis；
- 输出 `prediction: [B, H, C]`；
- 不依赖 dense-row anchor、teacher、EMA、nested residual 或 target-query path。

论文叙事边界：

- 这是一个 unified multi-horizon architecture contribution；
- 它直接挑战 fixed-horizon per-horizon 训练的必要性；
- 它不是 early-stop、best-val、teacher distillation 或手工 horizon routing。

### Contribution 2: StageB Future-Aware Reliability Mechanism

StageB 尚未设计完成。它必须基于 A6-LBF-r256，而不是旧 StageA 失败候选。预期问题是：

> 在已经成立的 unified forecast operator 上，future-aware supervision 或 reliability-aware allocation
> 是否能进一步提升稳定性与机制解释力？

StageB 进入实现前必须重新完成 Step 2-6，包括 problem definition、existence evidence、idea、
theory check、design 和 narrative gate。

## Evidence Snapshot

### A6-LBF-r256 vs fixed-horizon per-horizon TimeAlign

Protocol: official-last；datasets: ETTh2 / ETTm1 / Weather；horizons: 96/192/336/720。

| Dataset | A6-LBF MSE wins | Mean MSE change |
| --- | ---: | ---: |
| ETTh2 | 4/4 | `-10.89%` |
| ETTm1 | 3/4 | `-1.46%` |
| Weather | 2/4 | `-0.36%` |
| Overall | 9/12 | `-4.82%` |

### A6-LBF-r256 vs official unified TimeAlign

| Dataset | A6-LBF MSE wins | Mean MSE change |
| --- | ---: | ---: |
| ETTh2 | 4/4 | `-3.39%` |
| ETTm1 | 3/4 | `-1.01%` |
| Weather | 4/4 | `-1.19%` |
| Overall | 11/12 | `-1.92%` |

## Method Boundary

Accepted into current mainline:

- `official` TimeAlign baseline；
- `A6-LBF-r256` learned-basis forecast operator；
- official-last protocol；
- multi-prefix evaluation on 96/192/336/720。

Archived or inactive:

- A2/A3 nested decoders；
- A4 reliability diagnostics；
- A5 target-query / continuous fixed-basis designs；
- A6-DER capacity ceiling；
- A6-QBR query-bilinear readout；
- A6S/A6ST/A7DG/A8TAG stability and teacher routes；
- pre-cleanup B0 pressure ablation。

## Active Files

| File | Purpose |
| --- | --- |
| `baselines/timealign_official/models/TimeAlign.py` | clean official + A6-LBF model |
| `baselines/timealign_official/train_repo.py` | clean training and evaluation adapter |
| `scripts/remote/run_phase5_a6_lbf_r256_main.sh` | clean A6-LBF-r256 remote runner |
| `docs/code-explanation/phase5-clean-timealign-a6-lbf.md` | code explanation |
| `docs/stage-ledgers/phase5-timealign-interface.md` | active StageA/StageB ledger |
| `docs/research-roadmap.md` | active roadmap |

## Next Step

1. Treat StageA as fixed.
2. Do not revive archived StageA code paths.
3. Design StageB from A6-LBF-r256 with a fresh narrative gate.
4. Only after StageB design is written and approved, implement new StageB code and launch remote experiments.
