# Paper Mainline

本文档记录当前论文主线。旧 StageA 候选、诊断和中间 rollback 细节已从主线文档移出，保存在
`docs/archive/phase5-stage-a/` 与 `analysis/`。

## Current Position

| Field | Content |
| --- | --- |
| `paper_target` | 高水平 SCI 期刊时间序列预测论文 |
| `working_title` | Horizon-Agnostic Supervision Scheduling for Unified Multi-Horizon Forecasting |
| `current_stage` | Phase5 StageA clean A6 validated；StageB paused after negative diagnostics |
| `active_carrier` | `A6-LBF-r256` |
| `active_stage_ledger` | `docs/stage-ledgers/phase5-timealign-interface.md` |
| `current_11_step` | StageA clean validation passed; StageB rollback to Step 2/3 only if a new non-generic problem is found |
| `paper_core_status` | A6-LBF-r256 pure operator 是当前唯一 paper-core method；StageB 暂无可实现贡献 |

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
- 不依赖 dense-row anchor、teacher、EMA、nested residual、target-query path 或 future-recon-branch。

论文叙事边界：

- 这是一个 unified multi-horizon architecture contribution；
- 它直接挑战 fixed-horizon per-horizon 训练的必要性；
- 它不是 early-stop、best-val、teacher distillation 或手工 horizon routing。

### Contribution 2 Candidate: Prefix-Native Objective

StageB 尚未成为正式贡献。B1/B3 reliability route 已证明 raw future-unit weighting 会被
forecast-distance confounder 污染，不能作为 method implementation。

TimeAlign dependency route 的最新结论是：

- artifact-only dependency audit 显示 A6-LBF 在 same TimeAlign align/recon setting 下相对 official
  unified TimeAlign 有 `11/12` MSE wins，mean MSE `-1.94%`；
- no-align/no-recon dependency ablation 显示纯 head/operator arm `no_align_no_recon` 相对 current
  A6-LBF mean MSE 仅 `+0.07%`，且有 `7/12` MSE wins；
- `align_no_recon` 的 mean MSE 略好 `-0.04%`，但 effect size 太小，不能单独支撑一个新的
  basis-aware alignment 方法。

因此 Contribution 1 的 head/operator 证据已经更强：A6-LBF-r256 不只是 inherited TimeAlign
alignment/reconstruction 的 artifact。当前代码也已将 A6-LBF 收束为 pure learned-basis forecast
operator：official TimeAlign baseline 保留 future reconstruction/alignment，A6-LBF 不再包含该 branch
或对应 auxiliary losses。

因此曾经提出的 Contribution 2 候选问题是：

> A6-LBF-r256 已经把 prediction head 改成 learned-basis coefficient space；训练目标是否也应该从
> generic time-domain point loss / generic auxiliary loss，转成与 prefix-native label autocorrelation
> 和 learned-basis residual 结构一致的 objective？

但 B6-PLO Step 2/3 diagnostic 已返回负证据：

- train-label PCA top32 与 DCT top32 几乎相同：ETTh2 `0.917/0.889`，ETTm1 `0.939/0.930`，
  Weather `0.832/0.831`；
- A6 learned basis top32 弱于 DCT：label coverage 为 ETTh2 `0.675`、ETTm1 `0.690`、Weather
  `0.251`，residual coverage 为 ETTh2 `0.287`、ETTm1 `0.110`、Weather `0.081`。

StageB 当前不得实现 prefix-native objective。该方向容易退化为 generic low-frequency/frequency auxiliary
loss，难以区别 FreDF/TransDF。论文主线应暂时以 Contribution 1 为唯一核心方法，并用 B1/B3/B4/B6
作为严谨的负诊断边界。

## Evidence Snapshot

### A6-LBF-r256 vs fixed-horizon per-horizon TimeAlign

Protocol: official-last；datasets: ETTh2 / ETTm1 / Weather；horizons: 96/192/336/720。

| Dataset | A6-LBF MSE wins | Mean MSE change |
| --- | ---: | ---: |
| ETTh2 | 4/4 | `-10.53%` |
| ETTm1 | 3/4 | `-1.64%` |
| Weather | 2/4 | `-0.22%` |
| Overall | 9/12 | `-4.13%` |

### A6-LBF-r256 vs official unified TimeAlign

| Dataset | A6-LBF MSE wins | Mean MSE change |
| --- | ---: | ---: |
| ETTh2 | 4/4 | `-2.78%` |
| ETTm1 | 3/4 | `-1.20%` |
| Weather | 4/4 | `-1.26%` |
| Overall | 11/12 | `-1.75%` |

### Clean A6 validation after removing future-recon branch

The clean rerun at `analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706/` validates the active
implementation: `effective_w_recon=0.0`, `effective_w_align=0.0`, `readout_mode=learned-basis-forecast-operator`,
`basis_rank=256`, and `pred_loss_mode=multi-prefix`.

Relative to the historical A6-LBF-r256 artifact, the clean rerun changes mean MSE by only `+0.20%` overall
(`6/12` MSE wins). Therefore the future reconstruction/alignment branch removal improves contribution boundary
clarity without materially changing the accepted StageA evidence.

### A6-LBF-r256 no-align/no-recon dependency ablation

Protocol: official-last；datasets: ETTh2 / ETTm1 / Weather；horizons: 96/192/336/720。

| Arm | Mean MSE vs current | MSE wins vs current | Decision |
| --- | ---: | ---: | --- |
| `no_align_recon` | `+0.07%` | 7/12 | inherited align not required |
| `align_no_recon` | `-0.04%` | 8/12 | recon not required; tiny align benefit only |
| `no_align_no_recon` | `+0.07%` | 7/12 | pure A6-LBF operator remains competitive |

## Method Boundary

Accepted into current mainline:

- `official` TimeAlign baseline；
- `A6-LBF-r256` pure learned-basis forecast operator；
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
| `scripts/analyze_phase5_a6_clean_operator_rerun.py` | clean A6 validation analyzer |
| `docs/code-explanation/phase5-clean-timealign-a6-lbf.md` | code explanation |
| `docs/code-explanation/phase5-clean-a6-rerun-analysis.md` | clean A6 validation analyzer explanation |
| `docs/stage-ledgers/phase5-timealign-interface.md` | active StageA/StageB ledger |
| `docs/research-roadmap.md` | active roadmap |
| `docs/experiments/phase5-stage-b-prefix-native-label-objective-diagnostic.md` | next StageB diagnostic protocol |
| `analysis/phase5_stage_b_timealign_dependency_audit_20260706/` | TimeAlign dependency audit |
| `analysis/phase5_stage_b_timealign_dependency_ablation_20260706/` | no-align/no-recon dependency ablation |
| `analysis/phase5_stage_b_prefix_native_objective_diagnostic_20260706/` | B6 negative diagnostic |
| `analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706/` | clean A6 validation report |

## Next Step

1. Treat StageA clean A6-LBF-r256 as fixed.
2. Do not revive archived StageA code paths.
3. Treat B5 basis-aware alignment as deferred, not the next implementation target.
4. Do not implement B6 objective under current evidence.
5. Consolidate the paper around Contribution 1 unless a new StageB Step 2/3 problem is found.
