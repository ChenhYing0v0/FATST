# Paper Mainline

本文档记录当前论文主线。旧 StageA 候选、诊断和中间 rollback 细节已从主线文档移出，保存在
`docs/archive/phase5-stage-a/` 与 `analysis/`。

## Current Position

| Field | Content |
| --- | --- |
| `paper_target` | 高水平 SCI 期刊时间序列预测论文 |
| `working_title` | Horizon-Agnostic Supervision Scheduling for Unified Multi-Horizon Forecasting |
| `current_stage` | Phase5 StageA clean A6 validated；StageB B9-FSN-SCF local implementation smoke passed |
| `active_carrier` | `A6-LBF-r256` |
| `active_stage_ledger` | `docs/stage-ledgers/phase5-timealign-interface.md` |
| `current_11_step` | StageB Step 7 B9-FSN-SCF minimal implementation and local smoke passed |
| `paper_core_status` | A6-LBF-r256 pure operator 是当前唯一 accepted paper-core method；B9-FSN-SCF 是可进入 small gate 的第二贡献候选 |

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
loss，难以区别 FreDF/TransDF。B6 因此作为严谨的负诊断边界保留。

当前新的 StageB candidate 是 `B7-UPO`: unified prefix optimization。它不再问 label/basis 是否需要
frequency-like auxiliary objective，而是问 A6-LBF 的 unified forecast operator 是否被 nested
multi-prefix objective 公平、稳定地优化。初步 offline diagnostic 显示，当前 `multi-prefix` loss 使 `0-96`
steps 获得 `336-720` tail steps 的 `14.39x` scalar supervision weight；segment-level A6 gains vs fixed
TimeAlign 从 early `-3.57%` 收窄到 tail `-0.16%`。该方向与 StageA 的 unified prediction 叙事更连贯，
但仍只是 `problem_candidate`：Weather 是反例，且还缺少 gradient/task-interference evidence。

根据用户对 StageB 主贡献的约束，B7 当前降级为 small objective candidate。随后提出的 architecture
candidate 是 `B8-FQA`: Future-Query Aligned Basis Operator。它的核心问题是：A6-LBF 已有
prefix-native learned-basis decoder，但 sample-specific coefficient vector 对 future positions 是不变的；
StageB 可以引入 future-position query/placeholder tokens，在进入 basis operator 前生成
target-position-aware coefficient modulation。该方向更适合作为第二个主创新点，因为它改变 representation
interface，而不是只改 loss。该判断已补充外部网络调研证据：TimeAlign、ElasTST、TimePerceiver 的 arXiv
或 official repository 资料分别支撑 future alignment、future placeholders/masks 与 target-query
decoder 的机制可行性；SRP++ 仅作为本地 note 辅助证据。

`B8-OCD` coefficient-space oracle diagnostic 已返回负向控制结果：learned basis 的 segment-specific
correction 相比 global correction 有明显 headroom，但 DCT control 的绝对 residual reduction 更强。Rank
64 下 learned basis 的 segment reduction 为 ETTh2 `79.05%`、ETTm1 `72.77%`、Weather `61.91%`，而
DCT control 为 ETTh2 `87.61%`、ETTm1 `91.85%`、Weather `91.18%`。因此当前证据不足以说明 B8
对应的是 A6 learned-basis coefficient interface 特有的 architecture problem，StageB 不能实现 B8-FQA，
应回到 Step 2/3 重新定义 architecture-level 第二贡献问题。

用户明确排除 residual-style architecture 作为 paper-core route。因此新的 StageB candidate 是
`B9-FSN-SCF`: Stage-Native Coefficient Field。它不做 `y=A6(x)+correction`，而是让 future stage
信息在 basis projection 前进入 primary coefficient/operator path。`B9-SGC` stage-gradient diagnostic
已给出正向问题证据：四个 future stage losses 对同一个 A6 `coeff[b,c]` 的梯度方向相似度很低，mean
pairwise cosine 为 ETTh2 `0.072`、ETTm1 `0.171`、Weather `0.048`，early-tail cosine 为
`0.041/0.112/0.014`。这说明 single coefficient state 同时服务多个 future stages 时存在 native stage
pressure。

Step 4-6 设计门已通过：B9-FSN-SCF 将 A6 的 `coeff: [B,C,K]` 扩展为
`coeff_field: [B,C,S,K]`，其中 `S=4` 对应当前 multi-prefix stages；每个 stage 在 prediction 前生成
自己的 coefficient field，再与同一组 `learned_temporal_basis` 做 projection。该设计通过 zero-gated
multiplicative coefficient modulation 保持 A6 function-preserving fallback。

Step 7 最小实现与本地 smoke 已通过：`stage-native-coefficient-field` 与
`stage-native-coefficient-field-no-stage` 均可训练/评估；B9/no-stage 对 A6 的初始 fallback max abs 为
`0.0`，`H=96` 与 `H=720` prefix consistency max abs 也为 `0.0`。B9 仍未成为 accepted method；下一步
只能做 remote small gate，比较 `a6_clean`、`b9_fsn_scf`、`b9_no_stage`。

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
| `docs/code-explanation/phase5-stage-b-b9-fsn-scf.md` | B9-FSN-SCF implementation explanation |
| `docs/stage-ledgers/phase5-timealign-interface.md` | active StageA/StageB ledger |
| `docs/research-roadmap.md` | active roadmap |
| `docs/experiments/phase5-stage-b-prefix-native-label-objective-diagnostic.md` | next StageB diagnostic protocol |
| `docs/experiments/phase5-stage-b-unified-prefix-optimization-diagnostic.md` | active B7 unified prefix optimization diagnostic protocol |
| `docs/experiments/phase5-stage-b-future-query-aligned-basis-architecture.md` | B8 rejected architecture candidate protocol |
| `docs/experiments/phase5-stage-b-native-future-stage-operator.md` | active B9-FSN-SCF Step 4-6 protocol |
| `scripts/remote/run_phase5_stage_b_b9_fsn_scf_small_gate.sh` | B9-FSN-SCF remote small gate runner |
| `scripts/analyze_phase5_stage_b_b9_fsn_scf_small_gate.py` | B9-FSN-SCF small gate analyzer |
| `scripts/sync_phase5_stage_b_b9_fsn_scf_small_gate_results.sh` | B9-FSN-SCF result sync/analyze wrapper |
| `analysis/phase5_stage_b_timealign_dependency_audit_20260706/` | TimeAlign dependency audit |
| `analysis/phase5_stage_b_timealign_dependency_ablation_20260706/` | no-align/no-recon dependency ablation |
| `analysis/phase5_stage_b_prefix_native_objective_diagnostic_20260706/` | B6 negative diagnostic |
| `analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706/` | clean A6 validation report |
| `analysis/phase5_stage_b_unified_prefix_optimization_20260707/` | B7 problem-candidate diagnostic |
| `analysis/phase5_stage_b_future_query_aligned_architecture_research_20260707/` | B8 architecture direction research |
| `analysis/phase5_stage_b_b8_ocd_coefficient_oracle_20260707/` | B8-OCD negative oracle diagnostic |
| `analysis/phase5_stage_b_b9_stage_gradient_diagnostic_20260707/` | B9-SGC positive problem-candidate diagnostic |

## Next Step

1. Treat StageA clean A6-LBF-r256 as fixed.
2. Do not revive archived StageA code paths.
3. Treat B5 basis-aware alignment as deferred, not the next implementation target.
4. Do not implement B6 objective under current evidence.
5. Defer B7 objective optimization as a small contribution candidate.
6. Do not implement B8-FQA under current evidence.
7. Launch B9-FSN-SCF remote small gate only after commit/push and GPU preflight; do not launch full matrix before the small gate returns.
