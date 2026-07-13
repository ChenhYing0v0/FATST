# Paper Mainline

## Current Position

| Field | Content |
| --- | --- |
| `paper_target` | 高水平 SCI 期刊时间序列预测论文 |
| `working_title` | Projective Forecasting: Decoder-Objective Co-Design for Unified Varied-Horizon Forecasting |
| `current_stage` | `StageC-UVHF` active；StageB 已归档 |
| `current_11_step` | Step 4-6：PMFO/PIR prior-art、theory与narrative design gate |
| `source_evidence` | A6-LBF-r256 historical/source-faithful performance |
| `mechanism_control` | frozen `A6-LBF-natural-baseline` dataset profiles |
| `test_reference` | 3 datasets × 3 seeds × 8 horizons，72/72 complete |
| `active_ledger` | `docs/stage-ledgers/stage-c-unified-forecasting-redesign.md` |
| `paper_core_status` | 两个 contribution slots 均为 candidate，尚未通过 narrative/effectiveness gate |

## Research Thesis

论文研究问题不是“为几个 benchmark horizons 分别训练或 condition 一个 head”，而是：

> 一个共享模型如何表示一族可限制、可细化的未来预测，并用与该函数族一致的风险定义进行训练，
> 从而在任意 requested horizon 上保持连续、统一且可比较的行为？

requested horizon 在当前主线中只定义输出域与计算域，不作为 learned semantic feature。禁止将离散
horizon ID、benchmark-specific embedding、per-horizon expert 或 per-horizon hyperparameter 作为核心机制。

## Contribution Slots

### Contribution 1 Candidate: Projective Multiresolution Forecast Operator

PMFO 让 history 一次映射到 nested future function-space coefficients，再按 requested domain 做
restriction/refinement。目标性质：

- exact nested-prefix consistency；
- $H$ 不进入 learned coefficient path；
- 将 single dense rank-256 future subspace重构为可限制、可细化的 nested spaces；
- 通过local support/refinement减少dense `[H,256]` basis evaluation，而不是重复A6已有的`basis[:H]` slicing；
- contribution 来自 refinement algebra 与 computation contract，不是“又一个 continuous basis decoder”。

当前状态：`problem_gate_passed / narrative_pending`。FlowState、TimePerceiver、ElasTST 等 prior art 已压缩
novelty空间；必须通过专项prior-art与refinement proof gate。

[Diagnostic status] D1-v1作废；D1-v2在3 datasets x 3 seeds上通过PMFO problem gate。当前A6 Encoder
保留为首轮carrier，但single dense basis不作为最终PMFO预设；尚无method performance evidence。

### Contribution 2 Candidate: Projective Increment Risk

Horizon measure 是有效的 deployment-risk 定义，但 simple step weighting 已接近 ElasTST，不足以独立成文。
PIR 候选在 PMFO 的 nested projections 上分解 coarse trajectory 与 refinement increments，使 loss unit 与
decoder unit一致，并由 deployment measure 决定 increment risk，而不是重复平均多个 overlapping prefixes。

当前状态：`problem_gate_passed_conditional / narrative_pending`。必须解释projected increments为何在
continuous deployment measure下超越raw harmonic weighting；否则退回training protocol，不强行包装为
Contribution 2。

[Diagnostic status] D1-v2 aggregate PIR problem gate通过，但证据具有measure boundary：log-uniform强、
uniform弱而跨dataset、benchmark projected excess 0/3。SC2只以`problem_gate_passed_conditional`进入Step4-6。

## Frozen Baseline Evidence

natural profile：

- Weather: `patch_num=12, d_model=64, d_ff=128`；
- ETTm1: `patch_num=24, d_model=32, d_ff=64`；
- ETTh2: `patch_num=12, d_model=64, d_ff=128`。

contract hash:
`254d85d47a9e5b7c212f8a8b88decf17a0328a1ea1df324c9cc65be4c672a50c`。
profile 与 checkpoint 均由 validation 预先冻结，test 不参与选择。完整表见
`analysis/stage_c_natural_baseline_test_20260713/natural_baseline_test_report.md`。

## Contribution Boundary

[Fact] A6 先生成`coeff: [B,C,256]`，再使用`basis[:H]: [H,256]`直接计算H步输出；它已经满足domain-only
horizon、exact prefix consistency与output-side O(HK) computation。因此“避免先生成H720”不是新问题。
真正未解决的是：history memory是否保留多尺度可预测信息，以及single global dense basis是否能提供
nested refinement、local support与operator-aligned risk decomposition。

[Decision] A6 Encoder与learned basis均不被预设为PMFO最终组件。D1必须分别审计`memory: [B,C,P,D]`
的information sufficiency和`basis: [720,256]`的capacity/localization geometry。若Encoder信息充分，优先
只重构decoder；只有frozen probes证明history信息已经丢失时，才允许增加最小multiscale encoder interface。

[Decision] 旧 StageB coefficient conditioning、STBO、GRU future composition、unit-specific retrieval 与
encoder repair 均不再是 active candidate。历史失败只按各自 failure attribution 使用，不能被扩大为未经
测试的方向级结论，也不能因为 archive 中代码仍存在而自动复活。

## Main Experiment Logic

1. 固定 natural A6 baseline 与 test reference；
2. D1-A验证label/residual nested structure，D1-B验证A6 Encoder information sufficiency，D1-C验证
   learned basis geometry，同时审计measure/projected gradients；
3. 分别通过 PMFO、PIR 的 Step 4-6 narrative/theory gate；
4. 单 dataset最小 implementation gate；
5. `2x2` factorial 分离 decoder 与 training 的独立主效应；
6. 3 datasets × 3 seeds × dense horizons full matrix；
7. 第二 backbone与 official native baselines 做 generality gate。

任何 candidate 若在 problem或narrative gate失败，回滚 Step 2/3；不得通过叠加 Encoder、MoE、auxiliary
loss 或更多 tuning 来掩盖失败。

## Canonical Active Artifacts

- `docs/stage-ledgers/stage-c-unified-forecasting-redesign.md`
- `docs/research-roadmap.md`
- `docs/experiments/stage-c-pmfo-pir-problem-diagnostic.md`
- `analysis/stage_c_contribution_research_reset_20260713/stage_c_contribution_deep_audit.md`
- `analysis/stage_c_natural_baseline_test_20260713/natural_baseline_test_report.md`
- `analysis/stage_c_d1_pmfo_pir_offline_20260713/`（v1 invalid audit evidence）
- `analysis/stage_c_d1_pmfo_pir_offline_v2_20260713/research_interpretation.md`

2026-07-13 reset 前主线完整 snapshot 位于
`docs/archive/pre-stage-c-reset-20260713/`，仅作历史审计。
