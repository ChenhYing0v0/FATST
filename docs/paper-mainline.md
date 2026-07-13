# Paper Mainline

## Current Position

| Field | Content |
| --- | --- |
| `paper_target` | 高水平 SCI 期刊时间序列预测论文 |
| `working_title` | Projective Forecasting: Decoder-Objective Co-Design for Unified Varied-Horizon Forecasting |
| `current_stage` | `StageC-UVHF` active；StageB 已归档 |
| `current_11_step` | PMFO-RCT v1 Step 7B failed；rollback Step 4 redesign |
| `source_evidence` | A6-LBF-r256 historical/source-faithful performance |
| `mechanism_control` | frozen `A6-LBF-natural-baseline` dataset profiles |
| `test_reference` | 3 datasets × 3 seeds × 8 horizons，72/72 complete |
| `future_validation_suite` | ETTh1/ETTh2/ETTm1/ETTm2/Weather；ETTh1/ETTm2 profile calibration pending |
| `active_ledger` | `docs/stage-ledgers/stage-c-unified-forecasting-redesign.md` |
| `paper_core_status` | SC1仍是open contribution slot，但PMFO-RCT v1关闭；SC2-MIPR held until operator refrozen |

## Research Thesis

论文研究问题不是“为几个 benchmark horizons 分别训练或 condition 一个 head”，而是：

> 一个共享模型如何表示一族可限制、可细化的未来预测，并用与该函数族一致的风险定义进行训练，
> 从而在任意 requested horizon 上保持连续、统一且可比较的行为？

requested horizon 在当前主线中只定义输出域与计算域，不作为 learned semantic feature。禁止将离散
horizon ID、benchmark-specific embedding、per-horizon expert 或 per-horizon hyperparameter 作为核心机制。

## Contribution Slots

### Contribution 1 Slot: Projective Forecast Operator Redesign

PMFO的具体`narrative_ready`候选为`PMFO-RCT`。它从A6 history memory建立future interval tree，按
`90 -> 30 -> 10 -> 5 -> 1`逐层生成scaling/detail coefficients，并用fixed orthogonal contrast保证fine
detail不能改写parent coarse projection。目标性质：

- exact refinement recovery与nested-prefix consistency；
- $H$ 只prune与prefix相交的tree nodes，不进入learned state/query/router；
- parent-to-child shared state transition + orthogonal detail complement + local support；
- contribution来自future-side refinement conservativity与domain execution，不是“又一个wavelet/continuous
  basis decoder”。

PMFO-RCT v1已完成其falsification职责：theory/local invariants成立，但Step 7B相对A6的dense-MSE macro为
`-1.0955%`，三dataset均退化，故不能成为paper core。组件归因并不相同：conservative synthesis相对
no-conservation在三dataset一致改善（macro `+2.3393%`），保留为redesign证据；recursive transition相对
no-transition仅`+0.0486%`且跨dataset不一致，v1 claim撤回；structured decoder相对matched dense的
`+0.7193%`只是弱信号。

[Decision] 关闭范围仅是固定`90/30/10/5/1` mixed-radix partition、v1 state transition和整体替换A6
readout的组合。Contribution 1 slot与projectivity/conservation问题仍开放；回到Step 4重审function-class
containment、future partition与history-to-node interface。Step 7B没有操纵Encoder，不能据此认定Encoder不足。

### Contribution 2 Candidate: Measure-Induced Projective Risk

SC2保留`PIR` slot ID，formal objective收紧为`MIPR`。raw horizon measure的exact risk为
$e^TW_\mu e$；MIPR定义$\widetilde W_\mu=\sum_lQ_lW_\mu Q_l$，在PMFO refinement blocks上保留
within-scale weighting并删除cross-scale coupling。它是decoder-aligned structured surrogate，不是比raw
risk“更measure-aligned”的等价改写。

当前状态：`narrative_ready / effectiveness_pending / held_after_SC1_rollback`。L2下quadratic algebra成立；
Huber/L1没有exact block-metric等价，首轮不实现。`log_uniform_h` off-block energy为`0.205154`，
`uniform_h/benchmark_h`只有`0.003456/0.002480`，因此贡献主场景必须是continuous dense-horizon
deployment，不能只靠四个benchmark horizons。

[Diagnostic status] D1-v2 aggregate PIR problem gate通过，但证据具有measure boundary：log-uniform强、
uniform弱而跨dataset、benchmark projected excess 0/3。该历史边界已在Step4-6收紧为MIPR与
same-measure raw control。

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

[Decision] horizon与resolution必须分离：$H$只定义prefix domain，refinement level定义同一future
function的分辨率。禁止令短H选择fine branch、长H选择coarse branch。任何显式输出H个值的方法都有
$\Omega(HC)$写出下界，PMFO只claim避免out-of-prefix atoms与global dense synthesis，不claim sublinear
total generation。

[Decision] D1已分别审计`memory: [B,C,P,D]`的information sufficiency和`basis: [720,256]`的
capacity/localization geometry。当前保留A6 Encoder、替换dense basis/operator；只有后续stable probes证明
history信息已经丢失时，才允许回滚并审计最小multiscale encoder interface。

[Decision] 旧 StageB coefficient conditioning、STBO、GRU future composition、unit-specific retrieval 与
encoder repair 均不再是 active candidate。历史失败只按各自 failure attribution 使用，不能被扩大为未经
测试的方向级结论，也不能因为 archive 中代码仍存在而自动复活。

[Decision] Step 7B将“结构正确”与“预测有效”明确分离：15/15 trained invariants通过说明实现与algebra无误，
但不补偿三dataset performance gate失败。当前归因为exact v1 `readout_or_head_design_wrong`，而非
`optimization_or_numeric_pathology`、Encoder方向失败或conservation方向失败。

## Main Experiment Logic

1. 固定 natural A6 baseline 与 test reference；
2. D1-A验证label/residual nested structure，D1-B验证A6 Encoder information sufficiency，D1-C验证
   learned basis geometry，同时审计measure/projected gradients；
3. PMFO-RCT与MIPR已分别通过Step 4-6 narrative/theory gate；
4. Step 7A local invariants通过；Step 7B使用frozen full-H720 pointwise L1完成15-run architecture controls；
5. PMFO-RCT v1 effectiveness失败，回滚Step 4；MIPR、factorial与full matrix全部暂停；
6. 新SC1候选必须先解释A6 function class、fixed partition与interface问题，再重新过Step 4-6；
7. 只有新SC1通过screening后，才恢复same-measure raw versus MIPR、`2x2` factorial、3-seed full matrix；
8. 第二 backbone与 official native baselines 最后做 generality gate。

未来candidate screening固定扩展到ETTh1、ETTh2、ETTm1、ETTm2、Weather。五dataset用于cross-dataset
generality，seeds2021/2022/2023用于stochastic confirmation；两者不能互相替代。ETTh1/ETTm2必须先完成
validation-only natural profile freeze。

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
- `analysis/stage_c_step46_pmfo_pir_theory_gate_20260713/step46_design_and_prior_art.md`
- `analysis/stage_c_step7a_pmfo_rct_local_20260713/step7a_local_gate_report.md`
- `analysis/stage_c_step7b_pmfo_rct_20260713/step7b_screening_report.md`
- `analysis/stage_c_step7b_pmfo_rct_20260713/failure_attribution_addendum.md`
- `docs/experiments/stage-c-five-dataset-validation-policy.md`
- `docs/code-explanation/stage-c-pmfo-rct-step7a.md`

2026-07-13 reset 前主线完整 snapshot 位于
`docs/archive/pre-stage-c-reset-20260713/`，仅作历史审计。
