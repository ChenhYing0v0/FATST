# Paper Mainline

## Current Position

| Field | Content |
| --- | --- |
| `paper_target` | 高水平 SCI 期刊时间序列预测论文 |
| `working_title` | Projective Forecasting: Decoder-Objective Co-Design for Unified Varied-Horizon Forecasting |
| `current_stage` | `StageC-UVHF` active；StageB 已归档 |
| `current_11_step` | Step 2-3：PMFO/PIR problem-existence diagnostics |
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
- 支持 prefix-restricted computation，而不是总先生成 H720；
- contribution 来自 refinement algebra 与 computation contract，不是“又一个 continuous basis decoder”。

当前状态：`problem/design candidate`。FlowState、TimePerceiver、ElasTST 等 prior art 已压缩 novelty 空间；
必须先通过 nested increment problem diagnostic 和专项 prior-art gate。

### Contribution 2 Candidate: Projective Increment Risk

Horizon measure 是有效的 deployment-risk 定义，但 simple step weighting 已接近 ElasTST，不足以独立成文。
PIR 候选在 PMFO 的 nested projections 上分解 coarse trajectory 与 refinement increments，使 loss unit 与
decoder unit一致，并由 deployment measure 决定 increment risk，而不是重复平均多个 overlapping prefixes。

当前状态：`problem candidate`。必须证明 projected increments 提供 raw harmonic step weighting之外的
稳定信息；否则退回 training-only DRO/measure protocol，且不强行包装为 Contribution 2。

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

[Fact] A6 已通过 full trajectory restriction 获得 exact prefix consistency，因此“修复 prefix
inconsistency”不是新问题。[Fact] A6 无论 requested horizon 多短都生成 H720，因此 horizon-adaptive
computation 与 refinable representation 仍未解决。

[Decision] 旧 StageB coefficient conditioning、STBO、GRU future composition、unit-specific retrieval 与
encoder repair 均不再是 active candidate。历史失败只按各自 failure attribution 使用，不能被扩大为未经
测试的方向级结论，也不能因为 archive 中代码仍存在而自动复活。

## Main Experiment Logic

1. 固定 natural A6 baseline 与 test reference；
2. offline diagnostics 验证 nested representation 与 measure-gradient problem；
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

2026-07-13 reset 前主线完整 snapshot 位于
`docs/archive/pre-stage-c-reset-20260713/`，仅作历史审计。
