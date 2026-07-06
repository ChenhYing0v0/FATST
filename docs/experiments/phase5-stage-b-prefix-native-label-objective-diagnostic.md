# Phase5 StageB: Prefix-Native Label Objective Diagnostic

`current_step`: StageB Step 2/3 problem definition and existence diagnostic.

本文档定义 `B6-PLO`。它不是一个新 loss 的实现说明，而是 StageB 在 B1/B3/B4 之后的
rollback 入口：先验证 A6-LBF-r256 是否真的存在 architecture-specific objective mismatch，再决定是否
进入 Step 4-6 method design。

## 11-step Record

| Field | Content |
| --- | --- |
| `current_step` | StageB Step 2/3：prefix-native label/basis objective diagnostic |
| `problem` | A6-LBF 已在 learned-basis coefficient space 中预测，clean code 也已移除 future-recon-branch；剩余问题是 objective 仍为 time-domain prefix point loss |
| `existence_evidence` | B1/B3 排除了 raw reliability weighting；B4 表明 inherited align/recon 不是必要性能来源，并触发 A6 clean code cut |
| `idea` | 诊断 train-label autocorrelation、learned-basis projection coverage、coefficient-space residual 是否形成稳定 objective problem |
| `theory_check` | 若 label residual 在 A6 learned basis / train-only label basis 中有稳定结构，则 objective 可以与 forecast operator 对齐；若没有，则继续堆 loss 只是 auxiliary engineering |
| `design` | offline diagnostic only；使用已有 A6 artifacts 和 train split labels，不训练新模型 |
| `narrative_gate` | `diagnostic_not_enough_pause_b6` |
| `effectiveness_gate` | not applicable before method implementation |
| `artifacts` | `analysis/phase5_stage_b_prefix_native_objective_diagnostic_20260706/` |
| `decision` | 不实现 B6 objective；PCA/label-basis 信号主要被 DCT low-frequency control 解释，A6 learned basis 没有形成额外优势 |

## Why This Replaces B5 As Next Step

[Fact] B4 dependency ablation shows `no_align_no_recon` mean MSE only `+0.07%` vs current A6-LBF and wins `7/12`
settings. This means the inherited TimeAlign alignment/reconstruction path is not a strong bottleneck.

[Fact] The active A6-LBF code path now removes the future reconstruction/alignment branch and forces
`w_recon=w_align=0.0`. Official TimeAlign keeps that branch only for baseline reproduction.

[Inference] If we still implement basis-aware alignment now, the paper risk is high: it can look like another small
auxiliary-loss variant rather than a necessary architecture change.

[Hypothesis] The stronger next problem is objective mismatch: A6-LBF is now a clean prefix-native learned-basis
forecast operator, but supervision still treats every time step through a generic point loss. A StageB contribution
should first prove that the label/residual structure has a stable basis-space signal that the current objective ignores.

## Diagnostic Questions

1. Train-label structure:
   - Does the train split label matrix have a low-rank or autocorrelated basis structure across prefix horizons?
   - Is this structure stable across ETTh2, ETTm1, and Weather?

2. Learned-basis compatibility:
   - How much label energy is covered by A6-LBF `learned_temporal_basis[:H]`?
   - Does coverage degrade systematically at horizons or datasets where A6 residuals are worse?

3. Residual in coefficient space:
   - Project current A6 prediction residuals into the learned basis or a train-only label basis.
   - Test whether residual energy concentrates in specific components rather than only increasing with forecast
     distance.

4. Control comparison:
   - Use B4 `current_align_recon` and `no_align_no_recon` as historical controls for inherited-align dependence.
   - Use the active clean A6 code path as the current carrier for any new diagnostic summaries.
   - If the same coefficient residual structure appears after the clean cut, the problem is head/objective related
     rather than inherited-align related.

## Required Inputs

| Input | Role |
| --- | --- |
| A6-LBF train/eval config | dataset identity, horizons, normalization protocol |
| train split labels | train-only label autocorrelation and basis diagnostics |
| returned A6 metrics/artifacts | residual and horizon-level comparison |
| historical `current_align_recon` and `no_align_no_recon` controls | separate objective/head signal from inherited alignment |
| active clean A6 code path | current carrier after future branch removal |

Large prediction arrays should remain outside git. If a diagnostic needs residual tensors, sync only the minimal
needed summaries or regenerate small summaries on the remote side.

## Candidate Metrics

| Metric | Computation | Meaning |
| --- | --- | --- |
| label basis energy | cumulative energy of train-label covariance/eigen components | whether labels have compressible temporal structure |
| prefix basis stability | similarity of basis vectors or subspaces across horizons/prefixes | whether a prefix-native objective is coherent |
| learned-basis coverage | projection energy of labels/residuals onto A6 `learned_temporal_basis[:H]` | whether current learned basis matches target structure |
| coefficient residual concentration | residual energy share by basis component | whether errors are structured beyond step distance |
| control delta | metric difference between `current_align_recon` and `no_align_no_recon` | whether the signal depends on inherited alignment |

All metrics must define tensor source, shape, normalization, and aggregation in the generated analysis report.

## Narrative Gate

`B6-PLO` may enter Step 4-6 method design only if:

1. The diagnostic finds a stable train-only label or residual basis signal on at least two datasets.
2. The signal is not reducible to forecast step/distance monotonicity.
3. The signal connects directly to A6-LBF's prefix-native learned-basis forecast operator.
4. The proposed method can be distinguished from generic auxiliary-loss work by optimizing an architecture-matched
   basis/coefficient target, not by adding an unrelated frequency/domain loss.
5. The method has a clean inference path: no future labels or future encoder are required at test time.

## Rollback Rule

If B6 fails the Step 2/3 diagnostic, StageB should pause. The paper should proceed with A6-LBF-r256 as the main
architecture contribution and use B1/B3/B4 as negative diagnostic evidence rather than stacking another weak mechanism.

## Returned Diagnostic

[Decision] B6 Step 2/3 diagnostic completed at
`analysis/phase5_stage_b_prefix_native_objective_diagnostic_20260706/`.

[Fact] Train labels are highly compressible, but the compression is almost entirely generic low-frequency structure:
PCA top32 vs DCT top32 is ETTh2 `0.917/0.889`, ETTm1 `0.939/0.930`, Weather `0.832/0.831`.

[Fact] The A6 learned temporal basis from pure `no_align_no_recon` checkpoints is weaker than DCT at top32:
label coverage is ETTh2 `0.675`, ETTm1 `0.690`, Weather `0.251`; residual coverage is ETTh2 `0.287`,
ETTm1 `0.110`, Weather `0.081`.

[Decision] `diagnostic_not_enough_pause_b6`. Do not implement a B6 objective now. The evidence would make the method
look like a generic low-frequency/frequency auxiliary loss, which weakens distinction from FreDF/TransDF-style work.
