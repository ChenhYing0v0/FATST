# StageC PMFO/PIR Problem-Existence Diagnostic Protocol

## Status

| Field | Value |
| --- | --- |
| `candidate` | `SC1-PMFO` / `SC2-PIR` |
| `role` | `diagnostic_only` |
| `current_step` | Step 2-3 |
| `method_training_authorized` | `false` |
| `carrier` | frozen `A6-LBF-natural-baseline` |
| `rollback` | problem evidence fails -> Step 2 redefinition |

## What We Plan To Test

检验真实 label 与 natural-baseline residual 是否具有稳定的 nested multiresolution increment structure，
以及 horizon measure 的差异是否在该结构上产生超越 raw step reweighting 的可解释 gradient/risk差异。

## Why It Matters

PMFO 和 PIR 的论文叙事都依赖同一个前提：未来轨迹可以在嵌套函数空间中分成 coarse trajectory 与
progressive refinements。若这一分解只在某个 dataset、某组 basis 或某个 seed 上成立，方法不应实现。

## Artifact Construction

1. 只使用 train split labels 与已冻结 checkpoint 的 predictions/residuals；
2. datasets: Weather、ETTm1、ETTh2；seeds: 2021、2022、2023；
3. nested levels 先用固定 DCT/wavelet-like orthogonal controls构造，不学习 decoder；
4. measures: `delta_720`、`uniform_contiguous_H`、`log_uniform_H`；benchmark set只作 diagnostic control；
5. 所有统计按 dataset/seed/level/horizon region 完整输出。

## Required Metrics

- `increment_energy_share`: 每层 $\|\Delta_\ell y\|_2^2 / \|y\|_2^2$；
- `residual_increment_energy_share`: 对 baseline error 的同一统计；
- `cross_seed_cv`: 上述能量份额跨 seed CV；
- `measure_gradient_cosine`: 不同 measure 下同一 module 的 flattened gradient cosine；
- `measure_gradient_norm_ratio`: 相对 `delta_720` 的 norm ratio；
- `projected_vs_raw_excess_separation`: projected increment risk相对raw step weights新增的可解释分离量；
- `reconstruction_error`: nested projections重构原 trajectory/error 的误差。

每一列必须在 analyzer/code explanation 中给出 source tensor、公式和含义。

## Gates

PMFO problem gate：

- 至少 2/3 datasets 上，level-wise energy order 与 residual refinement pattern 跨 3 seeds稳定；
- fixed nested basis显著优于 random-orthogonal/no-refinement controls；
- 结论不能只由 H720 late region step difficulty解释。

PIR problem gate：

- 至少 2/3 datasets 上，不同 deployment measures 产生非平凡、稳定的 gradient direction变化；
- projected increments必须提供 raw harmonic step weighting之外的解释量；
- 若只复现 ElasTST-style weights 的必然差异，判定 `simple_measure_alignment_only`，不进入方法设计。

## Failure Attribution

若 projection 数值不稳定、重构误差异常或 gradient extraction 受 checkpoint/data pipeline污染，标记
`diagnostic_invalid_for_direction_rejection`。只有 stable diagnostic + required controls 仍失败，才允许把
exact PMFO/PIR problem candidate 回滚 Step 2；不得扩大为所有 projective decoder/training 的方向级拒绝。
