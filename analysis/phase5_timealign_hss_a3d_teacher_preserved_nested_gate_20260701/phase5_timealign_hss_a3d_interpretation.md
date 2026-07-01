# Phase5 A3D Teacher-Preserved Nested Gate Interpretation

## What Was Tested

A3D 测试 `teacher_preserved_nested_primary`：student 仍是 A3C 的
`checkpoint-initialized-nested-segment-decoder` primary nested head，但训练时加入 frozen H1
`target_set_decoder_multiprefix` teacher consistency。

该实验检验的 hypothesis 是：

> A3C 的主要问题不是 nested structure 本身，而是 row-slice warm-start 不能在训练期保留 H1 learned
> function；加入 teacher consistency 后，nested primary 应优于 A3C/A2，并接近或超过 H1/H1C。

## Key Results

| Arm | vs A2 Nested | vs A3C Warm | vs H1 Target-Set | vs H1C Row-Gated | Wins vs H1C |
| --- | ---: | ---: | ---: | ---: | ---: |
| `teacher_preserved_nested_w03` | `-0.66%` | `-0.73%` | `-0.06%` | `-0.48%` | `8/12` |
| `teacher_preserved_nested_w10` | `-0.62%` | `-0.69%` | `-0.02%` | `-0.44%` | `5/12` |

Dataset pattern:

- ETTh2：两个 arms 都明显优于 A3C/H1/H1C，`w10` 相对 H1 为 `-0.23%`、相对 H1C 为 `-1.65%`；
- Weather：`w03` 小幅优于 A3C/H1/H1C；
- ETTm2：两个 arms 都弱于 H1C，且相对 A3C 小幅变差。

Training diagnostics:

- teacher loss 在所有 dataset/arm 上下降，说明 teacher consistency 实际生效；
- `w10` 的 teacher loss 更低，但并没有带来更好的 overall gate；
- `w03` 是当前更稳的 A3D arm。

## Decision

A3D 是 `partial_pass`，但不是 paper-core pass。

它证明：

1. teacher preservation 机制有效：相对 A3C/A2 有稳定小幅收益；
2. primary nested interface 仍有继续研究价值：A3D 已接近或略超 H1/H1C；
3. 但 A3D 的收益不够均匀，ETTm2 仍失败，因此不能直接作为 `Capacity-Preserving Prefix-Aware
   Interface` 的最终方法。

## Next Step

下一步进入 A3E target-conditioned nested primary，但必须修正实验边界：

- A3C 已证明 warm-start alone 无效；
- 因此 A3E 不能把 warm-start 当作机制贡献；
- `target_conditioned_nested_warm` 只用于与 A3C 对齐 initialization，隔离 target conditioning 的增量；
- `target_conditioned_nested_scratch` 作为 diagnostic/control，判断 target-conditioned nested 是否完全依赖 H1
  initialization。

A3E 的核心 hypothesis 应写成：

> requested target set / prefix condition 是否应直接进入 primary nested head，而不是只在 loss 或
> dense projection 前后调节。
