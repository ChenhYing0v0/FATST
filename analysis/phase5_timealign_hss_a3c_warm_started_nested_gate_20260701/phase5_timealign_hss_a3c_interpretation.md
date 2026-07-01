# Phase5 A3C Warm-Started Nested Gate Interpretation

## What Was Tested

A3C 测试 `checkpoint-initialized-nested-segment-decoder`：student 仍是 primary nested
prediction interface，但从 H1 `target_set_decoder_multiprefix` checkpoint 加载 shared carrier，
并把 H1 `proj_x` 的 learned row slices 写入 nested segment heads。

该实验检验的 hypothesis 是：

> A2 nested primary interface 的主要瓶颈是 learned capacity 缺失；若从 H1 trained head 做
> warm-start，nested primary 应显著优于 A2/A3B，并接近或超过 H1/H1C。

## Key Results

| Reference | ALL Mean Relative MSE | Wins |
| --- | ---: | ---: |
| A2 nested | `+0.07%` | `4/12` |
| A3B residual | `-4.06%` | `12/12` |
| H1 target-set | `+0.68%` | `3/12` |
| H1C row-gated | `+0.25%` | `5/12` |
| fixed reference | `-3.07%` | `7/12` |

Dataset-level pattern:

- ETTh2：相对 A2 基本持平 `+0.03%`，相对 H1/H1C 仍落后；
- ETTm2：相对 A2 略好 `-0.11%`，相对 H1 略好 `-0.12%`，但相对 H1C 仍基本持平；
- Weather：相对 A2/H1/H1C 均小幅变差。

Training:

- ETTh2 best epoch 是 `1`，last-val 相对 best-val drift `+6.77%`；
- ETTm2 / Weather best epoch 是 `6`，last-val drift 分别约 `+0.54%`、`+0.46%`；
- 三个 dataset 都实际加载了 H1 checkpoint，warm-start compatible keys 存在，因此该结果不应解释为
  warm-start 未生效。

## Decision

A3C 不通过 paper-core effectiveness gate。

它证明：

1. primary nested 比 residual nested 更合理：A3C 相对 A3B 全部 `12/12` wins；
2. 但 `row-slice warm-start` 不是足够的 capacity-preserving mechanism：A3C 相对 A2 只有
   `+0.07%`，没有形成稳定增益；
3. A3C 不能超过 H1/H1C，因此不能支撑 `Capacity-Preserving Prefix-Aware Interface` 的主贡献。

因此，A3C 只否定 `warm-started primary nested` 分支，不否定 Stage A interface 主线。

## Next Step

按照 Stage Ledger 的 candidate queue，下一步进入 A3D：

> `teacher_preserved_nested_primary_decoder`

理由：

- A3C 说明单纯参数迁移不足；
- A3D 用 H1 teacher consistency 保留 H1 target-set learned function，同时让 nested primary head
  学 prefix-consistent decomposition；
- 它是对 A3C 失败点的最小机制修复，比立即引入 target-conditioned nested structure 更符合当前
  11-step rollback。

A3D gate：

- 至少优于 A3C 和 A2 nested；
- paper-core gate 要求接近或超过 H1/H1C；
- 若只复制 H1 而不能改善 prefix/nested behavior，则降级为 teacher-preservation diagnostic。
