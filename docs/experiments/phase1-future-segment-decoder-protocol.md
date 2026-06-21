# Phase 1 Future-Segment Decoder Protocol

## 目的

[Fact] Phase0 已选择 `PatchEncoderFixedHead` 作为 canonical internal base，并显示 fixed
direct head 存在可量化但中等幅度的 prefix inconsistency 与 segment-wise oracle variation。

[Inference] Phase1 的目标不是证明 `variable-horizon` 本身，而是验证一个更具体的问题：

> static flatten head 是否把未来预测过程过度压缩为一次固定投影；显式 future segment
> states 是否能在 one-to-one horizon setting 下改善 forecast quality 或 segment-level
> error profile。

## 不检验什么

Phase1-A 暂不检验：

- one-model for all horizons；
- mixed-horizon training；
- ElasTST-style placeholder / structured mask；
- future teacher branch；
- MoE routing。

这些机制会混入额外变量。如果 Phase1-A 不能在 one-to-one horizon setting 中证明
future-side decoder 有价值，则后续 one-model、future-aware 和 MoE 都缺少稳定基础。

## 诊断假设

### FixedHead bottleneck

`PatchEncoderFixedHead` 的 output path 是：

$$
Z \in \mathbb{R}^{(BC) \times N \times d}
\rightarrow
\text{Flatten}(Z)
\rightarrow
\hat{Y} \in \mathbb{R}^{B \times H \times C}.
$$

[Hypothesis] 这种 head 可以很强，因为它给每个 output step 一组独立 linear weights；
但它没有显式 future-side state，也没有为 future segment、future-aware alignment 或
future-side routing 留出自然接口。

### Segment state alternative

Phase1-A 的最小替代是：

$$
Q_J \in \mathbb{R}^{J \times d},
\quad
U_J = \text{CrossAttn}(Q_J, Z, Z),
\quad
\hat{Y}_{a_j:b_j}=O(U_j).
$$

其中 $J=\lceil H / S\rceil$，$S$ 是 segment length。默认 $S=48$。

[Inference] 该设计把未来输出从 “position rows in a large linear matrix” 改为
“query-conditioned future segment states”。这使后续机制有明确接入点：

- future-aware alignment 可以约束 $U_J$ 或 $S_J$；
- MoE 可以在 $U_j$ 上做 segment-level routing；
- one-model compatibility 可以在 $Q_J$ 和 requested horizon 上检验，而不是强行复用
  fixed flatten head。

## Phase1-A 模型

| Model | Role |
| --- | --- |
| `PatchEncoderFixedHead` | Phase0 selected base |
| `PatchEncoderSegmentQueryHead` | 最小 future segment query decoder |

第一版 `PatchEncoderSegmentQueryHead` 只替换 output head，保持 encoder、RevIN、dataset
split、loss、optimizer 和训练 protocol 与 `PatchEncoderFixedHead` 对齐。

## 数据集与矩阵

Datasets:

- `ETTh2`
- `ETTm1`
- `Weather`

Horizons:

$$
H \in \{96,192,336,720\}.
$$

Primary seed:

$$
2021.
$$

如果 Phase1-A 出现正向结果，再对敏感设置补 seeds `{2021,2022,2023}`。

## 指标

Primary:

- MSE
- MAE

Diagnostics:

- `metrics_by_horizon.csv`
- `metrics_by_segment.csv`
- parameter count
- segment state cosine similarity
- segment-wise comparison against Phase0 fixed head

[Inference] 如果 SegmentQueryHead 只提升平均 MSE 但 segment states 完全同质化，则论文故事较弱；
如果平均 MSE 接近但 segment-level error profile 更稳定，也只能作为后续 future-aware/MoE 的
接口证据，不能单独作为强性能 claim。

## 通过条件

Phase1-A 通过需要满足至少一类强证据：

1. [Performance] 在至少两个 dataset/horizon 设置上超过 fixed head，且没有大面积退化。
2. [Segment profile] 对远端 segment 或高误差 segment 有稳定改善，并能通过
   `metrics_by_segment.csv` 定位。
3. [Mechanism interface] segment states 呈现非退化差异，为 Phase2 alignment 或 Phase3
   MoE routing 提供有效载体。

如果只满足第 3 条，Phase1 不能直接作为论文核心创新，只能作为后续机制的基础设施。

## 失败与转向

如果 `PatchEncoderSegmentQueryHead` 明显弱于 fixed head：

- 先检查是否因为参数量过小导致 underfit；
- 增加 dense parameter-control 或 per-segment adapter；
- 若仍失败，则不应继续围绕 decoder 讲故事，应转向 future-aware state 或 external
  baseline reproduction。

如果 SegmentQueryHead 只在 h720 有改善：

- 将主故事收敛到 long-horizon future segment modeling；
- one-model compatibility 延后。

如果 SegmentQueryHead 在短 horizon 上改善但 h720 退化：

- 检查 segment length 是否过粗；
- 尝试 multi-scale segment queries，但只在最小模型失败后执行。

## Phase1-B 边界

只有 Phase1-A 通过后才进入 Phase1-B：

- `train H=720, evaluate prefixes`
- mixed-horizon training over `{96,192,336,720}`
- prefix consistency and short-prefix MSE

Phase1-B 的目标是检验 one-model compatibility，不是替代 Phase1-A 的机制验证。
