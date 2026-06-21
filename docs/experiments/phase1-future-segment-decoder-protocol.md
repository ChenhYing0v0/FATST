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

## Phase1-A 第一轮结果

[Fact] 远程 gate 已完成，结果报告见：
`analysis/phase1_segment_decoder_gate_20260621/phase1_segment_decoder_gate_report.md`。

[Strong Evidence] `PatchEncoderSegmentQueryHead` 未通过：

- main MSE comparison: `0/12` wins；
- segment-level MSE comparison: `0/30` wins；
- relative MSE degradation: `+2.16%` 到 `+15.17%`，平均 `+6.79%`。

[Decision] 当前 `PatchEncoderSegmentQueryHead` 不能作为论文核心 decoder 创新点，也不能进入
Phase1-B one-model compatibility。更不能在它上面直接叠加 future-aware 或 MoE，否则后续机制
会变成补偿一个弱 decoder。

[Rollback] 回退到长研究执行模板第 5-6 步：重新评估理论可行性并重新设计方案。

下一轮候选应保留 fixed head 的强 readout capacity，只把 future-side interface 作为 adapter
或 conditioning 加进去。优先考虑：

1. `FixedHeadAdapter`: 保留 fixed flatten head，在 readout 前后加入轻量 future-segment adapter。
2. `SegmentQueryDenseHead`: segment state 只负责 conditioning，每个 segment 仍保留
   parameter-controlled dense readout。
3. `StepQueryHead`: step-level query 作为更细粒度方案，但需要控制参数和计算量。

## Phase1-A.2: Fixed-Head Adapter Gate

### 回退后的问题定义

[Strong Evidence] `PatchEncoderSegmentQueryHead` 的失败并不是因为 segment queries 完全同质化；
它的 query cosine 分布并未退化为全相同。更直接的问题是 readout capacity：

$$
\text{SegmentQueryHead}: U_j \rightarrow \mathbb{R}^{48}
$$

使用 shared segment output head，而 `PatchEncoderFixedHead` 使用：

$$
\text{Flatten}(Z) \rightarrow \mathbb{R}^{H}
$$

为每个 output step 保留独立 linear row。Phase1-A 第一轮的参数比例在 h720 只有 fixed
head 的 `0.128`，因此它很可能是在删除强 readout 后再尝试用 segment state 补偿。

[Decision] Phase1-A.2 不再直接替换 fixed head，而是检验一个更窄的问题：

> future-segment interface 是否能在保留 fixed flatten head 主路径的前提下，提供有效
> conditioning 或 residual correction？

### 候选模型

`PatchEncoderFixedHeadAdapter` 的 forward path 为：

$$
Z=E_\theta(X),
$$

$$
\hat{Y}^{base}=\text{FixedHead}(Z),
$$

$$
U_J=A_\theta(Q_J,Z),
$$

$$
(\gamma,\beta)=R_\theta(U_J),
$$

$$
\hat{Y}=\hat{Y}^{base}\odot(1+\gamma)+\beta.
$$

其中：

- $Q_J$ 是 future segment queries；
- $A_\theta$ 是 one-layer cross-attention adapter；
- $R_\theta$ 是 segment-wise affine head；
- $R_\theta$ 的最后一层 zero initialization，使初始 forward 等价于 fixed head。

[Inference] 这个设计把机制风险拆开：如果该模型退化，说明仅靠 history-derived
future-segment adapter 不足；如果它提升，则证明 future-side interface 可以在不破坏
fixed readout capacity 的情况下发挥作用。

### 诊断指标

除 Phase1-A 的 MSE/MAE 和 segment metrics 外，Phase1-A.2 额外记录：

| Artifact | 含义 |
| --- | --- |
| `adapter_delta_stats.csv` | adapter prediction 与 base prediction 的差异幅度 |
| `adapter_query_similarity.csv` | segment query 是否保持非退化差异 |

`adapter_delta_stats.csv` 中的核心列：

- `delta_mse_to_base`: $\hat{Y}$ 与 $\hat{Y}^{base}$ 的 MSE。
- `delta_mae_to_base`: $\hat{Y}$ 与 $\hat{Y}^{base}$ 的 MAE。
- `mean_abs_gamma`: affine multiplicative term 的平均绝对值。
- `mean_abs_beta`: additive residual term 的平均绝对值。
- `delta_to_base_mae_ratio`: adapter 修正幅度相对 base prediction 幅度的比例。

### 通过条件

Phase1-A.2 通过至少需要满足：

1. [Performance] main MSE 至少 `4/12` wins，且平均 relative MSE 不为明显正退化；
2. [Segment profile] 或者 segment-level wins 明显集中在远端/high-error segments；
3. [Mechanism] `delta_to_base_mae_ratio` 非零，排除 adapter 没有实际参与的假阳性。

若仅有微小 MSE 提升但 adapter delta 接近零，则视为训练波动，不作为 decoder 创新点。

### 失败后的回退

如果 `PatchEncoderFixedHeadAdapter` 仍不通过：

- 不再优先扩大 decoder capacity；
- 回退到长研究模板第 3-5 步，重新评估 “history-only future decoder” 问题是否足够；
- 下一候选应转向 `future-aware teacher/student alignment`，即用 training-only future
  signal 学习可推理的 future latent state，而不是继续只用 history-derived queries。

## Phase1-A.2 结果

[Fact] 远程 gate 已完成，结果报告见：
`analysis/phase1_fixed_adapter_gate_20260621/phase1_fixed_adapter_gate_report.md`。

[Evidence] `PatchEncoderFixedHeadAdapter` 相比 `PatchEncoderFixedHead` 的主指标结果为：

- main MSE wins: `7/12`；
- segment-level MSE wins: `15/30`；
- relative MSE change range: `-2.02%` 到 `+3.74%`；
- mean relative MSE change: `+0.20%`；
- mean adapter delta/base MAE ratio: `0.3200`。

[Decision] 该结果是 `partial_pass`，不是完整通过。它证明两件事：

1. [Strong Evidence] 第一轮 `SegmentQueryHead` 失败主要来自 readout capacity 损失；
   一旦保留 fixed head 主路径，future-side adapter 不再系统性失败。
2. [Strong Evidence] 当前 history-only future-segment adapter 的收益不够稳定，平均 MSE
   仍略退化，不能作为论文的核心 decoder 创新点。

[Inference] 因此，Phase1 不应继续靠增加 adapter 容量来硬推 decoder 主线。更合理的回退点
是长研究模板第 3-5 步：重新评估问题是否应从 “decoder readout form” 转为
“训练时 future signal 如何塑造可推理 future latent state”。

[Next] 下一候选应转向 `Future-Aware FixedHeadAdapter`：

- 保留 fixed head 和 adapter interface；
- 训练时加入 future teacher branch；
- 用 teacher/student alignment 约束 adapter state 或 affine residual；
- 推理时只保留 history-derived student path，严格做 leakage audit。

## Phase1-A.3: Future-Aware Adapter Gate

### 问题定义

[Fact] Phase1-A.2 说明 history-only future segment adapter 已经会实际修正 base prediction，
但它的收益不稳定，平均 MSE 仍为正退化。

[Hypothesis] 问题可能不在 future-side interface 本身，而在该 interface 只由历史窗口
和 learnable queries 学习，缺少关于 future distribution 的训练信号。TimeAlign 的证据
提示：训练阶段可以用 ground-truth future 构造 teacher representation，再约束预测分支的
student representation 向 future distribution 靠近。

### 候选模型

`PatchEncoderFutureAwareAdapter` 保留 Phase1-A.2 的推理路径：

$$
X \rightarrow Z \rightarrow \hat{Y}^{base}, U^{student}, \gamma,\beta
\rightarrow \hat{Y}.
$$

训练时额外构造 teacher：

$$
S^{teacher}=T_\psi(Y_{1:H},Q_H),
$$

并把 student adapter state 投影后对齐到 stop-gradient teacher：

$$
S^{student}=P_\theta(U^{student}),
$$

$$
\mathcal{L}_{align}
=1-\cos(S^{student},\operatorname{sg}(S^{teacher})).
$$

teacher branch 还需要重构 normalized future：

$$
\hat{Y}^{teacher}=R_\psi(S^{teacher}),
\quad
\mathcal{L}_{recon}=\operatorname{MSE}(\hat{Y}^{teacher},Y^{norm}).
$$

总 loss：

$$
\mathcal{L}
=
\mathcal{L}_{pred}
+\lambda_{align}\mathcal{L}_{align}
+\lambda_{recon}\mathcal{L}_{recon}.
$$

默认第一轮：

- `align_weight = 0.05`
- `recon_weight = 0.05`
- `segment_len = 48`
- teacher branch 只在 training / diagnostic 中执行

### Leakage Audit

必须验证：

$$
f_\theta(X) = f_\theta(X,Y)
$$

其中右侧的 $Y$ 只能用于诊断输出 teacher loss，不允许改变 `prediction`。

训练脚本写入：

- `future_alignment_stats.csv`
  - `alignment_loss`
  - `reconstruction_loss`
  - `teacher_student_cosine`
  - `prediction_leakage_max_abs`

`prediction_leakage_max_abs > 1e-7` 直接判定失败。

### 通过条件

Phase1-A.3 通过需要同时满足：

1. [Leakage] `prediction_leakage_max_abs <= 1e-7`。
2. [Performance] 相比 `PatchEncoderFixedHead` 至少 `6/12` main MSE wins 且 mean relative
   MSE < 0。
3. [Mechanism] teacher/student cosine 或 alignment loss 证明 student state 确实受到
   future teacher 约束；若 forecast 改善但 alignment 不工作，论文故事不成立。

如果只超过 `PatchEncoderFixedHeadAdapter` 但不能超过 fixed head，只能说明 alignment
修补了 adapter，不足以成为论文核心。
