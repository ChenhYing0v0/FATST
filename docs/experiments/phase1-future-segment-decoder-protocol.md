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

## Phase1-A.3 结果

[Fact] 远程 gate 已完成，结果报告见：
`analysis/phase1_future_aware_adapter_gate_20260621/phase1_future_aware_adapter_gate_report.md`。

[Evidence] `PatchEncoderFutureAwareAdapter` 相比 `PatchEncoderFixedHead`：

- main MSE wins: `4/12`；
- mean relative MSE: `+0.16%`；
- relative MSE range: `-2.32%` 到 `+2.41%`。

[Evidence] 相比 `PatchEncoderFixedHeadAdapter`：

- main MSE wins: `5/12`；
- mean relative MSE: `-0.01%`。

[Strong Evidence] leakage audit 通过：

- `max_prediction_leakage_abs = 0.0`。

[Evidence] teacher/student state 发生了有效耦合：

- mean teacher/student cosine: `0.4287`。

[Decision] 该结果为 `partial_pass`，不是完整通过。它说明：

1. training-only future branch 在工程和 leakage 边界上成立；
2. teacher/student alignment 有实际作用；
3. 但当前设置没有稳定超过 fixed head，因此还不能作为论文核心机制。

[Diagnosis] 当前最明确的修补点是 reconstruction loss 的尺度失衡。Weather 的 mean
reconstruction loss 约为 `645.08`，而 ETTh2 约为 `1.05`、ETTm1 约为 `0.40`。
这说明 $\mathcal{L}_{recon}$ 在不同 dataset 上不可比，`recon_weight=0.05` 可能在
Weather 上显著压过 prediction objective。

[Next] 下一步不应扩大 teacher branch。应回到长研究模板第 5-6 步，修补理论可行性和方案：

- `ScaleNormalizedRecon`: 用 target energy 或 segment energy 归一化 reconstruction loss；
- 或 `AlignOnly`: 暂时设置 `recon_weight=0`，只保留 teacher state alignment；
- 对二者做小矩阵 gate，先验证 Weather 退化是否来自 reconstruction scale。

## Phase1-A.4 Repair Gate 设计

### 待解决问题

[Fact] Phase1-A.3 的 teacher/student coupling 与 leakage boundary 成立，但
`reconstruction_loss` 在不同 dataset 上不可比。Weather 的 reconstruction loss 约为
ETTh2 的数百倍，这会让同一个 `recon_weight=0.05` 在不同数据集上代表完全不同的
optimization pressure。

[Hypothesis] 如果 future-aware supervision 本身有价值，那么修正 auxiliary loss 尺度后，
forecasting performance 应至少不再被 Weather 系统性拖累；如果修正后仍不稳定，则问题
可能不在 loss scale，而在 teacher state 与 prediction objective 的对齐方式。

### 候选 idea

`AlignOnly`：

$$
\mathcal{L}
=
\mathcal{L}_{pred}
+\lambda_{align}\mathcal{L}_{align}.
$$

该候选保留 teacher state 和 alignment，但不让 reconstruction loss 进入训练目标。
它回答：teacher reconstruction 是否反而干扰了 forecasting。

`ScaleNorm`：

$$
\mathcal{L}_{recon}^{norm}
=
\frac{
\operatorname{MSE}(\hat{Y}^{teacher,norm},Y^{norm})
}{
\operatorname{mean}((Y^{norm})^2)+\epsilon
},
$$

$$
\mathcal{L}
=
\mathcal{L}_{pred}
+\lambda_{align}\mathcal{L}_{align}
+\lambda_{recon}\mathcal{L}_{recon}^{norm}.
$$

该候选回答：reconstruction branch 是否仍需要保留，只是要按 target energy 校准尺度。

### 实验设计

对比模型：

- `PatchEncoderFixedHead`
- `PatchEncoderFixedHeadAdapter`
- `PatchEncoderFutureAwareAlignOnly`
- `PatchEncoderFutureAwareScaleNorm`

实验矩阵：

- datasets: `ETTh2`, `ETTm1`, `Weather`
- horizons: `96`, `192`, `336`, `720`
- seed: `2021`
- epochs: `100`

远程 runner：

- `scripts/remote/run_phase1_future_aware_repair_gate.sh`

分析脚本：

- `scripts/analyze_phase1_future_aware_repair_gate.py`

### 通过条件

Phase1-A.4 通过需要：

1. [Leakage] 所有 future-aware repair candidates 的 `prediction_leakage_max_abs <= 1e-7`。
2. [Performance] 最优 repair candidate 相比 `PatchEncoderFixedHead` 至少 `6/12`
   main MSE wins，且 mean relative MSE < 0。
3. [Mechanism] teacher/student cosine 显示 alignment 仍有效；`ScaleNorm` 的
   `raw_reconstruction_loss` 可以大，但 `reconstruction_loss` 应变成可比尺度。

如果只达到 `repair_partial`，则说明 future-aware 方向仍有机制信号，但不足以作为论文核心；
下一步应回到长研究模板第 3-5 步，重新评估 future-aware claim 的问题定义或转向新架构。

## Phase1-A.4 结果

[Fact] 远程 repair gate 已完成，结果报告见：
`analysis/phase1_future_aware_repair_gate_20260622/phase1_future_aware_repair_gate_report.md`。

[Evidence] `PatchEncoderFutureAwareAlignOnly` 是较好 repair candidate：

- vs `PatchEncoderFixedHead`: main MSE wins `4/12`，mean relative MSE `+0.04%`；
- vs `PatchEncoderFixedHeadAdapter`: main MSE wins `5/12`，mean relative MSE `-0.13%`；
- leakage audit: `max_prediction_leakage_abs = 0.0`。

[Evidence] `PatchEncoderFutureAwareScaleNorm` 修正了 reconstruction loss 尺度，但没有带来
稳定性能收益：

- vs `PatchEncoderFixedHead`: main MSE wins `4/12`，mean relative MSE `+0.12%`；
- vs `PatchEncoderFixedHeadAdapter`: main MSE wins `5/12`，mean relative MSE `-0.05%`；
- Weather raw reconstruction loss 仍可达数百到上千，但 normalized `reconstruction_loss`
  已压到约 `0.33-0.57`，说明 scale normalization 生效。

[Evidence] 聚合结构显示收益不稳定：

- `AlignOnly` vs fixed by dataset: ETTh2 `-0.43%`，ETTm1 `+0.13%`，Weather `+0.44%`；
- `ScaleNorm` vs fixed by dataset: ETTh2 `-0.38%`，ETTm1 `+0.39%`，Weather `+0.35%`；
- 两个 repair 在 horizon `192` 和 `720` 平均改善，但在 `96` 和 `336` 平均退化；
- Weather 的 4 个 horizon 对 fixed head 均未获胜。

[Decision] Phase1-A.4 为 `repair_partial`，不是 pass。该结果说明：

1. Phase1-A.3 的 scale imbalance 诊断是真问题；
2. `ScaleNorm` 可以修正 auxiliary loss 的数值尺度；
3. 但修正后仍不能稳定超过 fixed head，说明当前 future-aware adapter 的问题不只是
   reconstruction scale；
4. 该方向不应继续直接叠加 MoE 或增加 teacher capacity。

[Rollback] 按长研究模板，应回到 step 3-5：重新评估当前问题定义是否值得研究，以及
future-aware signal 是否应通过更基础的 decoder/state architecture 承载，而不是继续修补
当前 affine adapter。

## Phase1-A.5: Step-Specific State Decoder Reset

[Fact] 详细 reset 文档见：
`docs/experiments/phase1-step-specific-state-decoder-reset.md`。

[Decision] Phase1-A.5 不再继续修补 post-head affine adapter，也不直接叠加 MoE。
它回到长研究模板 step 1-6，重新定义一个更具体的问题：

> fixed head 是否把所有 future steps 绑定到同一个 history representation，使不同 future
> segments 只能通过 output rows 区分，而不能在进入 readout 前形成 step/segment-specific
> representations？

### 理论动机

[Strong Evidence] SRP++ 指出 multi-step forecasting 中 step-invariant representation 可能
构成 expressiveness bottleneck。TIMEPERCEIVER 则说明 target positions 应进入 decoder
computation，而不是只作为 output dimension。

[Inference] 本项目 Phase1-A 的负结果与该判断一致：

- `PatchEncoderSegmentQueryHead` 失败，说明不能删除 fixed head 的 dense readout capacity；
- `PatchEncoderFixedHeadAdapter` 和 future-aware repair 只得到 partial result，说明 post-head
  correction 太靠后；
- 因此下一步应保留 fixed head rows，但在 readout 前构造 segment-specific representation。

### 候选模型

下一候选暂命名为 `PatchEncoderStepSpecificStateAdapter`：

$$
Z=E_\theta(X),
$$

$$
U_j=A_\theta(q_j,Z),
$$

$$
\tilde{Z}_j=Z\odot(1+\gamma_j)+\beta_j,
$$

$$
\hat{Y}_{a_j:b_j}=W_{a_j:b_j}\operatorname{Flatten}(\tilde{Z}_j).
$$

其中：

- $q_j$ 是 future segment query；
- $U_j$ 是 future segment state；
- $\gamma_j,\beta_j \in \mathbb{R}^{d}$，由 $U_j$ 生成；
- $W_{a_j:b_j}$ 复用 fixed head 对应 segment 的 readout rows；
- $\gamma_j,\beta_j$ zero initialization，使初始 forward 等价于 fixed head。

### 第一版 gate

对比模型：

- `PatchEncoderFixedHead`
- `PatchEncoderFixedHeadAdapter`
- `PatchEncoderStepSpecificStateAdapter`

实验矩阵保持不变：

- datasets: `ETTh2`, `ETTm1`, `Weather`
- horizons: `96`, `192`, `336`, `720`
- seed: `2021`
- segment length: `48`

通过条件：

1. [Performance] 相比 `PatchEncoderFixedHead` 至少 `6/12` main MSE wins，且 mean relative
   MSE < 0。
2. [Stability] 不出现任一 dataset 全 horizon 系统性退化。
3. [Mechanism] segment-conditioned $\gamma,\beta$ 或 $\tilde{Z}_j-Z$ 的统计不能完全退化；
   segment state similarity 需要显示可分性。
4. [Capacity control] 参数量应与 `PatchEncoderFixedHeadAdapter` 可比；否则必须加入
   parameter-control。

[Boundary] 只有 Phase1-A.5 通过后，future-aware alignment 才应该重新进入；届时应对齐
$U_j$ 或 $\tilde{Z}_j$，而不是继续对齐 post-head adapter。MoE 也只能作为
$T_\theta(Z,U_j)$ 的 conditional operator 进入，而不是作为失败 decoder 的参数补偿。

## Phase1-A.5 结果

[Fact] Phase1-A.5 完整 gate 已在 `529_Lab-3090` 完成：

- remote output: `/home/yingch/exp_outputs/r-2026-fatst/phase1_step_specific_state`
- local raw artifacts: `analysis/phase1_step_specific_state_gate_20260622/raw/`
- local report: `analysis/phase1_step_specific_state_gate_20260622/phase1_step_specific_state_gate_report.md`
- matrix: `PatchEncoderFixedHead`, `PatchEncoderFixedHeadAdapter`,
  `PatchEncoderStepSpecificStateAdapter` x `ETTh2`, `ETTm1`, `Weather` x
  `96`, `192`, `336`, `720`
- seed: `2021`
- selected GPUs: `1`, `2`

主结果：

| Comparison | MSE wins | Mean relative MSE | Range | Zero-win datasets |
| --- | ---: | ---: | --- | --- |
| vs `PatchEncoderFixedHead` | 7/12 | +0.39% | -3.15% to +8.22% | none |
| vs `PatchEncoderFixedHeadAdapter` | 8/12 | +0.19% | -2.31% to +6.77% | none |

诊断结果：

- mean_abs_gamma: `0.604776`
- mean_abs_beta: `0.078682`
- mean segment activation cosine: `0.964393`

[Decision] Phase1-A.5 是 `partial`，不是 pass。它满足 “存在机制活动” 和
“没有 dataset 全面失效” 两类弱证据，但没有满足 main pass 条件：相对两个 control 的 mean
relative MSE 均为正。尤其 ETTh2 h96/h192 的退化说明 pre-head state modulation
会破坏 fixed head 在短 horizon 上已经学到的稳定 readout。

[Inference] 这说明当前问题定义仍不够准确。fixed head 的缺陷不是简单的
“所有 future steps 共享一个 representation”，否则 A.5 应该在保留 readout rows 后稳定
改进。更可能的情况是：dense fixed head 已经通过 output rows 隐式编码了相当强的
step-specific readout，轻量 latent FiLM 容易干扰这种 readout，而不是补足它。

[Rollback] 回退到长研究模板 step 2-3。下一步应重新定义 decoder/output strategy 的
核心问题，再决定是否进入新的 step 4 idea。当前不进入 Phase1-B one-model compatibility，
不在 A.5 上继续叠 future-aware alignment，也不引入 MoE 作为补偿。
