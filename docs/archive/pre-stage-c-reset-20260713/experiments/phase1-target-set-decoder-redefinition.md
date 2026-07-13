# Phase1-R Target-Set Decoder Redefinition

## 定位

[Decision] 本文档不是 Phase1-A.7 的又一个 fixed-head patch。A.1-A.6 已经连续证明：
围绕 horizon-specific fixed head 追加 lightweight future-side module 很难稳定超过
`PatchEncoderFixedHead`。因此本轮回到长研究模板 step 1-6，重新定义 decoder 创新点。

新的候选方向暂命名为 `Target-Set Forecasting Decoder`，本地候选实现名暂定：
`PatchEncoderTargetSetDecoder`。

## Step 1: 调研分析

### 来自本项目的证据

[Strong Evidence] Phase0 已确认 fixed direct head 存在问题，但问题不是灾难性的：

- prefix consistency 存在可量化 mismatch；
- segment-wise checkpoint oracle 显示同一 `0-720` 区间内局部最佳 checkpoint 会变化；
- `pred_len=720` checkpoint 在三个数据集的全区间平均 MSE 仍是最强。

[Strong Evidence] Phase1-A.1 到 A.6 的连续结果说明：

| Candidate | 结论 |
| --- | --- |
| `PatchEncoderSegmentQueryHead` | 直接替换 fixed head，`0/12` wins，readout capacity collapse |
| `PatchEncoderFixedHeadAdapter` | `7/12` wins，但 mean relative MSE `+0.20%` |
| `PatchEncoderFutureAwareAdapter` | leakage-free，但 vs fixed head mean relative MSE `+0.16%` |
| `PatchEncoderFutureAwareAlignOnly` | repair 后仍仅 `4/12` wins vs fixed head |
| `PatchEncoderStepSpecificStateAdapter` | 机制活跃，但 mean relative MSE `+0.39%` vs fixed head |
| `PatchEncoderTrajectoryBasisResidual` | residual 非零，但 mean relative MSE `+0.67%` vs fixed head |

[Inference] 这些负结果不是简单说明 decoder 方向无价值，而是说明当前问题定义不对：
我们一直在 one-to-one setting 中试图击败已经针对每个 horizon 单独优化的 specialist head。
这种目标偏向 fixed head，因为它允许每个 horizon 拥有一套独立 output rows 和一套独立训练过程。

### 来自文献的证据

[Strong Evidence] `TIMEPERCEIVER` 的关键启发不是“再加一个 cross-attention block”，而是把
预测任务定义为 target index set $J$ 上的 generalized temporal prediction。target timestamp
对应 query token，decoder 的输入不再是隐式 output dimension。

[Strong Evidence] `ElasTST` 说明 varied-horizon forecasting 的核心是 horizon-invariant
inference：请求更长 horizon 不应改变已经存在的 prefix forecast。其 future placeholders、
structured mask 和 horizon reweighting 都服务于 target positions 进入模型计算。

[Strong Evidence] `QDF` 指出 direct multi-step forecasting 的标准 MSE 把 future steps
当作独立等权任务，忽略 future label autocorrelation。这说明问题可能在 training objective
和 target-step dependency，而不只是 decoder block。

[Strong Evidence] `SRP++` 指出 step-invariant representation 有 expressiveness bottleneck，
但本项目 A.5 说明简单 pre-head modulation 不足以解决它。更合理的解释是：step specificity
需要和 target-set formulation 绑定，而不是作为 horizon-specific head 的局部修正。

[Strong Evidence] `Seg-MoE` / `MoHETS` 支持 segment-level routing 和 heterogeneous operators，
但它们也说明 MoE 需要稳定 routing unit。A.1-A.6 尚未产生这样的 unit，因此 MoE 不能提前叠加。

## Step 2: 待解决问题

旧问题定义是：

> fixed head 是否缺少 future-side decoder / state / residual？

该定义已经被 A.1-A.6 证明过宽。新的问题定义是：

> 当前主流 horizon-specific direct forecasting 把 target horizon 当作训练脚本级别的外部设置，
> 而不是模型输入的一部分；这导致每个 horizon 训练一套 specialist head，无法统一 target
> positions、prefix consistency 和 future-step dependency。我们需要一个 target-set indexed
> decoder，使预测目标 $T=\{\tau_1,\dots,\tau_m\}$ 显式进入模型计算与训练目标。

用符号表示，当前 fixed-head specialist 是：

$$
Z=E_\theta(X), \qquad
\hat{Y}_{1:H}^{(H)} = W_H\operatorname{Flatten}(Z),
$$

其中 $H$ 决定模型实例或训练 run，$W_H$ 是 horizon-specific parameter。不同 $H$ 下，
同一个 future step $\tau$ 可能由不同参数、不同训练轨迹和不同 optimum 产生：

$$
\hat{Y}_{\tau}^{(96)} \neq \hat{Y}_{\tau}^{(720)}.
$$

目标是把 horizon 外部设置改写为 target set 输入：

$$
T=\{\tau_1,\dots,\tau_m\},
$$

$$
Q_T = q_\phi(T),
$$

$$
\hat{Y}_T = D_\theta(Z,Q_T,M_T),
$$

其中 $M_T$ 是 target-query interaction policy。这样同一个 future position 的语义由
$q_\phi(\tau)$ 绑定，而不是由某个 horizon-specific output row 隐式决定。

## Step 3: 问题是否真实且值得研究

[Strong Evidence] 问题真实存在：

1. Prefix consistency diagnostics 已显示同一 prefix 在不同 horizon request 下会发生偏移。
2. Segment oracle 显示不同 checkpoint 在不同 future sub-interval 上局部最优，但 h720
   specialist 又在全局平均上最强，说明 horizon specialization 与 segment specialization
   同时存在。
3. A.1-A.6 均未能在 one-to-one setting 下稳定超过 fixed specialist head，说明继续补丁式
   优化 specialist head 的边际收益很低。

[Decision] 该问题值得研究，但 paper story 必须避免过强承诺：

- 不能声称 “variable-horizon 必然提升 accuracy”；
- 不能把 “一个模型替代四个模型” 单独作为高水平论文核心；
- 必须证明 target-set formulation 至少满足以下之一：
  1. 在单模型设置下接近或超过 horizon-specific specialist；
  2. 显著改善 prefix consistency / target-set consistency；
  3. 作为 future-aware state 或 MoE routing 的稳定 carrier，并在后续机制中转化为性能收益。

[Inference] 与 A.1-A.6 相比，新方向的研究价值在于它改变任务接口，而不是修补 fixed head。
它承认 horizon-specific specialist 很强，因此第一阶段不要求在每个 one-to-one setting 下
直接击败 specialist；第一阶段要衡量的是 amortization gap、consistency gain 和是否产生
可承载后续机制的 target-side states。

## Step 4: Candidate Idea

候选 idea：`Target-Set Forecasting Decoder`。

核心设计原则：

1. `H` 不再只作为训练脚本参数，而是由 target set $T$ 输入模型。
2. 每个 target position 生成 query token $q_\phi(\tau,H)$ 或 $q_\phi(\tau)$。
3. target queries 通过 cross-attention 从 history patch states $Z$ 读取信息。
4. target queries 之间允许受控 self-attention，但必须有 structured mask，避免 prefix
   输出被额外 future query 任意改写。
5. 输出端使用 shared local projection 或 patch decoder，而不是每个 horizon 一套大 linear head。
6. 训练时使用 mixed target-set sampling，从 `{96,192,336,720}` 或 prefix subsets 中采样
   target set。

最小数据流：

$$
Z = E_\theta(X) \in \mathbb{R}^{B \times N \times d},
$$

$$
Q_T = \operatorname{TargetEmbed}(T) \in \mathbb{R}^{B \times |T| \times d},
$$

$$
U_T = \operatorname{CrossAttn}(Q_T,Z,Z),
$$

$$
\tilde{U}_T = \operatorname{MaskedSelfAttn}(U_T, M_T),
$$

$$
\hat{Y}_T = O_\theta(\tilde{U}_T).
$$

第一版可以把 $T$ 定义为 future segments 而不是 every-step tokens，以控制成本：

$$
T=\{[1,48],[49,96],\dots\}.
$$

每个 target segment query 输出一个 patch：

$$
\hat{Y}_{a_j:b_j}=O_\theta(\tilde{U}_j).
$$

## Step 5: 理论可行性

[Hypothesis] 如果 fixed head 的优势主要来自 horizon-specific dense rows，那么 target-set
decoder 在 one-to-one setting 下未必立刻超过 fixed head。但它可能在 mixed-horizon training
中获得额外样本共享，使同一个 target position 的 representation 更一致。

[Hypothesis] prefix consistency 可由 target-set interface 直接约束。对于 $T_s \subset T_l$，
要求：

$$
\hat{Y}_{T_s} = \Pi_{T_s}\hat{Y}_{T_l},
$$

或者在训练中加入 soft consistency：

$$
\mathcal{L}_{cons}
=
\left\|
\hat{Y}_{T_s}
-
\operatorname{stopgrad}(\Pi_{T_s}\hat{Y}_{T_l})
\right\|_2^2.
$$

[Hypothesis] QDF-style future covariance 可以作为第二阶段 objective，而不是第一版 architecture
的一部分。第一版先记录 per-step / per-segment error covariance，判断 standard MSE 是否
压制 target-query decoder 的收益。

[Risk] 最大风险是 A.1 的 capacity collapse 重现。为降低该风险，第一版不应使用过窄 shared
MLP head；至少需要以下 capacity controls：

- target query cross-attention 后使用 two-layer patch decoder；
- 与 `PatchEncoderFixedHead` 对齐 encoder width 和 training budget；
- 加入 parameter-count report；
- 对比 `H=720 fixed head prefix`，而不是只对比四个 horizon-specific specialists。

[Risk] 若 target-set decoder 只能换来 consistency 而不能接近 specialist accuracy，则它不能
单独作为 paper-core。此时可作为 future-aware / MoE 的 carrier 继续评估，但必须通过后续
性能 gate 才能保留。

## Step 6: 第一版实验协议

第一版不再是 one-to-one-only gate，而是 `mixed target-set gate`。

### Models

| Model | 作用 |
| --- | --- |
| `PatchEncoderFixedHead` | horizon-specific specialist upper reference |
| `PatchEncoderFixedHead-H720-prefix` | single longest-horizon baseline，评估 prefix reuse |
| `PatchEncoderTargetSetDecoder` | target-set candidate |

### Training

- datasets: `ETTh2`, `ETTm1`, `Weather`
- target sets: `{96,192,336,720}` and optional prefix subsets
- seed: `2021`
- seq_len: `336`
- first gate epochs: same as Phase1 remote gate unless smoke requires shorter run
- mixed-horizon sampling: each batch samples one target set from `{96,192,336,720}`
- optional consistency pair: sample `(T_s,T_l)` with $T_s \subset T_l$ every fixed number of steps

### Metrics

Primary:

- MSE / MAE at horizons `{96,192,336,720}`
- relative MSE vs horizon-specific `PatchEncoderFixedHead`
- relative MSE vs `PatchEncoderFixedHead-H720-prefix`

Consistency:

- prefix mismatch MSE between `T_s` and prefix of `T_l`
- truth alignment audit, to rule out window mismatch

Target-state diagnostics:

- target query state cosine by future distance
- segment-wise MSE and relative MSE
- parameter count and trainable parameter ratio
- per-step / per-segment residual covariance, for possible QDF follow-up

### Gate

第一版 target-set decoder 通过条件分为 compatibility pass 和 paper-core pass。

Compatibility pass:

1. single model vs horizon-specific `PatchEncoderFixedHead` mean relative MSE 不超过 `+1.0%`；
2. 任一 dataset 平均退化不超过 `+3.0%`；
3. 相比 `H=720 fixed head prefix`，h96/h192 prefix MSE 不退化，最好改善；
4. prefix consistency mismatch 相比 Phase0 fixed-head mismatch 明显下降；
5. target states 不完全同质，且参数量低于四个 specialist heads 总和。

Paper-core pass:

1. mean relative MSE vs horizon-specific `PatchEncoderFixedHead` < 0，或
2. consistency 明显改善且 future-aware / MoE follow-up 把该 target-side state 转化为稳定
   forecast gain。

[Decision] 若第一版只达到 compatibility pass，则不能立即作为论文核心，但可以作为后续
future-aware/MoE 的 carrier。若连 compatibility pass 都达不到，应暂停 decoder 主线，转向
重新评估 future-aware objective 或 external baseline reproduction。

## 当前边界

- 不复刻完整 ElasTST。
- 不引入 future covariates。
- 不引入 MoE。
- 不使用旧仓库代码。
- 不要求第一版直接赢过所有 horizon-specific specialists，但必须明确报告 amortization gap。
- 不把 one-model flexibility 本身包装成性能贡献；必须通过上述 gate 才能继续。
