# 候选研究路线图

## 定位

[Fact] 本路线图基于 Zotero `FSA` 子集 12 篇 notes 的第一版机制整理：
TimeAlign、ProtoTS、TIMEPERCEIVER、TFPS、ElasTST、QDF、AME-TS、TimeEmb、
DTAF、SRP++、Seg-MoE、MoHETS。

[Fact] 当前文档不是最终架构规划，也不是 paper claim。它的作用是把当前研究方向
收敛为一个可证伪的 research program：哪些机制值得继续，哪些机制只是诊断入口，
以及每一阶段应该如何避免把多个变量混在一起解释。

## 长研究执行模板

[Decision] 本项目后续所有长研究阶段固定使用以下闭环，而不是按“先实现、再解释”的
方式推进：

1. 调研分析。
2. 提出待解决问题。
3. 评估问题是否值得研究，以及问题是否真实存在。
4. 提出 idea。
5. 评估 idea 的理论可行性。
6. 设计具体方案与实验协议。
7. 实现方案。
8. 远程训练。
9. 评估结果。
10. 判断是否通过：同时看 performance evidence 和 paper narrative 是否成立。
11. 若不通过，评估应回退至哪一步，然后继续循环。

该模板是本路线图的执行边界：

- step 1-3 决定一个问题是否进入实现，而不是默认所有直觉都值得做实验；
- step 4-6 必须先形成可证伪假设、数学数据流和实验协议，再进入 coding；
- step 7-9 只负责产生可审计 evidence，不自动构成通过；
- step 10 必须同时判断指标收益、机制解释和论文故事；
- step 11 必须给出明确 rollback point，例如回到问题定义、理论可行性、方案设计或实现细节。

[Inference] 对当前 roadmap 而言，这意味着 `Future-Segment Decoder`、
`Future-Aware Mechanism` 和 `Future-Side MoE` 不是三段顺序堆叠的模块，而是三个候选
research loops。一个 loop 未通过时，应先判断回退点，而不是继续把下一个机制叠上去。

[Strong Evidence] 经过 Phase0 baseline gate、targeted controls、seed variance、
prefix consistency 和 segment-wise checkpoint oracle 后，`PatchEncoderFixedHead`
被选为 Phase1 的 canonical internal base。固定 direct head 暴露了可量化问题，但
问题幅度是中等的，而不是灾难性的。

[Inference] 因此，本文路线不应继续把 “variable-horizon” 本身作为第一创新点的
核心卖点。更合理的主线是：

> long-term forecasting 的核心问题不是只缺少更强 encoder，也不是简单支持可变
> 输出长度，而是 future positions / future segments / future states 没有被作为
> 一个明确的预测过程来建模。

据此，当前候选论文主张收敛为三个互相耦合的创新点：

1. `Future-Segment Decoder`: 替代 static fixed head，显式构造 future-side states。
2. `Future-Aware Mechanism`: 用 training-only future signal 约束可推理的 future latent state。
3. `Future-Side MoE`: 在 future segment / future state 上做 conditional operators。

`Variable-horizon` 和 `prefix consistency` 仍然保留，但它们的角色从主 claim 降级为：

- 诊断 fixed head 是否存在输出策略问题；
- 检查 future-side decoder 是否具备 one-model compatibility；
- 作为附加能力，而不是第一轮机制成败标准。

## Phase0 证据更新

### Baseline 选择

[Fact] Phase0 gate 比较了 `DLinear`、`PatchEncoderFixedHead`、
`SegTSFTDenseFixedHead`。在 follow-up controls 和 seed variance 后，
`PatchEncoderFixedHead` 被选为 canonical internal base。

[Fact] `PatchEncoderFixedHead` 是 clean PatchTST-style base，不是 exact PatchTST
paper reproduction。它的价值在于：encoder 简洁、性能合理、fixed head 缺陷清楚，
便于后续只替换 output / decoder side。

### Prefix consistency

[Strong Evidence] Fixed direct head 暴露了可量化 prefix issue：

- `Weather / H=96`: h720 prefix 比 h96 fixed head 劣化 `+4.79%` MSE。
- `ETTm1 / H=96`: h720 prefix 比 h96 fixed head 劣化 `+4.70%` MSE。
- 最大 fixed-head prediction mismatch 为 `ETTm1 / H=192` 的 `0.044742` MSE。
- `truth_alignment_mse = 0.0`，说明该现象不是数据窗口错位造成的。

[Inference] 这支持 fixed head 存在问题，但不足以单独支撑
“variable-horizon decoder 一定显著提升性能”。

### Segment-wise checkpoint oracle

[Strong Evidence] 在统一 `0-720` 预测区间内，每 48 step 一个 segment，短 checkpoint
用 rolling autoregression 扩展到 720 后，`pred_len=720` checkpoint 在三个数据集的
全区间平均 MSE 都是最优：

| Dataset | h96 avg MSE | h192 avg MSE | h336 avg MSE | h720 avg MSE | Best |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | 0.461915 | 0.418412 | 0.413482 | 0.407403 | 720 |
| ETTm1 | 0.463471 | 0.428760 | 0.416765 | 0.412788 | 720 |
| Weather | 0.326617 | 0.331844 | 0.327533 | 0.323127 | 720 |

[Strong Evidence] 但 h720 并不是所有 segment 的局部最优：

| Dataset | h96 wins | h192 wins | h336 wins | h720 wins |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | 0 | 4 | 1 | 10 |
| ETTm1 | 2 | 0 | 5 | 8 |
| Weather | 3 | 2 | 0 | 10 |

[Inference] 该结果否定了一个过强假设：短 horizon 模型并不天然统治短期预测。
它同时说明另一个更有价值的问题：同一个 static fixed head 很难在所有 future segment
上都达到局部最优。因此，Phase1 应验证 future-side decoder，而不是直接追求
one-model for all horizons。

## 统一问题表述

令历史窗口为：

$$
X \in \mathbb{R}^{B \times L \times C},
$$

预测目标为：

$$
Y_{1:H} \in \mathbb{R}^{B \times H \times C}.
$$

常规 direct forecasting 可写为：

$$
\hat{Y}_{1:H} = D_\theta(E_\theta(X), H),
$$

其中 $E_\theta(X)$ 是 history representation，$D_\theta$ 通常是 fixed output head。
这个形式隐含三个假设：

1. [Inference] 同一个 history representation 足以支持所有 future steps。
2. [Inference] future step 之间的结构只需要通过 output dimension 或 loss 被动体现。
3. [Inference] 不同 future segments 与 temporal regimes 可以共享同一套 prediction operator。

当前路线改写为：

$$
Z = E_\theta(X),
$$

$$
U_H = A_\theta(Z, Q_H, G_H),
$$

$$
S_H = P_\theta(U_H),
$$

$$
\tilde{U}_H = C_\theta(U_H, S_H, Q_H),
$$

$$
\hat{Y}_{1:H}=O_\theta(\tilde{U}_H).
$$

其中：

- $Z$: history-derived state。
- $Q_H$: future position 或 future segment queries。
- $G_H$: future segment grouping / dependency policy。
- $U_H$: decoder 生成的 future-side hidden states。
- $S_H$: future-aware latent state，可在训练时被 future signal 约束。
- $C_\theta$: conditional operator，可由 dense adapter 或 MoE 实现。
- $O_\theta$: final output projection。

该数据流的底层约束是：

- 推理时不能访问 ground-truth future。
- 第一阶段必须先证明 decoder-side 改动在 one-to-one horizon setting 下有效。
- one-model for all horizons 是后续 compatibility gate，不是 Phase1-A 的成败标准。
- MoE routing 必须服务于 future states 或 future segments，不能只是增加参数量。

## 创新点一：Future-Segment Decoder

### 为什么不再以 variable-horizon 为主线

[Fact] ElasTST 提出了 varied-horizon forecasting，通过 placeholder、structured mask、
tunable RoPE、multi-scale patch 和 horizon reweighting 追求 horizon-invariant inference。

[Inference] 该问题形式优雅，但它不等价于更好的 forecasting performance。更长 horizon
请求下保持 prefix 不变，是 consistency 约束；而 forecasting 的主要困难还包括
future segment 的误差增长、远端 pattern 漂移、不同 temporal regime 的 operator 差异。

[Strong Evidence] 本项目 Phase0 结果显示：

- h720 checkpoint 并不在短 prefix 上普遍失败；
- 短 horizon checkpoint 也不稳定统治短期 segment；
- fixed head 的 prefix 问题存在，但幅度中等。

因此，Phase1 不应直接复刻 ElasTST 式 variable-horizon strategy，也不应把
“one model for all horizon” 作为第一轮目标。

### 需要回答的新问题

[Hypothesis] 关键问题是 fixed head 是否把未来预测过度压缩成一次静态投影：

$$
\hat{Y}_{1:H} = W \cdot \text{Flatten}(Z).
$$

更合理的 decoder 应显式生成 future-side states：

$$
U_{1:J}=A_\theta(Z,Q_{1:J}),
$$

其中 $J$ 可以是 future segment 数量。例如在 $H=720$、segment length 为 48 时，
$J=15$。

每个 segment state 再生成对应区间：

$$
\hat{Y}_{a_j:b_j}=O_\theta(U_j).
$$

这样 decoder 的研究问题从 “输出长度是否可变” 转为：

1. 不同 future segment 是否需要不同 readout state？
2. segment state 是否能改善 error-by-segment profile？
3. segment state 是否为后续 future-aware alignment 和 MoE routing 提供稳定载体？

### Phase1-A 候选

Phase1-A 使用 one-to-one horizon training。每个 horizon 单独训练，避免把 decoder
有效性与 mixed-horizon optimization 混在一起。

候选模型：

| Model | 作用 |
| --- | --- |
| `PatchEncoderFixedHead` | Phase0 selected base |
| `PatchEncoderSegmentQueryHead` | 用 future segment queries 替代 fixed flatten head |
| `PatchEncoderHorizonConditionedHead` | 可选，对 fixed head 加 horizon/segment conditioning |

第一轮不引入 MoE，不引入 teacher future branch。

### Phase1-A 通过条件

- 在 one-to-one setting 下，MSE/MAE 不低于 fixed head，最好在 ETTm1 或 Weather 上提升。
- segment-wise MSE 更均衡，或在远端 horizon segment 上有明确改善。
- 参数量提升需要受控；若参数增加明显，必须加入 dense parameter-control。
- decoder-side states 不能完全退化为同质表示，需要通过 state similarity 或 segment
  sensitivity 诊断。

### Phase1-B: One-Model Compatibility Gate

只有 Phase1-A 通过后，才进入 one-model compatibility：

1. `train H=720, evaluate prefixes`;
2. `mixed horizon training`，从 `{96,192,336,720}` 采样 horizon；
3. 检查 prefix consistency、short-prefix MSE 和 full-horizon MSE。

[Inference] 如果 Phase1-A 无法优于 fixed head，则 one-model for all horizons 没有继续
投入的必要；如果 Phase1-A 有收益，Phase1-B 才能检验该 decoder 是否自然支持
multi-horizon deployment。

## 创新点二：Future-Aware Mechanism

### 问题

[Inference] forecasting 的训练数据包含 ground-truth future $Y$，但推理时不可见。
合理的 future-aware 机制不是泄漏未来，而是用 training-only signal 学习一个可由
history 和 future query 推断的 future latent space。

TimeAlign 提供了关键启发：训练时可以用 future-side reconstruction branch 得到
$S_H^{teacher}$，再约束 prediction branch 的 $S_H^{student}$ 靠近 future-side
representation。ProtoTS 和 TIMEPERCEIVER 则提示 future pattern / target query 可以
成为 decoder 的显式接口。

### 候选数学形式

训练时定义 teacher future state：

$$
S_H^{teacher}=T_\psi(Y_{1:H},Q_H),
$$

推理可用的 student future state：

$$
S_H^{student}=P_\theta(U_H).
$$

总 loss：

$$
\mathcal{L}
=
\mathcal{L}_{pred}(\hat{Y},Y)
+
\lambda \mathcal{L}_{align}(S_H^{student},\text{stopgrad}(S_H^{teacher})).
$$

关键边界：

- $S_H^{teacher}$ 只在训练时出现。
- $S_H^{student}$ 必须只依赖 $X$、$Z$、$Q_H$ 或允许的 covariates。
- alignment 应约束 future pattern / distribution，而不是直接复制 $Y$ 的数值。

### 与 Future-Segment Decoder 的对齐

[Hypothesis] future-aware signal 最适合落在 decoder-side future segment states 上，
而不是只约束 encoder output $Z$。原因是 $Z$ 是 history-level state，而预测困难主要
在 $h=1,\ldots,H$ 的 future positions 或 segments 上展开。

因此 Phase2 的默认接入点是：

$$
U_H = A_\theta(Z,Q_H),
\quad
S_H^{student}=P_\theta(U_H),
\quad
\hat{Y}_{1:H}=O_\theta(U_H,S_H^{student}).
$$

### Phase2 通过条件

- alignment distance 下降，同时 forecast-relevant metrics 改善。
- 改善可以定位到特定 horizon segment、turning point、high-frequency component 或
  long-horizon error。
- 推理路径通过 leakage audit：去掉 teacher branch 后完全不依赖 $Y$。
- 如果 latent metric 改善但 forecast 不变，则 future-aware state 只能视为无效 proxy。

## 创新点三：Future-Side MoE

### 问题

[Inference] MoE 的必要性不能只由“不同样本需要不同专家”支撑。对本项目更强的论证是：
Future-Segment Decoder 已经把预测过程分解为多个 future segment states；这些 states
可能对应不同 temporal mechanisms，因此 conditional operators 应该放在 future side。

这与 Seg-MoE、AME-TS、MoHETS、TFPS、DTAF 的共同证据一致：

- routing granularity 应尊重 time series continuity，不能默认 token-wise。
- expert identity 应有结构含义，否则 specialization 难解释。
- expert 可以是不同 operator bias，而不一定都是同构 MLP。
- MoE 可以处理 pattern shift、non-stationarity 或 residual stabilization。

### 候选数据流

把 routing 放在 future segment state 上：

$$
r_j=\text{Router}_\theta(U_j,S_j,q_j),
$$

$$
\tilde{U}_j=\sum_{k=1}^{K}r_{j,k}E_k(U_j),
$$

$$
\hat{Y}_{a_j:b_j}=O_\theta(\tilde{U}_j).
$$

这个位置比 input-token MoE 更符合本项目主张：expert 选择直接对应未来预测机制，而不是
只对过去 tokens 做条件变换。

### Expert 的合理角色

[Hypothesis] 首轮不应直接堆叠大规模 experts。更可审计的候选包括：

- `TrendOperator`: 偏向低频趋势或 smooth projection。
- `SeasonalOperator`: 偏向 periodic / frequency structure。
- `LocalResidualOperator`: 偏向短期扰动和 residual correction。
- `LinearStateOperator`: 作为低参数 state transition baseline。

[Inference] 这些 operator bias 不必一开始全部实现。Phase3 可以先用 lightweight
homogeneous experts 验证 routing 位置，再在有证据时引入 heterogeneous experts。

### Phase3 通过条件

- MoE 改善不能只来自参数量，必须对比 same-parameter dense control。
- routing entropy 不能塌缩，也不能完全平均。
- routing pattern 应与 future segment、future-aware state cluster 或 error profile 有
  对应关系。
- fixed routing、random routing、no future-state routing 应作为 ablations。

## 分阶段收敛计划

### Phase 0：建立可证伪基线

状态：已完成。

结论：

- canonical internal base: `PatchEncoderFixedHead`。
- fixed head 有可量化 prefix issue，但不足以单独支撑 variable-horizon 主线。
- segment oracle 显示 h720 全局平均最强，但局部 segment 存在不同 checkpoint 最优。
- Phase1 应从 future-side decoder 的 one-to-one gate 开始。

关键文档：

- `docs/experiments/phase0-baseline-selection.md`
- `docs/experiments/phase0-experiment-protocol.md`
- `analysis/phase0_prefix_consistency_report_20260621.md`
- `analysis/phase0_segment_oracle_20260621/phase0_segment_oracle_summary_zh.md`

### Phase 1：Future-Segment Decoder Gate

目标：验证 future segment states 是否比 fixed flatten head 更合理。

Phase1-A:

- one-to-one horizon training；
- datasets: 先用 ETTh2、ETTm1、Weather；
- horizons: `{96,192,336,720}`；
- 对比 `PatchEncoderFixedHead`、`PatchEncoderSegmentQueryHead`，可选
  `PatchEncoderHorizonConditionedHead`；
- 不引入 MoE，不引入 future teacher branch。

[Fact] 第一轮 `PatchEncoderSegmentQueryHead` 已完成并未通过：
`analysis/phase1_segment_decoder_gate_20260621/phase1_segment_decoder_gate_report.md`。

[Strong Evidence] 它在 12/12 个主指标设置和 30/30 个 segment-level 设置上均弱于
`PatchEncoderFixedHead`，平均 MSE 退化 `+6.79%`。这说明简单地用 segment query
cross-attention 替换 fixed flatten head 会损失 readout capacity。

[Decision] Phase1 不进入 one-model compatibility，也不在当前 SegmentQueryHead 上叠加
future-aware 或 MoE。当前应回退到第 5-6 步，重新设计保留 fixed-head capacity 的
future-side interface。

下一轮优先候选：

- `FixedHeadAdapter`: 保留 fixed flatten head，只在 readout 后加入 future-segment affine adapter。
- `SegmentQueryDenseHead`: segment query 提供 conditioning，但每个 segment 保留更强 dense readout。
- `StepQueryHead`: 更细粒度 step-level query，作为高成本备选。

Phase1-A.2 当前执行候选：

- `PatchEncoderFixedHeadAdapter`。
- 主路径仍是 `Flatten(Z) -> Linear(..., H)`，即不删除 Phase0 selected base 的 readout。
- future segment queries 通过 cross-attention 生成 adapter state。
- adapter 输出 $\gamma,\beta$，对 fixed-head normalized forecast 做：

$$
\hat{Y}=\hat{Y}^{base}\odot(1+\gamma)+\beta.
$$

- adapter final projection 采用 zero initialization，使初始 forward 等价于 fixed head。

[Hypothesis] 如果该候选通过，说明 future-side interface 的问题仍成立，只是第一轮替换
readout 的设计过激；如果它也失败，则 history-only future decoder 主线需要降级，
下一步应优先转向 future-aware teacher/student alignment，而不是继续堆叠 decoder。

Phase1-A.2 通过条件：

- main MSE 至少 `4/12` wins，且平均 relative MSE 不出现明显退化；
- 或 segment-level wins 集中改善远端/high-error segment；
- `adapter_delta_stats.csv` 显示 adapter 对 base prediction 有非零有效修正，避免把训练波动
  误判为机制收益。

[Fact] Phase1-A.2 `PatchEncoderFixedHeadAdapter` 已完成：
`analysis/phase1_fixed_adapter_gate_20260621/phase1_fixed_adapter_gate_report.md`。

[Evidence] 结果为：

- main MSE wins: `7/12`；
- segment-level MSE wins: `15/30`；
- mean relative MSE change: `+0.20%`；
- relative MSE range: `-2.02%` 到 `+3.74%`；
- mean adapter delta/base MAE ratio: `0.3200`。

[Decision] `PatchEncoderFixedHeadAdapter` 是 `partial_pass`，但不足以成为论文核心创新。
它支持一个更精确的判断：future-side interface 不是无效的，但仅靠 history-derived
segment adapter 难以稳定提升性能。

[Inference] Phase1 的下一步不应继续堆 adapter capacity，而应进入 Future-Aware Gate：
用 training-only future signal 学习可推理的 future latent state，并检验这种 supervision
是否能把 adapter/interface 从弱修正变成稳定机制收益。

Phase1-A.3 当前执行候选：

- `PatchEncoderFutureAwareAdapter`。
- 推理路径与 `PatchEncoderFixedHeadAdapter` 一致，不接收 future target。
- 训练时额外编码 ground-truth future segment 得到 $S^{teacher}$。
- history-derived adapter state 经过 projection 得到 $S^{student}$。
- 使用 $\mathcal{L}_{align}$ 约束 $S^{student}$ 靠近 stop-gradient teacher。
- 使用 $\mathcal{L}_{recon}$ 防止 teacher branch 变成无意义 anchor。

第一轮默认：

- `align_weight = 0.05`
- `recon_weight = 0.05`
- `segment_len = 48`

Phase1-A.3 通过条件：

- `prediction_leakage_max_abs <= 1e-7`；
- 相比 `PatchEncoderFixedHead` 至少 `6/12` main MSE wins；
- mean relative MSE < 0；
- alignment diagnostics 显示 teacher/student state 发生有效耦合。

[Fact] Phase1-A.3 `PatchEncoderFutureAwareAdapter` 已完成：
`analysis/phase1_future_aware_adapter_gate_20260621/phase1_future_aware_adapter_gate_report.md`。

[Evidence] 结果为：

- vs `PatchEncoderFixedHead`: main MSE wins `4/12`，mean relative MSE `+0.16%`；
- vs `PatchEncoderFixedHeadAdapter`: main MSE wins `5/12`，mean relative MSE `-0.01%`；
- leakage audit: `max_prediction_leakage_abs = 0.0`；
- mean teacher/student cosine: `0.4287`。

[Decision] Phase1-A.3 是 `partial_pass`，不是 paper-core pass。它证明 future-aware
teacher/student alignment 可以在无泄漏条件下运行并实际耦合 student state，但当前性能
证据不足。

[Diagnosis] 第一轮 future-aware 方案暴露了 reconstruction loss scale imbalance：
Weather 的 reconstruction loss 约 `645.08`，显著高于 ETTh2 的 `1.05` 和 ETTm1 的
`0.40`。这会让同一个 `recon_weight=0.05` 在不同 dataset 上产生完全不同的优化压力。

[Next] 不应直接扩大模型。下一步应做最小修补 gate：

- `ScaleNormalizedRecon`: 将 $\mathcal{L}_{recon}$ 按 target/segment energy 归一化；
- `AlignOnly`: 设置 `recon_weight=0`，检验是否 alignment 本身足够；
- 如果两者仍不能稳定超过 fixed head，再考虑转向新架构或重新定义 future-aware claim。

Phase1-A.4 当前执行候选：

- `PatchEncoderFutureAwareAlignOnly`。
- `PatchEncoderFutureAwareScaleNorm`。

[Decision] Phase1-A.4 是对 Phase1-A.3 的 step 5-6 rollback，而不是新一轮模型扩容。
它只回答一个具体问题：future-aware alignment 的失败是否主要来自
$\mathcal{L}_{recon}$ 的尺度不可比。

`AlignOnly` 的训练目标为：

$$
\mathcal{L}
=
\mathcal{L}_{pred}
+\lambda_{align}\mathcal{L}_{align}.
$$

`ScaleNorm` 的 reconstruction objective 为：

$$
\mathcal{L}_{recon}^{norm}
=
\frac{
\operatorname{MSE}(\hat{Y}^{teacher,norm},Y^{norm})
}{
\operatorname{mean}((Y^{norm})^2)+\epsilon
}.
$$

Phase1-A.4 通过条件：

- 所有 repair candidates 的 `prediction_leakage_max_abs <= 1e-7`；
- 最优 repair candidate 相比 `PatchEncoderFixedHead` 至少 `6/12` main MSE wins；
- mean relative MSE < 0；
- teacher/student coupling 仍存在，且 normalized reconstruction loss 不再被 Weather
  的 raw scale 主导。

[Inference] 若 Phase1-A.4 仍不能达到通过条件，则 future-aware adapter 方向应降级：
它可作为 diagnostic evidence，但不应作为论文核心。下一步应回到 step 3-5 重新判断
“future latent state alignment” 是否是正确问题，或转向更基础的 decoder/state architecture。

Phase1-B:

- 仅在 Phase1-A 通过后执行；
- 检查 `train H=720, evaluate prefixes` 和 mixed-horizon training；
- 目标是 one-model compatibility，不是主效果 claim。

### Phase 2：Future-Aware Gate

目标：验证 training-only future state 是否提供有效监督。

- 在 Phase1-A 最稳 decoder 上加入 teacher future branch。
- 对比 no-alignment、latent alignment、segment-level alignment。
- 严格检查推理路径无 future leakage。

### Phase 3：Future-Side MoE Gate

目标：验证 conditional operators 是否必要，以及 routing 是否与 future state 对齐。

- 首先在 future segment states 上放轻量 MoE。
- 对比 dense parameter-control、fixed routing、random routing、no future-state routing。
- 若 routing 有证据，再探索 heterogeneous operators。

### Phase 4：统一模型与论文主张

目标：只合并通过 gate 的机制。

- 若 decoder、future-aware、MoE 都有效：形成 unified future-segment forecasting framework。
- 若 decoder + future-aware 有效但 MoE 无效：论文主张收敛为 future-side decoder with
  future-state alignment。
- 若 decoder + MoE 有效但 alignment 无效：论文主张收敛为 future-side conditional
  operators。
- 若 decoder gate 不通过：项目不应继续围绕 decoder 讲故事，应回到 future-aware 或
  external baseline reproduction。

## 当前不应冒进的点

- 不应直接把 12 篇文献的模块全部堆进一个模型。
- 不应把 ElasTST 式 variable-horizon 作为默认主线。
- 不应一开始做 one-model for all horizons，并把结果作为 decoder 成败判断。
- 不应把 rolling autoregression 作为主创新；它只能是 baseline 或 diagnostic。
- 不应把 handcrafted descriptors 放入主路径，除非后续有明确证据和用户批准。
- 不应默认 future covariates 可用，因为 benchmark protocol 未必支持。
- 不应直接迁移旧仓库结果作为当前证据。
- 不应在没有 parameter-control 和 routing diagnostics 时宣称 MoE 是机制贡献。
- 不应只报告平均 MSE/MAE；本路线的核心证据必须包含 segment-level、horizon-level
  和 routing-level diagnostics。

## Baseline 边界

- SRSNet 是重点 comparison baseline。
- 旧 `R_2026_FSA` 中的 SRSNet 性能数据只能在用户批准后作为证据迁入或引用。
- 新仓库内的 baseline 复现应保留 native upstream evidence，再决定是否写本地 wrapper。
