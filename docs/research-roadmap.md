# 候选研究路线图

## 定位

[Fact] 本路线图基于 Zotero `FSA` 子集 12 篇 notes 的第一版机制整理：
TimeAlign、ProtoTS、TIMEPERCEIVER、TFPS、ElasTST、QDF、AME-TS、TimeEmb、
DTAF、SRP++、Seg-MoE、MoHETS。

[Fact] 当前文档不是最终架构规划，也不是实验结论。它的作用是把可选路线收敛为
一个可审计的 research program：哪些问题值得成为创新点，为什么它们在数学和
数据流上相互约束，以及后续应如何用最小实验排除弱路线。

[Inference] 一篇完整 SCI 论文的合理体量可以围绕三个互相耦合的创新点展开：
variable-horizon decoder、future-aware mechanism、MoE-style conditional
architecture。三者不能被解释为简单模块堆叠；更合理的论文主张应是：

> long-term forecasting 的核心困难不是只缺少更强 backbone，而是
> future positions、future states 与 heterogeneous temporal mechanisms
> 在同一个模型内没有被一致建模。

## 统一问题表述

令历史窗口为 $X \in \mathbb{R}^{B \times L \times C}$，预测目标为
$Y_{1:H} \in \mathbb{R}^{B \times H \times C}$。常规 direct forecasting
通常学习

$$
\hat{Y}_{1:H} = D_\theta(E_\theta(X), H),
$$

其中 $E_\theta(X)$ 是 history representation，$D_\theta$ 是 decoder 或 output
head。这个形式有三个隐含假设：

1. [Inference] 同一个 history representation 足以支持所有 future steps。
2. [Inference] future step 之间的依赖只需要通过 output dimension 或 loss 被动体现。
3. [Inference] 不同样本、不同 horizon segment、不同 temporal regime 可以共享同一套
   prediction operator。

12 篇种子文献分别从不同侧面挑战这些假设：

- SRP++ 指出 step-invariant representation 存在 expressiveness bottleneck。
- ElasTST 指出 varied-horizon inference 需要 horizon-invariance，否则长 horizon
  推理会改变短 horizon 已有位置的输出。
- TIMEPERCEIVER 把 target query 作为显式 prediction interface，说明 horizon 不应只
  是 output tensor 的 index。
- TimeAlign 说明 prediction-side hidden state 与 ground-truth future-side hidden
  state 存在 distribution mismatch。
- QDF 说明 future steps 的 loss 不是独立同权任务。
- Seg-MoE、AME-TS、MoHETS、TFPS、DTAF 说明 time series 的 expert specialization
  需要有 temporal structure、pattern shift 或 operator bias，而不是任意 softmax
  routing。

因此，本项目候选主张应从如下统一映射出发：

$$
\hat{Y}_{1:H}
= D_\theta\left(
Z,\ Q_H,\ S_H,\ R_H
\right),
$$

其中：

- $Z = E_\theta(X)$：history-derived state，来自过去窗口。
- $Q_H = \{q_1,\ldots,q_H\}$：future position queries，显式标识要预测的位置或
  horizon segment。
- $S_H$：future-aware latent state，训练时可被 future-side signal 约束，推理时只能
  由 $X$ 与 $Q_H$ 生成。
- $R_H$：routing / operator assignment，决定不同 future positions、segments 或
  regimes 使用哪些 conditional operators。

这不是架构定稿，而是后续所有候选模型都必须满足的数据流约束：

- 推理时不能访问 ground-truth future。
- 同一模型必须能处理多个 horizon，并能解释不同 horizon 的输出是否稳定。
- future-aware signal 必须进入 representation 或 decoder 的可验证位置，而不是只在
  口头上声明。
- MoE routing 必须服务于 future state 或 temporal operator difference，不能只是增加
  参数量。

## 创新点一：Variable-Horizon Decoder

### 问题

[Hypothesis] 如果 decoder 只是把 $Z$ flatten 后映射到固定 $H \times C$，那么模型
学习到的是 fixed-horizon projection。它可以在单个 benchmark horizon 上有效，但
不天然满足 variable-horizon 或 one model for multi-horizon 的要求。

更严格的要求是 horizon-invariance。对任意 $H_1 < H_2$，同一模型在预测 $H_1$ 和
$H_2$ 时，前 $H_1$ 个位置应尽量一致：

$$
\hat{Y}_{1:H_1}^{(H_1)}
\approx
\hat{Y}_{1:H_1}^{(H_2)}.
$$

ElasTST 的 placeholder + structured mask 给出一个重要原则：future positions 可以
显式进入模型，但更长 future placeholder 不应泄漏信息并改变较短 horizon 的输出。

### 候选数据流

候选 decoder 不应只接收一个全局 $Z$，而应接收 future queries：

$$
Q_H = g_\phi(1,2,\ldots,H),
\quad
U_H = A_\theta(Q_H, Z, M_H),
\quad
\hat{Y}_{1:H} = O_\theta(U_H).
$$

其中 $M_H$ 是 horizon mask 或 dependency policy。它要回答两个问题：

1. future queries 是否互相 attention？
2. 如果互相 attention，是否会破坏 horizon-invariance？

### 初步收敛

[Inference] 第一阶段不应直接追求复杂 decoder。更稳的路线是定义三类最小候选：

- `FixedHead`: 传统 fixed horizon head，作为下界。
- `QueryDecoder`: future query cross-attends to history state，但 query 之间不互相泄漏。
- `SegmentQueryDecoder`: 以 horizon segment 为 query/routing unit，兼容后续 MoE。

[Strong Evidence] SRP++ 和 ElasTST 共同支持一个判断：multi-horizon 的关键不是
“输出长度可变”这么简单，而是 future step-specific representation 与
horizon-invariance 必须同时检查。

### 必要诊断

- Prefix consistency: 比较 $H_1$ 推理与 $H_2$ 推理的前缀输出差异。
- Step-specificity: 检查不同 $q_h$ 诱导的 hidden states 是否可分，而不是全部退化为
  shared representation。
- Error-by-horizon: 不只报告平均 MSE/MAE，要报告 horizon position 或 segment 层面的
  error profile。

## 创新点二：Future-Aware Mechanism

### 问题

[Inference] forecasting 的训练数据包含 ground-truth future $Y$，但推理时不可见。
合理的 future-aware 机制不是泄漏未来，而是用 training-only signal 学习一个可由
history 和 future query 推断的 future latent space。

TimeAlign 提供了关键启发：训练时可以用 future-side reconstruction branch 得到
$H_y$，再约束 prediction branch 的 $H_x$ 靠近 future-side representation。ProtoTS
和 TIMEPERCEIVER 则提示 future pattern / target query 可以成为 decoder 的显式接口。

### 候选数学形式

训练时可以定义 teacher future state：

$$
S_H^{teacher} = T_\psi(Y_{1:H}, Q_H),
$$

推理可用的 student future state：

$$
S_H^{student} = P_\theta(Z, Q_H).
$$

future-aware loss 不是替代 forecasting loss，而是约束二者的 latent alignment：

$$
\mathcal{L}
=
\mathcal{L}_{pred}(\hat{Y}, Y)
+
\lambda \mathcal{L}_{align}(S_H^{student}, \text{stopgrad}(S_H^{teacher})).
$$

这里的关键边界是：

- $S_H^{teacher}$ 只在训练时出现。
- $S_H^{student}$ 必须只依赖 $X$、$Q_H$ 或允许的 covariates。
- alignment 的目标应是 future distribution / pattern，而不是复制 $Y$ 的数值本身。

### 与 variable-horizon decoder 的对齐

[Hypothesis] future-aware 机制最适合约束 decoder 的 query-side hidden states，而不是
只约束 encoder output $Z$。原因是 $Z$ 是 history-level state，而未来差异主要发生在
$h=1,\ldots,H$ 的 target positions 或 horizon segments 上。

因此更合理的数据流是：

$$
U_H = A_\theta(Q_H, Z),
\quad
S_H = P_\theta(U_H),
\quad
\hat{Y}_{1:H} = O_\theta(U_H, S_H).
$$

这使 future-aware signal 与 variable-horizon decoder 在同一张量层面对齐：
它们都作用于 future query states，而不是分别附着在不相干模块上。

### 必要诊断

- Alignment quality: $S_H^{student}$ 与 $S_H^{teacher}$ 的距离是否下降。
- Forecast relevance: alignment 改善是否对应 high-frequency、turning point 或远端
  horizon error 改善。
- Leakage audit: 去掉 teacher branch 后推理路径是否完全不依赖 $Y$。
- Proxy check: 如果 alignment 只改善 latent metric 但不改善 forecast，说明
  future-aware state 可能只是无效 proxy。

## 创新点三：MoE-Style Conditional Architecture

### 问题

[Inference] MoE 的必要性不能只由“不同样本需要不同专家”来支撑。对本项目更强的论证是：
variable horizon 和 future-aware state 已经把预测过程分解为多个 future query /
segment states；这些 states 可能对应不同 temporal mechanisms，因此需要 conditional
operators。

这与 Seg-MoE、AME-TS、MoHETS、TFPS、DTAF 的共同证据一致：

- routing granularity 应尊重 time series continuity，不能默认 token-wise。
- expert identity 应有结构含义，否则 specialization 难解释。
- expert 可以是不同 operator bias，而不一定都是同构 MLP。
- MoE 可以处理 pattern shift、non-stationarity 或 residual stabilization。

### 候选数据流

把 routing 放在 future query / segment state 上，而不是随意放在 encoder token 上：

$$
r_h = \text{Router}_\theta(U_h, S_h, q_h),
\quad
\tilde{U}_h = \sum_{k=1}^{K} r_{h,k} E_k(U_h),
\quad
\hat{Y}_h = O_\theta(\tilde{U}_h).
$$

如果使用 segment routing，则：

$$
r_j = \text{Router}_\theta(U_{a_j:b_j}, S_{a_j:b_j}),
\quad
\tilde{U}_{a_j:b_j} = \sum_{k=1}^{K} r_{j,k} E_k(U_{a_j:b_j}).
$$

这个位置比 input-token MoE 更符合本项目主张：expert 选择直接对应未来预测机制，而不是
只对过去 token 做条件变换。

### Expert 的合理角色

[Hypothesis] 首轮不应直接堆叠大规模 experts。更可审计的候选包括：

- `TrendOperator`: 偏向低频趋势或 smooth projection。
- `SeasonalOperator`: 偏向 periodic / frequency structure。
- `LocalResidualOperator`: 偏向短期扰动和 residual correction。
- `LinearStateOperator`: 作为低参数、可解释的 state transition baseline。

[Inference] 这些 operator bias 不必一开始全部实现。第一阶段可以先用同构 lightweight
experts 验证 routing 位置，再在有证据时引入 heterogeneous experts。

### 与前两个创新点的对齐

MoE 不应作为第三个孤立模块。它必须回答：

1. variable-horizon decoder 产生的 $U_H$ 是否存在 horizon/segment-specific
   mechanism difference？
2. future-aware state $S_H$ 是否能提供比 raw history state 更稳定的 routing signal？
3. routing pattern 是否与 horizon error、future state cluster 或 temporal regime 有对应关系？

如果这些问题没有证据支持，MoE 只能算参数扩张，不能算核心创新。

### 必要诊断

- Routing entropy: expert 使用是否塌缩或完全平均。
- Horizon-routing relation: 不同 horizon segment 是否形成可解释 routing pattern。
- State-routing relation: routing 是否与 future-aware state 的 cluster/trajectory 有一致性。
- Expert ablation: 去掉某类 expert 或固定 routing 后，哪些 horizon 或 regime 受损。
- Parameter-control: 与同参数 dense model 比较，排除“只是参数更多”的解释。

## 三者的候选统一框架

当前最稳的候选不是一次性定义最终模型，而是保留一个可逐级开启的框架：

$$
Z = E_\theta(X),
$$

$$
U_H = A_\theta(Q_H, Z, M_H),
$$

$$
S_H = P_\theta(U_H),
$$

$$
\tilde{U}_H = \text{MoE}_\theta(U_H, S_H, Q_H),
$$

$$
\hat{Y}_{1:H} = O_\theta(\tilde{U}_H).
$$

训练时可选 teacher branch：

$$
S_H^{teacher}=T_\psi(Y_{1:H}, Q_H),
\quad
\mathcal{L}_{align}(S_H, S_H^{teacher}).
$$

这个统一框架的逻辑是：

1. $Q_H$ 解决 future position 被隐式处理的问题。
2. $M_H$ 约束 variable-horizon inference 的前缀一致性。
3. $S_H$ 让 future-aware signal 落在 future query state 上。
4. MoE 以 $U_H$、$S_H$、$Q_H$ 为 routing evidence，服务于 future mechanism 分化。

[Speculative] 如果后续实验显示 $S_H$ 对 routing 没有帮助，则 future-aware 与 MoE 应分离；
如果 $Q_H$ 已经足够解释 horizon 差异，则 MoE 可能只需要作为 segment-level adapter；
如果 horizon-invariance 与 future-aware alignment 冲突，则需要重新设计 mask 或 teacher
state 的粒度。

## 分阶段收敛计划

### Phase 0：建立可证伪基线

目标：不要先发明复杂模型，先证明当前问题确实存在。

- 实现或接入 `FixedHead` baseline。
- 定义 multi-horizon protocol：同一模型覆盖多个 $H$，或最长 $H_{max}$ 下评估多个
  prefix horizons。
- 记录 error-by-horizon、prefix consistency、parameter count。
- Baseline selection 见 `docs/experiments/phase0-baseline-selection.md`：当前推荐
  `PatchTST-style channel-independent patch encoder + FixedHead` 作为 internal base，
  `DLinear` 作为 sanity floor，`SRSNet` 作为重点 external comparison。
- Experiment protocol 见 `docs/experiments/phase0-experiment-protocol.md`；当前 Phase 0
  gate 比较 `DLinear`、`PatchEncoderFixedHead`、`SegTSFTDenseFixedHead`，dataset matrix
  为 ETTh2、ETTm1、Weather，horizon matrix 为 `{96,192,336,720}`。

通过条件：

- 固定 head 在 variable-horizon 或 prefix consistency 上暴露可量化问题。
- 如果问题不存在，后续 decoder 创新需要重新论证。

### Phase 1：Variable-Horizon Decoder Gate

目标：验证 future query / structured mask 是否是必要改动。

- 对比 `FixedHead`、`QueryDecoder`、`SegmentQueryDecoder`。
- 不引入 MoE，不引入 teacher future branch。
- 重点看 prefix consistency、远端 horizon error、短端 horizon 是否被牺牲。

通过条件：

- query/segment decoder 在至少一个核心 dataset 上改善 horizon stability，且不靠明显增加参数。

### Phase 2：Future-Aware Gate

目标：验证 training-only future state 是否提供有效监督。

- 在最稳 decoder 上加入 teacher future branch。
- 对比 no-alignment、latent alignment、可能的 frequency/segment alignment。
- 严格检查推理路径无 future leakage。

通过条件：

- alignment 改善 forecast-relevant metrics，而不只是 latent distance。
- 改善应能定位到特定 horizon 或 pattern，而不是平均指标微弱波动。

### Phase 3：MoE Gate

目标：验证 conditional operators 是否必要，以及 routing 是否与 future state 对齐。

- 首先在 future query / segment state 上放轻量 MoE。
- 对比 dense parameter-control、fixed routing、random routing、no future-state routing。
- 若 routing 有证据，再探索 heterogeneous operators。

通过条件：

- MoE 改善不能只来自参数量。
- routing diagnostics 与 horizon segment、future-aware state 或 error profile 有对应关系。

### Phase 4：统一模型与论文主张

目标：只把通过 gate 的机制合并成最终模型。

- 若三者均有效：形成 unified variable-horizon future-aware MoE framework。
- 若 future-aware 有效但 MoE 无效：论文主张应收敛到 decoder + future-state learning。
- 若 MoE 有效但 alignment 无效：论文主张应收敛到 horizon-conditioned conditional operator。
- 若 decoder gate 不通过：整个项目不应继续围绕 variable-horizon 讲故事。

## 当前不应冒进的点

- 不应直接把 12 篇文献的模块全部堆进一个模型。
- 不应把 handcrafted descriptors 放入主路径，除非后续有明确证据和用户批准。
- 不应默认 future covariates 可用，因为 benchmark protocol 未必支持。
- 不应直接迁移旧仓库结果作为当前证据。
- 不应在没有 parameter-control 和 routing diagnostics 时宣称 MoE 是机制贡献。
- 不应只报告平均 MSE/MAE；本路线的核心证据必须包含 horizon-level 和 routing-level
  diagnostics。

## Baseline 边界

- SRSNet 是重点 comparison baseline。
- 旧 `R_2026_FSA` 中的 SRSNet 性能数据只能在用户批准后作为证据迁入或引用。
- 新仓库内的 baseline 复现应保留 native upstream evidence，再决定是否写本地 wrapper。
