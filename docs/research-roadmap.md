# FATST Research Roadmap

## 定位

[Fact] 本仓库是 `R_2026_FSA` 的 clean successor，但不默认继承旧仓库代码、
配置、实验输出或未审计记忆。

[Fact] 当前目标是围绕 time series forecasting 形成一篇高水平 SCI 期刊论文。候选
方向包括：

1. `one model for multi-horizon forecasting`
2. `future-aware architecture`
3. `MoE-style conditional computation`

[Decision] 这三个方向不是三个可以直接堆叠的模块，而是一个统一模型框架的候选构件。
每个构件必须先通过独立 research loop，证明它解决的问题真实、理论上合理、工程上可实现，
并且能带来 performance evidence 或 paper narrative evidence。

## 长研究执行模板

[Decision] 本项目后续所有长研究阶段固定使用以下 11-step loop：

1. 调研分析。
2. 提出待解决问题。
3. 评估问题是否值得研究，以及问题是否真实存在。
4. 提出 idea。
5. 评估 idea 的理论可行性。
6. 设计方案。
7. 实现方案。
8. 远程训练。
9. 评估结果。
10. 判断是否通过：同时看性能收益与论文故事是否成立。
11. 若不通过，评估应回退至哪一步，然后继续循环。

该模板是推进边界，而不是文档装饰：

- step 1-3 决定问题是否值得进入实现；
- step 4-6 必须形成可证伪 hypothesis、数学数据流、实验协议和通过/回退条件；
- step 7-9 只负责产生可审计 evidence，不自动构成通过；
- step 10 必须同时判断 `performance evidence` 和 `paper narrative`；
- step 11 必须给出明确 rollback point，例如回到问题定义、理论可行性、方案设计或实现细节。

[Decision] 后续任何 decoder、future-aware、MoE 或新 architecture 候选，都必须显式标注
当前处于上述哪一步。一个机制未通过时，不继续叠加下一个机制；先判断应回退到哪一步。

### 阶段记录格式

[Decision] 每一轮长研究都必须在 protocol、experiment report 或 roadmap 中留下同构记录：

| Field | Required Content |
| --- | --- |
| `current_step` | 当前处于 11-step loop 的哪一步 |
| `problem` | 待解决问题，以及它为什么不是已有实验已经否定的问题 |
| `existence_evidence` | 问题真实存在的 artifact、公式推导或可复查现象 |
| `idea` | 核心 idea，不超过一个主机制 |
| `theory_check` | 数据流、数学约束、可能成立的原因和反例 |
| `design` | model/training/evaluation 的最小方案 |
| `gate` | pass/fail/rollback 条件 |
| `artifacts` | 代码版本、输出路径、报告和表格 |
| `decision` | 是否通过；若不通过，回退到哪一步 |

[Decision] 该记录格式用于防止两类错误：一是问题尚未证明真实就进入实现；二是某个
partial signal 失败后继续叠加 future-aware 或 MoE。后续每个候选必须先明确自己在这个
loop 中的角色，再进入远程训练。

## 当前总判断

[Strong Evidence] Phase0 与 Phase1-A.1 到 A.6 已经给出一个关键结论：
`PatchEncoderFixedHead` 是当前 canonical internal base，且 horizon-specific fixed head
非常强。围绕它追加轻量 decoder patch、adapter、future-aware alignment 或 output residual，
都没有稳定形成 paper-core 级别收益。

[Inference] 因此，当前不应继续把 “variable-horizon 本身” 作为第一创新点，也不应直接在
A.5/A.6 上叠加 future-aware 或 MoE。更合理的收敛方向是：

> time series forecasting 的缺口不只是 encoder 不够强，也不只是输出长度不灵活；
> 更底层的问题是 target future positions / segments 没有作为预测过程的显式输入来建模。

[Decision] 当前 Phase1 已从 `Future-Segment Decoder patching` 回退到
`Target-Set Forecasting Decoder` 的问题重定义。该方向先验证 one-model / target-set
interface 是否能以可控 amortization gap 接近 horizon-specific specialists，并改善
prefix consistency。只有该 interface 至少达到 compatibility pass，后续 future-aware 和 MoE
才有稳定 carrier。

[Decision] `PatchEncoderPrefixRiskWeighted` remote gate 已达到 compatibility pass：
one-model target-set interface 在不改变 architecture 的情况下，将 mean relative MSE 从第一版
`+0.62%` 推到 `-0.43%`，且 H720-prefix h96/h192 相比 fixed H720-prefix 改善 `-2.46%`。
这说明 target-set carrier 是可用的，mixed-horizon objective 确实是一个 active bottleneck。

[Decision] 但 R.3 仍不是 paper-core pass。它本质是 objective weighting，不是足够完整的
decoder/process mechanism；并且 `ETTm1 / h96` 仍退化 `+2.83%`，long horizon 也没有全面领先。
因此当前阶段不能停在“调 loss 通过 compatibility”上，而应进入下一轮 step 1-6：
围绕 target-side state 与 future-aware mechanism，提出一个更强的、可解释的 forecasting
process innovation。

下一轮必须先回答：

1. prefix-risk weighting 暴露出的早期 future risk 是否对应可建模的 future-state uncertainty；
2. target-side state $U_T$ 是否能承载 training-only future signal，而不是只作为 readout FiLM；
3. 新 idea 是否能在不破坏 R.3 compatibility carrier 的条件下，把 future-aware signal 转化为
   horizon/segment 可定位的 forecast gain。

## Phase0: Canonical Base

状态：已完成。

### 结论

[Fact] Phase0 gate 比较了：

- `DLinear`
- `PatchEncoderFixedHead`
- `SegTSFTDenseFixedHead`

[Decision] `PatchEncoderFixedHead` 被选为 canonical internal base。它是 clean
PatchTST-style base，而不是 exact PatchTST paper reproduction。选择它的原因是：

- encoder 简洁；
- baseline performance 合理；
- fixed direct head 的问题可诊断；
- 后续可以只替换或研究 output / decoder side。

### 关键证据

[Strong Evidence] Fixed direct head 存在可量化 prefix issue：

| Diagnostic | Evidence |
| --- | --- |
| `Weather / H=96` | h720 prefix 比 h96 fixed head 劣化 `+4.79%` MSE |
| `ETTm1 / H=96` | h720 prefix 比 h96 fixed head 劣化 `+4.70%` MSE |
| max fixed-head prediction mismatch | `ETTm1 / H=192`, `0.044742` MSE |
| truth alignment audit | `truth_alignment_mse = 0.0` |

[Strong Evidence] Segment-wise checkpoint oracle 显示，短 horizon checkpoint 并不天然统治
短期预测；`pred_len=720` checkpoint 在三个数据集的 `0-720` 全区间平均 MSE 均为最优。
但 h720 也不是所有 segment 的局部最优。

[Inference] 这说明 fixed head 的问题真实存在，但幅度是中等的。它不足以单独支撑
“variable-horizon decoder 一定能显著提升性能”的强 claim。

### 证据入口

- `docs/experiments/phase0-baseline-selection.md`
- `docs/experiments/phase0-experiment-protocol.md`
- `analysis/phase0_prefix_consistency_report_20260621.md`
- `analysis/phase0_segment_oracle_20260621/phase0_segment_oracle_summary_zh.md`

## Phase1: Decoder / Target Interface

### 历史路线结论

[Fact] Phase1-A.1 到 A.6 已完成多轮候选 gate：

| Candidate | Main Outcome | Decision |
| --- | --- | --- |
| `PatchEncoderSegmentQueryHead` | `0/12` wins, mean MSE `+6.79%` | fail |
| `PatchEncoderFixedHeadAdapter` | `7/12` wins, mean MSE `+0.20%` | partial |
| `PatchEncoderFutureAwareAdapter` | `4/12` wins, mean MSE `+0.16%` | partial |
| `PatchEncoderFutureAwareAlignOnly` | `4/12` wins, mean MSE `+0.04%` | repair partial |
| `PatchEncoderStepSpecificStateAdapter` | `7/12` wins, mean MSE `+0.39%` | partial |
| `PatchEncoderTrajectoryBasisResidual` | `5/12` wins, mean MSE `+0.67%` | partial |

[Strong Evidence] 这些结果共同说明：

1. 直接替换 fixed flatten head 容易造成 readout capacity collapse。
2. 保留 fixed head 后做 output-space 或 pre-head lightweight modulation 可以产生局部信号，
   但不能稳定超过 specialist fixed head。
3. future-aware alignment 可以无泄漏运行并形成 teacher/student coupling，但当前承载结构
   太弱，无法转化为稳定 forecast gain。
4. output-space low-rank trajectory residual 非零，但幅度过小，不足以构成 paper-core。

[Decision] A.1-A.6 后不进入旧 Phase1-B，不在 A.5/A.6 上继续叠加 future-aware 或 MoE。
当前回退到 11-step loop 的 step 2-3：重新定义 decoder 创新点要解决的真实问题。

### Phase1-R: Target-Set Forecasting Decoder

状态：第一版 remote gate 已完成，接近 compatibility pass，但严格条件未通过。

核心文档：

- `docs/experiments/phase1-target-set-decoder-redefinition.md`
- `docs/code-explanation/phase1-target-set-decoder.md`

[Decision] 新的问题定义是：

> horizon-specific direct forecasting 把 target horizon 当作训练脚本级别的外部设置，
> 而不是模型输入的一部分；这导致每个 horizon 训练一套 specialist head，无法统一
> target positions、prefix consistency 和 future-step dependency。

当前 fixed-head specialist 可写为：

$$
Z=E_\theta(X),
\qquad
\hat{Y}_{1:H}^{(H)}=W_H\operatorname{Flatten}(Z).
$$

不同 $H$ 下，同一个 future step $\tau$ 可能由不同参数和不同训练轨迹产生：

$$
\hat{Y}_{\tau}^{(96)} \neq \hat{Y}_{\tau}^{(720)}.
$$

Phase1-R 将 horizon 外部设置改写为 target set 输入：

$$
T=\{\tau_1,\dots,\tau_m\},
$$

$$
Q_T=q_\phi(T),
$$

$$
\hat{Y}_T=D_\theta(E_\theta(X),Q_T,M_T).
$$

其中 $M_T$ 是 target-query interaction policy。

### 第一版设计边界

[Decision] 第一版 `PatchEncoderTargetSetDecoder` 不复刻完整 ElasTST，不引入 future
covariates，不引入 MoE，不引入 future teacher branch。

[Decision] 第一版应优先验证 target-set interface，而不是追求一次性统一所有机制：

- datasets: `ETTh2`, `ETTm1`, `Weather`
- target horizons: `{96,192,336,720}`
- seq_len: `336`
- primary seed: `2021`
- base encoder: 与 `PatchEncoderFixedHead` 保持一致
- training: mixed target-set sampling
- evaluation: 分别报告 `{96,192,336,720}` 的 MSE/MAE、segment metrics、prefix consistency

[Inference] 第一版实现需要避免 A.1 的 capacity collapse。Target query 不能只经过一个
过窄 shared patch MLP。更合理的最小结构是保留 dense history readout capacity，同时让
target segment state 对 readout 进行 conditioning。

候选张量流：

$$
X \in \mathbb{R}^{B \times L \times C},
$$

$$
Z = E_\theta(X) \in \mathbb{R}^{BC \times N \times d},
$$

$$
Q_T = \operatorname{TargetEmbed}(T) \in \mathbb{R}^{BC \times J \times d},
$$

$$
U_T = \operatorname{TargetDecoder}(Q_T,Z),
$$

$$
r = R_\theta(\operatorname{Flatten}(Z)) \in \mathbb{R}^{BC \times d_r},
$$

$$
(\gamma_j,\beta_j)=F_\theta(U_j),
$$

$$
\hat{Y}_{a_j:b_j}
=
O_\theta(r\odot(1+\gamma_j)+\beta_j).
$$

[Hypothesis] 该结构比 A.1 更合理，因为每个 target segment 仍能访问 flattened history
readout，而不是只依赖一个低容量 segment token；同时它又不同于 horizon-specific fixed head，
因为 target segment 通过 $Q_T$ 和 $U_T$ 显式进入预测过程。

### Mask 与 consistency policy

[Decision] 第一版默认使用 `independent target queries` 或等价的 prefix-stable policy。
这样当 $T_s \subset T_l$ 时，同一个 target segment 的计算不会因为额外 future queries
加入而改变。

[Inference] 这比一开始使用复杂 target self-attention 更保守，但更符合当前目标：先验证
target-set interface 是否能降低 prefix mismatch，再考虑 target-query interaction 是否能带来
额外收益。

### Phase1-R Gate

Compatibility pass:

1. single mixed target-set model vs horizon-specific `PatchEncoderFixedHead` 的 mean relative
   MSE 不超过 `+1.0%`。
2. 任一 dataset 平均退化不超过 `+3.0%`。
3. 相比 `H=720 fixed head prefix`，h96/h192 prefix MSE 不退化，最好改善。
4. prefix consistency mismatch 相比 Phase0 fixed-head mismatch 明显下降。
5. target states 不完全同质，且参数量低于四个 specialist heads 总和。

Paper-core pass:

1. mean relative MSE vs horizon-specific `PatchEncoderFixedHead` < 0；或
2. consistency 明显改善，并且后续 future-aware / MoE 能把 target-side state 转化为稳定
   forecast gain。

[Decision] 若第一版只达到 compatibility pass，它可以作为后续 future-aware/MoE carrier，
但不能立即作为论文核心。若连 compatibility pass 都达不到，应暂停 decoder 主线，回到
step 2-5 重新评估问题定义或理论可行性。

### Phase1-R Gate Result

[Fact] 第一版 `PatchEncoderTargetSetDecoder` remote gate 已完成：

- report: `analysis/phase1_target_set_decoder_gate_20260622/phase1_target_set_decoder_gate_report.md`
- remote output: `/home/yingch/exp_outputs/r-2026-fatst/phase1_target_set_decoder`
- code commit: `3f7ead2`
- selected GPUs: `1`, `2`

[Evidence] 结果：

| Metric | Value |
| --- | ---: |
| MSE wins vs `PatchEncoderFixedHead` | `5/12` |
| MAE wins vs `PatchEncoderFixedHead` | `5/12` |
| mean relative MSE | `+0.62%` |
| ETTh2 mean relative MSE | `-0.27%` |
| ETTm1 mean relative MSE | `+1.99%` |
| Weather mean relative MSE | `+0.13%` |
| max prefix mismatch MSE | `5.112619e-14` |
| mean target state cosine | `0.359800` |

[Evidence] H720-aligned prefix reference 的平均 h96/h192 relative MSE 为 `-0.85%`，但严格
逐项看 ETTh2 h96 为 `+0.19%`、ETTm1 h96 为 `+0.17%`，未满足“不退化”的 strict gate。

[Decision] 该候选是 `near_miss_not_compatibility_pass`：单模型 amortization gap 控制在
`+1.0%` 内，没有 dataset 平均退化超过 `+3.0%`，prefix consistency 在数值上接近精确成立，
target states 也没有同质化；但 H720-prefix strict no-degradation 条件未完全满足。

[Decision] 它不是 paper-core pass。mean relative MSE 仍为正，`ETTm1 / h96` 退化 `+5.63%`，
短 horizon 平均退化 `+2.74%`。下一步不能把 one-model flexibility 当作主贡献，也不能直接
进入 MoE；应回到 step 5-6 做 short-horizon / prefix reuse 的 targeted repair。

### Phase1-R.1: Prefix Residual Repair

[Decision] 下一步 targeted repair 是 `PatchEncoderTargetSetPrefixResidual`。它不改变
target-set interface，而是在前若干 target segments 上增加 zero-initialized dense residual
readout：

$$
\hat{y}_{a_j:b_j}
=
O_\theta(r\odot(1+\gamma_j)+\beta_j)
+
\Delta^{prefix}_{a_j:b_j},
\quad j<K.
$$

[Hypothesis] 第一版 near-miss 的主要问题不是 prefix consistency，而是 short-prefix capacity。
给 h96/h192 对应前 2-4 个 segments 一个共享 residual readout，可以修复 h96/ETTm1 的
amortization gap，同时保持同一 prefix segment 在不同 requested horizon 下输出一致。

[Gate] 第一轮只要求 targeted compatibility repair：

- H720-aligned h96/h192 strict no-degradation 必须通过；
- h96 mean relative MSE 必须明显低于第一版 `+2.74%`；
- overall mean relative MSE 不能超过第一版 `+0.62%`；
- prefix mismatch 仍应接近 0；
- 若 repair 只靠大 residual 破坏 target-set state 解释，则不能进入 future-aware/MoE。

[Fact] Local smoke 已通过，`prefix_residual_stats.csv` 正常写出，prefix mismatch 仍为数值零级别。

[Fact] Remote gate 已完成：

- report: `analysis/phase1_target_set_prefix_residual_gate_20260622/phase1_target_set_prefix_residual_gate_report.md`
- code commit: `bb164ff`
- selected GPUs: `1`, `2`

[Evidence] 结果：

| Metric | Value |
| --- | ---: |
| MSE wins vs `PatchEncoderFixedHead` | `1/12` |
| mean relative MSE | `+2.03%` |
| h96 mean relative MSE | `+4.52%` |
| H720-prefix h96/h192 mean relative MSE | `+1.20%` |
| max prefix mismatch MSE | `1.54947e-14` |
| mean target state cosine | `0.836250` |
| mean prefix residual MAE norm | `0.289765` |

[Decision] `PatchEncoderTargetSetPrefixResidual` 未通过。它保持了 prefix consistency，但
short-prefix MSE 进一步恶化，target states 更同质，prefix residual 幅度不小，说明该 path
更像 uncontrolled correction，而不是 capacity-preserving repair。

[Rollback] 不应继续调大 prefix residual，也不应在该 repair 上叠加 MoE。当前应回到 step 3-5：
重新判断 target-set decoder 的主要瓶颈是 architecture capacity、mixed-horizon optimization，
还是 target-query interaction policy。下一轮若继续 decoder 主线，应优先考虑 objective-level
或 interaction-level 设计，而不是 output residual patch。

### Phase1-R.2: Causal Target Interaction

状态：remote gate 已完成，未通过。

核心文档：

- `docs/experiments/phase1-causal-target-interaction-design.md`

[Problem] 第一版 target-set decoder 为了 prefix stability 采用 independent target queries。
这保证了 `H=96` 与 `H=720` prefix 在数值上相同，但也把 future target segments 当成
conditioned-on-history 后的独立输出，无法显式建模 future-step / segment dependency。

[Evidence] 该问题值得测试的原因是：

- `PatchEncoderTargetSetDecoder` 已经解决 prefix mismatch，但仍有 `+0.62%` mean relative MSE；
- `PatchEncoderTargetSetPrefixResidual` 增加 dense residual 后恶化到 `+2.03%`，说明简单补
  output capacity 不是正确方向；
- `QDF` 与 `SRP++` 的 notes 都支持 future steps 不是独立同质任务；
- `ElasTST` 的结构化 mask 与 `TimePerceiver` 的 target query 共同支持 target-side
  decoder interface。

[Idea] 在 target-to-history cross-attention 后加入 causal target self-attention：

$$
U_T=\operatorname{CrossAttn}(Q_T,Z,Z),
$$

$$
V_j=\operatorname{CausalTargetAttn}(U_j,U_{\le j}),
$$

$$
\hat{Y}_{a_j:b_j}
=
O_\theta(r\odot(1+\gamma(V_j))+\beta(V_j)).
$$

[Theory Check] 因为第 $j$ 个 target segment 只读取 $U_{\le j}$，所以 extended horizon 中的
later target queries 不会改写 earlier prefix：

$$
V_j(T_{1:J})=V_j(T_{1:K}),\quad j\le K\le J.
$$

这给出了比 independent queries 更强的 future process modeling，同时保留 prefix consistency。
它不是 rolling autoregression，也不依赖 future ground truth。

[Gate] `PatchEncoderCausalTargetInteraction` 的第一轮 gate：

- mean relative MSE 不差于第一版 target-set decoder 的 `+0.62%`；
- h96 mean relative MSE 低于 `+2.74%`；
- H720-prefix h96/h192 relative MSE 不差于 `-0.85%`，或 strict no-degradation count 改善；
- prefix mismatch 仍为数值零级别；
- mean target-state cosine 不向 prefix residual 的 `0.836250` collapse。

[Rollback] 若 causal target interaction 只保持 consistency 但不提升 MSE，则瓶颈更可能是
mixed-horizon objective / task weighting，而不是 architecture interaction；应回到 step 3-5
设计 objective-level candidate，不继续加深 target interaction 或启动 MoE。

[Fact] Local smoke 已通过：

- output root: `artifacts/runs/smoke_phase1_causal_target_interaction`
- dataset: `ETTh2`
- target horizons: `{96,192,336,720}`
- command uses `target_interaction_layers=1`
- artifact check: `metrics_by_target_horizon.csv`, `prefix_consistency.csv`,
  `target_state_similarity.csv`, and `effective_config.json` written

[Evidence] smoke prefix mismatch remains numerical zero-level:

| Prefix | Mismatch MSE |
| --- | ---: |
| `96/720` | `4.62182e-16` |
| `192/720` | `7.53289e-16` |
| `336/720` | `1.44997e-15` |

[Fact] Remote gate 已完成：

- report: `analysis/phase1_causal_target_interaction_gate_20260622/phase1_causal_target_interaction_gate_report.md`
- code commit: `830ffc6`
- selected GPU: `2`
- note: GPU1 在 Weather 阶段已空闲，但当前顺序 runner 已启动 Weather，未中途拆分以避免同一路径重复写入。

[Evidence] 结果：

| Metric | Value |
| --- | ---: |
| MSE wins vs `PatchEncoderFixedHead` | `4/12` |
| MAE wins vs `PatchEncoderFixedHead` | `5/12` |
| mean relative MSE | `+1.40%` |
| h96 mean relative MSE | `+3.62%` |
| H720-prefix h96/h192 mean relative MSE | `-0.14%` |
| max prefix mismatch MSE | `4.52311e-14` |
| mean target state cosine | `0.424291` |

[Evidence] Dataset-level split shows why it fails:

| Dataset | Mean relative MSE |
| --- | ---: |
| `ETTh2` | `-0.03%` |
| `ETTm1` | `-0.45%` |
| `Weather` | `+4.68%` |

[Decision] `PatchEncoderCausalTargetInteraction` 未通过。它保留了 prefix consistency，也让
`ETTm1` 从第一版 target-set 的 `+1.99%` 变为 `-0.45%`，说明 target interaction 不是完全
无效；但 Weather 全 horizon 系统性退化，整体 mean relative MSE 高于第一版 target-set 的
`+0.62%`，h96 也从 `+2.74%` 恶化到 `+3.62%`。

[Rollback] 当前不应继续加深 target interaction，也不应在该 state 上启动 future-aware 或 MoE。
更合理的回退点是 step 3-5：重新判断 mixed-horizon training objective 是否才是主要瓶颈。
下一轮候选应优先测试 objective-level / task-weighting path，例如 horizon-balanced 或
covariance-aware weighting，而不是继续改 architecture。

### Phase1-R.3: Prefix-Risk Weighted Objective

状态：remote gate 已完成，达到 compatibility pass，但不是 paper-core pass。

核心文档：

- `docs/experiments/phase1-prefix-risk-weighted-objective-design.md`

[Problem] 第一版 target-set decoder 已经做到 prefix consistency，但 h96 与部分 H720-prefix
strict no-degradation 仍失败。R.1 output residual 和 R.2 causal target interaction 都没有稳定
改善，说明继续加 readout capacity 或 target interaction 不是当前最合理路径。

[Hypothesis] 当前瓶颈可能来自 mixed-horizon objective。普通 MSE 没有显式表达早期 future
prefix 是所有 requested horizons 共享、也是 consistency-sensitive 的风险区域。给 early
future steps 更高优化权重，可能修复 h96 和 prefix reuse，而不改模型结构。

[Idea] 使用 prefix-risk weighted MSE：

$$
w_t =
\frac{(t / H_{max})^{-\alpha}}
{
\frac{1}{H_{max}}\sum_{s=1}^{H_{max}}(s / H_{max})^{-\alpha}
},
$$

$$
\mathcal{L}_{prefix}
=
\frac{1}{BHC}
\sum_{b,t,c}w_t(\hat{y}_{b,t,c}-y_{b,t,c})^2.
$$

[Design] 第一轮候选 `PatchEncoderPrefixRiskWeighted` 保持第一版
`PatchEncoderTargetSetDecoder` architecture：

- `target_interaction_layers=0`
- `prefix_residual_segments=0`
- `step_loss_weighting=prefix_risk`
- `step_loss_alpha=0.5`

[Gate] 第一轮 pass 条件：

- mean relative MSE 不差于第一版 target-set 的 `+0.62%`；
- h96 mean relative MSE 低于 `+2.74%`；
- Weather mean relative MSE 不比第一版 target-set 的 `+0.13%` 多退化超过 `+1.0%`；
- H720-prefix h96/h192 不差于 `-0.85%`；
- prefix mismatch 仍为数值零级别。

[Rollback] 若只改善 h96 但牺牲 Weather 或 long horizons，则它只能说明存在 risk tradeoff，
不能作为 paper-core。下一步应转向更原则化的 covariance-aware objective，或重新考虑 base
architecture，而不是继续手工调权重。

[Fact] Local smoke 已通过：

- output root: `artifacts/runs/smoke_phase1_prefix_risk_weighted`
- dataset: `ETTh2`
- target horizons: `{96,192,336,720}`
- command uses `step_loss_weighting=prefix_risk`, `step_loss_alpha=0.5`
- artifact check: `metrics_by_target_horizon.csv`, `prefix_consistency.csv`,
  `target_state_similarity.csv`, and `effective_config.json` written

[Evidence] smoke prefix mismatch remains numerical zero-level:

| Prefix | Mismatch MSE |
| --- | ---: |
| `96/720` | `0.0` |
| `192/720` | `0.0` |
| `336/720` | `4.45529e-16` |

[Fact] Remote gate 已完成：

- report: `analysis/phase1_prefix_risk_weighted_gate_20260622/phase1_prefix_risk_weighted_gate_report.md`
- code commit: `e5f6b9e`
- selected GPUs: `1`, `2`
- remote output: `/home/yingch/exp_outputs/r-2026-fatst/phase1_target_set_decoder/PatchEncoderPrefixRiskWeighted`

[Evidence] 结果：

| Metric | Value |
| --- | ---: |
| MSE wins vs `PatchEncoderFixedHead` | `8/12` |
| MAE wins vs `PatchEncoderFixedHead` | `8/12` |
| mean relative MSE | `-0.43%` |
| h96 mean relative MSE | `+0.87%` |
| H720-prefix h96/h192 mean relative MSE | `-2.46%` |
| max prefix mismatch MSE | `5.3671e-14` |
| mean target state cosine | `0.313276` |

[Evidence] Dataset-level split:

| Dataset | Mean relative MSE |
| --- | ---: |
| `ETTh2` | `-0.66%` |
| `ETTm1` | `+0.33%` |
| `Weather` | `-0.97%` |

[Evidence] Horizon-level split:

| Horizon | Mean relative MSE |
| --- | ---: |
| `96` | `+0.87%` |
| `192` | `-2.00%` |
| `336` | `-0.98%` |
| `720` | `+0.38%` |

[Decision] `PatchEncoderPrefixRiskWeighted` 达到 compatibility pass。它满足 R.3 预设 gate：
overall mean relative MSE 优于第一版 target-set 的 `+0.62%`，h96 mean relative MSE 从
`+2.74%` 降到 `+0.87%`，Weather 不再退化，H720-prefix h96/h192 从 `-0.85%` 改善到
`-2.46%`，prefix mismatch 保持数值零级别。

[Decision] 该结果说明 one-model target-set decoder 路线可以保留为后续 mechanism carrier。
但它不能单独作为论文核心创新，因为主要改动是 loss weighting，尚未给出充分的
forecasting-process architecture 贡献；同时 `ETTm1 / h96` 仍比 specialist fixed head 差
`+2.83%`，h720 在 `ETTh2` 和 `ETTm1` 仍略退化。

[Rollback / Continue] 当前不回退到 base architecture，也不继续手工调 `alpha`。下一轮进入
Phase2 的 step 1-6：研究 future-aware state 是否能解释并修复 remaining horizon/segment
error，使用 R.3 作为 compatibility carrier；若 Phase2 无法把 latent alignment 转成 forecast
gain，再回退到 problem definition 或重新选择 base architecture。

## Phase2: Future-Aware Mechanism

状态：`PatchEncoderFutureStateAlignment` remote gate 已完成，未通过；回退到 step 3-5
诊断 future-state alignment conflict。

核心文档：

- `docs/experiments/phase2-future-state-alignment-design.md`

[Inference] future-aware 的合理形式不是泄漏 future，而是用 training-only future signal
约束一个推理时可由 history 和 target set 推断的 future latent state。

候选形式：

$$
S_T^{teacher}=T_\psi(Y_T,Q_T),
$$

$$
S_T^{student}=P_\theta(U_T),
$$

$$
\mathcal{L}
=
\mathcal{L}_{pred}
+
\lambda\mathcal{L}_{align}
(S_T^{student},\operatorname{stopgrad}(S_T^{teacher})).
$$

[Decision] Phase2 的默认接入点必须是 R.3 兼容 carrier 产生的 target-side state $U_T$，
而不是 A.5/A.6 的 fixed-head patch，也不是旧的 output adapter。future-aware 的目标不是
再修一次 prefix consistency，而是证明 training-only future signal 能产生一个推理时可由
history 与 target set 估计的 future latent state，并且这个 state 对 forecast error 有可定位贡献。

### Phase2-A: Future-State Alignment

[Problem] `PatchEncoderPrefixRiskWeighted` 的 target state $U_T$ 是由 history state $Z$ 与
target query $Q_T$ 推断得到：

$$
U_j=D_\theta(Q_j,Z).
$$

它已经能作为 one-model carrier，但并没有显式校准到真实 future segment 的 latent
distribution。旧 Phase1 future-aware adapter 失败的原因不是 future-aware signal 完全无效，
而是它作用在没有 target-set carrier 的 fixed-head patch 上，无法进入真正的 target-side
forecasting process。

[Evidence] R.3 的剩余误差主要集中在 `ETTm1/h96` `+2.83%`、`ETTm1/h720` `+1.09%`、
`ETTh2/h720` `+0.75%` 和 `Weather/h96` `+0.64%`。同时 target-state geometry 没有 collapse：
`ETTh2/h720` mean target-state cosine 为 `+0.918`，而 `ETTm1/h96` 为 `-0.559`，
`Weather/h96` 为 `-0.108`。这说明 $U_T$ 有数据集/segment 信号，但还缺少 future-side
distribution anchor。

[Idea] 第一版候选 `PatchEncoderFutureStateAlignment` 使用 training-only future teacher
编码 ground-truth future segment：

$$
S_j^Y=T_\psi(Y_{a_j:b_j},q_j),
$$

并从 inference-time target state 得到 student state：

$$
S_j^X=P_\theta(U_j).
$$

训练时加入 local alignment 与 relation alignment：

$$
\mathcal{L}
=
\mathcal{L}_{prefix\_risk}
+
\lambda_{local}
\sum_j
(1-\cos(S_j^X,\operatorname{sg}(S_j^Y)))
+
\lambda_{rel}
\left\|
\operatorname{sim}(S^X)-
\operatorname{sg}(\operatorname{sim}(S^Y))
\right\|_F^2.
$$

推理时仍只使用：

$$
\hat{Y}_{a_j:b_j}
=
O_\theta(r\odot(1+\gamma(U_j))+\beta(U_j)).
$$

[Theory Check] 该设计与 `TimeAlign` 的 future-side teacher 一致，但 alignment 目标从全局
history representation 改成 target-set decoder state $U_T$；与 `TimePerceiver` 的 target query
接口一致；与 `SRP++` 的 step/segment-specific representation claim 一致；并延续 R.3/QDF
暴露出的 future-step risk/objective 问题。

[Gate] `PatchEncoderFutureStateAlignment` 必须先保持 R.3 carrier 不被破坏：

- mean relative MSE vs `PatchEncoderFixedHead` 仍不差于 R.3 的 `-0.43%`；
- 任一 dataset 平均不比 R.3 差超过 `+0.3%`；
- prefix mismatch 保持数值零级别；
- prediction leakage max abs <= `1e-7`；
- teacher/student alignment 统计必须写出并随训练改善。

Paper-core candidate pass 需要进一步满足：

- vs R.3 至少 `7/12` dataset-horizon settings 获胜，或 mean relative MSE vs R.3 改善
  至少 `0.5%`；
- R.3 的弱项中至少两个改善：`ETTm1/h96`, `ETTm1/h720`, `ETTh2/h720`, `Weather/h96`；
- 改善能定位到 horizon、segment、frequency 或 target-state alignment diagnostics。

[Rollback] 若 alignment metric 改善但 MSE/MAE 不改善，说明 future teacher 只是 auxiliary
proxy，不是有效 paper-core mechanism；应回到 step 2-5，考虑 covariance-aware objective 或
更换 base architecture。若 Weather 系统性退化，先诊断 alignment conflict，不直接进入 MoE。

[Fact] Local smoke 已通过：

- output root: `artifacts/runs/smoke_phase2_future_state_alignment`
- dataset: `ETTh2`
- target horizons: `{96,192,336,720}`
- command uses `step_loss_weighting=prefix_risk`, `future_teacher_layers=1`,
  `future_align_weight=0.02`, `future_relation_weight=0.01`,
  `future_recon_weight=0.001`
- artifact check: `metrics_by_target_horizon.csv`, `prefix_consistency.csv`,
  `target_state_similarity.csv`, `future_alignment_stats.csv`,
  `future_leakage_audit.json`, and `effective_config.json` written

[Evidence] smoke prefix mismatch remains numerical zero-level:

| Prefix | Mismatch MSE |
| --- | ---: |
| `96/720` | `0.0` |
| `192/720` | `0.0` |
| `336/720` | `4.74343e-16` |

[Evidence] smoke leakage audit passes for all evaluated horizons:

| Horizon | prediction leakage max abs |
| --- | ---: |
| `96` | `0.0` |
| `192` | `0.0` |
| `336` | `0.0` |
| `720` | `0.0` |

[Fact] Remote gate 已完成：

- report: `analysis/phase2_future_state_alignment_gate_20260622/phase2_future_state_alignment_decision_report.md`
- code commit: `2f3dc42`
- selected GPUs: `1`, `2`
- remote output: `/home/yingch/exp_outputs/r-2026-fatst/phase2_future_state_alignment/PatchEncoderFutureStateAlignment`

[Evidence] vs horizon-specific `PatchEncoderFixedHead`:

| Metric | Value |
| --- | ---: |
| MSE wins | `6/12` |
| MAE wins | `6/12` |
| mean relative MSE | `+0.84%` |
| ETTh2 mean relative MSE | `+4.54%` |
| ETTm1 mean relative MSE | `-0.96%` |
| Weather mean relative MSE | `-1.04%` |

[Evidence] vs R.3 carrier `PatchEncoderPrefixRiskWeighted`:

| Metric | Value |
| --- | ---: |
| MSE wins | `7/12` |
| MAE wins | `7/12` |
| mean relative MSE | `+1.29%` |
| ETTh2 mean relative MSE | `+5.25%` |
| ETTm1 mean relative MSE | `-1.29%` |
| Weather mean relative MSE | `-0.07%` |

[Evidence] Mechanism diagnostics:

| Diagnostic | Value |
| --- | ---: |
| max prefix mismatch MSE | `4.72701e-14` |
| max prediction leakage abs | `0.0` |
| mean teacher/student cosine | `0.762778` |
| mean local alignment loss | `0.237222` |
| mean relation alignment loss | `0.074249` |

[Decision] `PatchEncoderFutureStateAlignment` 未通过 Phase2 gate。它不是 leakage failure，也
不是 prefix consistency failure；它对 `ETTm1` 全 horizon 有稳定改善，对 `Weather` 三个长
horizon 有轻微改善，但 `ETTh2` 全 horizon 明显退化，导致 R.3 carrier 被破坏。

[Rollback] 当前不应在该 state 上继续叠加 MoE，也不应简单增大 teacher capacity。回退点是
step 3-5：诊断 alignment conflict。下一轮候选应先回答 uniform future alignment 为什么与
`ETTh2` dynamics 冲突，可能方向包括 scale-normalized teacher reconstruction、scheduled/gated
alignment、uncertainty-weighted alignment，或 horizon/dataset conflict diagnostics。

Phase2 pass 条件：

- prediction path leakage audit 通过；
- alignment distance 下降并转化为 forecast-relevant improvement；
- 改善能定位到 horizon、segment、turning point、high-frequency component 或 long-horizon error；
- 若 latent metric 改善但 MSE/MAE 不变，则该 future-aware state 只能视为无效 proxy。

## Phase3: Future-Side MoE

状态：暂停，等待 Phase1-R/Phase2 产生稳定 target-side state。

[Inference] MoE 的必要性不能只由“不同样本需要不同专家”支撑。本项目更强的论证应是：
target-set decoder 已经把预测过程分解为 future target/segment states，这些 states 可能对应
不同 future mechanisms，因此 conditional operators 应放在 future side。

候选数据流：

$$
r_j=\operatorname{Router}_\theta(U_j,S_j,q_j),
$$

$$
\tilde{U}_j=\sum_{k=1}^{K}r_{j,k}E_k(U_j),
$$

$$
\hat{Y}_{a_j:b_j}=O_\theta(\tilde{U}_j).
$$

Phase3 pass 条件：

- 改善不能只来自参数量，必须对比 same-parameter dense control；
- routing entropy 不能塌缩，也不能完全平均；
- routing pattern 应与 future segment、future-aware state cluster 或 error profile 对齐；
- fixed routing、random routing、no future-state routing 必须作为 ablations。

## Phase4: Unified Model And Paper Claim

[Decision] 最终模型只合并通过 gate 的机制：

- 若 Phase1-R + Phase2 + Phase3 都通过：形成 unified target-set future-aware MoE forecasting framework。
- 若 Phase1-R + Phase2 通过但 MoE 不通过：论文主张收敛为 target-set decoder with future-state alignment。
- 若 Phase1-R + Phase3 通过但 Phase2 不通过：论文主张收敛为 future-side conditional operators。
- 若 Phase1-R 不通过：项目不应继续围绕 decoder 讲故事，应回到 problem definition 或 external
  baseline reproduction。

## 当前禁止冒进项

- 不把 12 篇文献的模块直接堆进一个模型。
- 不把 ElasTST 式 variable-horizon 当作默认主线。
- 不把 one-model flexibility 本身包装成性能贡献。
- 不把 rolling autoregression 作为主创新；它只能是 baseline 或 diagnostic。
- 不在没有 stable target-side state 的情况下启动 MoE。
- 不默认 future covariates 可用。
- 不直接迁移旧仓库结果作为当前证据。
- 不只报告平均 MSE/MAE；本路线必须包含 horizon-level、segment-level、consistency 和
  state/routing diagnostics。

## Baseline 边界

- SRSNet 是重点 comparison baseline。
- 旧 `R_2026_FSA` 中的 SRSNet 性能数据只能在用户批准后作为证据迁入或引用。
- 新仓库内的 external baseline 复现应优先保留 native upstream evidence，再决定是否写
  local wrapper。
