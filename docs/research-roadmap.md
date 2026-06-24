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

[Fact] Phase2-A conflict diagnosis 已完成：

- report:
  `analysis/phase2_alignment_conflict_diagnosis_20260623/phase2_alignment_conflict_diagnosis_report.md`
- script: `scripts/analyze_phase2_alignment_conflict.py`

[Evidence] failure pattern 更符合 target-state geometry conflict，而不是单纯 reconstruction
scale imbalance：

| Diagnostic | Value |
| --- | ---: |
| ETTh2 mean teacher/student cosine | `0.6363` |
| ETTm1 mean teacher/student cosine | `0.8300` |
| Weather mean teacher/student cosine | `0.8220` |
| MSE delta vs teacher/student cosine Pearson r | `-0.8866` |
| MSE delta vs local alignment loss Pearson r | `+0.8866` |
| MSE delta vs reconstruction loss Pearson r | `-0.2911` |

[Inference] `Weather` 的 reconstruction loss 极大，但只在 h96 轻微退化，其他 horizon
改善；因此 raw reconstruction scale 不是 Phase2-A 失败的充分解释。`ETTh2` 的主要问题是
teacher state 与 student target state 的 geometry 本来就不兼容，uniform alignment 把
$U_T$ 强行拉向一个对 forecasting 不利的 anchor。

### Phase2-R.1: Confidence-Weighted Future Alignment

状态：实现中。该候选不是扩大 teacher capacity，而是修补 Phase2-A 暴露的 alignment
conflict。

[Problem] Phase2-A 证明 training-only future teacher 可以安全接入 decoder state，但
uniform alignment 把所有 future segment 都当作同等可靠的 teacher anchor。若某个 segment
的 future reconstruction error 高，说明 teacher state 对该 segment 的 future semantics
解释能力弱；继续强制 student target state 对齐这个 anchor，可能把 $U_T$ 拉离对 forecasting
有用的方向。

[Idea] 用 teacher reconstruction error 估计 segment-level confidence：

$$
e_j=
\frac{
\operatorname{MSE}(\tilde{Y}^{norm}_j,Y^{norm}_j)
}{
\operatorname{mean}((Y^{norm}_j)^2)+\epsilon
},
\qquad
c_j=\max(c_{min},\exp(-e_j/\tau)).
$$

local alignment 和 relation alignment 只在 teacher 可置信时施加强约束：

$$
\mathcal{L}_{local}^{conf}
=
\frac{\sum_j c_j(1-\cos(S^X_j,\operatorname{sg}(S^Y_j)))}
{\sum_j c_j+\epsilon}.
$$

relation alignment 使用 pair weight $\sqrt{c_i c_j}$。teacher reconstruction loss 可通过
`target_energy` 归一化，避免 auxiliary loss 在不同 dataset 上不可比。

[Theory Check] 该设计与 Phase2-A 的核心边界一致：`future_y` 仍然不进入 prediction path；
confidence 只改变 training-time auxiliary gradient。若它能修复 `ETTh2` 退化并保留
`ETTm1/Weather` 收益，则说明 future-aware supervision 的关键不是“更多 future task”，而是
“只对可靠 future state 做 decoder-state calibration”。若仍失败，则 future teacher 方向很可能
只是 auxiliary proxy，不适合作为论文主机制。

[Gate] 使用 `PatchEncoderFutureStateAlignmentConfWeighted`：

- `future_recon_normalization=target_energy`;
- `future_align_weighting=reconstruction_confidence`;
- `future_confidence_floor=0.05`;
- 其他结构保持 Phase2-A 不变。

通过条件：

- prediction leakage audit 仍为 `<= 1e-7`；
- prefix mismatch 仍为数值零级别；
- vs R.3 的 `ETTh2` 平均退化收敛到 `<= +0.3%`；
- 同时不能牺牲 `ETTm1/Weather` 的已有正信号；
- 若只改善 latent metric 而不改善 MSE/MAE，则回退到 step 2-3，重新定义 decoder 问题。

Phase2-R.1 结果决策树：

1. **通过**：`ETTh2` 冲突被修复，且 `ETTm1/Weather` 正信号保留。
   - [Continue] 保留 future-aware 主线，但论文叙事必须收敛为
     “reliability-aware future-state calibration”，而不是泛泛的 future teacher。
   - [Next] 先做 ablation：`uniform` vs `confidence`、`none` vs `target_energy`、
     `confidence_floor` sensitivity，再决定是否进入 Phase3 MoE。
2. **部分通过**：只修复 `ETTh2`，但牺牲 `ETTm1/Weather`；或只保留
   `ETTm1/Weather`，但 `ETTh2` 仍明显退化。
   - [Rollback] 回到 step 3-5。该结果说明 future signal 存在 dataset-dependent
     conflict，不能作为 universal decoder state calibration。
   - [Next] 只允许做一个诊断实验：按 dataset/horizon 的 confidence、teacher recon
     error、target-state cosine 和 MSE delta 建立 conflict map；不得直接加 MoE。
3. **失败**：leakage/prefix 通过，但 MSE/MAE 没有稳定收益，或只改善 latent diagnostics。
   - [Rollback] 回到 step 2-3：重新定义 decoder 策略问题。当前 future teacher
     只能作为 auxiliary proxy，不是 paper-core。
   - [Pivot Candidate] 转向 output-process / error-process decoder，而不是继续对齐
     latent future state。更具体地，研究问题应从“预测 future latent state”改为
     “decoder 如何建模 horizon-wise error growth / covariance / residual process”。

[Fact] Output/error-process decoder problem diagnosis 已完成：

- report:
  `analysis/output_error_process_diagnosis_20260623/output_error_process_diagnosis_report.md`
- figures:
  `analysis/output_error_process_diagnosis_20260623/*_h720_step_relative_mse.png`
- script: `scripts/analyze_output_error_process_problem.py`

[Evidence] H720 step-wise 误差曲线显示，当前问题不是单个平均 MSE 可以解释的：

| Dataset | Segment | R.3 vs FixedHead | Phase2-A vs R.3 | Interpretation |
| --- | --- | ---: | ---: | --- |
| ETTh2 | `1-96` | `-1.80%` | `+6.42%` | R.3 early gain 被 Phase2-A 抹掉 |
| ETTh2 | `193-336` | `+4.07%` | `+2.54%` | 两阶段都恶化中段 |
| ETTm1 | `337-720` | `+3.03%` | `-1.73%` | R.3 late weak region 被 Phase2-A 修复 |
| Weather | `1-96` | `-4.04%` | `+0.52%` | R.3 early gain 被 Phase2-A 轻微抹掉 |

[Inference] 若 Phase2-R.1 失败，下一步不应继续增强 future teacher 或引入 MoE。更合理的
decoder 问题是：one-model decoder 如何建模 forecast output 的非均匀 error growth、
step-region covariance 和 residual process。候选机制应把 prediction trajectory / residual
trajectory 作为被解码对象，而不是只独立输出每个 segment 的 point values。

[Design] Phase2-B fallback design 已定义：

- doc: `docs/experiments/phase2-output-error-process-decoder-design.md`
- candidate family: `Target-Conditioned Error-Process Decoder`
- tentative implementation name: `PatchEncoderErrorProcessDecoder`

核心区别：

- 不重复 Phase1-A.6 的 static low-rank position basis；
- residual process 由 target-set state $U_j$、target segment feature $q_j$ 和 compact
  error-process state $c_{j-1}$ 共同生成；
- 目标是建模 forecast output/error process，而不是继续对齐 latent future teacher。

[Decision Update: 2026-06-23] Phase2-R.1 remote gate 已同步并分析，结论为 fail：

- vs R.3 MSE wins: `7/12`，但 mean relative MSE 为 `+1.28%`；
- `ETTh2` mean relative MSE 为 `+5.08%`，未修复核心冲突；
- `ETTm1` 保持正信号 `-1.28%`，但 `Weather` 变为 `+0.04%`；
- leakage 为 `0`，prefix mismatch 为 `4.7318994e-14`。

[Decision] future-state alignment 当前只能作为 auxiliary proxy 证据，不能作为 paper-core
decoder 创新，也不能在其上继续叠 MoE。按照 step 11 rollback rule，当前回退到 step 2-3：
将 decoder 问题从“对齐 latent future state”改为“建模 forecast output/error process”。
Phase2-B `PatchEncoderErrorProcessDecoder` 已完成本地 smoke、artifact validator 和
`ETTh2/ETTm1/Weather` remote gate。

当前 Phase2-B artifacts：

- model class: `PatchEncoderErrorProcessDecoder`
- trainer switch: `--model-variant error_process`
- code explanation: `docs/code-explanation/phase2-error-process-decoder.md`
- remote wrapper: `scripts/remote/run_phase2_error_process_decoder_gate.sh`

Phase2-B remote gate 结果：

- vs R.3 MSE wins: `4/12`，MAE wins: `4/12`；
- mean relative MSE vs R.3: `+1.12%`；
- `ETTh2`: `+4.15%`，`ETTm1`: `-1.24%`，`Weather`: `+0.44%`；
- focus H720 regions wins: `1/4`；
- prefix mismatch 通过，base + residual decomposition 通过；
- mean residual/base MAE ratio 为 `0.00305975`，说明 residual 受控但贡献很弱。

[Decision Update: 2026-06-23] `PatchEncoderErrorProcessDecoder` 未通过 Phase2-B gate。
该失败不是实现安全性问题：prefix consistency、decomposition 和 residual magnitude 均受控；
失败来自 forecast improvement 没有转化，且 ETTh2/Weather 仍退化。当前不能把
error-process state 作为 paper-core，也不能在该 residual state 上叠加 MoE。

[Rollback] 当前回到 11-step loop 的 step 2-3：重新评估 decoder 问题是否主要来自
mixed-horizon objective / step-region covariance，而不是继续强化 target interaction、
future-state alignment 或 residual capacity。下一步候选应优先验证 objective-level 机制
是否真实存在，例如 step covariance weighting 或 horizon-region loss shaping；若该问题也
不存在，应停止 decoder 主线并回到 base architecture / paper contribution 重新选择。

### Phase2-C: Objective Pressure / Step-Covariance

状态：objective-level diagnostic 已完成；进入 step 4-6 设计阶段。

核心文档：

- diagnostic report:
  `analysis/phase2_objective_pressure_diagnostic_20260623/phase2_objective_pressure_diagnostic_report.md`
- design:
  `docs/experiments/phase2-step-covariance-objective-design.md`
- script:
  `scripts/analyze_phase2_objective_pressure.py`

[Problem] Phase2-A/R/B 连续证明，继续强化 latent future-state alignment 或 residual state
不能稳定转化为 forecast gain。更底层的问题可能是 mixed-horizon one-model training
本身的 objective pressure：不同 horizon 共享 early prefix，但 middle/late regions 仍有
独立 dynamics；普通 uniform MSE 与单调 prefix-risk 都没有显式建模 step-region covariance。

[Existence Evidence] Phase2-C diagnostic 对比了 R.3 `PatchEncoderPrefixRiskWeighted`
和 uniform `PatchEncoderTargetSetDecoder`：

- R.3 vs uniform MSE wins: `11/12`；
- mean relative MSE vs uniform: `-1.03%`；
- mean h96 relative MSE vs uniform: `-1.81%`；
- H720-prefix h96/h192 mean relative MSE vs uniform: `-1.62%`；
- segment-level wins vs uniform: `24/30`；
- horizon loss multiplier 与 R.3 main-horizon delta 的 Pearson r 为 `-0.5530`；
- segment pressure-share delta 与 R.3 segment delta 的 Pearson r 为 `-0.6804`。

[Strong Evidence] objective problem 真实存在：R.3 在不改 architecture 的情况下明显优于
uniform target-set。但 naive prefix-risk 不是 paper-core，因为它把 normalized pressure share
过度集中到 `1-96`：

| Region | Uniform Share | Prefix-Risk Share | Share Delta |
| --- | ---: | ---: | ---: |
| `1-96` | `0.4798` | `0.7217` | `+50.43%` |
| `97-192` | `0.2298` | `0.1540` | `-32.98%` |
| `193-336` | `0.1571` | `0.0775` | `-50.71%` |
| `337-720` | `0.1333` | `0.0469` | `-64.85%` |

[Idea] 下一候选是 `Step-Covariance Balanced Objective`。它不改变 inference path，而是把
forecast target 视为 region-structured trajectory，对 early prefix、middle transition 和
long-tail regions 施加 region-normalized loss，并用 coverage balance 与 target-derived
covariance/novelty prior 决定 region weights。

候选目标形式：

$$
\mathcal{L}_{scb}
=
\sum_{r\in\mathcal{R}_H}
\lambda_r
\frac{1}{|r\cap[1,H]|C}
\sum_{t\in r\cap[1,H]}\sum_{c=1}^{C}
(\hat{y}_{t,c}-y_{t,c})^2.
$$

其中：

$$
\lambda_r
\propto
\left(p_r^{uniform}\right)^{-\beta}
\left(u_r+\epsilon\right)^{\eta}.
$$

[Theory Check] 该方向与 QDF-style future-step dependency 的观点一致，但当前实现目标更窄：
不是直接做完整未来协方差建模，而是先验证 one-model target-set decoder 是否需要
region/covariance-aware objective。若它能超过 R.3，论文叙事可以从“调 loss”上升到
“forecasting process has step-region covariance and needs objective-side calibration”。
若它不能超过 R.3，则 objective 主线也应停止，转向 base architecture 或外部 baseline
reproduction。

[Gate] Phase2-C candidate 必须以 R.3 为主要比较对象：

- mean relative MSE vs R.3 < `0`；
- vs R.3 wins 至少 `7/12`；
- 任一 dataset mean 不比 R.3 退化超过 `+0.3%`；
- 至少两个 fixed-specialist gap 改善：
  `ETTm1/h96`, `ETTm1/h720`, `ETTh2/h720`, `Weather/h96`；
- H720 middle/late regions 不退化，同时 h96 保持 competitive；
- prefix mismatch 保持数值零级别。

[Decision] 当前不进入 Phase3 MoE。下一步允许实现 Phase2-C 的 objective-level candidate，
并先做 `region_balanced` smoke，再决定是否加入 target covariance / novelty prior。

[Decision Update: 2026-06-23] `region_balanced` objective 已实现并通过本地 smoke：

- code: `baselines/patch_encoder_target_set_decoder/train.py`
- mode: `--step-loss-weighting region_balanced`
- smoke output:
  `artifacts/runs/smoke_phase2_region_balanced/PatchEncoderRegionBalanced/ETTh2/mixed_h96_h192_h336_h720/seed2021`
- artifact check: `metrics_by_target_horizon.csv`, `prefix_consistency.csv`,
  `objective_weight_stats.csv`, `effective_config.json`
- prefix mismatch MSE: `8.284057610079363e-15`, `8.386538432301766e-15`,
  `3.621962433886803e-15`
- objective-weight audit: four regions are rebalanced to weighted pressure share `0.25`

[Next] 按项目规则，远程 gate 前必须先 commit/push，然后在 `529_Lab-3090` 上 `git pull`，
检查 GPU 显存，再用 `scripts/remote/run_phase2_region_balanced_gate.sh` 跑完整
`ETTh2/ETTm1/Weather` matrix。gate 主要比较对象是 R.3，而不是只比较 FixedHead。

[Contingency] 若 `region_balanced` 不是 clear pass，但仍保留 R.3 compatibility carrier，
下一步允许回到 step 4-6 细化 `step_covariance_balanced` objective，而不是立即放弃
objective route。预案文档：
`docs/experiments/phase2-covariance-objective-contingency.md`。若 `region_balanced`
明确破坏 compatibility，则 objective route 应停止，回到 step 2-3 重新选择 base
architecture 或 external baseline problem。

[Decision Update: 2026-06-23] `PatchEncoderRegionBalanced` remote gate 已同步并分析，
结论为 fail：

- report:
  `analysis/phase2_region_balanced_gate_20260623/phase2_region_balanced_decision_report.md`
- MSE wins vs R.3: `2/12`；
- mean relative MSE vs R.3: `+1.53%`；
- dataset mean relative MSE vs R.3:
  `ETTh2 -0.29%`, `ETTm1 +3.19%`, `Weather +1.70%`；
- MSE wins vs uniform target-set: `4/12`；
- mean relative MSE vs uniform target-set: `+0.47%`；
- prefix mismatch max MSE: `5.2042527595328944e-14`。

[Inference] 单纯 equal-region coverage balance 不是有效 paper-core，也不是 R.3 的可靠修补。
它把 early prefix pressure 从 uniform 的 `0.4798` 降到 `0.25`，保留了 prefix consistency，
但明显损害 `ETTm1/h96`, `Weather/h96` 和多数 horizon。这说明 R.3 的收益不能被简化为
“让四个 regions 等权”。更可能的结论是：early prefix 确实需要更高 pressure，但 middle/late
regions 的补偿必须由 source-grounded dependency/novelty evidence 决定，而不是手工等权。

[Rollback] 当前回到 11-step loop 的 step 2-3。不要继续手调 region multipliers，也不要直接
在失败的 `region_balanced` 上叠 MoE。若保留 objective 主线，下一步必须先做离线
covariance/novelty diagnostic：用 training targets 估计 region novelty，并检验它是否能解释
R.3 与 `region_balanced` 的 segment-level gain/loss pattern。只有该诊断成立，才允许进入
`step_covariance_balanced` 的 step 4-6；若诊断不成立，应停止 objective-only 主线，转向
base architecture 或 external baseline problem。

[Decision Update: 2026-06-23] Phase2-C.1 offline covariance/novelty diagnostic
已完成：

- report:
  `analysis/phase2_covariance_novelty_diagnostic_20260623/phase2_covariance_novelty_diagnostic_report.md`
- script:
  `scripts/analyze_phase2_covariance_novelty.py`
- code explanation:
  `docs/code-explanation/phase2-covariance-novelty-diagnostic.md`

核心结果：

- R.3 segment delta vs novelty share Pearson: `-0.7219`；
- R.3 segment delta vs prefix pressure share Pearson: `-0.6909`；
- `region_balanced` delta vs novelty deficit Pearson: `+0.6253`；
- aggregate R.3 delta vs novelty share Pearson: `-0.6714`；
- aggregate `region_balanced` delta vs novelty deficit Pearson: `+0.6253`。

[Inference] novelty 解释力支持一个更窄的 objective-side continuation：R.3 的 early
prefix emphasis 并非随机调参收益，`region_balanced` 失败也不是因为 target-set interface
坏掉，而是 equal-region coverage 把三个数据集中 novelty 最高的 `1-96` region 压得过低。

[Decision] 当前进入 11-step loop 的 step 4-6：允许设计 `step_covariance_balanced`
candidate。该 candidate 必须保持 architecture/inference path 不变，使用 train-split
static novelty prior，固定少量超参，并以 R.3 作为 primary baseline。若它不能超过 R.3，
objective-only 主线应停止。

[Caveat] 该诊断不是 paper-core 证明。`region_balanced` delta vs novelty deficit 的
Spearman 只有 `0.1538`，说明 novelty proxy 的排序解释力有限；下一轮训练必须验证
forecast gain，而不能只报告更好的 novelty alignment。

[Decision Update: 2026-06-23] Phase2-C.2 `step_covariance_balanced` 已实现并通过
本地 smoke。该候选与 QDF / MetaDF 相关，但只采用 diagonal heterogeneous task
weighting 的保守部分，不实现完整 off-diagonal quadratic objective 或 bilevel/meta
weight learning。

Implementation:

- train switch:
  `--step-loss-weighting step_covariance_balanced`
- run name:
  `PatchEncoderStepCovarianceBalanced`
- remote runner:
  `scripts/remote/run_phase2_step_covariance_balanced_gate.sh`
- sync wrapper:
  `scripts/sync_phase2_step_covariance_balanced_results.sh`
- code explanation:
  `docs/code-explanation/phase2-step-covariance-balanced-objective.md`

本地 smoke：

- output:
  `artifacts/runs/smoke_phase2_step_covariance_balanced/PatchEncoderStepCovarianceBalanced/ETTh2/mixed_h96_h192_h336_h720/seed2021`
- weighted pressure share:
  `1-96 = 0.4807`, `97-192 = 0.1951`, `193-336 = 0.1640`, `337-720 = 0.1603`
- prefix mismatch MSE:
  `96/720 = 8.455293790696292e-15`,
  `192/720 = 8.434740536231167e-15`,
  `336/720 = 3.5504944524786947e-15`

[Next] 按远程实验策略，commit/push 后在 `529_Lab-3090` `git pull`，检查 GPU 显存，
再运行完整 Phase2-C.2 gate。Primary baseline 仍是 R.3。

[Decision Update: 2026-06-23] Phase2-C.2 remote gate 已完成，结论为 fail：

- report:
  `analysis/phase2_step_covariance_balanced_gate_20260623/phase2_step_covariance_balanced_decision_report.md`
- interpretation:
  `analysis/phase2_step_covariance_balanced_gate_20260623/phase2_step_covariance_balanced_interpretation.md`
- MSE wins vs R.3: `2/12`;
- mean relative MSE vs R.3: `+0.76%`;
- dataset mean relative MSE vs R.3:
  `ETTh2 -0.09%`, `ETTm1 +1.35%`, `Weather +1.03%`;
- MSE wins vs uniform target-set: `12/12`;
- mean relative MSE vs uniform target-set: `-0.28%`;
- max prefix mismatch MSE: `5.182444710459706e-14`。

[Inference] 该结果保留了 QDF 相关性的弱证据：非等权 objective 比 uniform MSE 好。但
static novelty-aware diagonal weighting 不足以超过 R.3，尤其没有保住 early-prefix gain。

[Rollback] 当前停止 objective-only simplification path。不要继续对 `beta/eta` 做宽 sweep，
也不要在该 objective carrier 上叠 MoE。若继续 QDF 方向，应先把完整 QDF-style
off-diagonal / learned quadratic objective 作为 external baseline 或 diagnostic 复现；否则回到
base architecture / external baseline selection。

[Decision Update: 2026-06-23] Phase2-D QDF off-diagonal diagnostic 已完成，结论为 pass。

Artifacts:

- report:
  `analysis/phase2_qdf_offdiag_diagnostic_20260623/phase2_qdf_offdiag_diagnostic_report.md`
- script:
  `scripts/analyze_phase2_qdf_offdiag_diagnostic.py`
- experiment plan:
  `docs/experiments/phase2-qdf-offdiag-reproduction-path.md`

11-step loop 判断：

- `current_step`: Step 2-3 rollback check；
- `problem`: diagonal / static objective proxy 已失败，但 QDF 的完整 learned quadratic
  objective 尚未被等价测试；
- `existence_evidence`: QDF loss 的轴语义是 `[B*D, P]`，本地 `[B*D, 4]` future-region
  diagnostic 显示 strong off-diagonal dependence；
- `decision`: 进入 QDF upstream reproduction gate，不继续 diagonal objective sweep。

核心统计：

| Dataset | Mean abs offdiag corr | Max abs offdiag corr | Offdiag corr Fro share |
| --- | ---: | ---: | ---: |
| `ETTh2` | `0.7103` | `0.8127` | `0.6057` |
| `ETTm1` | `0.8585` | `0.8897` | `0.6888` |
| `Weather` | `0.7342` | `0.8066` | `0.6193` |

[Decision] 下一步实验路径是 native QDF upstream reproduction：

1. 先在 QDF 官方仓库原生流程中跑 `meta_type=all`；
2. 若成本允许，再补 `diag` 和 `off_diag` controls；
3. 只在 full/off-diagonal 明确优于 own MSE/diag control 后，才考虑 source-informed
   local component；
4. 若 QDF upstream gate 失败，objective route 回滚到 Step 2，转向 base architecture 或
   external baseline selection。

[Implementation Update] QDF upstream reproduction tooling 已实现：

- `scripts/remote/run_phase2_qdf_upstream_gate.sh`
- `scripts/remote/check_phase2_qdf_upstream_progress.sh`
- `scripts/sync_phase2_qdf_upstream_results.sh`
- `scripts/analyze_phase2_qdf_upstream_gate.py`

默认远程输出为 `/home/yingch/exp_outputs/r-2026-fatst/phase2_qdf_upstream_gate`。第一轮
`META_TYPES=all` 只能完成 full-QDF metric collection；最终 gate 需要 `diag` control 才能
判断是否 pass。

[Decision Update: 2026-06-24] Phase2-E1/E2 已完成，objective-matrix route 暂停。

Key returned evidence:

- `PatchEncoderOffdiagBlockQuadratic` vs R.3:
  `1/12` MSE wins, mean relative MSE `+0.0464%`;
- QDF learned precision residual alignment:
  `qdf_off_diag_precision` gap/non-gap ratio `0.8662`,
  `qdf_all_precision` gap/non-gap ratio `0.8853`;
- conclusion: QDF 可作为 future-step interaction 背景证据，但不再作为 local objective 的直接
  设计来源。

## Phase3-A: Prefix Specialist Tradeoff Diagnostic

[Decision Update: 2026-06-24] Phase3-A diagnostic 已完成，结论为 pass-as-diagnostic，不是
MoE pass。

Artifacts:

- analyzer:
  `scripts/analyze_phase3_prefix_specialist_tradeoff.py`;
- report:
  `analysis/phase3_prefix_specialist_tradeoff_20260624/phase3_prefix_specialist_report.md`;
- code explanation:
  `docs/code-explanation/phase3-prefix-specialist-tradeoff-diagnostic.md`。

11-step loop 判断：

- `current_step`: Step 9-10；
- `problem`: R.3 剩余 specialist gaps 来源不清；
- `existence_evidence`: R.3 aggregate gaps 分解为 short-only extra-window gaps 与 H720
  late-segment gaps；
- `idea`: 检查 prefix-consistent carrier 是否在同输入下牺牲 horizon-specialist prediction；
- `theory_check`: 若同输入 prefix conflict 存在，prediction/residual prefix mismatch 应非零；
- `design`: 对齐 `h96/h192/h336` 与 `h720` prefix 的前 `N_720` windows，并拆出
  short-only extra windows；
- `gate`: prefix identity pass，short gaps 可由 extra-window regime 解释，long gaps 可由
  late segment localization 解释；
- `decision`: diagnostic pass；下一步进入 Phase3-B regime/segment calibration design。

核心结果：

- max prediction prefix mismatch MSE:
  `5.382513303646484e-14`;
- short aggregate gaps:
  `ETTm1/96`, `Weather/96`；
- both short gaps are `short_extra_window_gap` rather than same-input prefix conflict；
- H720 segment gaps:
  `ETTh2 193-336`, `ETTh2 337-720`, `ETTm1 337-720`。

[Next] Phase3-B 不进入 full MoE，也不直接实现 output residual correction。下一步先做
Regime/Segment Mechanism Diagnostic：只检查困难 windows/segments 是否能被 prediction-before
features 识别。

1. short-only extra-window issue: 先确认 test split 末端/局部 regime 是否导致 higher residual
   energy 与可识别 input-regime signal；
2. H720 late-segment issue: 先判断 high-error late windows 是否能由 history/window-position
   signal 识别；
3. 若 diagnostic pass，Phase3-C 才允许设计 target-state / segment-operator conditioning；
   不采用 prediction 后的 arbitrary residual correction。

## Phase3-B: Regime/Segment Mechanism Diagnostic

[Decision Update: 2026-06-24] Phase3-B diagnostic 已完成，结论为 pass-as-diagnostic。

Artifacts:

- analyzer:
  `scripts/analyze_phase3_regime_segment_mechanism.py`;
- report:
  `analysis/phase3_regime_segment_mechanism_20260624/phase3_regime_segment_mechanism_report.md`;
- code explanation:
  `docs/code-explanation/phase3-regime-segment-mechanism-diagnostic.md`;
- next design doc:
  `docs/experiments/phase3-regime-segment-conditioned-target-operator.md`。

11-step loop 判断：

- `current_step`: Step 9-10 完成，进入 Step 4-6；
- `problem`: short-only extra windows 与 H720 late segments 有可定位 gaps；
- `existence_evidence`: Phase3-A 定位 gaps；Phase3-B 证明这些 failure groups 有
  prediction-before feature separation；
- `idea`: 用 history/window-position regime token 条件化 target-side segment operator；
- `theory_check`: 如果困难区域在 prediction 前可识别，就不需要 output residual correction；
- `design`: Phase3-C `Regime/Segment-Conditioned Target Operator`，作用在 output readout 前；
- `gate`: 修复 observed gaps，控制 non-gap degradation，保持 prefix consistency；
- `decision`: diagnostic pass；下一步实现 Phase3-C 最小候选。

核心结果：

| Setting | Feature | AUC | SMD |
| --- | --- | ---: | ---: |
| `ETTm1/96` short extra windows | `history_mean` | `0.997619` | `-3.221514` |
| `Weather/96` short extra windows | `window_index_norm` | `1.000000` | `2.599896` |
| `Weather/96` short extra windows | `history_std` | `0.979425` | `2.494574` |
| `ETTh2 337-720` high-error segment | `window_index_norm` | `0.845886` | `1.543815` |
| `ETTh2 337-720` high-error segment | `history_slope_abs_mean` | `0.828835` | `1.262857` |
| `ETTm1 337-720` high-error segment | `window_index_norm` | `0.786843` | `1.219849` |

[Decision] 这些结果打消了“只能靠 residual 输出修补”的核心顾虑：hard groups 在预测前已有可用
regime/segment signal。下一步可以做 conditioned target operator，但 residual/error 只能作为
diagnostic label，不应成为模型输出后的自由修正项。

## Phase3: Future-Side MoE

状态：继续暂停。Phase3-A 支持的是 regime/segment calibration 分支，不支持直接启动 MoE。

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
