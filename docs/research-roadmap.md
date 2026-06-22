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

## Phase2: Future-Aware Mechanism

状态：暂停，等待 Phase1-R 至少达到 compatibility pass。

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

[Decision] Phase2 的默认接入点必须是 Phase1-R 产生的 target-side state $U_T$，而不是
A.5/A.6 的 fixed-head patch。若 Phase1-R 不通过，future-aware 应重新定义 objective 或
carrier，而不是继续作为 adapter 修补项。

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
