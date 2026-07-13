# Phase1-A.6 Output-Process Rollback

## Step 1: 调研分析

[Fact] Phase1-A.1 到 A.5 已经排除了三条直接路线：

- 直接用 segment query decoder 替换 fixed head 会丢失 dense readout capacity。
- 在 fixed head 输出后做 segment affine correction 只能得到不稳定 partial result。
- 在 fixed head 前做 segment-conditioned latent modulation 有机制活动，但仍不能稳定超过
  `PatchEncoderFixedHead` 和 `PatchEncoderFixedHeadAdapter`。

[Fact] Phase1-A.5 的关键证据是：

| Comparison | MSE wins | Mean relative MSE |
| --- | ---: | ---: |
| vs `PatchEncoderFixedHead` | 7/12 | +0.39% |
| vs `PatchEncoderFixedHeadAdapter` | 8/12 | +0.19% |

[Fact] A.5 的收益不是均匀分布的：

- vs `PatchEncoderFixedHead`，h96/h192 平均退化，h336/h720 平均改善；
- ETTh2 平均退化 `+2.37%`，ETTm1 平均改善 `-1.24%`，Weather 接近持平；
- segment-level 上，`193-336` 和 `337-720` 平均改善，`1-96` 和 `97-192` 接近持平或退化。

[Inference] 这些结果说明 fixed head rows 已经承担了较强的 step-specific readout。
继续调制 shared latent state 容易破坏短 horizon 上已经稳定的 readout，而不是单调补足
fixed head 的缺陷。

## Step 2: 待解决问题

下一轮不再问：

> 是否需要为每个 future segment 构造新的 latent representation？

而是问：

> fixed direct head 是否缺少对 future output trajectory 的结构化残差建模，使每个 output row
> 独立学习，难以利用 future steps 之间的相关性、局部平滑性和 horizon-dependent error mode？

该问题更贴近当前 evidence：

- fixed head 强，说明 dense row readout 不能删除；
- A.5 的中长 horizon 局部收益说明 long-range segments 仍有可修正 error mode；
- QDF 指出标准 MSE 把 future steps 当作独立等权任务，忽略 label autocorrelation；
- TIMEPERCEIVER / ElasTST 都把 target positions 显式纳入预测过程，而不是只作为 output
  dimension；
- SRP++ 强调 step-specificity，但 A.5 说明本项目不能用侵入式 latent modulation 来实现它。

## Step 3: 问题是否真实且值得研究

[Strong Evidence] 问题真实存在，但幅度中等：

- prefix consistency 和 segment oracle 说明 fixed direct head 的 output strategy 有缺陷；
- A.5 的 segment-level 结果说明可修正区域主要在中长 horizon；
- A.5 的失败说明缺陷不是简单的 shared-latent bottleneck。

[Decision] 该问题值得继续研究，但必须收窄为 output-process 问题，而不是笼统的
future-side decoder 问题。论文故事也应相应调整：

> Long-term forecasting should not only learn a stronger encoder or a static direct head.
> A strong direct head should be treated as the base trajectory readout, while the remaining
> forecasting process needs structured, horizon-aware residual modes over future positions.

## Step 4: Idea

候选暂命名为 `TrajectoryBasisResidualDecoder`，实现名暂定：
`PatchEncoderTrajectoryBasisResidual`。

核心思想：

1. 保留 `PatchEncoderFixedHead` 的 dense direct readout，作为 base trajectory。
2. 在 output trajectory 上加入 low-rank, position-aware residual basis，而不是重新调制
   encoder latent state。
3. residual 由 history state 产生系数，由 future target positions 产生 basis，使 residual
   在 future steps 上是结构化的，而不是每个 output row 独立修正。
4. 用 zero initialization / negative gate bias / residual penalty 保证模型从 fixed head
   起步，避免破坏短 horizon readout。

## Step 5: 理论可行性

令 encoder 输出为：

$$
Z=E_\theta(X),
$$

fixed head 输出为：

$$
\hat{Y}^{base}_{1:H}=W_H\operatorname{Flatten}(Z).
$$

构造 future position embedding：

$$
p_t = \operatorname{PE}(t/H, t), \qquad t=1,\dots,H,
$$

并由 position MLP 产生 $K$ 个 trajectory basis：

$$
B_H = B_\phi(p_{1:H}) \in \mathbb{R}^{H \times K}.
$$

由 history representation 产生 channel-wise 或 global coefficients：

$$
A = A_\phi(\operatorname{Pool}(Z)) \in \mathbb{R}^{C \times K}.
$$

trajectory residual 为：

$$
\Delta Y_{t,c}=\sum_{k=1}^{K} B_{t,k} A_{c,k}.
$$

最终输出：

$$
\hat{Y}_{1:H}=\hat{Y}^{base}_{1:H}+g_H \odot \Delta Y_{1:H}.
$$

其中 $g_H$ 可以是 horizon/segment-aware gate，初始接近 0。训练目标为：

$$
\mathcal{L}
= \operatorname{MSE}(\hat{Y},Y)
+ \lambda_r \|\Delta Y\|_2^2
+ \lambda_s \|\nabla_t^2 \Delta Y\|_2^2.
$$

[Inference] 这个结构与 A.2/A.5 的差异在于：

- A.2 是 output-space affine correction，segment 内仍缺少 trajectory-level coupling；
- A.5 是 latent-space modulation，会干扰 fixed head readout；
- A.6 只在 fixed head 输出上学习低秩、平滑、position-aware residual，直接针对
  future output process。

[Hypothesis] 如果 fixed head 的主要问题是 output rows 独立、缺少 future-step covariance
结构，那么 low-rank trajectory residual 应能在中长 horizon 改善 MSE，同时通过 residual gate
保护短 horizon。

[Risk] 如果 residual basis 只学习到接近 0 的校正，说明 fixed head 已经足够；如果 residual
在短 horizon 上继续造成退化，说明 output-space residual 也不是当前主线，应回退到 objective
或训练协议，而不是继续扩容。

## Step 6: 第一版实验方案

第一版仍使用 one-to-one horizon training，避免 mixed-horizon training 干扰机制判断。

对比模型：

- `PatchEncoderFixedHead`
- `PatchEncoderFixedHeadAdapter`
- `PatchEncoderStepSpecificStateAdapter`
- `PatchEncoderTrajectoryBasisResidual`

实验矩阵：

- datasets: `ETTh2`, `ETTm1`, `Weather`
- horizons: `96`, `192`, `336`, `720`
- seed: `2021`
- optional follow-up seeds: `2022`, `2023`，仅在 single-seed gate 通过后执行。

最小配置建议：

- basis count: `K=8`
- residual gate init: negative bias，使初始 residual 接近 0
- residual penalty: `lambda_r=1e-4`
- smoothness penalty: `lambda_s=1e-4`
- residual basis MLP: two-layer lightweight MLP

必须记录：

- main MSE/MAE；
- residual energy: $\|\Delta Y\|_2 / \|\hat{Y}^{base}\|_2$；
- residual smoothness: $\|\nabla_t^2 \Delta Y\|_2$；
- basis coefficient norm；
- segment-wise relative MSE，特别是 `1-96`、`97-192`、`193-336`、`337-720`；
- parameter count 和 parameter ratio。

第一版 pass 条件：

1. [Performance] 相比 `PatchEncoderFixedHead` mean relative MSE < 0，且至少 `7/12`
   main MSE wins。
2. [Control] 相比 `PatchEncoderFixedHeadAdapter` mean relative MSE < 0，且至少 `6/12`
   main MSE wins。
3. [Stability] h96/h192 平均不能显著退化；若 short horizons 继续像 A.5 一样退化，则失败。
4. [Mechanism] residual energy 非零但受控；若 residual 近零且仍提升，应检查是否只是训练噪声。
5. [Paper story] 若收益只来自单个 dataset 或单个 horizon，不能作为 paper-core decoder。

建议 controls：

- no-smoothness residual；
- unstructured MLP residual，参数量匹配但不使用 position basis；
- random/frozen basis residual；
- residual-only without fixed base，用于验证 fixed base 是否必须保留。

## Step 7 之前的边界

[Decision] 在实现 A.6 前，不进入 Phase1-B，不重启 future-aware alignment，不引入 MoE。
A.6 若通过，才重新讨论：

- 是否把 basis/residual 变成 one-model multi-horizon interface；
- 是否用 future-aware teacher 监督 residual basis；
- 是否在 residual modes 上做 MoE-style conditional operators。

## Step 8-10: 远程训练、结果评估与决策

[Fact] Phase1-A.6 完整 gate 已完成：

- remote host: `529_Lab-3090`
- remote output: `/home/yingch/exp_outputs/r-2026-fatst/phase1_trajectory_basis_residual`
- local report: `analysis/phase1_trajectory_basis_residual_gate_20260622/phase1_trajectory_basis_residual_gate_report.md`
- models: `PatchEncoderFixedHead`, `PatchEncoderFixedHeadAdapter`,
  `PatchEncoderStepSpecificStateAdapter`, `PatchEncoderTrajectoryBasisResidual`
- datasets: `ETTh2`, `ETTm1`, `Weather`
- horizons: `96`, `192`, `336`, `720`
- seed: `2021`
- selected GPUs: `1`, `2`

主结果：

| Comparison | MSE wins | Mean relative MSE | Range | Zero-win datasets |
| --- | ---: | ---: | --- | --- |
| vs `PatchEncoderFixedHead` | 5/12 | +0.67% | -3.85% to +6.43% | none |
| vs `PatchEncoderFixedHeadAdapter` | 6/12 | +0.49% | -3.08% to +8.63% | none |
| vs `PatchEncoderStepSpecificStateAdapter` | 4/12 | +0.33% | -7.09% to +3.17% | none |

Horizon-level 结果显示该候选没有保护住关键稳定性条件：

- vs `PatchEncoderFixedHead`，h96 平均退化 `+1.86%`，h720 平均退化 `+1.33%`；
- vs `PatchEncoderFixedHeadAdapter`，h96 平均退化 `+2.12%`，h720 平均退化 `+1.56%`；
- vs `PatchEncoderStepSpecificStateAdapter`，h336/h720 均为 `0/3` wins。

机制诊断：

- mean residual/base MAE ratio: `0.002637`
- mean gate: `0.018039`
- mean coefficient L2: `0.292015`

[Decision] Phase1-A.6 判定为 `partial`，但不通过。它不是空实现：residual branch
确实被训练为非零，且在 ETTh2 h336/h720、ETTm1 h96/h192、Weather h336 等局部设置上
有收益。然而整体 mean relative MSE 相对三个 control 均为正，MSE wins 也不足以通过
预设 gate；同时 residual/base MAE ratio 仅约 `0.26%`，说明该 low-rank output residual
主要以弱扰动方式工作，未形成足以支撑 paper-core 的结构化 output-process 改善。

[Inference] 该结果反驳了 A.6 的强假设：当前 fixed head 的主要误差不只是缺少低秩、
平滑、future-position residual basis。若仅调大 residual gate 或减小 penalty，可能增加
残差幅度，但已有 h96/h720 退化表明这很可能放大不稳定性，而不是解决底层问题。

## Step 11: 回退点

[Rollback] A.6 不应进入 Phase1-B，也不应作为 future-aware 或 MoE 的承载结构。当前应回退到
长研究模板 step 2-3，重新评估 decoder 创新点的问题定义。

具体地说，A.1-A.6 的连续证据支持以下收敛判断：

1. fixed head 的 output strategy 有问题，但问题幅度中等；
2. 直接替换 dense head 会造成 readout capacity collapse；
3. post-head affine、pre-head state modulation、output-space low-rank residual 都只有
   partial signal，无法稳定超过 strong fixed-head base；
4. 因此，下一轮不应继续围绕 fixed head 做轻量补丁，而应重新提出更基础的 prediction
   process 问题，或者暂时把 decoder 方向降级，转向更有证据的新问题。

[Decision] 在没有新的问题定义前，暂停 one-model compatibility、future-aware alignment
和 Future-Side MoE 的继续实现。若后续仍保留 decoder 主线，需要回到 step 1-5，重新调研并
论证一个不依赖 weak residual patch 的结构性问题。
