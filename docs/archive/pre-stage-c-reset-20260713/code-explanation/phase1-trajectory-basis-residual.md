# Phase1-A.6 Trajectory Basis Residual 代码说明

`PatchEncoderTrajectoryBasisResidual` 对应 Phase1-A.6 output-process rollback。
它不是替换 fixed head，也不是在 readout 前调制 latent state，而是在 fixed head 输出后增加
结构化 trajectory residual。

## Forward 数据流

输入：

$$
x \in \mathbb{R}^{B \times L \times C}.
$$

如果启用 RevIN，先在 history window 上归一化。随后按 Phase0 base 的方式 patching：

$$
x \rightarrow \text{patches} \in \mathbb{R}^{(B C) \times N \times P}.
$$

patch embedding 和 Transformer encoder 得到：

$$
z \in \mathbb{R}^{(B C) \times N \times d}.
$$

fixed head 保持原始 dense readout：

$$
base = W \operatorname{Flatten}(z)
\in \mathbb{R}^{(B C) \times H}.
$$

新增 residual path 分为两部分。

首先，从 future positions 构造特征：

$$
p_t = [t/H,\ (t/H)^2,\ \sin(2\pi t/H),\ \cos(2\pi t/H)].
$$

`basis_mlp` 生成 trajectory basis：

$$
B = B_\phi(p_{1:H}) \in \mathbb{R}^{H \times K}.
$$

然后对 encoder state 做 patch mean pooling：

$$
u = \operatorname{Mean}_N(z) \in \mathbb{R}^{(B C) \times d},
$$

`coefficient_head` 输出每个 channel-series 的 basis coefficients：

$$
a = A_\phi(u) \in \mathbb{R}^{(B C) \times K}.
$$

raw residual 是：

$$
r = a B^\top \in \mathbb{R}^{(B C) \times H}.
$$

再乘上按 future step 学习的 gate：

$$
\tilde{r}_{:,t} = \sigma(g_t) r_{:,t}.
$$

最终 normalized prediction：

$$
\hat{y}^{norm} = base + \tilde{r}.
$$

reshape 回：

$$
\hat{Y}^{norm} \in \mathbb{R}^{B \times H \times C}.
$$

若启用 RevIN，`prediction` 和 `base_prediction` 都 denorm；`residual` 乘以同一个
history std，使它处于原始尺度。

## 初始化语义

`coefficient_head` 的最后一层 weight/bias 初始化为 0。因此训练开始时：

$$
a=0 \Rightarrow r=0 \Rightarrow \hat{Y}=\hat{Y}^{base}.
$$

`residual_gate_logits` 默认初始化为 `-4.0`，使 residual path 即使开始学习，也先以小幅度进入。
这对应 A.6 的核心边界：保护 fixed head 的短 horizon readout。

## Training Loss

训练主损失仍是 MSE：

$$
\mathcal{L}_{mse}=\operatorname{MSE}(\hat{Y},Y).
$$

额外记录并加入两个轻量正则项：

$$
\mathcal{L}_{res}=\|\tilde{r}^{norm}\|_2^2,
$$

$$
\mathcal{L}_{smooth}=\|\nabla_t^2 \tilde{r}^{norm}\|_2^2.
$$

总损失：

$$
\mathcal{L}
=\mathcal{L}_{mse}
+\lambda_r\mathcal{L}_{res}
+\lambda_s\mathcal{L}_{smooth}.
$$

默认 `lambda_r=lambda_s=1e-4`。

## Artifact 语义

`trajectory_residual_stats.csv` 记录：

- `residual_mse`: denorm 后 residual 的均方幅度；
- `residual_mae`: denorm 后 residual 的平均绝对幅度；
- `raw_residual_mae`: gate 之前 residual 的平均绝对幅度；
- `residual_to_base_mae_ratio`: residual 相对 base prediction 幅度；
- `residual_to_prediction_mae_ratio`: residual 相对 final prediction 幅度；
- `mean_gate`, `max_gate`: future-step residual gate；
- `coefficient_l2`: basis coefficients 的 L2 norm；
- `residual_norm_smoothness`: normalized residual 的二阶差分能量。

这些统计用于判断 residual 是否：

1. 近似为 0，说明 A.6 没有真正工作；
2. 过大并破坏 fixed head，说明 residual path 变成 uncontrolled correction；
3. 在中长 horizon 更活跃，符合 A.5 诊断出的 error mode。

## Code-Theory Consistency

[Intended theory] A.6 要验证 fixed head 的问题是否在 output-process residual structure，
而不是 shared latent state 本身。

[Code realization] 当前代码保留 fixed head 输出，新增低秩 position-aware residual，并用
zero-init 和 gate 让模型从 fixed head 起步。

[Proxy] 当前 basis 只由 deterministic future position features 生成，还不是完整
TIMEPERCEIVER-style target query decoder；它是最小 output-process probe。

[Falsification] 如果 full gate 中 h96/h192 继续显著退化，或 residual 近零但指标随机波动，
则 A.6 不能作为 paper-core decoder，应回退到 objective-level 或 training protocol，而不是
继续扩 residual capacity。
