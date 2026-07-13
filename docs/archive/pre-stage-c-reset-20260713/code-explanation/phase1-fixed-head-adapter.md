# Phase1 Fixed-Head Adapter 代码说明

## 目的

`PatchEncoderFixedHeadAdapter` 对应 Phase1-A.2 gate。它不是重新提出一个更复杂的
encoder，而是在 Phase0 选定的 `PatchEncoderFixedHead` 上测试一个更窄的问题：

> 保留 fixed flatten head 的强 readout capacity 后，future segment states 是否还能作为
> conditioning interface 改善预测？

该设计直接回应 Phase1-A 第一轮失败：`PatchEncoderSegmentQueryHead` 用 segment state
替换 fixed head 后大面积退化，主要风险是 readout capacity 被削弱。

## Forward 数据流

输入：

$$
X \in \mathbb{R}^{B \times L \times C}
$$

默认 $L=336$。

### 1. RevIN

如果启用 RevIN，输入先按 batch 和 channel 做历史窗口归一化：

$$
X^{norm} = \frac{X-\mu_X}{\sigma_X}
$$

推理输出在最后使用同一组 $\mu_X,\sigma_X$ denormalize。

### 2. Channel-independent patch encoding

代码将输入变为：

$$
X^{norm} \rightarrow \mathbb{R}^{(B C) \times 1 \times L}
$$

然后通过 replication padding、`unfold` 和 patch embedding 得到：

$$
P \in \mathbb{R}^{(B C) \times N \times d}
$$

其中 $N$ 是 patch 数，$d=d_{model}$。加入 learnable positional embedding 后进入
Transformer encoder：

$$
Z=E_\theta(P) \in \mathbb{R}^{(B C) \times N \times d}
$$

### 3. Fixed-head 主路径

主路径完全保留 Phase0 fixed head：

$$
\hat{Y}^{base,norm}
= \text{Linear}(\text{Flatten}(Z))
\in \mathbb{R}^{(B C) \times H}
$$

这一步是模型的高容量 readout。Phase1-A.2 不删除它。

### 4. Future-segment adapter

令 segment length 为 $S=48$，segment 数量为：

$$
J=\lceil H/S\rceil.
$$

模型维护 learnable segment queries：

$$
Q \in \mathbb{R}^{1 \times J \times d}.
$$

扩展到 batch-channel 维后，使用 cross-attention 从 history representation 中读取：

$$
U = A_\theta(Q,Z)
\in \mathbb{R}^{(B C) \times J \times d}.
$$

`adapter_head` 对每个 segment 输出 affine 参数：

$$
(\gamma,\beta)
=R_\theta(U)
\in \mathbb{R}^{(B C) \times J \times 2S}.
$$

裁剪到 horizon $H$ 后：

$$
\gamma,\beta \in \mathbb{R}^{(B C) \times H}.
$$

最终 normalized forecast 为：

$$
\hat{Y}^{norm}
=\hat{Y}^{base,norm}\odot(1+\gamma)+\beta.
$$

最后 reshape 回：

$$
\hat{Y}\in \mathbb{R}^{B \times H \times C}.
$$

## 初始化约束

`adapter_head` 最后一层使用 zero initialization：

$$
\gamma=0,\quad \beta=0
$$

因此初始 forward 等价于 `PatchEncoderFixedHead`：

$$
\hat{Y}^{norm}=\hat{Y}^{base,norm}.
$$

这不是为了保证最终一定不退化，因为训练会同时更新 fixed head 和 adapter；它的作用是避免
第一轮 `SegmentQueryHead` 那种从结构上先删除强 readout path 的风险。

## 诊断 artifacts

训练脚本除常规文件外额外写入：

| Artifact | 来源张量 | 含义 |
| --- | --- | --- |
| `adapter_query_similarity.csv` | `segment_queries` | segment query cosine similarity |
| `adapter_delta_stats.csv` | `prediction - base_prediction` | adapter 对 fixed-head base prediction 的实际修正幅度 |

`adapter_delta_stats.csv` 的列定义：

- `delta_mse_to_base`: $\hat{Y}$ 与 $\hat{Y}^{base}$ 的 MSE。
- `delta_mae_to_base`: $\hat{Y}$ 与 $\hat{Y}^{base}$ 的 MAE。
- `mean_abs_gamma`: $\gamma$ 的平均绝对值。
- `mean_abs_beta`: denormalized $\beta$ 的平均绝对值。
- `delta_to_base_mae_ratio`: adapter MAE 修正量相对 base prediction 平均绝对值的比例。

## Code-Theory Consistency Evaluation

Intended theory:

- fixed head 性能强，不能被低容量 segment readout 直接替换；
- 但 fixed head 缺少 future-side interface，不利于后续 future-aware alignment 和 MoE；
- 一个 identity-initialized segment adapter 可以在不破坏主 readout 的前提下测试该 interface。

Code realization:

- `fixed_head` 保留 `Flatten(Z) -> Linear(..., H)`；
- `segment_queries + cross-attention` 生成 future segment states；
- `adapter_head` 生成 $\gamma,\beta$，只 conditioning fixed-head output；
- `return_components=True` 暴露 `base_prediction`、`gamma`、`beta` 用于诊断。

Still proxy:

- 当前 adapter state 只由 history 和 learnable query 生成，没有 training-only future teacher；
- 因此它只能证明 history-derived future interface 是否有用，不能证明 future-aware
  latent alignment 的价值。

Falsification evidence:

- 若 main MSE 大面积退化，说明 adapter 干扰 fixed path 或优化不稳定；
- 若 main MSE 微弱改善但 `delta_to_base_mae_ratio` 接近零，说明机制未实际参与；
- 若 adapter 有非零修正但仍无收益，下一步应转向 future-aware teacher/student alignment，
  而不是继续增加 adapter 容量。
