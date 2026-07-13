# Phase1-A.5 Step-Specific State Adapter Code Explanation

## 目的

`PatchEncoderStepSpecificStateAdapter` 对应 Phase1-A.5 reset 中的候选：
在保留 `PatchEncoderFixedHead` dense readout rows 的前提下，把 future segment conditioning
从 output space 前移到 latent representation space。

它要检验的问题是：

> fixed head 是否把所有 future steps 绑定到同一个 history representation，使不同 future
> segments 只能通过 output rows 区分，而不能在进入 readout 前形成 step/segment-specific
> representations？

## Forward Flow

### 1. RevIN 与 patch encoder

代码位置：
[model.py](/Users/river/PaperResearch/Project/R_2026_FATST/baselines/patch_encoder_step_specific_state_adapter/model.py:186)

输入为：

$$
X \in \mathbb{R}^{B \times L \times C}.
$$

如果启用 RevIN，模型先对每个样本和变量沿时间维归一化。随后 `_encode` 将输入变为
channel-independent patch tokens：

$$
Z \in \mathbb{R}^{(B C) \times N \times d}.
$$

其中 `N` 是 patch 数，`d = d_model`。

### 2. Base fixed-head path

代码位置：
[model.py](/Users/river/PaperResearch/Project/R_2026_FATST/baselines/patch_encoder_step_specific_state_adapter/model.py:159)

base path 与 `PatchEncoderFixedHead` 保持同构：

$$
\hat{Y}^{base}_{norm}
= W_H \operatorname{Flatten}(Z).
$$

实现上，`head_linear` 保存完整 fixed-head rows：

$$
W_H \in \mathbb{R}^{H \times (N d)}.
$$

这一步的作用是保留 Phase0 selected base 的 readout capacity，并为诊断提供
`base_prediction`。

### 3. Future segment states

代码位置：
[model.py](/Users/river/PaperResearch/Project/R_2026_FATST/baselines/patch_encoder_step_specific_state_adapter/model.py:153)

模型维护 learnable `segment_queries`：

$$
Q \in \mathbb{R}^{1 \times J \times d},
$$

其中：

$$
J = \lceil H / S \rceil,
$$

`S` 是 `segment_len`，默认 `48`。`SegmentAdapterBlock` 使用 cross-attention 从 history
tokens `Z` 读取信息：

$$
U_j = A_\theta(q_j, Z).
$$

得到：

$$
U \in \mathbb{R}^{(B C) \times J \times d}.
$$

### 4. Pre-head state modulation

代码位置：
[model.py](/Users/river/PaperResearch/Project/R_2026_FATST/baselines/patch_encoder_step_specific_state_adapter/model.py:162)

`modulation_head` 从每个 segment state 生成 latent-space FiLM 参数：

$$
(\gamma_j,\beta_j)=R_\theta(U_j),
\qquad
\gamma_j,\beta_j \in \mathbb{R}^{d}.
$$

最后一层 zero initialization：
[model.py](/Users/river/PaperResearch/Project/R_2026_FATST/baselines/patch_encoder_step_specific_state_adapter/model.py:140)

因此初始时：

$$
\gamma_j=0,\quad \beta_j=0.
$$

### 5. Segment-wise fixed-row readout

代码位置：
[model.py](/Users/river/PaperResearch/Project/R_2026_FATST/baselines/patch_encoder_step_specific_state_adapter/model.py:169)

对每个 future segment，模型先构造 segment-specific representation：

$$
\tilde{Z}_j = Z \odot (1+\gamma_j) + \beta_j.
$$

然后只取 fixed head 中对应 horizon rows：

$$
\hat{Y}_{a_j:b_j}
= W_{a_j:b_j}\operatorname{Flatten}(\tilde{Z}_j).
$$

所有 segment 输出拼接后得到：

$$
\hat{Y}_{norm} \in \mathbb{R}^{(B C) \times H}.
$$

最后 reshape 回：

$$
\hat{Y} \in \mathbb{R}^{B \times H \times C}.
$$

如果 RevIN 启用，则对 `prediction` 和 `base_prediction` 都执行 denorm。

## Diagnostics

训练脚本记录以下新增诊断：

- `adapter_delta_stats.csv`: prediction 与 base prediction 的差异幅度。
- `adapter_query_similarity.csv`: learnable segment queries 是否退化为同质向量。
- `segment_state_activation_similarity.csv`: 输入相关的 segment states 是否彼此可分。
- `state_modulation_stats.csv`: `gamma`、`beta` 和 segment state norm 的整体与 per-segment
  统计。

代码位置：
[train.py](/Users/river/PaperResearch/Project/R_2026_FATST/baselines/patch_encoder_step_specific_state_adapter/train.py:281)

## Code-Theory Consistency

[Intended theory] Phase1-A.5 不是在 fixed head 输出后做校正，而是在 readout 前构造
step/segment-specific representation。

[Code realization] `_segment_readout` 在每个 segment 内使用
`z * (1 + gamma_j) + beta_j` 后，再调用 `head_linear.weight[start:end]` 和
`head_linear.bias[start:end]`。因此 fixed head rows 被复用，但它们看到的是 segment-specific
latent state。

[Proxy boundary] 当前第一版只使用 channel-independent latent FiLM，尚未引入 future-aware
teacher branch、MoE routing 或 heterogeneous operators。因此它只能验证 pre-head
state adaptation 是否值得继续，不能直接证明最终 unified model。

[Falsification evidence] 若 A.5 gate 中 `PatchEncoderStepSpecificStateAdapter` 无法达到
`6/12` main MSE wins、mean relative MSE < 0，或 Weather 等 dataset 全 horizon 退化，则应判定
该候选不足以作为 paper-core。若 MSE 有提升但 `state_modulation_stats.csv` 接近零且 segment
activation cosine 接近 1，则说明机制可能退化为固定 head 微扰，也不应直接进入 MoE。
