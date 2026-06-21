# Phase1 Future-Aware Adapter 代码说明

## 目的

`PatchEncoderFutureAwareAdapter` 是 Phase1-A.3 候选。它不是改变推理时的输入条件，而是在
训练时使用 ground-truth future 构造 teacher state，约束 history-derived adapter state。

直接动机来自两条证据：

- `PatchEncoderSegmentQueryHead` 删除 fixed head readout 后失败；
- `PatchEncoderFixedHeadAdapter` 保留 fixed head 后达到 `partial_pass`，但平均 MSE 仍略退化。

因此，Phase1-A.3 测试的问题是：

> future-side interface 是否需要 training-only future signal 才能稳定对齐 future
> distribution？

## Inference Forward Path

推理时只输入：

$$
X \in \mathbb{R}^{B \times L \times C}.
$$

RevIN 后，模型按 channel-independent 方式 patch：

$$
X \rightarrow P \in \mathbb{R}^{(BC) \times N \times d}.
$$

Transformer encoder 输出：

$$
Z \in \mathbb{R}^{(BC) \times N \times d}.
$$

fixed head 主路径：

$$
\hat{Y}^{base,norm}
=
\text{Linear}(\text{Flatten}(Z))
\in \mathbb{R}^{(BC) \times H}.
$$

adapter student state：

$$
U^{student}=A_\theta(Q,Z)
\in \mathbb{R}^{(BC) \times J \times d},
$$

其中 $J=\lceil H/S\rceil$，默认 $S=48$。

adapter affine output：

$$
(\gamma,\beta)=R_\theta(U^{student})
\in \mathbb{R}^{(BC) \times H}.
$$

最终 normalized prediction：

$$
\hat{Y}^{norm}
=
\hat{Y}^{base,norm}\odot(1+\gamma)+\beta.
$$

最后 denormalize 得到：

$$
\hat{Y}\in\mathbb{R}^{B \times H \times C}.
$$

## Training-Only Teacher Path

训练时额外传入：

$$
Y\in\mathbb{R}^{B \times H \times C}.
$$

模型使用历史窗口的 RevIN statistics 把 $Y$ 转成 $Y^{norm}$。随后按 segment 切分：

$$
Y^{norm}
\rightarrow
\mathbb{R}^{(BC) \times J \times S}.
$$

teacher branch 通过 segment embedding 和轻量 Transformer encoder 得到：

$$
S^{teacher}
\in
\mathbb{R}^{(BC) \times J \times d}.
$$

teacher reconstruction head 输出 normalized future reconstruction：

$$
\hat{Y}^{teacher,norm}
\in
\mathbb{R}^{(BC) \times H}.
$$

student adapter state 经过 projection：

$$
S^{student}=P_\theta(U^{student}).
$$

alignment loss：

$$
\mathcal{L}_{align}
=1-\cos(S^{student},\operatorname{sg}(S^{teacher})).
$$

reconstruction loss：

$$
\mathcal{L}_{recon}
=\operatorname{MSE}(\hat{Y}^{teacher,norm},Y^{norm}).
$$

训练总 loss：

$$
\mathcal{L}
=
\mathcal{L}_{pred}
+\lambda_{align}\mathcal{L}_{align}
+\lambda_{recon}\mathcal{L}_{recon}.
$$

默认第一轮 $\lambda_{align}=0.05$，$\lambda_{recon}=0.05$。

## Leakage Boundary

`forward(x)` 和 `forward(x, y=Y)` 必须产生相同 `prediction`。当 `y` 存在时，teacher
branch 只增加 diagnostic tensors 和 auxiliary losses，不参与 prediction computation。

训练脚本写入 `future_alignment_stats.csv`：

- `alignment_loss`: teacher/student state alignment loss。
- `reconstruction_loss`: teacher branch reconstruction MSE。
- `teacher_student_cosine`: normalized teacher/student state cosine。
- `prediction_leakage_max_abs`: `forward(x)` 与 `forward(x,y)` prediction 最大绝对差。

`prediction_leakage_max_abs > 1e-7` 直接说明实现不符合理论边界。

## Code-Theory Consistency Evaluation

Intended theory:

- history-only adapter interface 已有活动迹象，但缺少 future distribution supervision；
- teacher branch 可在训练时提供 future state anchor；
- 推理时仍只依赖 history-derived student state。

Code realization:

- `fixed_head` 和 adapter affine prediction path 与 Phase1-A.2 一致；
- `_teacher_state` 只在 `y is not None` 时执行；
- `teacher_state.detach()` 用于 alignment，避免 teacher 被 alignment loss 拉向 student；
- `teacher_reconstruction_head` 通过 reconstruction loss 保持 teacher state 的 future meaning；
- `future_alignment_stats.csv` 显式审计 leakage。

Still proxy:

- teacher state 当前来自 segment-level future values，不包含更复杂的 frequency/global relation
  alignment；
- alignment loss 是 cosine mean，不是 TimeAlign 的完整 local/global relation loss；
- 第一轮只验证 future-aware supervision 是否值得继续，不作为最终模型定型。

Falsification evidence:

- leakage audit 非零；
- alignment/reconstruction loss 下降但 MSE 不改善；
- 只超过 adapter、不超过 fixed head；
- 改善只集中在单个 dataset/horizon，缺少跨设置稳定性。
