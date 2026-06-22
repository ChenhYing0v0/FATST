# Phase1-R Target-Set Decoder 代码说明

`PatchEncoderTargetSetDecoder` 对应 Phase1-R 的 target-set decoder hypothesis。
它不是继续修补 horizon-specific fixed head，而是把 requested future target segments
显式输入模型。第一版为了保证 prefix consistency，不把 requested horizon scalar 写入
单个 segment feature；horizon 只通过 target set 中包含多少 segment 进入模型。

## Forward 数据流

输入：

$$
x \in \mathbb{R}^{B \times L \times C}.
$$

若启用 RevIN，先在 history window 上归一化。随后沿用 Phase0 patch encoder：

$$
x \rightarrow \text{patches}
\in \mathbb{R}^{(B C) \times N \times P}.
$$

patch embedding 和 Transformer encoder 得到 history patch states：

$$
Z=E_\theta(x)
\in \mathbb{R}^{(B C) \times N \times d}.
$$

给定 requested horizon $H$，代码按 `segment_len=48` 构造 target segments：

$$
T_H=\{[1,48],[49,96],\dots\}.
$$

每个 target segment 生成 8 维 deterministic feature：

$$
[\text{start},\text{end},\text{center},\text{width},0,\text{segment id},
\sin(2\pi\text{center}/720),\cos(2\pi\text{center}/720)].
$$

第 5 维是保留位，当前固定为 0。因此同一 segment 在 `H=96` 和 `H=720` 请求下拥有相同
query，prefix consistency 不会被 horizon scalar 主动破坏。

`target_feature_embedding` 和 `target_pos_embedding` 得到 target queries：

$$
Q_T \in \mathbb{R}^{(B C) \times J \times d}.
$$

`TargetDecoderBlock` 只做 target-to-history cross-attention：

$$
U_T=\operatorname{CrossAttn}(Q_T,Z,Z)
\in \mathbb{R}^{(B C) \times J \times d}.
$$

第一版不做 target self-attention，因此同一个 prefix segment 的计算不会因为请求更长
horizon 而被后续 target query 改写。这是当前 prefix-stable policy 的工程实现。

## Causal Target Interaction

`--target-interaction-layers K` 启用 Phase1-R.2 的 causal target interaction。默认
`K=0`，即保持第一版 `PatchEncoderTargetSetDecoder` 的 independent target-query policy。

启用后，target-to-history cross-attention 先得到：

$$
U_T=\operatorname{CrossAttn}(Q_T,Z,Z)
\in\mathbb{R}^{(BC)\times J\times d}.
$$

随后 `CausalTargetInteractionBlock` 在 target segments 维度上做 masked self-attention：

$$
V_j = \operatorname{SelfAttn}(U_j, U_{\le j}, U_{\le j}).
$$

代码中的 causal mask 是上三角 mask，禁止 segment $j$ 读取任意 later target segment
$k>j$。因此当 requested horizon 从 `H=96` 扩展到 `H=720` 时，前缀内第 $j$ 个 target
state 的可见集合不变：

$$
V_j(T_{1:J}) = V_j(T_{1:K}),\quad j \le K \le J.
$$

这保持了 prefix stability，同时允许模型在 latent target-state 空间建模 future segment
dependency。它不是 autoregression：模型没有把已预测的 future values 回灌到下一段，
也没有读取 ground-truth future；interaction 只发生在 target queries / target states 之间。

为了避免 A.1 `SegmentQueryHead` 的 readout capacity collapse，模型保留 dense history
readout：

$$
r = R_\theta(\operatorname{Flatten}(Z))
\in \mathbb{R}^{(B C) \times d_r}.
$$

target state 生成 FiLM 参数：

$$
(\gamma_j,\beta_j)=F_\theta(U_j),
\qquad
\gamma_j,\beta_j \in \mathbb{R}^{(B C) \times d_r}.
$$

每个 target segment 用同一个 shared segment output head：

$$
\hat{y}_{a_j:b_j}
=
O_\theta(r\odot(1+\gamma_j)+\beta_j).
$$

拼接全部 segment 后裁剪到 requested $H$，reshape 回：

$$
\hat{Y}_{1:H}\in\mathbb{R}^{B \times H \times C}.
$$

若启用 RevIN，最终 prediction denorm 回原始尺度。

## Prefix Residual Repair

`--prefix-residual-segments K` 启用 short-prefix targeted repair。默认 `K=0`，即完全复现
第一版 `PatchEncoderTargetSetDecoder`。

启用后，模型从同一个 history patch state 生成前 $K$ 个 target segments 的 residual：

$$
\Delta^{prefix}
=
R^{prefix}_\theta(\operatorname{Flatten}(Z))
\in \mathbb{R}^{(B C) \times (K \cdot S)}.
$$

其中 $S$ 是 `segment_len`。该 residual 只加到前 $K$ 个 segments：

$$
\hat{y}_{a_j:b_j}
=
O_\theta(r\odot(1+\gamma_j)+\beta_j)
+
\Delta^{prefix}_{a_j:b_j},
\quad j<K.
$$

`prefix_residual_head` 采用 zero initialization，因此训练起点仍等价于未修补版本。
这个 repair 的约束是：

- residual 只依赖 $Z$ 和 segment index，不依赖 requested horizon scalar；
- 同一个 prefix segment 在 `H=96` 与 `H=720` 请求下 residual 完全相同；
- 它只补 short-prefix dense readout capacity，不改变 target-set interface 的核心故事。

该机制直接针对 Phase1-R 的 near-miss 失败点：h96/h192 strict H720-prefix gate 未完全通过，
且 h96 相对 specialist 的 amortization gap 明显偏大。

## Mixed-Horizon Training

`train.py` 为每个 target horizon `{96,192,336,720}` 建立独立 `ForecastDataset` 和
`DataLoader`。每个 optimization step 随机采样一个 horizon，读取对应 batch，然后调用：

```python
pred = model(x, pred_len=horizon)
```

这种做法避免用 `pred_len=720` 数据集训练所有 horizon 时丢失短 horizon 可用窗口，同时保持
一个模型共享参数。validation 对每个 horizon 分别评估，再用 mean validation MSE 做
early stopping。

## Prefix-Risk Weighted Objective

`--step-loss-weighting uniform` 是默认设置，复现前序所有 target-set decoder 实验。

`--step-loss-weighting prefix_risk` 启用 Phase1-R.3 的 objective-level diagnostic。它不改变
模型结构，只改变训练 loss 中不同 future steps 的权重：

$$
w_t =
\frac{(t / H_{max})^{-\alpha}}
{
\frac{1}{H_{max}}\sum_{s=1}^{H_{max}}(s / H_{max})^{-\alpha}
}.
$$

训练 loss 为：

$$
\mathcal{L}
=
\frac{1}{BHC}
\sum_{b,t,c}w_t
(\hat{y}_{b,t,c}-y_{b,t,c})^2.
$$

当 `--step-loss-alpha 0` 时，该目标退化为 uniform MSE。默认 gate 使用
`--step-loss-alpha 0.5`，给 shared early prefix 更高优化压力。该机制用于判断
mixed-horizon target-set 的瓶颈是否来自 objective/task weighting，而不是 readout capacity 或
target interaction。它不是完整 QDF；若有效，才值得继续设计 covariance-aware objective。

## Artifact 语义

主 run 目录为：

```text
PatchEncoderTargetSetDecoder/{dataset}/mixed_h96_h192_h336_h720/seed{seed}
```

关键文件：

- `metrics_by_target_horizon.csv`: 单模型在每个 requested horizon 上的 MSE/MAE。
- `prefix_consistency.csv`: `pred_len=96/192/336` 与 `pred_len=720` prefix 的 prediction mismatch。
- `h{H}/metrics.json`: 某个 target horizon 的 test MSE/MAE。
- `h{H}/metrics_by_horizon.csv`: step-level MSE/MAE。
- `h{H}/metrics_by_segment.csv`: segment-level MSE/MAE。
- `h{H}/target_state_similarity.csv`: target segment states 的 pairwise cosine。
- `h{H}/target_conditioning_stats.csv`: $\gamma,\beta$ 与 target state norm 统计。
- `h{H}/prefix_residual_stats.csv`: 仅在启用 `--prefix-residual-segments` 时写出，记录
  normalized prefix residual 幅度。
- `h{H}/predictions_test.npz`: 仅在显式传入 `--save-predictions` 时写出，避免 Weather
  等大数据集在 remote gate 中把时间浪费在压缩后又删除的 heavy artifact 上。
- `training_log.csv`: 每个 epoch 的 train loss、各 horizon validation metrics 和 horizon sampling counts。
- `environment.json`: Python、torch、CUDA、device 和 parameter count。
- `effective_config.json`: 记录 `step_loss_weighting`、`step_loss_alpha`、target interaction
  和 prefix residual 等有效训练设置。

## Code-Theory Consistency

[Intended theory] Phase1-R 要验证 horizon 不应只作为训练脚本参数，而应作为 target set
输入模型。目标不是证明 one-model 一定更准，而是测量 amortization gap、prefix consistency
和 target-side state carrier quality。

[Code realization] 当前实现把 requested future positions 变成 future segment features，
通过 cross-attention 生成 target states，再让 target states condition dense history readout。

[Proxy] target set 目前是 segment-level，而不是 every-step token；target-query interaction
第一版采用 independent policy。Phase1-R.2 的 causal target interaction 只放开
past-to-current target-state dependency，仍禁止 later target segment 改写 earlier prefix。

[Repair boundary] prefix residual 是 short-prefix capacity repair，不是新的 paper-core claim。
若它只修复 h96/h192 strict gate 但无法降低整体 mean relative MSE，则它最多说明 target-set
interface 需要 capacity-preserving prefix path，不能单独作为最终创新点。

[Interaction boundary] causal target interaction 是 architecture-level hypothesis，用于检验
first target-set decoder 是否因 independent target queries 而丢失 future-step dependency。
若它只保持 prefix consistency 但 MSE 不改善，下一步应回到 objective-level 或 problem-level
诊断，而不是继续加更深 target interaction 或 MoE。

[Objective boundary] prefix-risk weighted objective 是 diagnostic，不是最终 paper-core claim。
如果它只把误差从 h96 转移到 Weather 或 long horizons，则说明 uniform target-set objective
存在 risk tradeoff，但还不足以支撑最终方法。若它稳定提升 h96、prefix reuse 和 overall MSE，
才进入 covariance-aware objective 或 future-aware/MoE carrier 的下一轮设计。

[Falsification] 如果 mixed target-set model 相比 horizon-specific `PatchEncoderFixedHead`
mean relative MSE 超过 `+1.0%`，或 target states 高度同质，或 prefix consistency 没有改善，
则该 target-set interface 不满足 compatibility pass，应回退到 problem/theory/design，而不是
直接叠加 future-aware 或 MoE。
