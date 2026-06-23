# Phase2-B Error-Process Decoder 代码说明

`PatchEncoderErrorProcessDecoder` 是 Phase2-B fallback 候选，实现在
`baselines/patch_encoder_target_set_decoder` 中。它不替换 R.1
`PatchEncoderFutureStateAlignmentConfWeighted` 的判断，只是在 R.1 远程结果暂未同步时，
先把下一条可验证路线工程化。

## Forward 数据流

输入 history：

$$
x\in\mathbb{R}^{B\times L\times C}
$$

先经过 RevIN、patch embedding 和 Transformer encoder：

$$
Z=E_\theta(x)\in\mathbb{R}^{(BC)\times N\times d}.
$$

给定 requested horizon $H$，模型按 `segment_len=48` 生成 $J=\lceil H/48\rceil$
个 target segment query，并通过 target-to-history decoder 得到：

$$
U_j=D_\theta(q_j,Z),\quad j=1,\dots,J.
$$

base path 与 R.3 target-set decoder 一致。首先从 history representation 得到 dense
readout：

$$
r=R_\theta(\operatorname{Flatten}(Z)).
$$

每个 target state 生成 FiLM 参数：

$$
(\gamma_j,\beta_j)=F_\theta(U_j).
$$

base normalized segment 为：

$$
\hat{Y}^{base,norm}_{a_j:b_j}
=O_\theta(r\odot(1+\gamma_j)+\beta_j).
$$

## Error-Process Residual

Phase2-B 的新增部分不是 static position basis。它把 residual 视为一个由 target-side
state 驱动的 future process：

$$
c_j=\operatorname{GRUCell}_\theta([U_j,q_j],c_{j-1}),
$$

$$
\Delta Y^{norm}_{a_j:b_j}
=g\cdot O^e_\theta([U_j,q_j,c_j]).
$$

其中 `error_residual_gate_logit` 初始化为 `-4.0`，所以初始 residual 很小；最后一层
`error_residual_head` 也 zero-init，使第一步训练近似 R.3 base path。

最终 normalized prediction 为：

$$
\hat{Y}^{norm}_{1:H}
=
\hat{Y}^{base,norm}_{1:H}
+
\Delta Y^{norm}_{1:H}.
$$

如果启用 RevIN，base prediction 与 final prediction 都在加 residual 后 denormalize；
diagnostic 中的 `error_residual` 则按 history `std` 转回 data scale，因此应满足：

$$
\hat{Y}^{base}+\Delta Y\approx\hat{Y}.
$$

## 关键 Tensor

`return_components=True` 时返回：

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `prediction` | `[B,H,C]` | final denormalized prediction |
| `base_prediction` | `[B,H,C]` | R.3 base path output |
| `error_residual` | `[B,H,C]` | data-scale additive residual |
| `error_residual_norm` | `[B,H,C]` | normalized residual used in loss |
| `error_process_states` | `[B,C,J,d_e]` | recurrent compact error-process states |
| `target_states` | `[B,C,J,d]` | inherited target-set states |
| `prefix_residual_norm` | `[B,H,C]` | inherited prefix residual branch, if enabled |

## Training Loss

主 loss 仍是 prediction MSE，可继续使用 `prefix_risk` step weighting：

$$
\mathcal{L}_{pred}
=
\operatorname{MSE}_w(\hat{Y},Y).
$$

error-process 只加两个轻量 regularizer：

$$
\mathcal{L}_{energy}
=
\|\Delta Y^{norm}\|_2^2,
$$

$$
\mathcal{L}_{smooth}
=
\|\nabla_t^2\Delta Y^{norm}\|_2^2.
$$

默认远程 gate 使用：

- `error_energy_weight=1e-4`;
- `error_smoothness_weight=1e-4`;
- `error_process_dim=64`;
- `error_process_layers=1`.

## Artifact 解释

每个 `h{H}/` evaluation 目录会额外写入 `error_process_stats.csv`：

| Column | Meaning |
| --- | --- |
| `residual_base_mae_ratio` | residual MAE 相对 base prediction MAE 的比例 |
| `residual_energy` | normalized residual energy |
| `residual_second_diff_smoothness` | residual 二阶差分能量 |
| `error_process_state_norm` | compact process state 平均范数 |
| `segment_state_cosine` | 相邻 error-process states 的平均 cosine |
| `base_prediction_mse` | base path MSE |
| `final_prediction_mse` | final prediction MSE |
| `residual_gain_mse_pct` | residual 对 base MSE 的相对变化 |
| `prediction_decomposition_max_abs` | `base + residual - prediction` 的最大绝对误差 |

## 理论一致性检查

[Intended Theory] 当前问题是 horizon-wise output/error process 没有被显式建模，而不是
future latent state 一定需要更强 teacher。

[Code Realization] 代码把 R.3 base prediction 保留为 `base_prediction`，新增 recurrent
state $c_j$ 只负责产生 normalized residual process；residual 在 RevIN denorm 前加到 base
trajectory 上。

[Proxy] 当前 residual process 仍由 MSE 监督间接学习，没有显式建模 full step covariance。
QDF-style covariance objective 暂时只作为后续可能的 objective-level rollback，而不是本轮第一版
loss。

[Falsification] 如果 remote gate 显示 residual gain 只来自 uncontrolled large correction、
prefix mismatch 不再为数值零，或平均 MSE 不能稳定优于 R.3，则该方向不能作为 paper-core。
