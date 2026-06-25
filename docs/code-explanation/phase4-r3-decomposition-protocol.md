# Phase4 R.3 Decomposition Protocol

## 目的

本次更新不是提出新的 paper-core mechanism，而是为 Phase4 主线提供一个必要的
control protocol。此前 R.3 被误解为“简单 step weighting”，但代码路径显示它同时包含：

1. `horizon_mixed` 训练暴露：训练时从 `{96, 192, 336, 720}` 中采样一个 horizon；
2. `prefix_risk` step loss weighting：早期 future steps 获得更高 loss pressure；
3. 同一模型在所有 target horizons 上 evaluation。

因此，若不拆分 R.3，任何 HSS 改进都无法判断是在输给 mixed-horizon exposure、
prefix-risk weighting，还是输给两者的组合。

## Forward/Loss 路径

### `single_720_prefix_risk`

入口位于 `horizon_decoupled_supervision_loss`：

1. `pred`: shape `[B, 720, C]`，来自 target-set decoder；
2. `true`: shape `[B, 720, C]`，来自 `supervision_pred_len=720` 的训练 loader；
3. 调用 `weighted_mse_loss(pred, true, max_pred_len, [720], "prefix_risk", alpha)`；
4. `weighted_mse_loss` 构造 step weights：
   $w_t \propto (t / H_{\max})^{-\alpha}$，并按 `Hmax=720` 的均值归一化；
5. 返回 `unit_loss` 作为 `pred_loss`，但不启用 mixed-horizon sampling。

这个 strategy 的训练 horizon exposure 是单一 h720，因此它隔离了
`prefix_risk` weighting 本身。

### `horizon_mixed`

`horizon_mixed` 已存在于原训练代码中：

1. `horizon_mixed_training=True`；
2. 每个 training step 从 `target_horizons` 中随机采样一个 horizon；
3. 使用 `weighted_mse_loss(..., mode="uniform")`；
4. `train_steps_h96/h192/h336/h720` 记录实际采样次数。

这个 strategy 隔离 mixed-horizon exposure，不引入 prefix-risk weighting。

### `r3_prefix_risk`

`r3_prefix_risk` 是 compound reference：

1. `horizon_mixed_training=True`；
2. 训练 horizon 从 `{96, 192, 336, 720}` 中采样；
3. loss 使用 `prefix_risk`；
4. 因此它同时包含 exposure 和 weighting 两个因素。

## 远程实验入口

`scripts/remote/run_phase4_r3_decomposition_gate.sh` 默认运行：

| Strategy | Run name | 作用 |
| --- | --- | --- |
| `full_time_mse` | `PatchEncoderFullTimeMSE720` | h720-only uniform baseline |
| `horizon_mixed` | `PatchEncoderHorizonMixedUniform` | mixed-horizon exposure-only |
| `single_720_prefix_risk` | `PatchEncoderSingle720PrefixRisk` | prefix-risk weighting-only |
| `r3_prefix_risk` | `PatchEncoderR3PrefixRisk` | compound reference |

默认数据集为 `ETTh2 Weather`，默认输出根目录为：

`/home/yingch/exp_outputs/r-2026-fatst/phase4_r3_decomposition_gate`

## Code-Theory Consistency

[Intended Theory] R.3 的强势应被拆成可解释因素，而不是作为黑盒 baseline。

[Code Realization] `single_720_prefix_risk` 保持 h720-only training，只替换 step loss；
`horizon_mixed` 保持 uniform loss，只改变训练 horizon exposure；`r3_prefix_risk` 保留为组合参照。

[Proxy Boundary] 该 protocol 仍然沿用当前 target-set carrier，不能证明最终 HSS 架构可行；
它只能回答 R.3 的优势主要来自 exposure、weighting，还是二者交互。

[Falsification] 如果 `horizon_mixed` 已接近 R.3，则 R.3 优势主要是 exposure；
如果 `single_720_prefix_risk` 接近 R.3，则主要是 weighting；如果二者都弱而 R.3 强，
则说明存在 interaction，需要在同等 compound protocol 下重新评估 HSS/gradient-routing。
