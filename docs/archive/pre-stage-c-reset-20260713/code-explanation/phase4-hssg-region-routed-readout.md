# Phase4-HSSG Region-Routed Readout

## 目的

HSSG-A 不是新的 horizon schedule，也不是 R.3 repair。它测试一个更具体的问题：

> h720-only prefix-risk supervision 已经有效；下一步是否应该控制这些 gradients 更新到哪些
> future-region readout subspace？

此前 OP-A 的 adapter-only finetune 失败，说明把可学习路径放在输出 residual adapter 上太晚、
capacity 太弱。HSSG-A 将 region-specific update path 上移到主预测路径的 readout 附近。

## Forward 路径

基础 target-set decoder 的主路径是：

1. `x`: `[B, L, C]`
2. `_encode(x)` 得到 `z`: `[B*C, Np, d_model]`
3. `_target_states(z, pred_len, segment_count)` 得到 `target_states`: `[B*C, S, d_model]`
4. `history_projector(z)` 得到 `history_readout`: `[B*C, readout_dim]`
5. `condition_head(target_states)` 得到 `gamma/beta`: `[B*C, S, readout_dim]`
6. `conditioned = history_readout[:, None, :] * (1 + gamma) + beta`
7. `segment_output(conditioned)` 得到 base segment values: `[B*C, S * segment_len]`

HSSG-A 在第 7 步旁路增加 `region_readout_heads`：

```text
conditioned
  -> region_readout_heads[early]  -> mask 1-96
  -> region_readout_heads[middle] -> mask 97-336
  -> region_readout_heads[late]   -> mask 337-720
```

每个 head 是一个 low-rank segment readout：

```text
LayerNorm(readout_dim)
Linear(readout_dim -> rank)
GELU
Dropout
Linear(rank -> segment_len)
```

最后一层 zero-init，因此初始 forward 等价于原 base。三个 region residual 按 step mask
加到 `segment_values`，得到最终预测。

## Loss 路径

新增 strategy：`hssg_region_routed_readout`。

训练仍然是 h720-only：

```text
train_horizons_effective = [720]
step_loss_weighting = prefix_risk
```

loss 使用同一套 prefix-risk weighted MSE：

```text
weighted_mse_loss(pred, true, max_pred_len, [720], "prefix_risk", alpha)
```

因此它与 `single_720_prefix_risk` 的 training objective 对齐；差异只在于是否存在
region-specific readout update path。

## Audit 输出

训练日志新增：

- `train_region_grad_norm_early`
- `train_region_grad_norm_middle`
- `train_region_grad_norm_late`

`supervision_trace.csv` 新增：

- `region_routed_early_steps`
- `region_routed_middle_steps`
- `region_routed_late_steps`
- `region_routed_mean_abs_residual`

evaluation 阶段若存在 `region_readout_residual`，会写出：

```text
h*/region_readout_stats.csv
```

该文件按 `all`、`1-96`、`97-336`、`337-720` 报告 residual MSE/MAE/max abs。

## Code-Theory Consistency

[Intended Theory] HSS 不只决定 future unit 的 loss weight，也决定该 unit 的 gradient
更新哪些参数子空间。

[Code Realization] 不改变 horizon sampling；只在 readout 主路径旁增加 early/middle/late
三个 region-specific low-rank path。由于每个 path 的 residual 被 step mask 限制，对应 region
的 prediction loss 才会给该 path 提供直接梯度。

[Proxy Boundary] 当前实现仍允许 shared encoder、target states、condition head 和 base
segment output 接收全局 prefix-risk loss 梯度；HSSG-A 只验证 region-specific readout path
是否提供额外可控 capacity，不是完整的梯度隔离理论实现。

[Falsification] 如果 HSSG-A 不优于 `single_720_prefix_risk`，且 audit 显示三个 region path
都收到非零梯度并产生 residual，则说明简单 region readout capacity 不足以解释 R.3 的优势。
如果 audit 显示 path collapse，则先回到初始化、scale 或 rank，而不是直接否定 HSSG。
