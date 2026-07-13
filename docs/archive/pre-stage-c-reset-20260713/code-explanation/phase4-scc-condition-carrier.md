# Phase4 SCC Condition Carrier

本文解释 Phase4-SCC-E1 中新增的 `condition_delta` carrier 与训练路径。

## 目标

前序 HSSG / adapter experiments 说明 scheduling signal 有局部效果，但 output-side residual
carrier 太弱，尤其无法修复 Weather 相对 R.3 的 gap。SCC-E1 将 routing pressure 从
detached readout / adapter 上移到主预测路径附近：

`target_states -> condition_head -> gamma/beta -> conditioned -> segment_output`

新增策略：

- `scc_condition_delta_detached`：`target_states.detach()` 进入 condition-delta head；
- `scc_condition_delta_state_open`：condition-delta auxiliary gradient 可以回到 target-state path。

## Forward Path

`PatchEncoderTargetSetDecoder` 新增 optional `condition_delta_head`：

1. base path 仍计算 `base_affine = condition_head(target_states)`；
2. condition carrier 计算 `condition_delta_affine = condition_delta_head(delta_source)`；
3. `scc_condition_delta_detached` 中 `delta_source = target_states.detach()`；
4. `scc_condition_delta_state_open` 中 `delta_source = target_states`；
5. `base_affine + condition_delta_affine` 得到最终 `gamma/beta`；
6. `condition_delta_residual` 使用 frozen `segment_output.weight.detach()` 投影到 prediction
   space，避免 auxiliary loss 直接把 pressure 写入 shared segment output。

`condition_delta_head` 最后一层 zero-init，因此初始 `condition_delta_residual = 0`，预测等价于
base path。这保证 SCC-E1 测的是 carrier/update path，而不是随机新增参数扰动。

## Loss Path

`condition_delta_carrier_routing_loss` 包含两部分：

- `time_loss`：对最终 prediction 使用 `prefix_risk`，保持与 single-prefix / R.3 control 的
  objective family 一致；
- `unit_loss`：用 residual-stability selector 找到 learnable-conflict steps，并计算
  `base_prediction.detach() + condition_delta_residual` 上的 masked MSE。

总损失：

$$
\mathcal{L} = \mathcal{L}_{prefix}(pred, y)
  + \lambda \mathcal{L}_{masked}(base\_pred.detach() + \Delta_{cond}, y)
$$

其中 noisy-conflict blocks 不提供额外 positive pressure。

## Diagnostics

训练输出新增：

- `condition_delta_active_steps`;
- `condition_delta_mean_abs_residual`;
- `train_condition_delta_grad_norm`;
- `checkpoint_selection_diagnostics.csv`。

`checkpoint_selection_diagnostics.csv` 记录 official `val_mean_mse` checkpoint 与
short/long/h720 oracle checkpoint 的差距，用于区分 carrier failure 和 checkpoint-selection
sensitivity。

## Code-Theory Consistency

[Theory] 如果 Weather/R.3 gap 来自 output-side carrier 太弱，那么将 routing pressure 上移到
condition/state carrier 应比 detached readout residual 更有效。

[Code] SCC auxiliary loss 只通过 `condition_delta_residual` 承接 learnable-conflict steps；
detached variant 不让 auxiliary pressure 回到 `target_states`，state-open variant 允许它回到
target-state path。

[Falsification] 如果两种 SCC strategy 在 official checkpoint 下仍无法接近 R.3，且 oracle
h720/long views 也无明显收益，则应回 Step 2/3，重新判断 Phase4 是否应转向 future-aware
architecture 或 pretraining。
