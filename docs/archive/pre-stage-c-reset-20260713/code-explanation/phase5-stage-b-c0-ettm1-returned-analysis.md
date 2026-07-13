# C0 ETTm1 Returned Encoder Control Analysis 代码说明

## Scope

`scripts/analyze_phase5_stage_b_c0_ettm1_carrier_deep.py` 对六臂返回结果补充 protocol、training dynamics 与
disjoint-segment 归因。它不修改模型，也不把 Encoder control 升级为 StageB method。

## Inputs

每个 arm 读取：

- `metrics_{last,best_val}_by_target_horizon.csv`：四个 prefix 的 MSE/MAE；
- `metrics_{last,best_val}_by_segment.csv`：H720 内的 disjoint segment metrics；
- `training_log.csv`：每 epoch train loss 与 validation mean MSE；
- 先前 clean A6 ETTm1 metrics：检查 accepted control 的 exact reproduction。

## Derived outputs

### `c0_protocol_sensitivity.csv`

- `dropout_d02_vs_d09_p1`：固定 P1-D256-F256，只改变 dropout；
- `dropout_d02_vs_d09_p5`：固定 P5-D52-F2048，只改变 dropout；
- `p5_f2048_vs_f256_d09`：固定 P5/dropout，只改变 token MLP `d_ff` capacity；
- `relative_mse_pct=(candidate_mse/baseline_mse-1)*100`，负值代表 candidate 更好；
- `MEAN` 是四个 horizon relative changes 的算术均值。

### `c0_training_dynamics.csv`

- `best_epoch`：`val_mean_mse` 最小 epoch；
- `last_vs_best_val_pct=(last_val/best_val-1)*100`；
- `train_loss_reduction_pct=(last_train/first_train-1)*100`；
- `best_vs_last_test_mean_mse_pct`：四 horizons 上 best-val 相对 last 的 MSE 变化均值。

### `c0_segment_deltas.csv`

只选择 `target_horizon=720`，按 artifact 中的 `[segment_start,segment_end)` 比较 matched P5 与 P1。
该表检查 cumulative prefix MSE 是否被单个 future region 主导。

### `c0_reference_reproduction.csv`

逐 horizon 比较新 C0 P1-D256-F256-drop0.9 official-last 与既有 clean A6 ETTm1 metrics，并报告 MSE/MAE
absolute difference。

## Code-theory consistency

理论目标是区分 patch topology、capacity、dropout 与 selector。代码只做预注册 arms 之间的 direct
contrasts，不跨 seed 构造伪统计显著性，也不把单 seed 结果推广为所有 patchwise architectures 失败。

可证伪边界：若 matched P5 在另一 seed 反向，当前 exact-design rejection 的稳定性会下降；但预注册协议只在
small gate 通过后追加 seeds，而本次四个 dropout-selector settings 全部 `0/4` wins，因此没有授权追加矩阵。
