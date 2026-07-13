# StageC SC0-R1 Training-Control Code Explanation

## 1. Purpose And Boundary

SC0-R1 修复的是 standardized research carrier 的 training/checkpoint protocol，不是模型创新。SC0 已证明
fixed 20 epochs 在 ETTh2 三臂上产生 31.63%-44.95% 的 validation degradation；本次改动增加显式、
opt-in 的 validation-controlled stopping，并用完整三臂三 seed calibration 代替逻辑不足的
selected-arm-only confirmation。

历史 runner 不传 `--enable-early-stopping` 时，`train_repo.py` 仍执行原固定 epoch 语义。

## 2. Training Flow

每个 epoch 的主 tensor flow不变：

1. `batch_x: [B,720,C]` 进入 TimeAlign-derived token MLP encoder；
2. `hidden: [B,C,P,D]` 展平为 `[B,C,1536]`；
3. learned-basis operator生成 `prediction: [B,720,C]`；
4. full-720 L1只反传一次；
5. validation loader计算 full-720 MSE `val_mean_mse`。

新增 control flow发生在 validation之后：

- `early_stopping_update()` 比较 `val_mean_mse < best_val - min_delta`；
- 改善时复制当前 `model.state_dict()` 为CPU `best_state`，并清零counter；
- 未改善时counter加一；达到统一`patience=5`后终止trajectory；
- `checkpoint_policy=best-val` 将`best_state` strict load回模型，再导出validation metrics。

`last_state` 仍表示停止触发时的 terminal state，但SC0-R1不把它作为部署checkpoint或selector。

## 3. Effective Config And Training Log

`effective_config.json -> adapter` 新增/保留以下可审计字段：

- `enable_early_stopping`: 是否启用新控制流；
- `patience`: 连续未改善epoch上限；
- `early_stopping_min_delta`: 判定改善所需最小差值；
- `checkpoint_policy`: SC0-R1固定为`best-val`；
- `profile_hash`: 完整SC0-R1 JSON的SHA256。

`training_log.csv` 每个epoch新增：

- `early_stopping_enabled`：0/1开关；
- `early_stopping_patience`和`early_stopping_min_delta`：有效规则；
- `best_epoch_so_far`：当前best state来源epoch；
- `epochs_without_improvement`：patience counter；
- `stop_triggered`：该epoch是否触发终止。

这些字段只描述训练trajectory，不参与模型输入或prediction。

## 4. Multi-Seed Analyzer

`analyze_stage_c_sc0_r1_carrier_calibration.py`要求完整
`3 datasets × 3 arms × 3 seeds = 27 runs`。主要输出为：

- `sc0_r1_run_diagnostics.csv`：每个run的配置、参数、epoch与停止状态；
- `sc0_r1_validation_horizon_metrics.csv`：来源于validation prediction的dense-horizon MSE/MAE；
- `sc0_r1_seed_selection.csv`：每个seed内、每个arm的跨dataset normalized regret；
- `sc0_r1_aggregate_selection.csv`：pooled-mean与median-seed两种聚合selector；
- `sc0_r1_summary.json`：最终gate与frozen profile decision。

normalized regret仍以同seed、同dataset的最佳arm为分母。通过条件是：mean/median selector一致；selected
arm至少赢2/3 seeds；pooled每dataset regret不超过3%；任一seed-dataset regret不超过5%；全程不读取
test artifacts。

## 5. Code-Theory Consistency

- Intended theory：统一超参数应是统一的**规则**，而不是强迫不同dataset在同一epoch停止；同一
  validation-controlled rule可在不同dataset产生不同realized epoch，同时保持attribution可比。
- Code realization：所有dataset/arm/seed共用max20、patience5、min-delta0、best-val selector和同一hash。
- Proxy boundary：patience5来自旧seed2021 trajectory的离线gate，尚不能证明新seed稳定。
- Falsification：若27-run gate中mean/median selector不一致、winner少于2 seeds，或regret越界，则
  SC0-R1失败并回滚Step 2/3；不得逐dataset改变patience或epoch。

## 6. Verification

本地验证覆盖：

- patience synthetic state transition；
- SC0三臂结构、active/unused parameter与prefix consistency；
- synthetic 27-run analyzer complete/pass路径；
- ETTh2 one-batch CPU trajectory在`patience=1`下确定于epoch 2停止，并恢复epoch 1 best；
- validation-only predictions存在，test predictions不存在；
- legacy behavior由默认`enable_early_stopping=false`保持。
