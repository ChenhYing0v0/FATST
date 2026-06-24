# Phase4-R4.1：Horizon-Decoupled Supervision Protocol

`current_step`: 11-step loop 的 Step 6 complete；允许进入 Step 7 implementation。

## 目标

[Decision] 本 protocol 把 Phase4-R 的研究问题落到可实现接口：

> evaluation horizons 固定为 `96,192,336,720`，training supervision units 不由这些
> horizons 定义。

[Decision] 除 `D1_r3_prefix_risk` baseline 外，所有候选 training strategy 都默认在
`supervision_pred_len=720` 的完整 future sequence 上构造 loss。这样训练侧不再通过
`target_horizons` 采样 horizon；`target_horizons` 只用于 validation、test、prefix consistency
和报告。

## 当前代码事实

[Fact] 现有入口集中在：

- `baselines/patch_encoder_target_set_decoder/train.py`

[Fact] 现有训练流：

1. `--target-horizons` 解析为 evaluation/training 共用的 horizon list；
2. `build_loaders(..., horizons=target_horizons)` 为每个 horizon 建 train/val/test loader；
3. 每个 training step 使用 `rng.choice(target_horizons)` 选一个 horizon；
4. model 以该 horizon 作为 `pred_len` 输出 prediction；
5. `weighted_mse_loss` 按 `step_loss_weighting` 计算 prediction loss；
6. validation/test 对每个 target horizon 输出 metrics；
7. `training_log.csv` 记录 `train_steps_h{horizon}`；
8. `effective_config.json` 记录 `target_horizons` 和 objective stats。

[Inference] 这说明 Phase4-R 的第一处改动不应在 model architecture，而应在 train-side
supervision selection 和 loss construction。

## API 设计

### 新增参数

建议在 `baselines/patch_encoder_target_set_decoder/train.py::parse_args` 中新增：

| 参数 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- |
| `--supervision-strategy` | choice | `horizon_mixed` | training supervision strategy |
| `--supervision-pred-len` | int | `720` | training-side full future length |
| `--supervision-mask-ratio` | float | `0.5` | random mask 激活比例 |
| `--supervision-block-size` | int | `48` | block mask / interval 的基本粒度 |
| `--supervision-interval-min-blocks` | int | `1` | interval 最小 block 数 |
| `--supervision-interval-max-blocks` | int | `4` | interval 最大 block 数 |
| `--supervision-component-rank` | int | `16` | top component 数 |
| `--supervision-component-beta` | float | `0.25` | balanced component 权重强度 |
| `--supervision-component-alpha` | float | `0.5` | component loss 与 time loss 的 mixing |
| `--supervision-curriculum` | choice | `none` | curriculum schedule 名称 |
| `--supervision-trace-limit` | int | `2000` | trace CSV 最多记录多少 training steps |

`--supervision-strategy` 首轮 choices：

| Strategy | 对应候选 | 说明 |
| --- | --- | --- |
| `horizon_mixed` | legacy / compatibility | 保留现有 target-horizon sampling |
| `full_time_mse` | `D0_full_time_mse` | `pred_len=720`，full time-domain MSE |
| `r3_prefix_risk` | `D1_r3_prefix_risk` | `horizon_mixed + prefix_risk` 的显式别名 |
| `random_future_mask` | `D2_random_future_mask` | `pred_len=720`，随机 future position/block mask |
| `interval_supervision` | `D3_interval_supervision` | `pred_len=720`，随机 contiguous future interval |
| `component_basis_top` | `D4_component_basis_top` | `pred_len=720`，监督 train-label top components |
| `component_basis_balanced` | `D5_component_basis_balanced` | `pred_len=720`，component groups balanced pressure |
| `curriculum_units` | `D6_curriculum_units` | `pred_len=720`，从 coarse unit 过渡到 dense time-domain |

### 兼容规则

[Decision] `horizon_mixed` 和 `r3_prefix_risk` 是仅有允许 training loader 使用
`target_horizons` 的策略。它们存在的目的只是复现旧 baseline。

[Decision] `random_future_mask`、`interval_supervision`、`component_basis_top`、
`component_basis_balanced`、`curriculum_units` 必须满足：

- train loader 只使用 `supervision_pred_len`；
- model training forward 只使用 `pred_len=supervision_pred_len`；
- loss 的 active unit 不来自 `target_horizons`；
- validation/test/prefix consistency 仍使用 `target_horizons`。

### 推荐命令语义

R.3 baseline：

```bash
python baselines/patch_encoder_target_set_decoder/train.py \
  --model-variant target_set \
  --target-horizons 96,192,336,720 \
  --supervision-strategy r3_prefix_risk \
  --step-loss-weighting prefix_risk \
  --step-loss-alpha 0.5
```

Horizon-decoupled candidate：

```bash
python baselines/patch_encoder_target_set_decoder/train.py \
  --model-variant target_set \
  --target-horizons 96,192,336,720 \
  --supervision-strategy random_future_mask \
  --supervision-pred-len 720 \
  --supervision-mask-ratio 0.5
```

## 数据流设计

### Loader

现有 loader：

```text
train_loaders: dict[horizon, DataLoader]
val_loaders: dict[horizon, DataLoader]
test_loaders: dict[horizon, DataLoader]
```

Phase4-R 推荐改为：

```text
train_loaders:
  - baseline strategies: dict[target_horizon, DataLoader]
  - horizon-decoupled strategies: {supervision_pred_len: DataLoader}

val_loaders/test_loaders:
  - always dict[target_horizon, DataLoader]
```

这样 evaluation protocol 不变，而训练侧可以完全绕开 benchmark horizon。

### Training step

Baseline path：

```text
horizon = rng.choice(target_horizons)
x, y = next_batch(train_loaders, horizon)
pred = model(x, pred_len=horizon)
loss = weighted_mse_loss(pred, y, ...)
```

Horizon-decoupled path：

```text
x, y = next_batch(train_loaders, supervision_pred_len)
pred = model(x, pred_len=supervision_pred_len)
unit_spec = selector.sample(epoch, step, y, pred)
loss = supervision_loss(pred, y, unit_spec)
```

### Loss family

`full_time_mse`：

$$
\mathcal{L}=\operatorname{mean}((\hat{Y}_{1:720}-Y_{1:720})^2).
$$

`random_future_mask`：

$$
\mathcal{L}=
\frac{\sum_t M_t(\hat{Y}_t-Y_t)^2}{\sum_t M_t+\epsilon}.
$$

`interval_supervision`：

$$
\mathcal{L}=
\operatorname{mean}_{t\in[a,b]}((\hat{Y}_t-Y_t)^2).
$$

`component_basis_top`：

用 train split labels 学习 projection matrix $P_K$：

$$
Z=Y P_K,\qquad \hat{Z}=\hat{Y}P_K.
$$

Loss：

$$
\mathcal{L}=
(1-\alpha)\mathcal{L}_{time}
+
\alpha\operatorname{mean}\left(\frac{(\hat{Z}-Z)^2}{\operatorname{mean}(Z^2)+\epsilon}\right).
$$

`component_basis_balanced`：

使用 train-label eigenvalues $\lambda_k$：

$$
w_k=\operatorname{clip}((\lambda_k+\epsilon)^{-\beta}, w_{min}, w_{max}),
$$

并归一化到平均权重为 1：

$$
\mathcal{L}_{comp}=
\operatorname{mean}_k w_k(\hat{Z}_k-Z_k)^2.
$$

`curriculum_units`：

首轮只定义固定三阶段，不做复杂 scheduler：

| Phase | Epoch range | Strategy |
| --- | --- | --- |
| coarse | first 30% | `component_basis_top` 或 long block interval |
| mixed | middle 40% | `component_basis_balanced` + interval |
| dense | final 30% | `full_time_mse` |

## Trace 格式

新增 `supervision_trace.csv`，最多记录 `--supervision-trace-limit` 个 step。

字段：

| Column | Meaning |
| --- | --- |
| `epoch` | epoch index |
| `step_in_epoch` | epoch 内 step |
| `global_step` | 全局 training step |
| `strategy` | supervision strategy |
| `supervision_pred_len` | train-side pred_len |
| `unit_type` | `full_time`, `mask`, `interval`, `component`, `curriculum` |
| `active_steps` | time-domain active step 数 |
| `mask_ratio` | active step ratio |
| `interval_start` | interval 起点，1-based；非 interval 为 0 |
| `interval_end` | interval 终点，1-based；非 interval 为 0 |
| `component_rank` | active component 数 |
| `curriculum_phase` | curriculum phase |
| `loss_time` | time-domain loss |
| `loss_unit` | unit-specific loss |
| `loss_total` | final loss |

`training_log.csv` 新增 epoch-level 汇总：

- `train_supervision_strategy`;
- `train_supervision_pred_len`;
- `train_unit_loss`;
- `train_time_loss`;
- `train_active_step_ratio`;
- `train_component_rank`;
- `train_curriculum_phase`。

`effective_config.json` 新增：

- `supervision_strategy`;
- `supervision_pred_len`;
- `supervision_unit_config`;
- `training_evaluation_decoupled`;
- `evaluation_target_horizons`。

## 最小触达文件

Step 7 implementation 预计只触达：

| 文件 | 用途 |
| --- | --- |
| `baselines/patch_encoder_target_set_decoder/train.py` | args、loader、selector、loss、trace、effective_config |
| `docs/code-explanation/phase4-horizon-decoupled-supervision.md` | 代码解释，若实现发生 |
| `scripts/remote/run_phase4_horizon_decoupled_gate.sh` | remote runner |
| `scripts/remote/check_phase4_horizon_decoupled_progress.sh` | progress checker |
| `scripts/sync_phase4_horizon_decoupled_results.sh` | result sync |
| `scripts/analyze_phase4_horizon_decoupled_gate.py` | Step 9 分析脚本，远程结果返回后再写 |

[Decision] 第一轮不修改 `model.py`，不新增 architecture class，不修改 dataset split。

## 本地 smoke 设计

本地 smoke 只验证机制激活，不声称性能。

建议命令：

```bash
conda run -n r2026-fsa python baselines/patch_encoder_target_set_decoder/train.py \
  --dataset ETTh2 \
  --target-horizons 96,192,336,720 \
  --supervision-strategy random_future_mask \
  --supervision-pred-len 720 \
  --epochs 1 \
  --steps-per-epoch 2 \
  --max-eval-batches 1 \
  --run-name SmokePhase4RandomFutureMask \
  --output-root artifacts/runs/smoke_phase4_horizon_decoupled
```

Smoke pass 条件：

1. `effective_config.json` 中 `training_evaluation_decoupled=true`；
2. `target_horizons=[96,192,336,720]`；
3. `supervision_pred_len=720`；
4. `supervision_trace.csv` 存在；
5. trace 中 `unit_type` 与 strategy 匹配；
6. `metrics_by_target_horizon.csv` 仍覆盖 `96,192,336,720`；
7. `prefix_consistency.csv` 仍存在且可解析。

## 远程 gate 设计

首轮远程候选：

| Run name | Strategy |
| --- | --- |
| `PatchEncoderFullTimeMSE720` | `full_time_mse` |
| `PatchEncoderR3PrefixRisk` | `r3_prefix_risk` |
| `PatchEncoderRandomFutureMask` | `random_future_mask` |
| `PatchEncoderIntervalSupervision` | `interval_supervision` |
| `PatchEncoderComponentTop` | `component_basis_top` |
| `PatchEncoderComponentBalanced` | `component_basis_balanced` |
| `PatchEncoderCurriculumUnits` | `curriculum_units` |

Datasets：

- `ETTh2`;
- `ETTm1`;
- `Weather`。

Evaluation：

- `96`;
- `192`;
- `336`;
- `720`。

Remote output root：

```text
/home/yingch/exp_outputs/r-2026-fatst/phase4_horizon_decoupled
```

## Gate

Phase4-R4.1 通过条件：

1. protocol 明确 training/evaluation 解耦；
2. implementation touch surface 明确且不包含 architecture；
3. 每个 candidate strategy 都有 unit、loss、trace、smoke 条件；
4. Step 7 可以按本文直接实现；
5. 未把 training unit 重新定义为 `96,192,336,720` 的 horizon subset。

[Decision] 本 protocol 满足 R4.1 通过条件。R4.2 local implementation 已完成，下一步进入
R4.3 remote gate。

## 11-step Record

| Field | Content |
| --- | --- |
| `current_step` | Step 6 complete |
| `problem` | training supervision units should not be fixed to evaluation horizons |
| `existence_evidence` | R.3 objective pressure, reduced/full horizon control, label-basis audit, TransDF/QDF/ElasTST evidence |
| `idea` | horizon-decoupled future supervision units |
| `theory_check` | define $\mathcal{U}$ independent from $\mathcal{H}_{eval}$ and optimize over sampled units |
| `design` | add strategy selector, full-720 train loader, unit losses, trace, unchanged evaluation |
| `gate` | protocol must specify API, loss, trace, smoke, remote matrix, and no architecture change |
| `artifacts` | this document |
| `decision` | pass; proceed to Step 7 implementation |
