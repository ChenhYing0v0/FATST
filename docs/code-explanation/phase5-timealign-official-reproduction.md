# Phase5 TimeAlign Official Reproduction 代码说明

## 研究定位

这一步不是继续改 HSS 方法，而是回到 11-step loop 的 Step 1/6/7/8：
审计 TimeAlign carrier 的 source-faithfulness。原因是前一版 `timealign_carrier`
属于 source-informed local implementation，fixed-horizon 结果与论文表格存在明显差距。
在判断 HSS 是否成立之前，必须先排除 repo 实现、dataloader、checkpoint policy 等错位。

## 代码边界

`baselines/timealign_official/` vendored 官方 TimeAlign 代码：

- `models/TimeAlign.py`：官方模型 forward；
- `layers/`：官方 alignment、embedding、normalization；
- `data_provider/`：官方 ETT/Weather dataloader；
- `utils/`：官方 metric、LR schedule、工具函数；
- `scripts/`：官方 shell settings，用作 dataset/horizon preset 来源。

新增的 FATST 适配只在 `train_repo.py`：

- 解析 repo-local dataset root；
- 按官方脚本设置 dataset+horizon preset；
- 统一导出 `metrics_by_target_horizon.csv`、`metrics_by_segment.csv`、`training_log.csv`；
- 支持 fixed-horizon 与 unified-720 对比；
- 显式记录 checkpoint policy。

## Dataloader 兼容补丁

官方 `data_provider/data_loader.py` 只做两个运行兼容补丁：

1. `sktime` 改成 optional import，因为 ETT/Weather dataloader 不使用它；
2. `df_stamp.drop(['date'], 1)` 改成 `df_stamp.drop(columns=['date'])`，以兼容新版 pandas。

这两个补丁不改变 split、scaling、window construction 或 target tensor。

## Checkpoint Protocol

[Fact] 官方 `EarlyStopping` 的实际比较逻辑被注释，`__call__()` 每个 epoch 都保存 checkpoint。
官方 `train()` 末尾没有 reload best checkpoint，`test()` 的 checkpoint load 也被注释。因此官方默认
路径实际是 last-epoch evaluation。

`train_repo.py` 不直接修补官方源码，而是暴露两个显式 protocol：

- `official-last`：复现官方有效行为，用于 paper-faithful gap audit；
- `best-val`：使用 validation MSE 最优模型，用于 corrected research control。

这两个 protocol 必须分开分析。`official-last` 回答“为什么和论文/官方结果不一致”，
`best-val` 才回答“作为我们的 HSS carrier 是否合理”。

## Forward 与 Loss

adapter 调用官方 `TimeAlign.Model(args)`：

```text
batch_x: [B, seq_len, C]
batch_y_future: [B, pred_len, C]
outputs, recon, alignment_loss = model(batch_x, batch_y_future, is_training=True)
```

训练 loss 与官方一致：

$$
\mathcal{L}
=
\mathcal{L}^{L1}_{pred}
+ w_{recon}\mathcal{L}^{L1}_{recon}
+ w_{align}\mathcal{L}_{align}.
$$

validation 使用 MSE prefix mean；test 默认使用官方 `is_training=True` 路径。该路径会构建
future reconstruction branch，但 prediction branch 的 `outputs` 只由 history branch 产生。

## Official Preset

`train_repo.py` 按官方脚本记录不同 dataset/horizon 的差异：

- `ETTh2`: `d_model=32`, `d_ff=32`, `lr=0.0005`, `dropout=0.1`, `patch_num=48`;
- `ETTm2 h96/h192`: `d_model=128`, `d_ff=128`, `dropout=0.3`, `patch_num=12`;
- `ETTm2 h336/h720`: 同上但 `dropout=0.9`;
- `Weather h96/h192/h336`: `d_model=128`, `d_ff=256`, `dropout=0.1`,
  `local_margin=0.5`, `layer_norm=0`;
- `Weather h720`: `d_ff=128`, `dropout=0.5`。

unified-720 使用 h720 official preset，并在 test 时评估 `h96/h192/h336/h720` prefix。

## Runner

`scripts/remote/run_phase5_timealign_official_gate.sh` 默认运行：

- datasets: `Weather ETTm2 ETTh2`;
- fixed-horizon runs: `h96/h192/h336/h720`;
- unified run: `pred_len=720`，`target_horizons=96,192,336,720`;
- checkpoint policy: `official-last`;
- output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_official_gate`;
- GPUs: `1 2`;
- workload-aware queue：优先排 Weather/ETTm2，GPU 空出即补下一个 job。

## Analysis

`scripts/analyze_phase5_timealign_official_gate.py` 输出：

- `phase5_timealign_official_fixed_metrics.csv`;
- `phase5_timealign_official_unified_metrics.csv`;
- `phase5_timealign_official_unified_gap.csv`;
- `phase5_timealign_official_unified_gap_summary.csv`;
- `phase5_timealign_official_training_summary.csv`;
- `phase5_timealign_official_gate_report.md`。

其中 `training_summary` 会记录 `best_epoch`、`best_val_mean_mse`、
`last_val_mean_mse`、`last_minus_best_val_mse_pct`，用于量化 official-last
相对 best-val 的影响。

## Code-Theory Consistency

[Intended theory] 在设计 HSS 前，必须先确认 TimeAlign fixed-horizon carrier 的官方复现是否可信。
如果 fixed-horizon 自身不能接近官方/论文表现，HSS 的失败或成功都无法归因。

[Code realization] 当前代码以官方 dataloader/model/loss/preset 为主体，只加入 repo artifact
导出和 unified/fixed 对比入口。

[Proxy] `official-last` 是 source-faithful proxy；它不代表最合理训练 protocol。
`best-val` 是 corrected protocol；它不代表论文代码默认行为。

[Falsification] 若 `official-last` fixed-horizon 仍明显偏离论文，下一步应继续审计数据版本、
官方 commit、test path 与 script setting，而不是直接进入 HSS 设计。
