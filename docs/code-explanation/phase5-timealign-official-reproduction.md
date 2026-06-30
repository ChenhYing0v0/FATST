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

[Fact] 作者在 GitHub issue #2 中说明，论文结果使用固定训练轮数后的最终模型，而不是
validation-best checkpoint；理由是长时序预测中 validation/test 可能存在 distribution shift，
validation-best 有时会导致训练不足。因此这里把 `official-last` 视为 author-intended
paper protocol，而不是源码 bug。

`train_repo.py` 不直接修补官方源码，而是暴露两个显式 protocol：

- `official-last`：复现官方有效行为，用于 paper-faithful gap audit；
- `best-val`：使用 validation MSE 最优模型，用于 validation-selector diagnostic。

这两个 protocol 必须分开分析。`official-last` 回答“为什么和论文/官方结果不一致”，
`best-val` 回答 unified/fixed 结论是否依赖 validation selector。

## Forward 与 Loss

adapter 调用官方 `TimeAlign.Model(args)`：

```text
batch_x: [B, seq_len, C]
batch_y_future: [B, pred_len, C]
outputs, recon, alignment_loss = model(batch_x, batch_y_future, is_training=True)
```

默认训练 loss 与官方一致：

$$
\mathcal{L}
=
\mathcal{L}^{L1}_{pred}
+ w_{recon}\mathcal{L}^{L1}_{recon}
+ w_{align}\mathcal{L}_{align}.
$$

validation 使用 MSE prefix mean；test 默认使用官方 `is_training=True` 路径。该路径会构建
future reconstruction branch，但 prediction branch 的 `outputs` 只由 history branch 产生。

## D0 Head / Interface Diagnostic

[Fact] 官方 TimeAlign 的 prediction head 是 fixed-length projection：

```text
proj_x: Linear(d_model * patch_num, pred_len)
outputs: [B, pred_len, C]
```

unified-720 评估 `h96/h192/h336/h720` 时，只是对 `[B,720,C]` 的输出做 prefix crop。
这保证 tensor-level prefix consistency，但没有显式 requested-horizon interface。

为排除这个 confounder，`train_repo.py` 新增 `--pred-loss-mode`：

- `full`：默认值，保持官方 full-horizon prediction L1；
- `multi-prefix`：只把 prediction loss 改为 `mean(L_96,L_192,L_336,L_720)`。
- `balanced-step`：把 future axis 切成不重叠区间
  `1:96,97:192,193:336,337:720`，分别计算 loss 后平均；
- `stochastic-prefix`：每个 batch 从 `{96,192,336,720}` 中采样 prefix；
- `continuous-prefix`：每个 batch 从 `32,64,...,720` 这类连续 prefix pool 中采样 prefix。
  当 `prefix_samples` 不超过 pool 大小时，采样为 no-replacement，避免 H0B `k=2`
  退化成同一个 prefix 被重复监督。

这些模式都不修改 official TimeAlign forward，不修改 `recon_loss` 或 `alignment_loss`。
`multi-prefix` 检查 unified decrease 是否来自短 prefix 缺少直接 prediction supervision；
`balanced-step` 检查收益是否只是 region reweight；`stochastic-prefix` 和 `continuous-prefix`
检查 prefix supervision 是否能形成 train-time schedule。

训练日志同步导出：

- `train_prediction_l1`：实际用于反传的 prediction loss；
- `train_prediction_full_l1`：full 720 prediction L1；
- `train_prediction_h{horizon}_l1`：`multi-prefix` 模式下各 prefix 的 L1。
- `train_prediction_seg{start}_{end}_l1`：`balanced-step` 模式下各不重叠区间的 L1。

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

`scripts/remote/run_phase5_timealign_hss_d0_head_gate.sh` 运行 D0 head/interface gate：

- mode: unified only；
- loss modes: `full` 与 `multi-prefix`；
- datasets: `Weather ETTm2 ETTh2`；
- output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_d0_head_gate`。

`scripts/remote/run_phase5_timealign_hss_h0_prefix_gate.sh` 运行 H0 prefix-supervision gate：

- mode: unified only；
- loss modes: `full`, `multi-prefix`, `balanced-step`, `stochastic-prefix`,
  `continuous-prefix`；
- datasets: `Weather ETTm2 ETTh2`；
- default GPUs: `0 1 2`；
- output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_h0_prefix_gate`。

`scripts/remote/run_phase5_timealign_hss_h0b_schedule_gate.sh` 运行 H0B schedule robustness gate：

- mode: unified only；
- arms:
  - `stochastic_prefix_k2`: `stochastic-prefix` with `prefix_samples=2`;
  - `continuous_prefix_k2`: `continuous-prefix` with `prefix_samples=2`,
    `continuous_min_prefix=32`, `continuous_prefix_step=32`;
  - `continuous_prefix_pool96`: `continuous-prefix` with `prefix_samples=1`,
    `continuous_min_prefix=96`, `continuous_prefix_step=96`;
- datasets: `Weather ETTm2 ETTh2`；
- default GPUs: `0 1 2`；
- output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_h0b_schedule_gate`。

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

`scripts/analyze_phase5_timealign_hss_d0_head_gate.py` 输出：

- `phase5_timealign_hss_d0_metrics.csv`;
- `phase5_timealign_hss_d0_multi_prefix_gap.csv`;
- `phase5_timealign_hss_d0_summary.csv`;
- `phase5_timealign_hss_d0_training.csv`;
- `phase5_timealign_hss_d0_best_epoch.csv`;
- `phase5_timealign_hss_d0_head_gate_report.md`。

该分析只比较 `multi-prefix` 相对 `full` 的变化。若它显著缩小 ETTm2/Weather 的 unified
decrease，说明 head/interface confounder 必须先处理；若无效，再进入 D1 supervision
reliability diagnostic。

`scripts/analyze_phase5_timealign_hss_h0_prefix_gate.py` 输出：

- `phase5_timealign_hss_h0_metrics.csv`;
- `phase5_timealign_hss_h0_comparison.csv`;
- `phase5_timealign_hss_h0_summary.csv`;
- `phase5_timealign_hss_h0_training.csv`;
- `phase5_timealign_hss_h0_best_epoch.csv`;
- `phase5_timealign_hss_h0_prefix_gate_report.md`。

该分析同时比较每个 loss mode 相对 `full`、相对 `multi-prefix`、相对 fixed-horizon reference
的变化。H0 的关键不是只看是否超过 full，而是判断 schedule-like modes 是否能接近或超过
`multi-prefix`，从而支撑 horizon-agnostic supervision scheduling 叙事。

`scripts/analyze_phase5_timealign_hss_h0b_schedule_gate.py` 输出：

- `phase5_timealign_hss_h0b_metrics.csv`;
- `phase5_timealign_hss_h0b_comparison.csv`;
- `phase5_timealign_hss_h0b_summary.csv`;
- `phase5_timealign_hss_h0b_training.csv`;
- `phase5_timealign_hss_h0b_best_epoch.csv`;
- `phase5_timealign_hss_h0b_schedule_gate_report.md`。

该分析读取 H0 的 `phase5_timealign_hss_h0_metrics.csv` 作为 reference，并比较 H0B arms
相对 `full`、`multi-prefix`、`stochastic-prefix`、`continuous-prefix` 与 fixed specialist 的差异。

## Code-Theory Consistency

[Intended theory] 在设计 HSS 前，必须先确认 TimeAlign fixed-horizon carrier 的官方复现是否可信，
并排除 unified head/interface confounder。如果 unified decrease 只是因为 fixed 720 head
没有直接优化短 prefix，HSS 不能直接写成 future supervision reliability 问题。

[Code realization] 当前代码以官方 dataloader/model/preset 为主体，只加入 repo artifact
导出、unified/fixed 对比入口，以及 adapter-level `pred-loss-mode` diagnostic。默认 `full`
保持官方训练语义；其他 prefix modes 只改变 prediction loss 的 aggregation 或 train-time
prefix sampling。

[Proxy] `official-last` 是 source-faithful proxy，也是作者确认的 paper protocol。
`best-val` 是 validation-selector diagnostic；它不代表论文代码默认行为，也不被视为对
TimeAlign 官方训练策略的修正。

[Falsification] 若 `official-last` fixed-horizon 仍明显偏离论文，下一步应继续审计数据版本、
官方 commit、test path 与 script setting，而不是直接进入 HSS 设计。若 D0 `multi-prefix`
已经解释 unified decrease，则下一步应先研究 unified head/interface，而不是进入 D1/M1。
