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

## H1 Prefix-Aware Readout

H0/H0B 后，`train_repo.py` 新增 `--readout-mode`：

- `official`：默认值，保持官方 `proj_x: Linear(d_model * patch_num, pred_len)`；
- `prefix-conditioned-head`：在 `proj_x` 前加入 requested-prefix condition；
- `target-set-decoder`：使用同一 prefix-conditioned readout，但训练时按 target set 中的
  多个 requested prefixes 逐个生成 prediction。

对应的 tensor flow 在 `models/TimeAlign.py` 中保持 backbone 不变：

```text
x_hidden: [B, C, patch_num, d_model]
x_flat:   [B, C, d_model * patch_num]
prefix:   scalar target_prefix / pred_len
cond:     [1, 1, d_model * patch_num]
x_cond:   x_flat + x_flat * tanh(cond)
proj_x:   Linear(d_model * patch_num, pred_len)
output:   [B, pred_len, C]
```

`prefix_condition` 的最后一层 zero-init，因此训练开始时 `x_cond == x_flat`，不会立刻破坏
official head；如果 H1 有收益，收益应来自模型学会按 requested prefix 调整 readout。

对于非 `official` readout，evaluation 不再只用 h720 输出裁剪所有 horizon，而是分别用
`target_prefix=96/192/336/720` forward 后计算对应 prefix metric。这样 H1 的指标真正测试
“模型是否知道当前请求的 horizon”，而不是继续测试 h720 crop。

[Boundary] H1 的 `target-set-decoder` 名称只表示 target-set-conditioned training protocol，
并不是真正的 decoder head。它仍然使用 `proj_x: Linear(...,720)`，只是 projection 前的
hidden 被 requested prefix 调制。

## H1B Variable-Prefix Readout

H1B 新增两个真正改变 prediction head 结构的 `--readout-mode`：

- `target-set-prefix-head`：对 requested prefix 中的每个 future step 生成动态 linear weight，
  直接输出 `[B,H,C]`；
- `prefix-token-decoder`：把 requested future step token 作为 query，对 history patch tokens
  做 attention readout，直接输出 `[B,H,C]`。

`target-set-prefix-head` 的 tensor flow：

```text
x_hidden:  [B, C, patch_num, d_model]
x_flat:    [B, C, d_model * patch_num]
features:  [H, 2] = [step / 720, H / 720]
weights:   MLP(features) -> [H, d_model * patch_num]
output:    einsum(x_flat, weights) -> [B, H, C]
```

`prefix-token-decoder` 的 tensor flow：

```text
x_tokens:  [B, C, patch_num, d_model]
features:  [H, 2] = [step / 720, H / 720]
query:     MLP(features) -> [H, d_model]
key/value: Linear(x_tokens) -> [B, C, patch_num, d_model]
attention: softmax(query @ key) over patch_num
output:    Linear(attended context) -> [B, H, C]
```

这两个模式都不再先生成 `[B,720,C]` 再 crop。训练和评估仍按 requested prefix 循环调用
`model(..., target_prefix=H)`，但模型返回的 prediction length 就是 `H`。

## H1C Capacity-Preserving Prefix Decoder

H1C 保留 TimeAlign 原始 dense 720 projection，把 H1B 的问题从“是否直接生成 variable
prefix”改成“prefix condition 应该调制 dense readout 的哪个增量路径”。新增三个
`--readout-mode`：

- `dense-prefix-residual-adapter`：`proj_x(hidden)` 是 base path，prefix-conditioned low-rank
  residual 只作为 additive delta；
- `row-gated-dense-head`：`proj_x(hidden)` 是 base path，`[step/720, target_prefix/720]`
  生成 720 个 row-wise gate；
- `prefix-adapter-shared-dense`：在 `hidden` 上加入 prefix-conditioned low-rank adapter，
  再复用同一个 `proj_x` 输出 720 steps。

`dense-prefix-residual-adapter` 的 tensor flow：

```text
hidden:     [B, C, d_model * patch_num]
base:       proj_x(hidden) -> [B, C, 720]
condition:  Linear(target_prefix / 720) -> [1, 1, adapter_dim]
adapted:    GELU(residual_down(hidden)) * (1 + condition)
residual:   residual_up(adapted) -> [B, C, 720]
output:     base + residual -> [B, 720, C]
```

`row-gated-dense-head` 的 tensor flow：

```text
hidden:    [B, C, d_model * patch_num]
base:      proj_x(hidden) -> [B, C, 720]
features:  [720, 2] = [step / 720, target_prefix / 720]
gate:      1 + 0.1 * tanh(MLP(features)) -> [720]
output:    base * gate -> [B, 720, C]
```

`prefix-adapter-shared-dense` 的 tensor flow：

```text
hidden:     [B, C, d_model * patch_num]
condition:  Linear(target_prefix / 720) -> [1, 1, adapter_dim]
adapted:    GELU(hidden_adapter_down(hidden)) * (1 + condition)
hidden':    hidden + hidden_adapter_up(adapted)
output:     proj_x(hidden') -> [B, 720, C]
```

三个 H1C arms 的 last projection 或 residual-up 分支均为 zero-init / identity-preserving
设计：训练开始时尽量等价于 official dense head，之后才学习 prefix-aware delta。

## A2 SCI-Level Unified Interface

A2 不再继续 H1C 的 post-hoc residual/gate/adapter sweep，而是测试两个更明确的 unified
prediction interface contract：

- `dense-row-initialized-prefix-decoder`：直接输出 requested prefix `[B,H,C]`，但 base
  weight 使用 `proj_x.weight[:H]` 和 `proj_x.bias[:H]`，避免 H1B random variable head 的
  capacity collapse；
- `nested-segment-decoder`：按 target horizons 构造 nested segments，例如
  `[1:96]`、`[97:192]`、`[193:336]`、`[337:720]`，requested prefix 由共享 segment heads
  拼接得到，而不是完整生成 720 后裁剪。

`dense-row-initialized-prefix-decoder` 的 tensor flow：

```text
hidden:     [B, C, d_model * patch_num]
base rows:  Linear(hidden, proj_x.weight[:H], proj_x.bias[:H]) -> [B, C, H]
condition:  Linear(target_prefix / 720) -> [1, 1, adapter_dim]
delta:      zero-init low-rank delta(hidden, condition)[:, :, :H]
output:     base + delta -> [B, H, C]
```

这个 head 与 H1C 的区别是：H1C 仍完整生成 `[B,720,C]` 再参与 prefix loss，而 A2 直接按
requested prefix 读取 dense rows。它保留 dense-row initialization，但改变 output contract。

`nested-segment-decoder` 的 tensor flow：

```text
hidden:       [B, C, d_model * patch_num]
segment_1:    Linear(hidden) -> [B, C, 96]
segment_2:    Linear(hidden) -> [B, C, 96]
segment_3:    Linear(hidden) -> [B, C, 144]
segment_4:    Linear(hidden) -> [B, C, 384]
output(H):    concat needed segments and crop to H -> [B, H, C]
```

这个 head 直接测试 prefix-consistent / nested output contract：短 horizon 是长 horizon 的
前缀组成部分，而不是从 full 720 output 中被动裁剪。

## A3 Dense-Initialized Nested Interface Repair

A3 保留 A2 `nested-segment-decoder` 的 output contract，只修复一个机制漏洞：A2 的 segment
heads 是随机初始化，可能丢掉 official dense head 已经具备的 row-level readout capacity。

`dense-initialized-nested-segment-decoder` 的 tensor flow：

```text
hidden:       [B, C, d_model * patch_num]
segment_1:    Linear(hidden), initialized from proj_x.weight[0:96]     -> [B, C, 96]
segment_2:    Linear(hidden), initialized from proj_x.weight[96:192]   -> [B, C, 96]
segment_3:    Linear(hidden), initialized from proj_x.weight[192:336]  -> [B, C, 144]
segment_4:    Linear(hidden), initialized from proj_x.weight[336:720]  -> [B, C, 384]
output(H):    concat needed segments and crop to H -> [B, H, C]
```

它和 A2 nested 的 forward 完全一致，区别只在 initialization：每个 segment head 的
`weight` 与 `bias` 从 `proj_x` 对应 row slice 复制。这个实验测试 nested composition 的
收益是否被 A2 的 random segment initialization 掩盖。

注意：当前 A3-1 复制的是模型初始化时的 `proj_x` rows，而不是已训练 full head 或 H1/H1C
checkpoint 的 learned rows。因此它是 shallow initialization repair，不是严格意义上的
learned capacity preservation。若论文主线需要声明 capacity preservation，后续必须使用
teacher consistency、target-conditioned nested readout，或从已训练 checkpoint warm-start。

## A3B Target-Conditioned Nested Residual Interface

A3B 修复 A3-1 的设计错误：不再把随机初始化的 row-copy 当作 learned capacity。它保留
`proj_x` 作为 active dense base path，并只让 nested branch 学 residual。

`target-conditioned-nested-residual-decoder` 的 tensor flow：

```text
hidden:      [B, C, d_model * patch_num]
base_full:   proj_x(hidden) -> [B, C, 720]
base:        base_full[:, :, :H] -> [B, C, H]
condition:   Linear(target_prefix / 720) -> [1, 1, adapter_dim]
adapted:     GELU(nested_residual_down(hidden)) * (1 + condition)
segment_1:   zero-init Linear(adapted) -> [B, C, 96]
segment_2:   zero-init Linear(adapted) -> [B, C, 96]
segment_3:   zero-init Linear(adapted) -> [B, C, 144]
segment_4:   zero-init Linear(adapted) -> [B, C, 384]
residual(H): concat needed segments and crop to H -> [B, C, H]
output:      base + residual(H) -> [B, H, C]
```

Code-theory consistency：

- capacity preservation: true at function level, because zero-init residual makes initial output exactly
  equal to `proj_x(hidden)[:, :, :H]`;
- target condition: `target_prefix / 720` controls the residual path before segment heads;
- nested composition: residual is generated by horizon segments rather than one monolithic 720 adapter;
- limitation: this is still jointly trained dense capacity, not a teacher checkpoint. If A3B fails, the next
  valid repair is teacher consistency or checkpoint warm-start, not another initialization sweep.

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

`scripts/remote/run_phase5_timealign_hss_h1_readout_gate.sh` 运行 H1 readout gate：

- mode: unified only；
- arms:
  - `prefix_conditioned_stochastic_k2`: `readout-mode=prefix-conditioned-head`,
    `pred-loss-mode=stochastic-prefix`, `prefix_samples=2`;
  - `target_set_decoder_multiprefix`: `readout-mode=target-set-decoder`,
    `pred-loss-mode=multi-prefix`;
- datasets: `Weather ETTm2 ETTh2`；
- default GPUs: `0 1 2`；
- output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_h1_readout_gate`。

`scripts/remote/run_phase5_timealign_hss_h1b_variable_readout_gate.sh` 运行 H1B variable-readout gate：

- mode: unified only；
- arms:
  - `target_set_prefix_head_multiprefix`: `readout-mode=target-set-prefix-head`,
    `pred-loss-mode=multi-prefix`;
  - `prefix_token_decoder_multiprefix`: `readout-mode=prefix-token-decoder`,
    `pred-loss-mode=multi-prefix`;
- datasets: `Weather ETTm2 ETTh2`；
- default GPUs: `0 1 2`；
- output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_h1b_variable_readout_gate`。

`scripts/remote/run_phase5_timealign_hss_h1c_capacity_preserving_gate.sh` 运行 H1C
capacity-preserving decoder/head gate：

- mode: unified only；
- arms:
  - `dense_prefix_residual_adapter_multiprefix`: `readout-mode=dense-prefix-residual-adapter`,
    `pred-loss-mode=multi-prefix`;
  - `row_gated_dense_head_multiprefix`: `readout-mode=row-gated-dense-head`,
    `pred-loss-mode=multi-prefix`;
  - `prefix_adapter_shared_dense_multiprefix`: `readout-mode=prefix-adapter-shared-dense`,
    `pred-loss-mode=multi-prefix`;
- datasets: `Weather ETTm2 ETTh2`；
- default GPUs: `0 1 2`；
- output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_h1c_capacity_preserving_gate`。

`scripts/remote/run_phase5_timealign_hss_a2_interface_gate.sh` 运行 A2 unified-interface gate：

- mode: unified only；
- arms:
  - `dense_row_initialized_prefix_decoder_multiprefix`:
    `readout-mode=dense-row-initialized-prefix-decoder`,
    `pred-loss-mode=multi-prefix`;
  - `nested_segment_decoder_multiprefix`: `readout-mode=nested-segment-decoder`,
    `pred-loss-mode=multi-prefix`;
- datasets: `Weather ETTm2 ETTh2`；
- default GPUs: `0 1 2`；
- output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_a2_interface_gate`。

`scripts/remote/run_phase5_timealign_hss_a3_interface_repair.sh` 运行 A3 interface repair gate：

- mode: unified only；
- arm:
  - `dense_initialized_nested_segment_decoder_multiprefix`:
    `readout-mode=dense-initialized-nested-segment-decoder`,
    `pred-loss-mode=multi-prefix`;
- datasets: `Weather ETTm2 ETTh2`；
- default GPUs: `0 1 2`；
- output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_a3_interface_repair`。

`scripts/remote/run_phase5_timealign_hss_a3b_nested_residual_gate.sh` 运行 A3B nested residual gate：

- mode: unified only；
- arm:
  - `target_conditioned_nested_residual_decoder_multiprefix`:
    `readout-mode=target-conditioned-nested-residual-decoder`,
    `pred-loss-mode=multi-prefix`;
- datasets: `Weather ETTm2 ETTh2`；
- default GPUs: `0 1 2`；
- output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_a3b_nested_residual_gate`。

`scripts/remote/run_phase5_timealign_hss_a3c_warm_started_nested_gate.sh` 运行 A3C warm-started
primary nested gate：

- mode: unified only；
- arm:
  - `checkpoint_initialized_nested_segment_decoder_multiprefix`:
    `readout-mode=checkpoint-initialized-nested-segment-decoder`,
    `pred-loss-mode=multi-prefix`;
- warm-start source:
  `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_h1_readout_gate/official-last/TimeAlignOfficialUnified720_H1_target_set_decoder_multiprefix_official-last/{dataset}/mixed_h96_h192_h336_h720/seed2021/checkpoint.pt`；
- datasets: `Weather ETTm2 ETTh2`；
- default GPUs: `0 1 2`；
- output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_a3c_warm_started_nested_gate`。


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

`scripts/analyze_phase5_timealign_hss_h1_readout_gate.py` 输出：

- `phase5_timealign_hss_h1_metrics.csv`;
- `phase5_timealign_hss_h1_comparison.csv`;
- `phase5_timealign_hss_h1_summary.csv`;
- `phase5_timealign_hss_h1_training.csv`;
- `phase5_timealign_hss_h1_best_epoch.csv`;
- `phase5_timealign_hss_h1_readout_gate_report.md`。

该分析同时比较 H1 arms 相对 H0 `full`、H0 `multi-prefix`、H0B
`stochastic_prefix_k2` 与 fixed specialist 的差异。H1 的 primary gate 是 ETTm2 fixed gap
是否缩小，同时不能牺牲 ETTh2 unified benefit 或 Weather no-harm。

`scripts/analyze_phase5_timealign_hss_h1b_variable_readout_gate.py` 输出：

- `phase5_timealign_hss_h1b_metrics.csv`;
- `phase5_timealign_hss_h1b_comparison.csv`;
- `phase5_timealign_hss_h1b_summary.csv`;
- `phase5_timealign_hss_h1b_training.csv`;
- `phase5_timealign_hss_h1b_best_epoch.csv`;
- `phase5_timealign_hss_h1b_variable_readout_gate_report.md`。

该分析比较 H1B arms 相对 H0 `full`、H0B `stochastic_prefix_k2`、H1
`target_set_decoder_multiprefix` 与 fixed specialist 的差异。H1B 的关键不是只超过 H0B，
而是必须超过 H1 target-set conditioned 720 projection，证明真正 decoder/head 改造有额外价值。

`scripts/analyze_phase5_timealign_hss_h1c_capacity_preserving_gate.py` 输出：

- `phase5_timealign_hss_h1c_metrics.csv`;
- `phase5_timealign_hss_h1c_comparison.csv`;
- `phase5_timealign_hss_h1c_summary.csv`;
- `phase5_timealign_hss_h1c_training.csv`;
- `phase5_timealign_hss_h1c_best_epoch.csv`;
- `phase5_timealign_hss_h1c_capacity_preserving_gate_report.md`。

该分析比较 H1C arms 相对 H0 `full`、H0B `stochastic_prefix_k2`、H1
`target_set_decoder_multiprefix` 与 fixed specialist 的差异。H1C 的 primary gate 是能否超过
H1 target-set conditioned 720 projection；仅比 H1B 好不能构成 pass。

`scripts/analyze_phase5_timealign_hss_a2_interface_gate.py` 输出：

- `phase5_timealign_hss_a2_metrics.csv`;
- `phase5_timealign_hss_a2_comparison.csv`;
- `phase5_timealign_hss_a2_summary.csv`;
- `phase5_timealign_hss_a2_training.csv`;
- `phase5_timealign_hss_a2_best_epoch.csv`;
- `phase5_timealign_hss_a2_interface_gate_report.md`。

`scripts/analyze_phase5_timealign_hss_a3_interface_repair.py` 输出：

- `phase5_timealign_hss_a3_metrics.csv`;
- `phase5_timealign_hss_a3_comparison.csv`;
- `phase5_timealign_hss_a3_summary.csv`;
- `phase5_timealign_hss_a3_decision.csv`;
- `phase5_timealign_hss_a3_training.csv`;
- `phase5_timealign_hss_a3_best_epoch.csv`;
- `phase5_timealign_hss_a3_interface_repair_report.md`。

A2 分析比较 A2 arms 相对 H0 `full`、H0B `stochastic_prefix_k2`、H1
`target_set_decoder_multiprefix`、H1C `row_gated_dense_head_multiprefix` 与 fixed specialist
的差异。A3 分析重点比较 dense-initialized nested 相对 A2 nested、H1、H1C 与 fixed
specialist 的差异。A3-1 的 primary gate 是先优于 A2 nested，再判断是否达到 H1/H1C
controls。

`scripts/analyze_phase5_timealign_hss_a3b_nested_residual_gate.py` 输出：

- `phase5_timealign_hss_a3b_metrics.csv`;
- `phase5_timealign_hss_a3b_comparison.csv`;
- `phase5_timealign_hss_a3b_summary.csv`;
- `phase5_timealign_hss_a3b_decision.csv`;
- `phase5_timealign_hss_a3b_training.csv`;
- `phase5_timealign_hss_a3b_best_epoch.csv`;
- `phase5_timealign_hss_a3b_nested_residual_gate_report.md`。

A3B 分析重点比较 target-conditioned nested residual 相对 A2 nested、A3-1 shallow、H1、H1C
与 fixed specialist 的差异。

`scripts/analyze_phase5_timealign_hss_a3c_warm_started_nested_gate.py` 输出：

- `phase5_timealign_hss_a3c_metrics.csv`;
- `phase5_timealign_hss_a3c_comparison.csv`;
- `phase5_timealign_hss_a3c_summary.csv`;
- `phase5_timealign_hss_a3c_training.csv`;
- `phase5_timealign_hss_a3c_best_epoch.csv`;
- `phase5_timealign_hss_a3c_warm_started_nested_gate_report.md`。

A3C 分析重点比较 warm-started primary nested 相对 A2 nested、A3B residual、H1、H1C 与
fixed specialist 的差异。

## A3C Warm-Started Primary Nested Interface

A3C 回到 primary nested prediction interface，不再把 nested 放在 residual path。它从已训练的
H1 `target_set_decoder_multiprefix` checkpoint warm-start：

```text
source checkpoint:
  shared TimeAlign carrier weights
  trained proj_x.weight / proj_x.bias

target model:
  readout_mode = checkpoint-initialized-nested-segment-decoder
  nested heads:
    head_1 <- source proj_x rows [0:96]
    head_2 <- source proj_x rows [96:192]
    head_3 <- source proj_x rows [192:336]
    head_4 <- source proj_x rows [336:720]

forward:
  hidden -> nested segment heads -> concat needed segments -> [B, H, C]
```

Code-theory consistency：

- learned capacity preservation: true as a warm-start diagnostic, because compatible shared weights and
  trained `proj_x` rows come from a completed H1 checkpoint;
- primary nested interface: true, because output is produced directly by nested heads rather than
  `proj_x + residual`;
- limitation: this introduces a two-stage dependency on H1. If it passes, the next question is whether the
  warm-started transfer can be simplified or justified as part of the final training protocol.

## Code-Theory Consistency

[Intended theory] 在设计 HSS 前，必须先确认 TimeAlign fixed-horizon carrier 的官方复现是否可信，
并排除 unified head/interface confounder。如果 unified decrease 只是因为 fixed 720 head
没有直接优化短 prefix，HSS 不能直接写成 future supervision reliability 问题。

[Code realization] 当前代码以官方 dataloader/model/preset 为主体，只加入 repo artifact
导出、unified/fixed 对比入口，以及 adapter-level `pred-loss-mode` diagnostic。默认 `full`
保持官方训练语义；其他 prefix modes 只改变 prediction loss 的 aggregation 或 train-time
prefix sampling。H1 的 `readout-mode` 只在 prediction head 前加入 requested-prefix condition，
不改 encoder、future autoencoder、alignment loss 或 official dataloader。H1B 进一步替换
prediction head，使 selected prefix 直接输出 `[B,H,C]`，但仍保留 TimeAlign backbone、
future autoencoder、alignment loss 与 official dataloader。H1C 回到 capacity-preserving
decoder/head：保留或复用 `proj_x: Linear(...,720)`，只让 prefix condition 控制 residual、gate
或 hidden adapter 这类增量路径。A2 进一步把问题从 post-hoc modulation 改成 output contract：
直接按 requested prefix 读取 dense rows，或用 nested segments 组成 prefix-consistent output。

[Proxy] `official-last` 是 source-faithful proxy，也是作者确认的 paper protocol。
`best-val` 是 validation-selector diagnostic；它不代表论文代码默认行为，也不被视为对
TimeAlign 官方训练策略的修正。

[Falsification] 若 `official-last` fixed-horizon 仍明显偏离论文，下一步应继续审计数据版本、
官方 commit、test path 与 script setting，而不是直接进入 HSS 设计。若 D0 `multi-prefix`
已经解释 unified decrease，则下一步应先研究 unified head/interface，而不是进入 D1/M1。
若 H1C 仍不能超过 H1 target-set conditioned 720 projection，则应回到 Step 2/3，重新判断
当前 post-hoc interface 族，而不是直接否定 interface 主轴或继续堆叠 future reliability schedule。
