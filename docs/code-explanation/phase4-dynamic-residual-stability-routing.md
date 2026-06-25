# Phase4 Dynamic Residual-Stability Routing

## Purpose

`dynamic_residual_stability_routing` 是 Phase4-RG-B 的最小 training strategy。它把 HSS 从
loss-only reweight 和 fixed late adapter 升级为 dynamic gradient routing：

> horizon-agnostic supervision scheduling 不只决定 future unit 的 loss pressure，
> 还决定该 unit 的 gradient 是否允许进入 detached adapter path。

本版本只验证最核心的 routing 机制。它不引入 future-aware module，也不改变 evaluation
horizons。

## Forward Flow

模型复用 `PatchEncoderTargetSetDecoder` 已有的 supervision adapter：

1. shared target-set decoder 产生 `base_prediction`。
2. `supervision_adapter_head(conditioned.detach())` 产生 `adapter_residual`。
3. 对 `dynamic_residual_stability_routing`，adapter 的 effective start step 强制为 `1`；
   fixed late adapter 的 `337` mask 不再生效。
4. training loss 使用两个路径：
   - dense base path: `base_prediction` 对 full 720 steps 做 MSE；
   - adapter path: `base_prediction.detach() + adapter_residual` 只在 learnable-conflict
     blocks 上做 auxiliary MSE。

由于 adapter 输入是 detached，adapter auxiliary 不会通过该路径更新 shared encoder、
target path、condition head 或 base readout。

对应代码位置：

- [baselines/patch_encoder_target_set_decoder/model.py](/Users/river/PaperResearch/Project/R_2026_FATST/baselines/patch_encoder_target_set_decoder/model.py:535)
- [baselines/patch_encoder_target_set_decoder/train.py](/Users/river/PaperResearch/Project/R_2026_FATST/baselines/patch_encoder_target_set_decoder/train.py:808)

## Routing Rule

`dynamic_residual_stability_routing_loss` 在每个 training batch 内执行：

1. 将 `pred_len=720` 分成 48-step blocks。
2. 对每个 block 计算 persistence residual 和 seasonal residual，默认 seasonal periods 为
   `24,48,96,168`。
3. 按 `novelty_mse` 选择 top `25%` high-novelty blocks。
4. 只在 selected blocks 内计算 batch-relative thresholds：
   - `gain_threshold`: selected `best_gain_over_persistence` 的 60% quantile proxy；
   - `smooth_threshold`: selected `residual_smoothness` median；
   - `variation_threshold`: selected `local_variation` median。
5. 分桶：
   - `learnable_conflict`: high gain and low smoothness；
   - `noisy_conflict`: high smoothness and high local variation；
   - `ambiguous_conflict`: 其余 selected blocks。
6. 如果某个 batch 没有 learnable block，选择 `gain / (1 + smoothness)` 最高的 selected block
   作为 fallback，避免 adapter auxiliary 完全不训练。

对应代码位置：

- [baselines/patch_encoder_target_set_decoder/train.py](/Users/river/PaperResearch/Project/R_2026_FATST/baselines/patch_encoder_target_set_decoder/train.py:808)

## Training Loss

总 loss 为：

$$
\mathcal{L}
=
\mathcal{L}_{base}(1{:}720)
+
\lambda \mathcal{L}_{adapter}(\mathcal{B}_{learnable}).
$$

其中：

- $\mathcal{L}_{base}$ 使用 full dense MSE，只训练 shared base path；
- $\mathcal{L}_{adapter}$ 使用 `base_prediction.detach() + adapter_residual`；
- $\mathcal{B}_{learnable}$ 是当前 batch 中 residual-stability router 判定为 learnable 的 blocks；
- noisy/ambiguous blocks 不进入 adapter auxiliary。

这个版本没有下调 full dense base loss，因此它与 S2 的 scalar downweight 保持清晰区别。

## Trace

`supervision_trace.csv` 新增字段：

- `residual_stability_learnable_blocks`;
- `residual_stability_noisy_blocks`;
- `residual_stability_ambiguous_blocks`;
- `residual_stability_mean_gain`;
- `residual_stability_mean_smoothness`;
- `residual_stability_mean_variation`;
- `residual_stability_noisy_suppression_ratio`;
- `adapter_active_steps`;
- `adapter_mean_abs_residual`。

这些字段用于 gate 检查：bucket 不能塌缩为单一类别，adapter residual 不能长期为零，
noisy suppression ratio 应能解释 Weather 与 ETTh2 的差异。

## Remote Runner

入口脚本：

`scripts/remote/run_phase4_dynamic_residual_stability_gate.sh`

默认 small gate：

- strategies:
  `dynamic_residual_stability_routing`, `full_time_mse`, `r3_prefix_risk`;
- datasets:
  `ETTh2`, `Weather`;
- target horizons:
  `96,192,336,720`;
- output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase4_dynamic_residual_stability_gate`。

## Code-Theory Consistency

[Theory] RG-A 失败说明 fixed late adapter route 太粗；residual-stability diagnostic 显示
Weather late 同时包含 learnable-conflict 和 noisy-conflict。合理的下一步不是修 adapter
强度，而是让 supervision unit 自己决定 gradient destination。

[Code] 本实现保留 dense base 的 full 720 MSE，只把 batch 内判定为 learnable 的 selected
blocks 路由到 detached adapter auxiliary。noisy blocks 不接 adapter pressure。

[Proxy] 当前 routing thresholds 是 batch-relative，可能比 offline diagnostic 的
dataset-relative quantiles 更噪声化。fallback learnable block 也可能让少数 noisy batch
仍训练 adapter。

[Falsification] 如果 Weather vs R.3 仍是 `0/4` collapse，或者 trace 显示 noisy/learnable
bucket 无法稳定分离，则说明 residual-stability proxy 不能作为在线 router，应回退 Step 5
重新设计 threshold calibration 或 state-conditioned gradient blocking。
