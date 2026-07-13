# Phase4-HSSG Learnability-Conditioned Region Routing

本文解释 `hssg_learnability_region_routing` 的代码路径。它是在 HSSG-A
`hssg_region_routed_readout` 失败后的 Step 6/7 最小重设计。

## 目标

HSSG-A 证明 region-routed readout 对 late/long region 有信号，但 fixed early/middle/late
path 会牺牲 short/early 优势。新策略保留 HSSG 主线，但把 region routing 从固定 step
region 升级为 learnability-conditioned gradient routing：

> HSS 不只决定 loss weight，也决定哪些 residual-stability learnable blocks 允许更新
> region-specific readout path；noisy/ambiguous blocks 不给 auxiliary pressure。

## Forward 路径

模型仍使用 `PatchEncoderTargetSetDecoder`：

1. `x -> z`：patch encoder 得到 history representation。
2. `target_states = _target_states(z, pred_len, segment_count)`。
3. `history_readout = history_projector(z)`。
4. `conditioned = history_readout[:, None, :] * (1 + gamma) + beta`。
5. shared base path：
   `base_segment_values = segment_output(conditioned)`。
6. region path：
   `region_readout_residual = region_readout_heads(conditioned.detach())`。
7. evaluation prediction：
   `prediction = base_segment_values + region_readout_residual`。

关键差异是 `region_routed_readout_detach_input=True`。因此 auxiliary loss 通过 region
path 反向传播时，只更新 `region_readout_heads.*`，不会沿这条路径更新 shared encoder、
`condition_head` 或 `segment_output`。

## Loss 路径

训练分支在 `learnability_region_routed_readout_loss()` 中完成：

1. `residual_stability_mask_and_stats()` 把 h720 future 切成 `block_size=48` 的 blocks。
2. 每个 block 计算 persistence residual 与 seasonal residual candidates。
3. 高 novelty blocks 中，根据 residual gain、smoothness、local variation 分成：
   - `learnable_blocks`
   - `noisy_blocks`
   - `ambiguous_blocks`
4. shared base loss：

   $$
   \mathcal{L}_{base} = \mathcal{L}_{prefix\_risk}(base\_pred, y)
   $$

5. region auxiliary loss：

   $$
   \mathcal{L}_{region} =
   \mathcal{L}_{masked}(base\_pred.detach() + region\_residual, y; \mathcal{B}_{learnable})
   $$

6. total loss：

   $$
   \mathcal{L} = \mathcal{L}_{base} + \lambda \mathcal{L}_{region}
   $$

这意味着 shared path 继续接受 h720-only prefix-risk pressure，region path 只吸收
learnable residual blocks。noisy/ambiguous blocks 不更新 region path，也不会通过
auxiliary loss 影响 shared path。

## Audit 输出

`supervision_trace.csv` 记录：

- `unit_type=learnability_region_routed_readout`;
- `residual_stability_learnable_blocks`;
- `residual_stability_noisy_blocks`;
- `residual_stability_ambiguous_blocks`;
- `residual_stability_noisy_suppression_ratio`;
- `region_routed_early_steps`;
- `region_routed_middle_steps`;
- `region_routed_late_steps`;
- `region_routed_mean_abs_residual`。

`training_log.csv` 记录：

- `train_region_grad_norm_early`;
- `train_region_grad_norm_middle`;
- `train_region_grad_norm_late`。

evaluation 阶段继续输出每个 target horizon 的 `region_readout_stats.csv`，用于判断 region
path 是否 collapse 或过强。

## Code-Theory Consistency

[Theory] HSSG-B/C 的假设是：short/prefix structure 应主要更新 shared base；late/noisy
future units 需要按 learnability 决定是否进入 region-specific path。

[Code] `base_pred` 的 prefix-risk loss 更新 shared path；`base_pred.detach() + region_residual`
的 masked auxiliary 只更新 detached-input region heads。

[Proxy] residual-stability score 仍是 heuristic proxy，不是真正的不确定性估计。它只利用
persistence/seasonal residual 的 gain 与 smoothness 判断 learnability。

[Falsification] 如果新策略仍只改善 late/long 而牺牲 h96/h192，说明当前 region readout
carrier 本身会干扰 prefix-stable prediction，应回到 Step 4/6 重新设计 carrier；如果
region grad norm 或 residual 近零，则优先回 Step 7 修实现或初始化。
