# Phase4 Late-Conflict Adapter Routing

## Purpose

`late_conflict_adapter_routing` 是 Phase4-RG 的第一个最小方法候选。它把 HSS 从
loss-only reweight 扩展为 gradient-routing：

> late/conflict supervision 不直接更新 shared readout/head，而是更新 zero-init adapter residual。

## Forward Flow

在 `PatchEncoderTargetSetDecoder.forward` 中：

1. history `x` 经过 RevIN、patch embedding 和 Transformer encoder，得到 `z`。
2. target segment query cross-attend `z`，得到 `target_states`。
3. `history_projector(z)` 得到 `history_readout`。
4. `condition_head(target_states)` 产生 `gamma` / `beta`。
5. shared path:
   `conditioned = history_readout * (1 + gamma) + beta`。
6. base head:
   `base_segment_values = segment_output(conditioned)`。
7. adapter path:
   `adapter_residual = supervision_adapter_head(conditioned.detach())`。
8. adapter residual 只在 `supervision_adapter_start_step` 之后生效，默认 step `337`。
9. final normalized prediction:
   `segment_values = base_segment_values + adapter_residual`。

`conditioned.detach()` 是关键：adapter auxiliary loss 不会通过 adapter path 反向更新 shared
encoder、target path 或 readout projector。

## Training Loss

`late_conflict_adapter_routing_loss` 使用两个预测：

- `base_pred`: 不含 adapter residual，用 full 720 dense MSE 训练 shared base；
- `adapter_pred = base_pred.detach() + adapter_residual`: 只在 late region 上训练 adapter residual。

总 loss 为：

$$
\mathcal{L}
=
\mathcal{L}_{base}(1{:}720)
+
\lambda \mathcal{L}_{adapter}(337{:}720).
$$

其中 $\mathcal{L}_{base}$ 更新 shared path，$\mathcal{L}_{adapter}$ 只更新 adapter head。

## Trace

`supervision_trace.csv` 额外记录：

- `unit_type=late_conflict_adapter`;
- `adapter_start_step`;
- `adapter_active_steps`;
- `adapter_mean_abs_residual`;
- `loss_time`;
- `loss_unit`;
- `loss_total`。

## Code-Theory Consistency

[Theory] Gradient conflict diagnostic 显示 Weather 的 late vs early 在 `readout_head` 与
`all_shared` 上出现明显低 cosine / negative share，因此 late supervision 不应继续直接污染
shared path。

[Code] Dense anchor 训练 base path；late auxiliary 通过 detached base + adapter residual 训练
adapter。

[Proxy] 当前只用固定 late start step `337`，还不是 train-side dynamic predictability router。
这是为了最小验证 gradient destination 是否比 scalar downweight 更有效。

[Falsification] 如果 Weather `late_337_720` segment 不改善，或 ETTh2 正信号丢失，则说明
fixed late adapter route 不是合适 carrier，应回退到 residual-stability proxy 或 dynamic routing。
