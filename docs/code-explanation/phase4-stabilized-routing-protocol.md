# Phase4 Stabilized Routing Protocol

## Purpose

`phase4_stabilized_routing_gate` 是 RG-B 失败后的 protocol-level 诊断。它不新增模型结构，
而是检验一个更基础的问题：

> 当前 HSS/routing 机制是否失败在 routing 本身，还是失败在 base carrier 从零训练太不稳定。

这个方向来自两个证据：

- RG-B trace 显示 router 没塌缩，但 Weather 仍无法超过 R.3。
- 当前和相邻 Phase4 gates 中，ETTh2/Weather 的 best validation epoch 系统性出现在
  epoch `1-5`，随后 train loss 继续下降而 validation 变差。

## SRP-Informed Protocol

SRP 代码中的 finetune path 使用两阶段思想：

1. 先训练 base model 并保存 checkpoint。
2. finetune 阶段加载 pretrain checkpoint。
3. 冻结 base parameters，只让当前 tuner/group 更新。

本仓库不复制 SRP 的 tuner API 或 pred_len grouping，只吸收 protocol claim：

> route-specific parameters should be trained on top of a stabilized base instead of competing with
> base learning from step zero.

## Local Implementation

`baselines/patch_encoder_target_set_decoder/train.py` 新增两个参数：

- `--init-checkpoint`: 在模型构造后加载一个 checkpoint，使用 `strict=False`。
- `--freeze-non-adapter`: 冻结所有非 `supervision_adapter_head.*` 参数。

当 `dynamic_residual_stability_routing` 使用这两个参数时：

1. `PatchEncoderTargetSetDecoder` 构造 full base + supervision adapter。
2. 从 full-time pretrain checkpoint 加载 shared/base weights。
3. adapter weights 缺失是预期行为，由当前模型初始化。
4. 非 adapter 参数冻结。
5. optimizer 只接收 `requires_grad=True` 的 adapter 参数。
6. training loss 仍调用 `dynamic_residual_stability_routing_loss`，但 dense base loss 不再更新
   shared base，只作为 logging/selection anchor；adapter auxiliary 更新 adapter path。

对应代码位置：

- [baselines/patch_encoder_target_set_decoder/train.py](/Users/river/PaperResearch/Project/R_2026_FATST/baselines/patch_encoder_target_set_decoder/train.py:78)
- [baselines/patch_encoder_target_set_decoder/train.py](/Users/river/PaperResearch/Project/R_2026_FATST/baselines/patch_encoder_target_set_decoder/train.py:1615)
- [baselines/patch_encoder_target_set_decoder/train.py](/Users/river/PaperResearch/Project/R_2026_FATST/baselines/patch_encoder_target_set_decoder/train.py:1781)

## Remote Runner

入口脚本：

`scripts/remote/run_phase4_stabilized_routing_gate.sh`

每个 dataset 顺序执行：

1. `PatchEncoderFullTimeMSE720Pretrain`
   - strategy: `full_time_mse`
   - lr: `1e-4`
   - epochs: `100`
   - patience: `10`
   - 保留 `checkpoint.pt`
2. `PatchEncoderStabilizedDynamicResidualRouting`
   - strategy: `dynamic_residual_stability_routing`
   - init checkpoint: pretrain run 的 `checkpoint.pt`
   - `--freeze-non-adapter`
   - lr: `1e-3`
   - epochs: `30`
   - patience: `5`

默认只跑 `ETTh2` 和 `Weather`，作为 small gate。

## Trace And Audit

`effective_config.json` 记录：

- `init_checkpoint_info.path`;
- `init_checkpoint_info.missing_keys`;
- `init_checkpoint_info.unexpected_keys`;
- `freeze_non_adapter_effective`;
- `supervision_unit_config.adapter_effective_start_step`。

`environment.json` 记录：

- `parameter_count`;
- `trainable_parameter_count`。

这些字段用于确认 finetune 阶段确实只训练 adapter，而不是重新训练整个 base。

## Code-Theory Consistency

[Theory] 如果 current carrier 从零训练快速过拟合，那么 routing 机制可能还没来得及形成有效
adapter before validation selects an early checkpoint。先稳定 base，再训练 route-specific
adapter，应该更符合 SRP-style 的分阶段优化假设。

[Code] 通过 `--init-checkpoint` 和 `--freeze-non-adapter` 实现 base-stabilized adapter-only
finetune，不改变 model forward，不复制 SRP tuner modules。

[Proxy] 当前 pretrain 使用 `full_time_mse`，不是 R.3 或更强 base；因此如果失败，只能说明
full-time stabilized base + RG-B adapter 不够，不等价于否定所有 pretraining。

[Falsification] 如果 stabilized routing 仍无法改善 Weather vs R.3，或 adapter-only finetune
没有显著改变 test metrics，则说明当前 adapter/routing path 容量或 supervision signal 不足，
下一步应回到 carrier design，而不是继续调 finetune lr。
