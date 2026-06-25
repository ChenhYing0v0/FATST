# Phase4 Protocol Calibration Gate

本文解释 `scripts/remote/run_phase4_protocol_calibration_gate.sh` 的实验目的和代码路径。

## 目标

前序 Phase4 多个模型都在 epoch `1-5` 达到 best validation，之后 train loss 继续下降但
validation 变差。这个现象会干扰 HSS / HSSG 的机制判断：一个 routing 或 readout 设计失败，
可能是机制不成立，也可能是 shared representation 在过快训练中已经过拟合。

Protocol Calibration Gate 的目标不是提出新结构，而是回答：

> 当前 HSSG 失败是否主要来自 learning rate / early-stopping protocol，而不是 routing
> carrier 本身？

## Runner 行为

`run_phase4_protocol_calibration_gate.sh` 是已有
`run_phase4_hssg_gradient_routing_gate.sh` 的 wrapper。

默认设置：

- learning rates: `0.0001 0.00005 0.00003`;
- strategies:
  - `single_720_prefix_risk`;
  - `r3_prefix_risk`;
  - `hssg_region_routed_readout`;
- datasets: 继承 base runner 默认 `ETTh2 Weather`;
- output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase4_protocol_calibration_gate`;
- 每个 LR 单独写入子目录，例如 `lr_0p00005`。

该 runner 不改模型结构、不改 loss 定义，只改变 learning rate。这样能把 protocol effect
和 mechanism effect 分开。

## 分析重点

远程结果返回后，应检查：

1. `best_epoch` 是否从 `1-5` 后移；
2. validation drift 是否下降；
3. `single_720_prefix_risk` 的 short/prefix gain 是否稳定；
4. HSSG-A 是否在较低 LR 下保住 ETTh2 h96/h192；
5. Weather h720 late segment 是否仍接近 R.3。

## Gate

[Pass] 如果较低 LR 让 HSSG-A 同时保住 h96/h192 并维持 Weather late gain，说明 HSSG 的
下一步应是 protocol-stabilized gradient routing。

[Fail] 如果较低 LR 只能延迟 early-best 但不改善 HSSG-A gate，说明失败主要来自 carrier；
下一步应停止 readout residual path，转向更新 `condition_head/target_states` 的受控子空间。

## Code-Theory Consistency

[Theory] 如果 early-best collapse 是主要瓶颈，降低 LR 应改善 validation trajectory，并给
region-routed path 更多可用训练空间。

[Code] wrapper 只传递 `LEARNING_RATE` 和 `OUTPUT_ROOT`，其余训练逻辑完全复用现有 HSSG
runner。

[Falsification] 如果 LR 降低后 HSSG-A 仍牺牲 short/early 或 Weather late，不能继续把失败
归因于 training protocol，应回到 representation carrier 设计。
