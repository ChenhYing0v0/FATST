# Active Baselines

- `timealign_official/`: StageC active carrier；clean A6 `timealign-token-mlp` Encoder + natural profiles保持
  frozen；A6 readout及PMFO-RCT四个Step7 controls获得活动实现授权，remote training仍由active protocol gate控制；
- `dlinear/`: 简单 external control，后续需按其 native protocol 使用。

其余 local architecture candidates 已移入 `baselines/archive/`。archive 中代码不构成 active method，
也不得绕过 StageC ledger 的 Step 2-6 gate 被重新启动。
