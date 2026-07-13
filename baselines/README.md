# Active Baselines

- `timealign_official/`: StageC active carrier；仅 clean A6 `timealign-token-mlp` +
  `learned-basis-forecast-operator` path 获得活动实验授权；
- `dlinear/`: 简单 external control，后续需按其 native protocol 使用。

其余 local architecture candidates 已移入 `baselines/archive/`。archive 中代码不构成 active method，
也不得绕过 StageC ledger 的 Step 2-6 gate 被重新启动。
