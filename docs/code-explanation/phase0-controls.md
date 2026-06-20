# Phase0 Targeted Controls 说明

## 目的

本次更新只服务于 Phase0 结果审计中的两个 targeted controls：

1. `DLinearOfficialInit`
   - 目的：检查 `ETTh2 / 720` 的 DLinear gap 是否来自我们此前的 average
     weight initialization。
   - 代码实现：`baselines/dlinear/model.py` 增加 `init_mode`。
   - `init_mode=average` 保持原 Phase0 行为。
   - `init_mode=pytorch_default` 保持 `nn.Linear` 默认初始化，用作 official-init
     control。

2. `PatchEncoderFixedHeadOfficialETT`
   - 目的：检查 `PatchEncoderFixedHead` 在 ETTh2 上落后 PatchTST paper 的主要原因是否
     是官方 ETTh2 small config mismatch。
   - 代码实现：`baselines/patch_encoder_fixed_head/train.py` 暴露 patch encoder 的关键
     hyperparameters。
   - control runner 使用：
     `d_model=16`, `n_heads=4`, `d_ff=128`, `dropout=0.3`,
     `patch_len=16`, `stride=8`, `encoder_layers=3`。

## 执行入口

`scripts/remote/run_phase0_controls.sh` 在 `529_Lab-3090` 上顺序执行：

- `DLinearOfficialInit × ETTh2 × {96,192,336,720}`
- `PatchEncoderFixedHeadOfficialETT × ETTh2 × {96,192,336,720}`

默认输出：

```text
/home/yingch/exp_outputs/r-2026-fatst/phase0_controls
```

默认日志：

```text
/home/yingch/exp_outputs/r-2026-fatst/phase0_controls/_logs
```

## 数据流影响

[Fact] Dataset split、scaler、loss、early stopping、artifact contract 均不变。

[Fact] `DLinearOfficialInit` 只改变 `Linear` layer initialization，不改变 forward
data flow：

```text
x -> moving average decomposition -> seasonal/trend Linear -> sum -> y_hat
```

[Fact] `PatchEncoderFixedHeadOfficialETT` 只改变 encoder hyperparameters，不改变 forward
data flow：

```text
x -> RevIN -> channel-independent patching -> TransformerEncoder -> flatten head -> RevIN denorm
```

## 一致性评价

[Strong Evidence] 这些 controls 是审计性实验，不是新的候选 backbone。若它们显著缩小论文
差距，说明 Phase0 主实验中观察到的 gap 主要来自 control variable，而不是数据加载或评估
错误。

[Speculative] 如果 controls 仍无法缩小差距，则下一步应检查 optimizer schedule、official
checkpoint selection、Transformer encoder implementation details，以及 seed variance。
