# StageC Active Training-Mode Gate

## Purpose

`baselines/timealign_official/models/TimeAlign.py` 仍保留若干历史分支，以便已落盘 checkpoint 与 failure
attribution可审计。但这些分支不应因为 parser 仍可见就成为新研究的隐式候选。

`train_repo.py` 现在默认只允许 StageC frozen carrier contract：

```text
mode=unified
encoder_mode=timealign-token-mlp
readout_mode=learned-basis-forecast-operator
pred_loss_mode=full
```

## Control Flow

`parse_args()` 解析参数后，先把四个会改变主研究语义的字段组成 `active_values`，与
`ACTIVE_STAGE_C_CONTRACT`逐项比较。任一字段不一致时，在 dataset loading、model construction 或 training
之前直接报错，因此 historical encoder/readout/multi-prefix loss不会误入新实验。

若确实需要审计历史 run，必须显式传入 `--allow-archived-research-modes`。该 flag 只解除入口保护，
不等于 active ledger 授权，也不能把归档模式提升为 paper-core candidate。

## Artifact Effect

该参数由现有 effective-config 记录路径随其他 CLI arguments 一起保存。baseline checkpoint evaluator
直接按 frozen checkpoint config构造模型，不经过 training parser，因此 72-row test reference 的加载语义
不受影响。

## Code-Theory Consistency

- intended rule：新 StageC 训练只基于 A6 natural carrier；
- code realization：默认 CLI contract在任何数据读取前拒绝旧模式；
- retained proxy：历史 branch代码仍位于 model file中，仅为 checkpoint/evidence compatibility；
- falsification：若默认 CLI 可以构造非 A6 mode，或 natural baseline evaluator无法加载旧 checkpoint，
  则 gate 实现失败。
