# Phase0 Gate 结果统计

## 数据来源

- [Fact] 远程主机：`529_Lab-3090`
- [Fact] 运行目录：`/home/yingch/projects/FATST`
- [Fact] 运行输出：`artifacts/runs/phase0`
- [Fact] 运行日志：`artifacts/logs/phase0_gate/phase0_gate_20260620_192324.nohup.log`
- [Fact] 实验代码 commit：`01b5baa7bcc5984823fce58acf6874e5c8a5f345`
- [Fact] seed：`2021`
- [Fact] 计划矩阵：3 models × 3 datasets × 4 horizons = 36 runs
- [Fact] 完成矩阵：35/36 runs
- [Fact] 缺失项：`SegTSFTDenseFixedHead / Weather / horizon=720`
- [Fact] 缺失原因：CUDA OOM during `loss.backward()`

完整逐项指标见 `analysis/phase0_gate_metrics_20260620.csv`。

## 共同完成设置上的总体比较

以下统计只使用三个候选都完成的 11 个 setting，排除缺失的
`Weather / horizon=720`，因此是当前最公平的比较口径。

| Model | Settings | Avg MSE | Avg MAE | MSE Wins |
| --- | ---: | ---: | ---: | ---: |
| PatchEncoderFixedHead | 11 | 0.315627 | 0.353756 | 7 |
| SegTSFTDenseFixedHead | 11 | 0.317427 | 0.354808 | 2 |
| DLinear | 11 | 0.357522 | 0.381112 | 2 |

[Strong Evidence] `PatchEncoderFixedHead` 是当前 Phase0 gate 的最稳候选：平均 MSE 最低，
且在 11 个共同 setting 中赢下 7 个。

[Strong Evidence] `SegTSFTDenseFixedHead` 与 `PatchEncoderFixedHead` 很接近，平均 MSE
只高约 0.57%，但它在 `Weather / 720` OOM，说明显存稳定性弱于
`PatchEncoderFixedHead`。

[Strong Evidence] `DLinear` 在少数短 horizon 上仍有竞争力，但长 horizon 特别是
`ETTh2 / 720` 明显落后，不适合作为后续机制实验的主 backbone。

## Dataset 平均表现

共同完成 setting 上按 dataset 分组：

| Dataset | Model | Avg MSE | Avg MAE |
| --- | --- | ---: | ---: |
| ETTh2 | PatchEncoderFixedHead | 0.369077 | 0.409543 |
| ETTh2 | SegTSFTDenseFixedHead | 0.370969 | 0.408505 |
| ETTh2 | DLinear | 0.461071 | 0.459979 |
| ETTm1 | PatchEncoderFixedHead | 0.350626 | 0.382272 |
| ETTm1 | SegTSFTDenseFixedHead | 0.352759 | 0.383558 |
| ETTm1 | DLinear | 0.358244 | 0.380527 |
| Weather | PatchEncoderFixedHead | 0.197694 | 0.241352 |
| Weather | SegTSFTDenseFixedHead | 0.198929 | 0.244878 |
| Weather | DLinear | 0.218494 | 0.276737 |

## 每个 setting 的 MSE winner

| Dataset | Horizon | Winner | MSE | MAE |
| --- | ---: | --- | ---: | ---: |
| ETTh2 | 96 | DLinear | 0.289550 | 0.355002 |
| ETTh2 | 192 | SegTSFTDenseFixedHead | 0.370638 | 0.399872 |
| ETTh2 | 336 | PatchEncoderFixedHead | 0.384115 | 0.421288 |
| ETTh2 | 720 | PatchEncoderFixedHead | 0.407403 | 0.443847 |
| ETTm1 | 96 | PatchEncoderFixedHead | 0.290475 | 0.344233 |
| ETTm1 | 192 | DLinear | 0.335446 | 0.366136 |
| ETTm1 | 336 | PatchEncoderFixedHead | 0.361540 | 0.390765 |
| ETTm1 | 720 | PatchEncoderFixedHead | 0.412788 | 0.420701 |
| Weather | 96 | PatchEncoderFixedHead | 0.147087 | 0.195054 |
| Weather | 192 | PatchEncoderFixedHead | 0.195208 | 0.241885 |
| Weather | 336 | SegTSFTDenseFixedHead | 0.249115 | 0.285867 |

## 初步判断

[Strong Evidence] 如果 Phase0 目标是选择一个性能强、结构简单、显存稳定、适合继续做
Variable-Horizon Decoder 与 future-aware 机制的基础，当前应优先选择
`PatchEncoderFixedHead`。

[Hypothesis] `SegTSFTDenseFixedHead` 可以作为 parallel reference，而不是主 baseline：
它的 dense modern block 在部分 setting 有优势，但整体收益没有显著超过
`PatchEncoderFixedHead`，且 `Weather / 720` 暴露显存风险。

[Speculative] 后续若仍想保留 `SegTSFTDenseFixedHead`，建议单独补跑
`Weather / 720`，使用更小 batch size 或等待 GPU 1 显存完全空闲；但该结果不应混入
当前 strict Phase0 gate，除非明确记录 batch-size protocol change。
