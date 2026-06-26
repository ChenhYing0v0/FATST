# Phase5 Official TimeAlign Interpretation

## 结论摘要

[Decision] `official_source_carrier_valid_but_checkpoint_policy_blocks_hss_claim`.

`official-last` 矩阵已经完成，说明 official-source TimeAlign carrier 的 fixed-horizon
表现明显强于上一版 repo-local implementation。但该结果还不能直接进入 HSS 设计，因为
ETTh2 的 last-epoch checkpoint policy 对 validation 产生了严重污染。

## Fixed-Horizon 复现质量

相对上一版 `baselines/timealign_carrier`，official-source fixed-horizon 结果整体更强：

| Dataset | Horizon | Local MSE | Official MSE | Relative MSE |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | 0.277606 | 0.269713 | -2.84% |
| ETTh2 | 192 | 0.346087 | 0.334599 | -3.32% |
| ETTh2 | 336 | 0.373400 | 0.374892 | +0.40% |
| ETTh2 | 720 | 0.426396 | 0.403252 | -5.43% |
| ETTm2 | 96 | 0.161239 | 0.154927 | -3.91% |
| ETTm2 | 192 | 0.215832 | 0.210239 | -2.59% |
| ETTm2 | 336 | 0.270883 | 0.263301 | -2.80% |
| ETTm2 | 720 | 0.347223 | 0.343566 | -1.05% |
| Weather | 96 | 0.149646 | 0.140427 | -6.16% |
| Weather | 192 | 0.198690 | 0.182234 | -8.28% |
| Weather | 336 | 0.244323 | 0.232655 | -4.78% |
| Weather | 720 | 0.316670 | 0.306776 | -3.12% |

[Strong Evidence] 之前 fixed-horizon 与论文表现差距较大，至少部分来自 repo-local 实现、
dataloader 或 official preset mismatch，而不是 TimeAlign 机制本身失败。

## Unified-vs-Fixed

| Dataset | Unified wins | Mean relative MSE | Mean relative MAE |
| --- | ---: | ---: | ---: |
| ETTh2 | 3/4 | -8.01% | -3.84% |
| ETTm2 | 0/4 | +3.72% | +2.12% |
| Weather | 0/4 | +1.05% | +1.87% |
| ALL | 3/12 | -1.08% | +0.05% |

[Fact] ETTm2 和 Weather 出现稳定 unified degradation；ETTh2 则相反，unified-720
在 h96/h192/h336 显著优于 fixed。

[Inference] 不能把问题简单写成 “unified multi-horizon 一定退化”。更准确的研究问题是：
TimeAlign-style future alignment 在 unified setting 下存在 dataset-dependent
checkpoint/protocol interaction；ETTm2/Weather 显示 multi-horizon pressure 伤害短中期，
但 ETTh2 的 official-last 结果可能被 checkpoint policy 放大或扭曲。

## Checkpoint Policy Risk

`official-last` 是 source-faithful reproduction，但不是可靠研究 protocol。ETTh2 的
last-vs-best validation gap 很大：

| Run | Best epoch | Last-vs-best val MSE |
| --- | ---: | ---: |
| ETTh2 fixed h96 | 6 | +1.16% |
| ETTh2 fixed h192 | 1 | +6.29% |
| ETTh2 fixed h336 | 1 | +15.73% |
| ETTh2 fixed h720 | 1 | +27.84% |
| ETTh2 unified720 | 1 | +20.76% |

相比之下，ETTm2 和 Weather 的 last-vs-best validation gap 大多小于 `2%`。

[Strong Evidence] ETTh2 的 unified advantage 不能直接作为 HSS 反证，因为 fixed 和 unified
都在 official-last 下严重偏离最佳 validation epoch，而且偏离幅度不一致。

## Decision

当前证据支持：

1. [Pass] official-source TimeAlign carrier 比 repo-local carrier 更可信；
2. [Pass] ETTm2/Weather 存在 unified degradation signal；
3. [Fail] 不能在 `official-last` 上做最终 HSS necessity claim；
4. [Next] 必须运行 `best-val` corrected control。

如果 `best-val` 后 ETTm2/Weather 仍退化，且 ETTh2 不再呈现由 checkpoint artifact 导致的反向结论，
则可以进入 TimeAlign-HSS：研究 horizon-agnostic supervision scheduling 如何调度
future reconstruction/alignment 的梯度压力。

如果 `best-val` 消除 unified degradation，则 TimeAlign 本身可能已经能处理 unified prefix
evaluation，HSS 必要性不足，应回 Step 2/3 寻找新的 carrier 或研究问题。
