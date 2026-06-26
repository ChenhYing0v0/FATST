# Phase5 Official TimeAlign Interpretation

## 结论摘要

[Decision] `official_source_carrier_valid_need_selector_sensitivity_check`.

`official-last` 矩阵已经完成，说明 official-source TimeAlign carrier 的 fixed-horizon
表现明显强于上一版 repo-local implementation。作者在 GitHub issue #2 中确认论文使用
fixed training epochs 后的 final model，而不是 validation-best checkpoint。因此
`official-last` 应视为 author-intended paper protocol，不应被称为源码错误。

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
TimeAlign-style future alignment 在 unified setting 下存在 dataset-dependent behavior；
ETTm2/Weather 显示 multi-horizon pressure 伤害短中期，而 ETTh2 显示 unified-720 对短中期
反而有益。该分裂需要通过 `best-val` diagnostic 检查是否由 selector 造成。

## Checkpoint Selector Diagnostic Need

`official-last` 是 source-faithful reproduction，也是作者确认的 paper protocol。尽管如此，
ETTh2 的 last-vs-best validation gap 很大，因此需要 `best-val` 作为 validation-selector
diagnostic，而不是作为对 TimeAlign 官方策略的 correction：

| Run | Best epoch | Last-vs-best val MSE |
| --- | ---: | ---: |
| ETTh2 fixed h96 | 6 | +1.16% |
| ETTh2 fixed h192 | 1 | +6.29% |
| ETTh2 fixed h336 | 1 | +15.73% |
| ETTh2 fixed h720 | 1 | +27.84% |
| ETTh2 unified720 | 1 | +20.76% |

相比之下，ETTm2 和 Weather 的 last-vs-best validation gap 大多小于 `2%`。

[Fact] 该 diagnostic 的目的不是替代论文 protocol，而是判断 unified/fixed 结论是否依赖 selector。

## Decision

当前证据支持：

1. [Pass] official-source TimeAlign carrier 比 repo-local carrier 更可信；
2. [Pass] ETTm2/Weather 存在 unified degradation signal；
3. [Hold] ETTh2 与 ETTm2/Weather 的结论方向相反，不能建立 global unified-degradation 叙事；
4. [Next] 必须运行 `best-val` validation-selector diagnostic。

如果 `best-val` 后 winner pattern 仍一致，则 checkpoint selector 不是主因，下一步应把研究问题
改成 dataset/state-dependent future alignment scheduling，或先做 look-back horizon sweep 对齐论文复现口径。

如果 `best-val` 消除 unified degradation，则 TimeAlign 本身可能已经能处理 unified prefix
evaluation，HSS 必要性不足，应回 Step 2/3 寻找新的 carrier 或研究问题。
