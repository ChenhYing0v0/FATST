# Phase5 TimeAlign-HSS D0 Interpretation

## Decision

[Decision] `head_interface_confounder_strong`.

D0 通过：`multi-prefix` prediction loss 在 `3 datasets x 4 horizons` 上全部优于
official unified-full loss，并显著缩小了 ETTm2/Weather 相对 fixed-horizon specialist 的
unified decrease。因此，当前不能直接把 TimeAlign unified decrease 主要解释为
future alignment / reconstruction supervision reliability 问题。

## Direct D0 Result

| dataset | settings | multi-prefix wins | mean relative MSE vs full | mean relative MAE vs full |
| --- | --- | --- | --- | --- |
| ETTh2 | 4 | 4/4 | -3.36% | -1.71% |
| ETTm2 | 4 | 4/4 | -1.57% | -0.98% |
| Weather | 4 | 4/4 | -1.17% | -1.06% |
| ALL | 12 | 12/12 | -2.03% | -1.25% |

[Strong Evidence] 这不是单一 horizon 或单一 dataset 的偶然改善。`multi-prefix` 在全部
12 个 setting 上降低 MSE。

## Fixed-Reference Reading

| dataset | full unified gap vs fixed | multi-prefix gap vs fixed | gap improvement | multi-prefix beats fixed |
| --- | --- | --- | --- | --- |
| ETTh2 | -8.01% | -11.05% | +3.05 pct-pt | 4/4 |
| ETTm2 | +3.72% | +2.06% | +1.66 pct-pt | 1/4 |
| Weather | +1.05% | -0.13% | +1.18 pct-pt | 2/4 |
| ALL | -1.08% | -3.04% | +1.96 pct-pt | 7/12 |

[Fact] Weather 的平均 gap 已从 `+1.05%` 改为 `-0.13%`，即从整体弱于 fixed 变为基本打平并
略优于 fixed。

[Fact] ETTm2 仍弱于 fixed，但 gap 从 `+3.72%` 缩小到 `+2.06%`。这说明 head/interface
不是 ETTm2 的完整解释，但已经解释了相当一部分退化。

[Fact] ETTh2 不仅没有被 multi-prefix 损伤，反而从 `-8.01%` 进一步提升到 `-11.05%`。
这满足 D0 gate 中“不能丢失 ETTh2 unified benefit”的要求。

## Training-Dynamics Reading

| dataset | full best epoch | multi-prefix best epoch | full last-best gap | multi-prefix last-best gap |
| --- | --- | --- | --- | --- |
| ETTh2 | 1 | 1 | +20.76% | +13.64% |
| ETTm2 | 3 | 3 | +1.00% | +1.68% |
| Weather | 5 | 7 | +0.25% | +0.07% |

[Inference] `multi-prefix` 同时改善 test metrics 与部分 validation dynamics：ETTh2 的
post-best drift 从 `20.76%` 降到 `13.64%`，Weather best epoch 从 `5` 后移到 `7` 且 drift
更低。它不是单纯最后 epoch 噪声。

[Counter-Evidence] ETTh2 仍然 epoch 1 最优，说明 early-best 问题没有被完全解决；ETTm2 的
last-best gap 略增，说明 `multi-prefix` 也不是完整 training protocol 修复。

## Research Implication

[Decision] 下一步不应直接进入 D1/M1，把问题写成“future supervision reliability scheduling”。
更合理的 rollback 是 Step 4/6：先设计一个 TimeAlign-compatible unified head/interface carrier。

新的主问题应表述为：

> TimeAlign 的 future-aware carrier 在 fixed-horizon forecasting 中很强，但其 fixed-length
> output head 并没有为 unified multi-horizon interface 设计。HSS 的第一层问题是让 unified
> prediction interface 接收与 evaluation-consistent 的 prefix supervision；第二层问题才是
> 调度 future reconstruction/alignment supervision 的 reliability 与 gradient path。

## Next Plan

1. 将 D0 从 diagnostic 升级为 `H0` carrier redesign 起点：`Prefix-Supervised TimeAlign`。
2. 第一轮方法不要改 architecture：保留 official TimeAlign head，正式化 `multi-prefix`
   prediction loss，并做 `best-val` 或 seed sensitivity 小检查。
3. 第二轮方法再考虑轻量 unified head/interface，例如 prefix-aware readout 或 target-set readout，
   但必须保持 TimeAlign future branch 与 alignment mechanism 可对照。
4. D1 supervision reliability diagnostic 延后：只有当 unified head/interface control 后仍存在
   ETTm2/Weather residual gap，才进入 D1/M1/M2。
