# Phase5 TimeAlign Carrier Gate Interpretation

## 结论

[Decision] `carrier_viable_but_hss_necessity_not_yet_proven`.

TimeAlign-style carrier 本身有价值，但本轮结果不能直接支撑
“unified multi-horizon 必然退化，所以需要 HSS 修复”的叙事。更准确的判断是：

- TimeAlign unified-720 是一个可继续研究的 strong carrier；
- unified-vs-fixed 没有形成稳定正 gap，整体反而是 unified 略好；
- 当前最明确的问题不是 training schedule，而是 `pred_len=720` 下的 validation/checkpoint
  selection 与 region-specific failure：ETTh2 h720 仍弱，Weather h96 仍弱。

## 关键事实

### Unified-vs-Fixed

[Fact] Unified-720 相对 fixed-horizon：

- ETTh2: `3/4` wins，mean relative MSE `-7.88%`；
- ETTm2: `0/4` wins，mean relative MSE `+1.47%`；
- Weather: `3/4` wins，mean relative MSE `-1.07%`；
- ALL: `6/12` wins，mean relative MSE `-2.49%`。

[Inference] 这不是一个“unified 明显退化”的结果。ETTm2 有轻微 unified penalty，
但 ETTh2 与 Weather 的 unified-720 反而更好。

### 与当前 R.3 reference 的横向关系

[Fact] 在 ETTh2/Weather 上，TimeAlign unified-720 相对 R.3：

| Dataset | Horizon | R.3 MSE | TimeAlign unified MSE | Relative |
| --- | --- | --- | --- | --- |
| ETTh2 | 96 | 0.300315 | 0.251752 | -16.17% |
| ETTh2 | 192 | 0.360485 | 0.302045 | -16.21% |
| ETTh2 | 336 | 0.378096 | 0.337939 | -10.62% |
| ETTh2 | 720 | 0.402909 | 0.426506 | +5.86% |
| Weather | 96 | 0.148917 | 0.151302 | +1.60% |
| Weather | 192 | 0.193101 | 0.193061 | -0.02% |
| Weather | 336 | 0.245888 | 0.242194 | -1.50% |
| Weather | 720 | 0.323368 | 0.311396 | -3.70% |

[Strong Evidence] TimeAlign unified-720 是一个真实 carrier candidate：

- ETTh2 short/mid 明显强于 R.3；
- Weather long 明显强于 R.3；
- 但 ETTh2 h720 与 Weather h96 是新的 failure points。

### Validation / Checkpoint Selection

[Fact] `TimeAlignCarrierFixedH720` 与 `TimeAlignCarrierUnified720` 的训练 loss 在同一 dataset
上完全一致；二者差异来自 validation target set 与 early stopping selector。

[Fact] Weather 上：

- fixed h720 使用 h720-only validation，best epoch 是 `1`；
- unified-720 使用 `h96/h192/h336/h720` mean validation，best epoch 是 `7`；
- unified h720 test MSE 比 fixed h720 低 `-1.67%`。

[Inference] Weather unified 的 h720 改善不能解释为 training schedule 的收益；它首先是
checkpoint selector 改变带来的效果。这个结果提示：Phase5 的下一个问题应包含 validation
protocol，而不是直接设计 HSS loss。

## 研究判断

[Decision] 不应直接进入 “TimeAlign + HSS schedule”。

原因：

1. unified-vs-fixed 没有出现稳定退化，HSS 缺少“修复统一模型”的直接问题支点；
2. TimeAlign unified-720 已经在 ETTh2 short/mid 与 Weather long 展示出强 carrier 信号；
3. 当前失败点是 selective 的：ETTh2 h720 和 Weather h96，而不是所有 horizon；
4. h720 上 fixed/unified 的差异首先来自 validation selector，不能误写成 training 策略贡献。

## 下一步

[Decision] 进入 Phase5-R1：TimeAlign Carrier Validation and Mechanism Control。

最小实验优先级：

1. `TimeAlignUnified720` validation selector audit：
   - 固定训练轨迹，比较 `val_h720`、`val_long_mean`、`val_all_mean` 对 test h720 / all-horizon
     的影响；
   - 目标是确认 unified-720 的收益是否主要来自 checkpoint selection。
2. TimeAlign mechanism ablation：
   - `w_align=0, w_recon=1`；
   - `w_align=0.1, w_recon=0`；
   - full TimeAlign；
   - 目标是确认收益来自 distribution-aware alignment，而不是更强的 patch MLP carrier 或
     reconstruction regularization。
3. 若 R1 通过，再设计 HSS：
   - 不是修复普遍 unified degradation；
   - 而是围绕 ETTh2 h720 / Weather h96 的 region-specific failure，研究哪些 future
     distribution constraints 应该监督、何时监督、以及其梯度应更新哪里。

