# B14 Prerequisite Contextual Patch Encoder Gate Report

## Small-Gate Result

| Arm | Mean MSE delta | MSE wins | ETTh2 | ETTm1 | Weather | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cpe_p16s8` | `+4.135%` | `1/12` | `+0.770%` | `+7.395%` | `+3.916%` | `fail` |
| `cpe_p48s24` | `+4.799%` | `0/12` | `+3.344%` | `+4.066%` | `+8.040%` | `fail` |

## Decision

[Decision] No arm passes the pre-registered carrier effectiveness gate.
Legacy A6 remains active；rollback to Step 5/6 failure attribution before B14.

## Failure Attribution

[Strong Evidence] 两个 arms 都没有 numeric pathology，但失败模式不同：`P16-S8` 在 ETTh2 接近
non-inferior，主要由 ETTm1 `+7.40%` 拖累；`P48-S24` 将 ETTm1 缩小至 `+4.07%`，却使 Weather达到
`+8.04%`。因此结果不支持“只要统一 patch count 即可”的假设，也不支持继续调单一 patch scale。

[Strong Evidence] `P16-S8` 在 ETTm1/Weather 有 `3.55M` parameters，并出现 train loss下降而 validation
持续恶化；`P48-S24` 为 `1.58M` parameters，过拟合减轻但 coarse patch损伤 Weather long horizons。

[Decision] 主要归因为 `readout_or_encoder_design_wrong`：full replacement把标准 PatchTST-style history
encoder直接接入 A6 flatten coefficient head，破坏了 accepted A6 carrier path。该结果关闭 exact tested
full contextual replacement，但不否定 canonical patch memory作为 B14 retrieval interface。

[Rollback] 回 Step 5/6。repair 使用 exact function-preserving hierarchical interface：accepted A6 carrier
state保持预测；parameter-free `P48-S24` normalized patches提供 `[B,C,30,48]` local memory。repair只通过
strict state/parameter/output/metric equivalence gate，不再训练另一个 encoder。

## Capacity And Runtime

| Arm | Dataset | P | D | Parameters | Mean epoch seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| `cpe_p16s8` | ETTh2 | 90 | 16 | 571824 | 14.55 |
| `cpe_p16s8` | ETTm1 | 90 | 128 | 3545552 | 130.11 |
| `cpe_p16s8` | Weather | 90 | 128 | 3545552 | 374.16 |
| `cpe_p48s24` | ETTh2 | 30 | 16 | 325616 | 14.45 |
| `cpe_p48s24` | ETTm1 | 30 | 128 | 1575888 | 100.06 |
| `cpe_p48s24` | Weather | 30 | 128 | 1575888 | 129.13 |
