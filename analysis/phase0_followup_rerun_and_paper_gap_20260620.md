# Phase0 补跑与论文差距审计

## 1. `SegTSFTDenseFixedHead / Weather / 720` 补跑

### 补跑设置

- [Fact] Remote host: `529_Lab-3090`
- [Fact] Code commit: `0054e33`
- [Fact] GPU: physical GPU 2, via `CUDA_VISIBLE_DEVICES=2`
- [Fact] Start-time GPU 2 memory: 18 MiB used / 24107 MiB free
- [Fact] Dataset root: `/home/yingch/dataset`
- [Fact] Output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase0_rerun_weather720`
- [Fact] Model: `SegTSFTDenseFixedHead`
- [Fact] Dataset: `Weather`
- [Fact] Horizon: `720`
- [Fact] Batch size: default `128`
- [Fact] Max epochs: `100`
- [Fact] Seed: `2021`

### 补跑结果

| Model | Dataset | Horizon | MSE | MAE | Status |
| --- | --- | ---: | ---: | ---: | --- |
| SegTSFTDenseFixedHead | Weather | 720 | 0.337790 | 0.341967 | complete |

[Strong Evidence] 原 gate 中该项 OOM 不是该 setting 在空闲 3090 上必然不可运行导致的。
它在 GPU 2 空闲时以相同 batch size 完整跑通，并写出完整 artifact。

[Strong Evidence] 原 gate 日志显示 OOM 发生时目标进程之外的 GPU memory pressure 很高；
本次补跑启动时 GPU 2 仅 18 MiB used，训练进程约 4.45 GiB used，最终完成。因此更合理的
解释是共享服务器 GPU 占用导致 OOM。

### 补全后的 Phase0 均值

补入该结果后，三个候选均有 12/12 settings：

| Model | Settings | Avg MSE | Avg MAE |
| --- | ---: | ---: | ---: |
| PatchEncoderFixedHead | 12 | 0.316252 | 0.352232 |
| SegTSFTDenseFixedHead | 12 | 0.319124 | 0.353738 |
| DLinear | 12 | 0.354915 | 0.379922 |

[Strong Evidence] 补全后 `PatchEncoderFixedHead` 仍是 Phase0 gate 中平均 MSE 最低的候选。

## 2. 与论文汇报值的差距

### 对照来源

- DLinear: `Are Transformers Effective for Time Series Forecasting?`, Table 2.
  URL: https://ojs.aaai.org/index.php/AAAI/article/view/26317/26089
- PatchTST: `A Time Series is Worth 64 Words`, supervised PatchTST Table 3.
  URL: https://openreview.net/pdf?id=Jbdc0vTOcol
- PatchTST official scripts:
  `PatchTST_supervised/scripts/PatchTST/etth2.sh`,
  `ettm1.sh`, `weather.sh`.
  URL: https://github.com/yuqinie98/PatchTST
- Seg-MoE: `Seg-MoE: Multi-Resolution Segment-wise Mixture-of-Experts for Time Series
  Forecasting Transformers`, Table 1 and configuration Table 8.
  URL: https://arxiv.org/html/2601.21641v1

### DLinear

| Scope | Avg MSE delta vs paper | Avg MAE delta vs paper | Main issue |
| --- | ---: | ---: | --- |
| 12 settings | +1.63% | +0.84% | mostly aligned |

Problematic point:

| Dataset | Horizon | Ours MSE/MAE | Paper MSE/MAE | MSE delta |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | 720 | 0.717386 / 0.602802 | 0.605 / 0.551 | +18.6% |

[Strong Evidence] Most DLinear results are very close to the paper values, especially ETTm1 and
Weather. This suggests dataset split, scaling, and evaluation are largely correct.

[Strong Evidence] Our DLinear implementation is not an exact official replica: official
`DLinear.py` keeps the average-weight initialization commented as a visualization option, while our
implementation initializes seasonal/trend linear weights to `1 / seq_len`. This is a real
architecture/initialization mismatch.

[Hypothesis] The isolated `ETTh2 / 720` degradation is more likely caused by initialization and/or
training-protocol sensitivity than by a broad dataloader bug. A focused control should rerun
DLinear with official random initialization, at least on `ETTh2 / 720`.

### PatchEncoderFixedHead vs PatchTST

| Scope | Avg MSE delta vs supervised PatchTST | Avg MAE delta | Main issue |
| --- | ---: | ---: | --- |
| 12 settings | +4.12% | +2.62% | ETTh2 config mismatch |

Problematic points:

| Dataset | Horizon | Ours MSE/MAE | PatchTST MSE/MAE | MSE delta |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | 0.307448 / 0.366091 | 0.274 / 0.337 | +12.2% |
| ETTh2 | 192 | 0.377340 / 0.406947 | 0.341 / 0.382 | +10.7% |
| ETTh2 | 336 | 0.384115 / 0.421288 | 0.329 / 0.384 | +16.8% |

[Strong Evidence] The ETTh2 gap is explained by hyperparameter mismatch. PatchTST official
`etth2.sh` uses a small ETTh2 configuration:
`d_model=16`, `n_heads=4`, `d_ff=128`, `dropout=0.3`.

[Strong Evidence] Our `PatchEncoderFixedHead` uses the larger unified configuration:
`d_model=128`, `n_heads=16`, `d_ff=256`, `dropout=0.2`.

[Strong Evidence] On ETTm1 and Weather, where our config aligns much more closely with official
PatchTST scripts, the gap is small and sometimes our result is slightly better. This argues against a
general implementation failure.

[Hypothesis] Before using Phase0 results as a paper-facing baseline claim, add an
`official_config` PatchEncoder control for ETTh2. For internal mechanism development, the unified
configuration remains acceptable if it is explicitly documented as an internal base rather than a
PatchTST reproduction.

### SegTSFTDenseFixedHead vs Seg-MoE

| Scope | Avg MSE delta vs Seg-MoE paper | Avg MAE delta | Main issue |
| --- | ---: | ---: | --- |
| 12 settings, including GPU2 rerun | +6.35% | +6.42% | not an exact Seg-MoE replica |

[Strong Evidence] `SegTSFTDenseFixedHead` should not be treated as a Seg-MoE reproduction. It
intentionally removes:

- segment-wise MoE
- token-wise MoE fallback
- router probabilities
- imbalance / balance loss
- one-for-all autoregressive forecasting

[Strong Evidence] Seg-MoE paper also uses training choices that differ materially from ours:
BF16, Huber prediction loss, auxiliary balance loss, AdamW with schedule, and A100 80GB hardware.

[Inference] Therefore, the 6% gap is not surprising and does not by itself falsify the dense TSFT
proxy. It does mean that `SegTSFTDenseFixedHead` cannot be used as a claimed Seg-MoE
reproduction.

## 3. Current decision implication

[Strong Evidence] After Weather-720補跑，`PatchEncoderFixedHead` remains the best internal
Phase0 base by average MSE, but its ETTh2 gap against PatchTST is config-induced and should be
addressed before any paper-facing comparison.

[Strong Evidence] The cleanest next action is not to abandon `PatchEncoderFixedHead`, but to add
two targeted controls:

1. `PatchEncoderFixedHeadOfficialETT`: official ETTh2 small config
   (`d_model=16`, `n_heads=4`, `d_ff=128`, `dropout=0.3`).
2. `DLinearOfficialInit`: remove forced average initialization and match official random
   initialization.

[Speculative] If these controls close the ETTh2 and DLinear-720 gaps, Phase0 can confidently select
`PatchEncoderFixedHead` as the internal base. If not, the remaining gap likely comes from lower-level
implementation differences in encoder/lr schedule/checkpoint selection and should be isolated before
mechanism experiments.
