# Phase0 Targeted Controls 结果

## 运行信息

- [Fact] Remote host: `529_Lab-3090`
- [Fact] Code commit: `4efb0cad339f47e7fd431fe19adc8c7a335f106a`
- [Fact] GPU: physical GPU 2
- [Fact] Output root: `/home/yingch/exp_outputs/r-2026-fatst/phase0_controls`
- [Fact] Log: `/home/yingch/exp_outputs/r-2026-fatst/phase0_controls/_logs/phase0_controls_20260620_230643.nohup.log`
- [Fact] Matrix:
  - `DLinearOfficialInit × ETTh2 × {96,192,336,720}`
  - `PatchEncoderFixedHeadOfficialETT × ETTh2 × {96,192,336,720}`
- [Fact] Status: 8/8 complete

完整逐项结果见 `analysis/phase0_controls_metrics_20260620.csv`。

## DLinearOfficialInit

| Horizon | Phase0 DLinear MSE/MAE | OfficialInit MSE/MAE | Paper MSE/MAE | MSE delta vs paper |
| ---: | ---: | ---: | ---: | ---: |
| 96 | 0.289550 / 0.355002 | 0.286505 / 0.351230 | 0.289 / 0.353 | -0.9% |
| 192 | 0.372272 / 0.410620 | 0.381870 / 0.416856 | 0.383 / 0.418 | -0.3% |
| 336 | 0.465078 / 0.471494 | 0.457791 / 0.467744 | 0.448 / 0.465 | +2.2% |
| 720 | 0.717386 / 0.602802 | 0.705321 / 0.597374 | 0.605 / 0.551 | +16.6% |

[Strong Evidence] Removing the forced average initialization improves `ETTh2 / 720` only slightly
(`0.717386 -> 0.705321`) and does not close the gap to the paper value `0.605`.

[Strong Evidence] The DLinear implementation is broadly aligned for `96/192/336`, but
`ETTh2 / 720` remains anomalous under both initialization modes.

[Inference] The large `ETTh2 / 720` gap is not primarily explained by weight initialization. Next
checks should focus on official script details that are still unmatched: exact upstream training loop,
checkpoint selection, learning-rate schedule if any, and possible variance across seeds.

## PatchEncoderFixedHeadOfficialETT

| Horizon | Phase0 Patch MSE/MAE | OfficialETT MSE/MAE | PatchTST paper MSE/MAE | MSE delta vs paper |
| ---: | ---: | ---: | ---: | ---: |
| 96 | 0.307448 / 0.366091 | 0.300591 / 0.355446 | 0.274 / 0.337 | +9.7% |
| 192 | 0.377340 / 0.406947 | 0.363394 / 0.396532 | 0.341 / 0.382 | +6.6% |
| 336 | 0.384115 / 0.421288 | 0.385106 / 0.415512 | 0.329 / 0.384 | +17.1% |
| 720 | 0.407403 / 0.443847 | 0.405443 / 0.442987 | 0.379 / 0.422 | +7.0% |

[Strong Evidence] Official ETTh2 small hyperparameters improve `h96`, `h192`, and `h720` modestly,
but do not close the PatchTST paper gap. `h336` is essentially unchanged in MSE.

[Strong Evidence] Therefore, the earlier hypothesis that ETTh2 gap is mainly caused by
`d_model/n_heads/d_ff/dropout` mismatch is incomplete.

## PatchTST implementation gap

From the official PatchTST supervised script and backbone:

- [Fact] Official ETTh2 script uses:
  `d_model=16`, `n_heads=4`, `d_ff=128`, `dropout=0.3`, `fc_dropout=0.3`,
  `head_dropout=0`, `patch_len=16`, `stride=8`, `batch_size=128`,
  `learning_rate=0.0001`, `train_epochs=100`.
- [Fact] Official backbone defaults include `res_attention=True` and `pre_norm=False`.
- [Fact] Official backbone uses a custom `TSTEncoderLayer` with `BatchNorm` by default, not the
  PyTorch `TransformerEncoderLayer` used in our clean implementation.
- [Fact] Our `PatchEncoderFixedHead` currently uses PyTorch `TransformerEncoderLayer` with
  `norm_first=True`, i.e. pre-norm LayerNorm behavior.

[Strong Evidence] The remaining PatchTST gap is more plausibly explained by encoder implementation
differences than by top-level hyperparameters alone.

[Inference] This means `PatchEncoderFixedHead` is a clean PatchTST-style internal base, but it is not
an exact PatchTST reproduction. It remains acceptable as an internal research backbone if documented
that way, but paper-facing PatchTST comparison should be run from the official repository or a closer
local reproduction.

## Decision update

[Strong Evidence] These controls do not overturn the Phase0 internal-base decision:
`PatchEncoderFixedHead` remains the best clean internal candidate by the existing Phase0 gate.

[Strong Evidence] They do change the interpretation of paper-gap risk:

1. DLinear `ETTh2 / 720` gap is not fixed by official initialization alone.
2. PatchTST ETTh2 gap is not fixed by official small top-level config alone.
3. Exact paper-facing baselines should not rely on the simplified internal controls.

[Hypothesis] The practical path is:

1. Keep `PatchEncoderFixedHead` as the internal base for Variable-Horizon Decoder experiments.
2. Treat official PatchTST/DLinear results as external comparison baselines reproduced in native
   upstream code before paper claims.
3. Do not spend Phase0 time turning the internal base into exact PatchTST unless later mechanism
   gains are ambiguous and require stricter attribution.
