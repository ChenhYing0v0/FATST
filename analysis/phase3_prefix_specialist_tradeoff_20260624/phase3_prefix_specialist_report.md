# Phase3-A Prefix Specialist Tradeoff Diagnostic

## Decision

[Decision] diagnostic supports continuing from objective-matrix failure to a prefix/specialist tradeoff analysis.

## Gate

- prefix_identity_pass: `True`
- short_gaps_are_extra_window_effects: `True`
- long_gaps_are_segment_localized: `True`
- supports_tradeoff_diagnostic: `True`

## Short-Horizon Alignment

| Dataset | Horizon | Gap type | Full gap | H720-prefix gap | Full MSE | Aligned MSE | Extra MSE | Extra vs aligned | Pred mismatch |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | no_mse_gap | False | False | 0.304796 | 0.249752 | 0.495421 | 98.365155 | 0.000000 |
| ETTh2 | 192 | no_mse_gap | False | False | 0.369043 | 0.293936 | 0.676445 | 130.133629 | 0.000000 |
| ETTh2 | 336 | h720_prefix_only_gap | False | True | 0.382910 | 0.326393 | 0.700964 | 114.760334 | 0.000000 |
| ETTm1 | 96 | short_extra_window_gap | True | False | 0.298685 | 0.284174 | 0.549860 | 93.494287 | 0.000000 |
| ETTm1 | 192 | no_mse_gap | False | False | 0.329662 | 0.318111 | 0.565950 | 77.909341 | 0.000000 |
| ETTm1 | 336 | no_mse_gap | False | False | 0.360729 | 0.355628 | 0.504195 | 41.775717 | 0.000000 |
| Weather | 96 | short_extra_window_gap | True | False | 0.148026 | 0.147463 | 0.156898 | 6.398844 | 0.000000 |
| Weather | 192 | no_mse_gap | False | False | 0.192409 | 0.190075 | 0.235821 | 24.067320 | 0.000000 |
| Weather | 336 | no_mse_gap | False | False | 0.244793 | 0.242797 | 0.295851 | 21.851122 | 0.000000 |

## H720 Segment Gaps

| Dataset | Segment | Relative MSE vs fixed | Is segment gap |
| --- | --- | ---: | --- |
| ETTh2 | 193-336 | 4.070552 | True |
| ETTh2 | 337-720 | 0.465728 | True |
| ETTm1 | 337-720 | 3.030044 | True |

## Interpretation

[Fact] Prefix prediction and truth alignment are checked on the overlapping `h720`-compatible windows. If mismatch is near zero, standalone short-horizon differences are not caused by inconsistent prediction prefixes on the same inputs.

[Decision Rule] If short-horizon gaps disappear on the `h720`-aligned subset but appear on the short-only extra windows, treat them as coverage/regime effects. If h720 gaps are concentrated in late segments, treat them as long-tail calibration effects.
