# Phase5 TimeAlign Carrier Gate Report

## Decision

[Decision] `no_clear_unified_gap_do_not_build_hss_yet`.

## Fixed-Horizon Metrics

| model | dataset | target_horizon | mse | mae |
| --- | --- | --- | --- | --- |
| TimeAlignCarrierFixedH96 | ETTh2 | 96 | 0.277606 | 0.333740 |
| TimeAlignCarrierFixedH192 | ETTh2 | 192 | 0.346087 | 0.376934 |
| TimeAlignCarrierFixedH336 | ETTh2 | 336 | 0.373400 | 0.402138 |
| TimeAlignCarrierFixedH720 | ETTh2 | 720 | 0.426396 | 0.450357 |
| TimeAlignCarrierFixedH96 | ETTm2 | 96 | 0.161239 | 0.247563 |
| TimeAlignCarrierFixedH192 | ETTm2 | 192 | 0.215832 | 0.284503 |
| TimeAlignCarrierFixedH336 | ETTm2 | 336 | 0.270883 | 0.320817 |
| TimeAlignCarrierFixedH720 | ETTm2 | 720 | 0.347223 | 0.375293 |
| TimeAlignCarrierFixedH96 | Weather | 96 | 0.149646 | 0.190094 |
| TimeAlignCarrierFixedH192 | Weather | 192 | 0.198690 | 0.238119 |
| TimeAlignCarrierFixedH336 | Weather | 336 | 0.244323 | 0.272599 |
| TimeAlignCarrierFixedH720 | Weather | 720 | 0.316670 | 0.326938 |

## Unified-vs-Fixed Gap

| dataset | target_horizon | fixed_mse | unified_mse | relative_mse_pct | relative_mae_pct | unified_win |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | 0.277606 | 0.251752 | -9.313227 | -3.743835 | True |
| ETTh2 | 192 | 0.346087 | 0.302045 | -12.725667 | -5.657504 | True |
| ETTh2 | 336 | 0.373400 | 0.337939 | -9.496745 | -3.957787 | True |
| ETTh2 | 720 | 0.426396 | 0.426506 | 0.025756 | -0.493480 | False |
| ETTm2 | 96 | 0.161239 | 0.166864 | 3.488613 | 0.895938 | False |
| ETTm2 | 192 | 0.215832 | 0.220812 | 2.307343 | 1.109799 | False |
| ETTm2 | 336 | 0.270883 | 0.271073 | 0.070005 | 0.447122 | False |
| ETTm2 | 720 | 0.347223 | 0.347223 | 0.000000 | 0.000000 | False |
| Weather | 96 | 0.149646 | 0.151302 | 1.106978 | 2.652587 | False |
| Weather | 192 | 0.198690 | 0.193061 | -2.833068 | -1.355976 | True |
| Weather | 336 | 0.244323 | 0.242194 | -0.871055 | 0.400649 | True |
| Weather | 720 | 0.316670 | 0.311396 | -1.665519 | -1.654706 | True |

## Gap Summary

| dataset | settings | unified_wins | mean_relative_mse_pct | mean_relative_mae_pct |
| --- | --- | --- | --- | --- |
| ETTh2 | 4 | 3 | -7.877471 | -3.463151 |
| ETTm2 | 4 | 0 | 1.466490 | 0.613215 |
| Weather | 4 | 3 | -1.065666 | 0.010638 |
| ALL | 12 | 6 | -2.492215 | -0.946433 |

## Gate Reading

- [Fact] This gate tests whether a TimeAlign-style carrier is viable before adding HSS.
- [Fact] A positive unified gap is useful only if fixed-horizon TimeAlign is a reasonable carrier.
- [Decision] If the fixed carrier is weak, do not build HSS on it; first repair or reject the carrier.
