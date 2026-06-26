# Phase5 TimeAlign Official Gate Report

## Decision

[Decision] `validation_selector_no_clear_unified_gap`.
[Fact] checkpoint_policy = `best-val`.

## Fixed-Horizon Metrics

| model | dataset | target_horizon | mse | mae |
| --- | --- | --- | --- | --- |
| TimeAlignOfficialFixedH96_best-val | ETTh2 | 96 | 0.270189 | 0.331739 |
| TimeAlignOfficialFixedH192_best-val | ETTh2 | 192 | 0.342799 | 0.381270 |
| TimeAlignOfficialFixedH336_best-val | ETTh2 | 336 | 0.374652 | 0.407160 |
| TimeAlignOfficialFixedH720_best-val | ETTh2 | 720 | 0.424630 | 0.448620 |
| TimeAlignOfficialFixedH96_best-val | ETTm2 | 96 | 0.154702 | 0.241409 |
| TimeAlignOfficialFixedH192_best-val | ETTm2 | 192 | 0.214428 | 0.285328 |
| TimeAlignOfficialFixedH336_best-val | ETTm2 | 336 | 0.263705 | 0.317605 |
| TimeAlignOfficialFixedH720_best-val | ETTm2 | 720 | 0.341299 | 0.372189 |
| TimeAlignOfficialFixedH96_best-val | Weather | 96 | 0.141384 | 0.182692 |
| TimeAlignOfficialFixedH192_best-val | Weather | 192 | 0.182212 | 0.220125 |
| TimeAlignOfficialFixedH336_best-val | Weather | 336 | 0.233735 | 0.263523 |
| TimeAlignOfficialFixedH720_best-val | Weather | 720 | 0.304501 | 0.317461 |

## Unified-vs-Fixed Gap

| dataset | target_horizon | fixed_mse | unified_mse | relative_mse_pct | relative_mae_pct | unified_win |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | 0.270189 | 0.250432 | -7.312528 | -3.061743 | True |
| ETTh2 | 192 | 0.342799 | 0.294316 | -14.143194 | -7.212637 | True |
| ETTh2 | 336 | 0.374652 | 0.329485 | -12.055747 | -5.818344 | True |
| ETTh2 | 720 | 0.424630 | 0.424630 | 0.000000 | 0.000000 | False |
| ETTm2 | 96 | 0.154702 | 0.169403 | 9.502713 | 5.653967 | False |
| ETTm2 | 192 | 0.214428 | 0.220913 | 3.024603 | 1.402350 | False |
| ETTm2 | 336 | 0.263705 | 0.269482 | 2.190773 | 1.459777 | False |
| ETTm2 | 720 | 0.341299 | 0.341299 | 0.000000 | 0.000000 | False |
| Weather | 96 | 0.141384 | 0.143570 | 1.546271 | 1.851402 | False |
| Weather | 192 | 0.182212 | 0.185283 | 1.685510 | 3.242004 | False |
| Weather | 336 | 0.233735 | 0.234742 | 0.430928 | 1.393543 | False |
| Weather | 720 | 0.304501 | 0.307318 | 0.924847 | 0.349335 | False |

## Gap Summary

| dataset | settings | unified_wins | mean_relative_mse_pct | mean_relative_mae_pct |
| --- | --- | --- | --- | --- |
| ETTh2 | 4 | 3 | -8.377867 | -4.023181 |
| ETTm2 | 4 | 0 | 3.679522 | 2.129023 |
| Weather | 4 | 0 | 1.146889 | 1.709071 |
| ALL | 12 | 3 | -1.183819 | -0.061695 |

## Checkpoint Diagnostics

| run_name | dataset | epochs_ran | best_epoch | best_val_mean_mse | last_val_mean_mse | last_minus_best_val_mse_pct |
| --- | --- | --- | --- | --- | --- | --- |
| TimeAlignOfficialFixedH96_best-val | ETTh2 | 10 | 6 | 0.217759 | 0.220274 | 1.155146 |
| TimeAlignOfficialFixedH192_best-val | ETTh2 | 10 | 1 | 0.292144 | 0.310520 | 6.289886 |
| TimeAlignOfficialFixedH336_best-val | ETTh2 | 10 | 1 | 0.384403 | 0.444864 | 15.728595 |
| TimeAlignOfficialFixedH720_best-val | ETTh2 | 10 | 1 | 0.629532 | 0.804793 | 27.839838 |
| TimeAlignOfficialUnified720_best-val | ETTh2 | 10 | 1 | 0.406879 | 0.491329 | 20.755807 |
| TimeAlignOfficialFixedH96_best-val | ETTm2 | 10 | 8 | 0.112163 | 0.112332 | 0.150409 |
| TimeAlignOfficialFixedH192_best-val | ETTm2 | 10 | 2 | 0.154591 | 0.156071 | 0.957405 |
| TimeAlignOfficialFixedH336_best-val | ETTm2 | 10 | 6 | 0.200106 | 0.201486 | 0.689594 |
| TimeAlignOfficialFixedH720_best-val | ETTm2 | 10 | 3 | 0.274820 | 0.279392 | 1.663679 |
| TimeAlignOfficialUnified720_best-val | ETTm2 | 10 | 3 | 0.184627 | 0.186479 | 1.003025 |
| TimeAlignOfficialFixedH96_best-val | Weather | 10 | 2 | 0.378200 | 0.381625 | 0.905788 |
| TimeAlignOfficialFixedH192_best-val | Weather | 10 | 9 | 0.438271 | 0.438938 | 0.152285 |
| TimeAlignOfficialFixedH336_best-val | Weather | 10 | 3 | 0.500492 | 0.503928 | 0.686399 |
| TimeAlignOfficialFixedH720_best-val | Weather | 10 | 3 | 0.592020 | 0.596658 | 0.783554 |
| TimeAlignOfficialUnified720_best-val | Weather | 10 | 5 | 0.489505 | 0.490709 | 0.246106 |

## Gate Reading

- [Fact] This baseline vendors official TimeAlign model and dataloader code.
- [Fact] `official-last` is the author-confirmed paper protocol; `best-val` is a validation-selector diagnostic, not a correction of the paper protocol.
- [Decision] If the unified-vs-fixed pattern is stable across both selectors, treat checkpoint policy as a sensitivity factor rather than the main mechanism.
