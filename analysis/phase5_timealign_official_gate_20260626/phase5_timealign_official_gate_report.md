# Phase5 TimeAlign Official Gate Report

## Decision

[Decision] `source_faithful_reproduction_ready_for_paper_gap_audit`.
[Fact] checkpoint_policy = `official-last`.

## Fixed-Horizon Metrics

| model | dataset | target_horizon | mse | mae |
| --- | --- | --- | --- | --- |
| TimeAlignOfficialFixedH96_official-last | ETTh2 | 96 | 0.269713 | 0.331056 |
| TimeAlignOfficialFixedH192_official-last | ETTh2 | 192 | 0.334599 | 0.374624 |
| TimeAlignOfficialFixedH336_official-last | ETTh2 | 336 | 0.374892 | 0.403989 |
| TimeAlignOfficialFixedH720_official-last | ETTh2 | 720 | 0.403252 | 0.435136 |
| TimeAlignOfficialFixedH96_official-last | ETTm2 | 96 | 0.154927 | 0.241132 |
| TimeAlignOfficialFixedH192_official-last | ETTm2 | 192 | 0.210239 | 0.280657 |
| TimeAlignOfficialFixedH336_official-last | ETTm2 | 336 | 0.263301 | 0.316952 |
| TimeAlignOfficialFixedH720_official-last | ETTm2 | 720 | 0.343566 | 0.371601 |
| TimeAlignOfficialFixedH96_official-last | Weather | 96 | 0.140427 | 0.178288 |
| TimeAlignOfficialFixedH192_official-last | Weather | 192 | 0.182234 | 0.220079 |
| TimeAlignOfficialFixedH336_official-last | Weather | 336 | 0.232655 | 0.262207 |
| TimeAlignOfficialFixedH720_official-last | Weather | 720 | 0.306776 | 0.316634 |

## Unified-vs-Fixed Gap

| dataset | target_horizon | fixed_mse | unified_mse | relative_mse_pct | relative_mae_pct | unified_win |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | 0.269713 | 0.249109 | -7.639189 | -3.427446 | True |
| ETTh2 | 192 | 0.334599 | 0.295961 | -11.547375 | -5.671175 | True |
| ETTh2 | 336 | 0.374892 | 0.326745 | -12.842884 | -6.257417 | True |
| ETTh2 | 720 | 0.403252 | 0.403252 | 0.000000 | 0.000000 | False |
| ETTm2 | 96 | 0.154927 | 0.167614 | 8.189114 | 4.953759 | False |
| ETTm2 | 192 | 0.210239 | 0.219519 | 4.414173 | 2.435117 | False |
| ETTm2 | 336 | 0.263301 | 0.269307 | 2.280787 | 1.100936 | False |
| ETTm2 | 720 | 0.343566 | 0.343566 | 0.000000 | 0.000000 | False |
| Weather | 96 | 0.140427 | 0.143170 | 1.953611 | 3.446695 | False |
| Weather | 192 | 0.182234 | 0.184864 | 1.443203 | 2.581480 | False |
| Weather | 336 | 0.232655 | 0.234520 | 0.801615 | 1.458763 | False |
| Weather | 720 | 0.306776 | 0.306776 | 0.000000 | 0.000000 | False |

## Gap Summary

| dataset | settings | unified_wins | mean_relative_mse_pct | mean_relative_mae_pct |
| --- | --- | --- | --- | --- |
| ETTh2 | 4 | 3 | -8.007362 | -3.839009 |
| ETTm2 | 4 | 0 | 3.721018 | 2.122453 |
| Weather | 4 | 0 | 1.049607 | 1.871735 |
| ALL | 12 | 3 | -1.078912 | 0.051726 |

## Checkpoint Diagnostics

| run_name | dataset | epochs_ran | best_epoch | best_val_mean_mse | last_val_mean_mse | last_minus_best_val_mse_pct |
| --- | --- | --- | --- | --- | --- | --- |
| TimeAlignOfficialFixedH96_official-last | ETTh2 | 10 | 6 | 0.217759 | 0.220274 | 1.155146 |
| TimeAlignOfficialFixedH192_official-last | ETTh2 | 10 | 1 | 0.292144 | 0.310520 | 6.289886 |
| TimeAlignOfficialFixedH336_official-last | ETTh2 | 10 | 1 | 0.384403 | 0.444864 | 15.728595 |
| TimeAlignOfficialFixedH720_official-last | ETTh2 | 10 | 1 | 0.629532 | 0.804793 | 27.839838 |
| TimeAlignOfficialUnified720_official-last | ETTh2 | 10 | 1 | 0.406879 | 0.491329 | 20.755807 |
| TimeAlignOfficialFixedH96_official-last | ETTm2 | 10 | 8 | 0.112163 | 0.112332 | 0.150409 |
| TimeAlignOfficialFixedH192_official-last | ETTm2 | 10 | 2 | 0.154591 | 0.156071 | 0.957405 |
| TimeAlignOfficialFixedH336_official-last | ETTm2 | 10 | 6 | 0.200106 | 0.201486 | 0.689594 |
| TimeAlignOfficialFixedH720_official-last | ETTm2 | 10 | 3 | 0.274820 | 0.279392 | 1.663679 |
| TimeAlignOfficialUnified720_official-last | ETTm2 | 10 | 3 | 0.184627 | 0.186479 | 1.003025 |
| TimeAlignOfficialFixedH96_official-last | Weather | 10 | 2 | 0.378200 | 0.381625 | 0.905788 |
| TimeAlignOfficialFixedH192_official-last | Weather | 10 | 9 | 0.438271 | 0.438938 | 0.152285 |
| TimeAlignOfficialFixedH336_official-last | Weather | 10 | 3 | 0.500492 | 0.503928 | 0.686399 |
| TimeAlignOfficialFixedH720_official-last | Weather | 10 | 3 | 0.592020 | 0.596658 | 0.783554 |
| TimeAlignOfficialUnified720_official-last | Weather | 10 | 5 | 0.489505 | 0.490709 | 0.246106 |

## Gate Reading

- [Fact] This baseline vendors official TimeAlign model and dataloader code.
- [Fact] The `official-last` policy is for source-faithful reproduction, not for corrected research selection.
- [Decision] If fixed-horizon `official-last` still diverges from the paper, audit data version, official commit/version, and script-level hyperparameters before designing HSS on top.
