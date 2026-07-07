# Phase5 StageB Unified Prefix Optimization Diagnostic

## Decision

[Decision] `prefix_imbalance_problem_candidate`.
[Current Step] StageB Step 2/3 problem-existence diagnostic after clean A6 validation.
[Candidate] `B7-UPO`: unified prefix optimization / nested-prefix supervision imbalance.

## Why This Is Not B6-PLO

B6 asked whether label/residual structure requires a learned-basis or frequency-space objective. This diagnostic asks a different question: whether the current unified multi-prefix loss over-weights short prefix steps and under-weights long-tail steps because nested horizons are averaged as tasks.

## Prefix Supervision Weight

For horizons `[96, 192, 336, 720]`, the current loss is the mean of per-prefix mean losses. A step `t` receives weight `mean(1/H for H >= t)`, so earlier steps are repeated in more prefix tasks.

| segment_start | segment_end | avg_prefix_weight | relative_to_tail_weight |
| --- | --- | --- | --- |
| 0 | 96 | 0.004998 | 14.392857 |
| 96 | 192 | 0.002393 | 6.892857 |
| 192 | 336 | 0.001091 | 3.142857 |
| 336 | 720 | 0.000347 | 1.000000 |

## Segment Gap Summary vs Fixed-Horizon TimeAlign

| dataset | bucket | segments | mse_wins | mean_relative_mse_pct | mean_relative_mae_pct | mean_relative_to_tail_weight |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | early_0_96 | 4 | 4 | -7.174338 | -3.879153 | 14.392857 |
| ETTh2 | mid_96_192 | 3 | 3 | -11.553475 | -5.629966 | 6.892857 |
| ETTh2 | late_192_336 | 4 | 4 | -12.976093 | -6.392091 | 2.875000 |
| ETTh2 | tail_336_720 | 4 | 2 | -0.691082 | -0.281599 | 1.000000 |
| ETTh2 | ALL | 15 | 13 | -7.868432 | -3.940085 | 6.250000 |
| ETTm1 | early_0_96 | 4 | 4 | -2.867076 | -0.877971 | 14.392857 |
| ETTm1 | mid_96_192 | 3 | 3 | -2.089187 | -0.467738 | 6.892857 |
| ETTm1 | late_192_336 | 4 | 2 | 0.223412 | 0.060532 | 2.875000 |
| ETTm1 | tail_336_720 | 4 | 0 | 1.472549 | 0.371047 | 1.000000 |
| ETTm1 | ALL | 15 | 9 | -0.730135 | -0.212585 | 6.250000 |
| Weather | early_0_96 | 4 | 3 | -0.655729 | 0.442499 | 14.392857 |
| Weather | mid_96_192 | 3 | 2 | -0.801021 | 0.235395 | 6.892857 |
| Weather | late_192_336 | 4 | 3 | -0.204498 | 0.111104 | 2.875000 |
| Weather | tail_336_720 | 4 | 4 | -1.257587 | -0.576158 | 1.000000 |
| Weather | ALL | 15 | 12 | -0.724954 | 0.041064 | 6.250000 |
| ALL | early_0_96 | 12 | 11 | -3.565714 | -1.438209 | 14.392857 |
| ALL | mid_96_192 | 9 | 8 | -4.814561 | -1.954103 | 6.892857 |
| ALL | late_192_336 | 12 | 9 | -4.319060 | -2.073485 | 2.875000 |
| ALL | tail_336_720 | 12 | 6 | -0.158707 | -0.162237 | 1.000000 |
| ALL | ALL | 45 | 34 | -3.107840 | -1.370535 | 6.250000 |

## Reading

- [Fact] Overall segment-level mean relative MSE vs fixed is `-3.11%`.
- [Fact] Early bucket mean relative MSE is `-3.57%`; tail bucket is `-0.16%`.
- [Fact] Tail-minus-early relative MSE gap is `+3.41%`; positive means A6 gains are weaker in the under-weighted tail.
- [Inference] If the tail gap is stable by dataset, B7 is a stronger StageB route than B6 because it targets unified multi-horizon training mechanics, not generic frequency auxiliary losses.
- [Rollback] If follow-up gradient/task diagnostics do not support horizon-task interference, do not implement a new loss; keep StageB paused.
