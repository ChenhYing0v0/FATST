# Phase5 TimeAlign-HSS H0 Prefix Gate

## Summary

| dataset | loss_mode | settings | wins_vs_full | mean_mse_vs_full | wins_vs_fixed | mean_mse_vs_fixed |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | full | 4 | 0 | 0.00 | 3 | -8.01 |
| ETTh2 | multi-prefix | 4 | 4 | -3.36 | 4 | -11.05 |
| ETTh2 | balanced-step | 4 | 4 | -2.03 | 4 | -9.86 |
| ETTh2 | stochastic-prefix | 4 | 4 | -3.39 | 4 | -11.07 |
| ETTh2 | continuous-prefix | 4 | 4 | -3.01 | 4 | -10.75 |
| ETTm2 | full | 4 | 0 | 0.00 | 0 | 3.72 |
| ETTm2 | multi-prefix | 4 | 4 | -1.57 | 1 | 2.06 |
| ETTm2 | balanced-step | 4 | 4 | -0.99 | 1 | 2.68 |
| ETTm2 | stochastic-prefix | 4 | 4 | -1.25 | 1 | 2.40 |
| ETTm2 | continuous-prefix | 4 | 4 | -1.07 | 1 | 2.59 |
| Weather | full | 4 | 0 | 0.00 | 0 | 1.05 |
| Weather | multi-prefix | 4 | 4 | -1.17 | 2 | -0.13 |
| Weather | balanced-step | 4 | 4 | -0.65 | 1 | 0.40 |
| Weather | stochastic-prefix | 4 | 4 | -1.07 | 2 | -0.03 |
| Weather | continuous-prefix | 4 | 4 | -0.94 | 2 | 0.10 |
| ALL | full | 12 | 0 | 0.00 | 3 | -1.08 |
| ALL | multi-prefix | 12 | 12 | -2.03 | 7 | -3.04 |
| ALL | balanced-step | 12 | 12 | -1.22 | 6 | -2.26 |
| ALL | stochastic-prefix | 12 | 12 | -1.90 | 7 | -2.90 |
| ALL | continuous-prefix | 12 | 12 | -1.67 | 7 | -2.69 |

## Reading Guide

- `balanced-step` tests whether D0 was just non-overlapping region reweighting.
- `stochastic-prefix` tests whether prefix supervision works as a train-time schedule over benchmark prefixes.
- `continuous-prefix` tests whether the schedule can move away from fixed benchmark horizon ids.
- H0 passes as a paper-story carrier only if a schedule-like mode approaches or beats `multi-prefix` while preserving ETTm2/Weather gap reduction and ETTh2 gains.

## Per-Horizon Rows

| dataset | horizon | loss_mode | mse | vs_full | vs_multi_prefix | vs_fixed |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | full | 0.249109 | 0.00 | 2.61 | -7.64 |
| ETTh2 | 96 | multi-prefix | 0.242773 | -2.54 | 0.00 | -9.99 |
| ETTh2 | 96 | balanced-step | 0.245116 | -1.60 | 0.96 | -9.12 |
| ETTh2 | 96 | stochastic-prefix | 0.242637 | -2.60 | -0.06 | -10.04 |
| ETTh2 | 96 | continuous-prefix | 0.243915 | -2.09 | 0.47 | -9.56 |
| ETTh2 | 192 | full | 0.295961 | 0.00 | 4.56 | -11.55 |
| ETTh2 | 192 | multi-prefix | 0.283057 | -4.36 | 0.00 | -15.40 |
| ETTh2 | 192 | balanced-step | 0.288543 | -2.51 | 1.94 | -13.76 |
| ETTh2 | 192 | stochastic-prefix | 0.282636 | -4.50 | -0.15 | -15.53 |
| ETTh2 | 192 | continuous-prefix | 0.284757 | -3.79 | 0.60 | -14.90 |
| ETTh2 | 336 | full | 0.326745 | 0.00 | 4.44 | -12.84 |
| ETTh2 | 336 | multi-prefix | 0.312852 | -4.25 | 0.00 | -16.55 |
| ETTh2 | 336 | balanced-step | 0.318731 | -2.45 | 1.88 | -14.98 |
| ETTh2 | 336 | stochastic-prefix | 0.311666 | -4.62 | -0.38 | -16.87 |
| ETTh2 | 336 | continuous-prefix | 0.314180 | -3.85 | 0.42 | -16.19 |
| ETTh2 | 720 | full | 0.403252 | 0.00 | 2.33 | 0.00 |
| ETTh2 | 720 | multi-prefix | 0.394071 | -2.28 | 0.00 | -2.28 |
| ETTh2 | 720 | balanced-step | 0.396931 | -1.57 | 0.73 | -1.57 |
| ETTh2 | 720 | stochastic-prefix | 0.395764 | -1.86 | 0.43 | -1.86 |
| ETTh2 | 720 | continuous-prefix | 0.393833 | -2.34 | -0.06 | -2.34 |
| ETTm2 | 96 | full | 0.167614 | 0.00 | 2.97 | 8.19 |
| ETTm2 | 96 | multi-prefix | 0.162777 | -2.89 | 0.00 | 5.07 |
| ETTm2 | 96 | balanced-step | 0.164772 | -1.70 | 1.23 | 6.35 |
| ETTm2 | 96 | stochastic-prefix | 0.163190 | -2.64 | 0.25 | 5.33 |
| ETTm2 | 96 | continuous-prefix | 0.164037 | -2.13 | 0.77 | 5.88 |
| ETTm2 | 192 | full | 0.219519 | 0.00 | 1.57 | 4.41 |
| ETTm2 | 192 | multi-prefix | 0.216118 | -1.55 | 0.00 | 2.80 |
| ETTm2 | 192 | balanced-step | 0.217437 | -0.95 | 0.61 | 3.42 |
| ETTm2 | 192 | stochastic-prefix | 0.216674 | -1.30 | 0.26 | 3.06 |
| ETTm2 | 192 | continuous-prefix | 0.217132 | -1.09 | 0.47 | 3.28 |
| ETTm2 | 336 | full | 0.269307 | 0.00 | 1.10 | 2.28 |
| ETTm2 | 336 | multi-prefix | 0.266372 | -1.09 | 0.00 | 1.17 |
| ETTm2 | 336 | balanced-step | 0.267437 | -0.69 | 0.40 | 1.57 |
| ETTm2 | 336 | stochastic-prefix | 0.267165 | -0.80 | 0.30 | 1.47 |
| ETTm2 | 336 | continuous-prefix | 0.267254 | -0.76 | 0.33 | 1.50 |
| ETTm2 | 720 | full | 0.343566 | 0.00 | 0.78 | 0.00 |
| ETTm2 | 720 | multi-prefix | 0.340916 | -0.77 | 0.00 | -0.77 |
| ETTm2 | 720 | balanced-step | 0.341414 | -0.63 | 0.15 | -0.63 |
| ETTm2 | 720 | stochastic-prefix | 0.342646 | -0.27 | 0.51 | -0.27 |
| ETTm2 | 720 | continuous-prefix | 0.342556 | -0.29 | 0.48 | -0.29 |
| Weather | 96 | full | 0.143170 | 0.00 | 1.23 | 1.95 |
| Weather | 96 | multi-prefix | 0.141425 | -1.22 | 0.00 | 0.71 |
| Weather | 96 | balanced-step | 0.142260 | -0.64 | 0.59 | 1.31 |
| Weather | 96 | stochastic-prefix | 0.141548 | -1.13 | 0.09 | 0.80 |
| Weather | 96 | continuous-prefix | 0.141756 | -0.99 | 0.23 | 0.95 |
| Weather | 192 | full | 0.184864 | 0.00 | 1.26 | 1.44 |
| Weather | 192 | multi-prefix | 0.182559 | -1.25 | 0.00 | 0.18 |
| Weather | 192 | balanced-step | 0.183633 | -0.67 | 0.59 | 0.77 |
| Weather | 192 | stochastic-prefix | 0.182893 | -1.07 | 0.18 | 0.36 |
| Weather | 192 | continuous-prefix | 0.183151 | -0.93 | 0.32 | 0.50 |
| Weather | 336 | full | 0.234520 | 0.00 | 1.20 | 0.80 |
| Weather | 336 | multi-prefix | 0.231734 | -1.19 | 0.00 | -0.40 |
| Weather | 336 | balanced-step | 0.232923 | -0.68 | 0.51 | 0.12 |
| Weather | 336 | stochastic-prefix | 0.232172 | -1.00 | 0.19 | -0.21 |
| Weather | 336 | continuous-prefix | 0.232463 | -0.88 | 0.31 | -0.08 |
| Weather | 720 | full | 0.306776 | 0.00 | 1.04 | 0.00 |
| Weather | 720 | multi-prefix | 0.303611 | -1.03 | 0.00 | -1.03 |
| Weather | 720 | balanced-step | 0.304918 | -0.61 | 0.43 | -0.61 |
| Weather | 720 | stochastic-prefix | 0.303464 | -1.08 | -0.05 | -1.08 |
| Weather | 720 | continuous-prefix | 0.303810 | -0.97 | 0.07 | -0.97 |
