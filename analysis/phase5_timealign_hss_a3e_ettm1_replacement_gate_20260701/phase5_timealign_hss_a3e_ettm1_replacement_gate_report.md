# Phase5 TimeAlign-HSS A3E ETTm1 Replacement Gate

Dataset universe: `ETTh2 + ETTm1 + Weather`. `ETTm1` replaces `ETTm2`; all ETTm1 references are rebuilt from remote raw metrics.

## Summary

| dataset | arm | settings | vs_a2_nested | vs_a3c_warm | vs_a3d_w03 | vs_h1_target_set | wins_vs_h1c | vs_h1c_row_gated | wins_vs_fixed | vs_fixed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | target_conditioned_nested_warm | 4 | -0.77 | -0.80 | 1.32 | 1.15 | 2 | -0.29 | 4 | -11.69 |
| ETTh2 | target_conditioned_nested_scratch | 4 | -1.24 | -1.25 | 0.85 | 0.67 | 4 | -0.76 | 4 | -12.07 |
| ETTm1 | target_conditioned_nested_warm | 4 | -0.47 | 0.18 | -0.70 | -0.62 | 3 | -0.53 | 3 | -1.82 |
| ETTm1 | target_conditioned_nested_scratch | 4 | 0.08 | 0.73 | -0.15 | -0.07 | 1 | 0.02 | 3 | -1.28 |
| Weather | target_conditioned_nested_warm | 4 | 0.15 | -0.13 | 0.14 | 0.07 | 1 | 0.06 | 2 | -0.05 |
| Weather | target_conditioned_nested_scratch | 4 | 0.03 | -0.26 | 0.02 | -0.06 | 4 | -0.06 | 2 | -0.17 |
| ALL | target_conditioned_nested_warm | 12 | -0.36 | -0.25 | 0.25 | 0.20 | 6 | -0.25 | 9 | -4.52 |
| ALL | target_conditioned_nested_scratch | 12 | -0.38 | -0.26 | 0.24 | 0.18 | 9 | -0.27 | 9 | -4.51 |

## Per-Horizon Rows

| dataset | horizon | arm | mse | vs_a2_nested | vs_a3c_warm | vs_a3d_w03 | vs_h1_target_set | vs_h1c_row_gated | vs_fixed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | target_conditioned_nested_warm | 0.245671 | 0.80 | -0.80 | 1.58 | 1.95 | 1.45 | -8.91 |
| ETTh2 | 192 | target_conditioned_nested_warm | 0.283596 | -0.27 | -0.95 | 2.07 | 2.30 | 0.61 | -15.24 |
| ETTh2 | 336 | target_conditioned_nested_warm | 0.305894 | -2.12 | -0.91 | 1.09 | 0.78 | -1.63 | -18.40 |
| ETTh2 | 720 | target_conditioned_nested_warm | 0.386372 | -1.50 | -0.55 | 0.54 | -0.43 | -1.59 | -4.19 |
| ETTh2 | 96 | target_conditioned_nested_scratch | 0.240463 | -1.34 | -2.90 | -0.57 | -0.22 | -0.70 | -10.84 |
| ETTh2 | 192 | target_conditioned_nested_scratch | 0.279352 | -1.76 | -2.43 | 0.54 | 0.76 | -0.89 | -16.51 |
| ETTh2 | 336 | target_conditioned_nested_scratch | 0.307399 | -1.64 | -0.42 | 1.59 | 1.27 | -1.15 | -18.00 |
| ETTh2 | 720 | target_conditioned_nested_scratch | 0.391408 | -0.22 | 0.74 | 1.85 | 0.86 | -0.31 | -2.94 |
| ETTm1 | 96 | target_conditioned_nested_warm | 0.271332 | -1.26 | 0.27 | -1.36 | -1.36 | -1.37 | -2.61 |
| ETTm1 | 192 | target_conditioned_nested_warm | 0.308800 | -0.50 | 0.20 | -0.70 | -0.65 | -0.58 | -4.13 |
| ETTm1 | 336 | target_conditioned_nested_warm | 0.347204 | -0.25 | 0.15 | -0.51 | -0.42 | -0.30 | -0.76 |
| ETTm1 | 720 | target_conditioned_nested_warm | 0.407631 | 0.14 | 0.08 | -0.22 | -0.06 | 0.15 | 0.23 |
| ETTm1 | 96 | target_conditioned_nested_scratch | 0.274960 | 0.06 | 1.61 | -0.04 | -0.04 | -0.05 | -1.31 |
| ETTm1 | 192 | target_conditioned_nested_scratch | 0.310703 | 0.12 | 0.82 | -0.09 | -0.04 | 0.03 | -3.54 |
| ETTm1 | 336 | target_conditioned_nested_scratch | 0.348264 | 0.05 | 0.45 | -0.21 | -0.12 | 0.01 | -0.45 |
| ETTm1 | 720 | target_conditioned_nested_scratch | 0.407438 | 0.10 | 0.04 | -0.27 | -0.10 | 0.10 | 0.18 |
| Weather | 96 | target_conditioned_nested_warm | 0.141415 | 0.09 | 0.04 | 0.05 | -0.13 | -0.01 | 0.70 |
| Weather | 192 | target_conditioned_nested_warm | 0.182593 | 0.06 | -0.11 | 0.17 | 0.09 | 0.00 | 0.20 |
| Weather | 336 | target_conditioned_nested_warm | 0.232073 | 0.18 | -0.19 | 0.16 | 0.10 | 0.11 | -0.25 |
| Weather | 720 | target_conditioned_nested_warm | 0.304216 | 0.29 | -0.28 | 0.19 | 0.21 | 0.14 | -0.83 |
| Weather | 96 | target_conditioned_nested_scratch | 0.141387 | 0.07 | 0.02 | 0.03 | -0.15 | -0.03 | 0.68 |
| Weather | 192 | target_conditioned_nested_scratch | 0.182443 | -0.02 | -0.19 | 0.09 | 0.01 | -0.08 | 0.11 |
| Weather | 336 | target_conditioned_nested_scratch | 0.231673 | 0.00 | -0.36 | -0.01 | -0.07 | -0.06 | -0.42 |
| Weather | 720 | target_conditioned_nested_scratch | 0.303519 | 0.06 | -0.50 | -0.04 | -0.02 | -0.09 | -1.06 |
