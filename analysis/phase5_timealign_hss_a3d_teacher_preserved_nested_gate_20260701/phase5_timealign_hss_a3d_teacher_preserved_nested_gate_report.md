# Phase5 TimeAlign-HSS A3D Teacher-Preserved Nested Gate

## Summary

| dataset | arm | settings | vs_a2_nested | vs_a3c_warm | vs_h1_target_set | wins_vs_h1c | vs_h1c_row_gated | wins_vs_fixed | vs_fixed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | teacher_preserved_nested_w03 | 4 | -2.06 | -2.09 | -0.17 | 4 | -1.59 | 4 | -12.82 |
| ETTh2 | teacher_preserved_nested_w10 | 4 | -2.12 | -2.14 | -0.23 | 4 | -1.65 | 4 | -12.86 |
| ETTm2 | teacher_preserved_nested_w03 | 4 | 0.07 | 0.18 | 0.05 | 0 | 0.22 | 1 | 1.87 |
| ETTm2 | teacher_preserved_nested_w10 | 4 | 0.13 | 0.24 | 0.12 | 0 | 0.29 | 1 | 1.94 |
| Weather | teacher_preserved_nested_w03 | 4 | 0.01 | -0.28 | -0.07 | 4 | -0.08 | 2 | -0.19 |
| Weather | teacher_preserved_nested_w10 | 4 | 0.12 | -0.16 | 0.04 | 1 | 0.03 | 2 | -0.08 |
| ALL | teacher_preserved_nested_w03 | 12 | -0.66 | -0.73 | -0.06 | 8 | -0.48 | 7 | -3.71 |
| ALL | teacher_preserved_nested_w10 | 12 | -0.62 | -0.69 | -0.02 | 5 | -0.44 | 7 | -3.67 |

## Per-Horizon Rows

| dataset | horizon | arm | mse | vs_a2_nested | vs_a3c_warm | vs_h1_target_set | vs_h1c_row_gated | vs_fixed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | teacher_preserved_nested_w03 | 0.241845 | -0.77 | -2.34 | 0.36 | -0.13 | -10.33 |
| ETTh2 | 192 | teacher_preserved_nested_w03 | 0.277847 | -2.29 | -2.95 | 0.22 | -1.43 | -16.96 |
| ETTh2 | 336 | teacher_preserved_nested_w03 | 0.302600 | -3.17 | -1.97 | -0.31 | -2.69 | -19.28 |
| ETTh2 | 720 | teacher_preserved_nested_w03 | 0.384301 | -2.03 | -1.08 | -0.97 | -2.12 | -4.70 |
| ETTh2 | 96 | teacher_preserved_nested_w10 | 0.240868 | -1.17 | -2.74 | -0.05 | -0.53 | -10.69 |
| ETTh2 | 192 | teacher_preserved_nested_w10 | 0.276764 | -2.67 | -3.33 | -0.17 | -1.81 | -17.28 |
| ETTh2 | 336 | teacher_preserved_nested_w10 | 0.302699 | -3.14 | -1.94 | -0.27 | -2.66 | -19.26 |
| ETTh2 | 720 | teacher_preserved_nested_w10 | 0.386312 | -1.51 | -0.57 | -0.45 | -1.60 | -4.20 |
| ETTm2 | 96 | teacher_preserved_nested_w03 | 0.161632 | -0.22 | 0.80 | 0.27 | 0.13 | 4.33 |
| ETTm2 | 192 | teacher_preserved_nested_w03 | 0.215645 | 0.05 | 0.34 | 0.04 | 0.23 | 2.57 |
| ETTm2 | 336 | teacher_preserved_nested_w03 | 0.266309 | 0.13 | -0.02 | -0.00 | 0.23 | 1.14 |
| ETTm2 | 720 | teacher_preserved_nested_w03 | 0.341657 | 0.32 | -0.40 | -0.09 | 0.29 | -0.56 |
| ETTm2 | 96 | teacher_preserved_nested_w10 | 0.161761 | -0.14 | 0.88 | 0.35 | 0.21 | 4.41 |
| ETTm2 | 192 | teacher_preserved_nested_w10 | 0.215831 | 0.14 | 0.42 | 0.13 | 0.32 | 2.66 |
| ETTm2 | 336 | teacher_preserved_nested_w10 | 0.266506 | 0.20 | 0.05 | 0.07 | 0.30 | 1.22 |
| ETTm2 | 720 | teacher_preserved_nested_w10 | 0.341733 | 0.34 | -0.38 | -0.07 | 0.31 | -0.53 |
| Weather | 96 | teacher_preserved_nested_w03 | 0.141340 | 0.03 | -0.01 | -0.18 | -0.06 | 0.65 |
| Weather | 192 | teacher_preserved_nested_w03 | 0.182276 | -0.11 | -0.28 | -0.08 | -0.17 | 0.02 |
| Weather | 336 | teacher_preserved_nested_w03 | 0.231705 | 0.02 | -0.35 | -0.06 | -0.05 | -0.41 |
| Weather | 720 | teacher_preserved_nested_w03 | 0.303637 | 0.10 | -0.47 | 0.02 | -0.05 | -1.02 |
| Weather | 96 | teacher_preserved_nested_w10 | 0.141569 | 0.19 | 0.15 | -0.02 | 0.10 | 0.81 |
| Weather | 192 | teacher_preserved_nested_w10 | 0.182508 | 0.01 | -0.16 | 0.05 | -0.04 | 0.15 |
| Weather | 336 | teacher_preserved_nested_w10 | 0.231953 | 0.12 | -0.24 | 0.05 | 0.06 | -0.30 |
| Weather | 720 | teacher_preserved_nested_w10 | 0.303823 | 0.16 | -0.41 | 0.08 | 0.01 | -0.96 |
