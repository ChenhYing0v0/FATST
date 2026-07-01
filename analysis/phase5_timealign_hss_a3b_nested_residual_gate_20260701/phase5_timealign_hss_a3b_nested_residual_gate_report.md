# Phase5 TimeAlign-HSS A3B Nested Residual Gate

## Summary

| dataset | arm | settings | vs_a2_nested | vs_a3_shallow | vs_h1_target_set | wins_vs_h1c | vs_h1c_row_gated | wins_vs_fixed | vs_fixed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | target_conditioned_nested_residual_decoder_multiprefix | 4 | 9.01 | 9.07 | 11.12 | 0 | 9.54 | 3 | -2.91 |
| ETTm2 | target_conditioned_nested_residual_decoder_multiprefix | 4 | 2.71 | 2.86 | 2.69 | 0 | 2.87 | 0 | 4.54 |
| Weather | target_conditioned_nested_residual_decoder_multiprefix | 4 | 1.53 | 1.50 | 1.45 | 0 | 1.44 | 0 | 1.33 |
| ALL | target_conditioned_nested_residual_decoder_multiprefix | 12 | 4.42 | 4.48 | 5.09 | 0 | 4.61 | 3 | 0.99 |

## Per-Horizon Rows

| dataset | horizon | arm | mse | vs_a2_nested | vs_a3_shallow | vs_h1_target_set | vs_h1c_row_gated | vs_fixed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | target_conditioned_nested_residual_decoder_multiprefix | 0.265811 | 9.06 | 9.20 | 10.30 | 9.77 | -1.45 |
| ETTh2 | 192 | target_conditioned_nested_residual_decoder_multiprefix | 0.308426 | 8.47 | 8.70 | 11.25 | 9.42 | -7.82 |
| ETTh2 | 336 | target_conditioned_nested_residual_decoder_multiprefix | 0.335319 | 7.30 | 7.37 | 10.47 | 7.83 | -10.56 |
| ETTh2 | 720 | target_conditioned_nested_residual_decoder_multiprefix | 0.436312 | 11.23 | 11.04 | 12.43 | 11.13 | 8.20 |
| ETTm2 | 96 | target_conditioned_nested_residual_decoder_multiprefix | 0.162459 | 0.29 | 0.60 | 0.78 | 0.65 | 4.86 |
| ETTm2 | 192 | target_conditioned_nested_residual_decoder_multiprefix | 0.221322 | 2.68 | 2.87 | 2.67 | 2.87 | 5.27 |
| ETTm2 | 336 | target_conditioned_nested_residual_decoder_multiprefix | 0.276205 | 3.85 | 3.93 | 3.72 | 3.95 | 4.90 |
| ETTm2 | 720 | target_conditioned_nested_residual_decoder_multiprefix | 0.354272 | 4.02 | 4.04 | 3.60 | 3.99 | 3.12 |
| Weather | 96 | target_conditioned_nested_residual_decoder_multiprefix | 0.143352 | 1.46 | 1.46 | 1.24 | 1.36 | 2.08 |
| Weather | 192 | target_conditioned_nested_residual_decoder_multiprefix | 0.185543 | 1.68 | 1.68 | 1.71 | 1.62 | 1.82 |
| Weather | 336 | target_conditioned_nested_residual_decoder_multiprefix | 0.235390 | 1.61 | 1.58 | 1.53 | 1.54 | 1.18 |
| Weather | 720 | target_conditioned_nested_residual_decoder_multiprefix | 0.307550 | 1.39 | 1.30 | 1.31 | 1.24 | 0.25 |
