# Phase5 TimeAlign-HSS A3C Warm-Started Nested Gate

## Summary

| dataset | arm | settings | vs_a2_nested | vs_a3b_residual | vs_h1_target_set | wins_vs_h1c | vs_h1c_row_gated | wins_vs_fixed | vs_fixed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | checkpoint_initialized_nested_segment_decoder_multiprefix | 4 | 0.03 | -8.23 | 1.96 | 2 | 0.52 | 4 | -10.98 |
| ETTm2 | checkpoint_initialized_nested_segment_decoder_multiprefix | 4 | -0.11 | -2.73 | -0.12 | 2 | 0.04 | 1 | 1.68 |
| Weather | checkpoint_initialized_nested_segment_decoder_multiprefix | 4 | 0.29 | -1.23 | 0.20 | 1 | 0.20 | 2 | 0.09 |
| ALL | checkpoint_initialized_nested_segment_decoder_multiprefix | 12 | 0.07 | -4.06 | 0.68 | 5 | 0.25 | 7 | -3.07 |

## Per-Horizon Rows

| dataset | horizon | arm | mse | vs_a2_nested | vs_a3b_residual | vs_h1_target_set | vs_h1c_row_gated | vs_fixed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | checkpoint_initialized_nested_segment_decoder_multiprefix | 0.247644 | 1.61 | -6.83 | 2.76 | 2.27 | -8.18 |
| ETTh2 | 192 | checkpoint_initialized_nested_segment_decoder_multiprefix | 0.286307 | 0.69 | -7.17 | 3.27 | 1.57 | -14.43 |
| ETTh2 | 336 | checkpoint_initialized_nested_segment_decoder_multiprefix | 0.308693 | -1.22 | -7.94 | 1.70 | -0.73 | -17.66 |
| ETTh2 | 720 | checkpoint_initialized_nested_segment_decoder_multiprefix | 0.388513 | -0.95 | -10.96 | 0.12 | -1.04 | -3.65 |
| ETTm2 | 96 | checkpoint_initialized_nested_segment_decoder_multiprefix | 0.160353 | -1.01 | -1.30 | -0.53 | -0.66 | 3.50 |
| ETTm2 | 192 | checkpoint_initialized_nested_segment_decoder_multiprefix | 0.214918 | -0.29 | -2.89 | -0.30 | -0.10 | 2.23 |
| ETTm2 | 336 | checkpoint_initialized_nested_segment_decoder_multiprefix | 0.266363 | 0.15 | -3.56 | 0.02 | 0.25 | 1.16 |
| ETTm2 | 720 | checkpoint_initialized_nested_segment_decoder_multiprefix | 0.343036 | 0.72 | -3.17 | 0.31 | 0.69 | -0.15 |
| Weather | 96 | checkpoint_initialized_nested_segment_decoder_multiprefix | 0.141354 | 0.04 | -1.39 | -0.17 | -0.05 | 0.66 |
| Weather | 192 | checkpoint_initialized_nested_segment_decoder_multiprefix | 0.182797 | 0.17 | -1.48 | 0.21 | 0.11 | 0.31 |
| Weather | 336 | checkpoint_initialized_nested_segment_decoder_multiprefix | 0.232511 | 0.36 | -1.22 | 0.29 | 0.30 | -0.06 |
| Weather | 720 | checkpoint_initialized_nested_segment_decoder_multiprefix | 0.305059 | 0.57 | -0.81 | 0.49 | 0.42 | -0.56 |
