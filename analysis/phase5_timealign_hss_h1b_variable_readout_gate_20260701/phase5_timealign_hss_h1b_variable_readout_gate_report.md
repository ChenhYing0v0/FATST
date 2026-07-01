# Phase5 TimeAlign-HSS H1B Variable Readout Gate

## Summary

| dataset | arm | settings | vs_h0_full | vs_h0b_stochastic_k2 | vs_h1_target_set | wins_vs_fixed | vs_fixed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | target_set_prefix_head_multiprefix | 4 | 8.43 | 12.22 | 14.28 | 2 | -0.24 |
| ETTh2 | prefix_token_decoder_multiprefix | 4 | 23.78 | 28.09 | 30.41 | 0 | 13.67 |
| ETTm2 | target_set_prefix_head_multiprefix | 4 | 13.58 | 15.42 | 15.73 | 0 | 17.91 |
| ETTm2 | prefix_token_decoder_multiprefix | 4 | 18.41 | 20.33 | 20.66 | 0 | 22.95 |
| Weather | target_set_prefix_head_multiprefix | 4 | 11.92 | 13.23 | 13.23 | 0 | 13.12 |
| Weather | prefix_token_decoder_multiprefix | 4 | 24.05 | 25.51 | 25.50 | 0 | 25.44 |
| ALL | target_set_prefix_head_multiprefix | 12 | 11.31 | 13.62 | 14.41 | 2 | 10.26 |
| ALL | prefix_token_decoder_multiprefix | 12 | 22.08 | 24.64 | 25.52 | 0 | 20.68 |

## Per-Horizon Rows

| dataset | horizon | arm | mse | vs_h0_full | vs_h0b_stochastic_k2 | vs_h1_target_set | vs_fixed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | target_set_prefix_head_multiprefix | 0.279690 | 12.28 | 14.98 | 16.06 | 3.70 |
| ETTh2 | 192 | target_set_prefix_head_multiprefix | 0.319262 | 7.87 | 12.84 | 15.16 | -4.58 |
| ETTh2 | 336 | target_set_prefix_head_multiprefix | 0.347132 | 6.24 | 11.25 | 14.36 | -7.40 |
| ETTh2 | 720 | target_set_prefix_head_multiprefix | 0.432805 | 7.33 | 9.80 | 11.53 | 7.33 |
| ETTh2 | 96 | prefix_token_decoder_multiprefix | 0.357483 | 43.50 | 46.97 | 48.34 | 32.54 |
| ETTh2 | 192 | prefix_token_decoder_multiprefix | 0.371898 | 25.66 | 31.44 | 34.15 | 11.15 |
| ETTh2 | 336 | prefix_token_decoder_multiprefix | 0.380730 | 16.52 | 22.02 | 25.43 | 1.56 |
| ETTh2 | 720 | prefix_token_decoder_multiprefix | 0.441248 | 9.42 | 11.94 | 13.71 | 9.42 |
| ETTm2 | 96 | target_set_prefix_head_multiprefix | 0.197481 | 17.82 | 21.38 | 22.51 | 27.47 |
| ETTm2 | 192 | target_set_prefix_head_multiprefix | 0.252950 | 15.23 | 17.05 | 17.35 | 20.32 |
| ETTm2 | 336 | target_set_prefix_head_multiprefix | 0.304147 | 12.94 | 14.13 | 14.21 | 15.51 |
| ETTm2 | 720 | target_set_prefix_head_multiprefix | 0.372242 | 8.35 | 9.12 | 8.85 | 8.35 |
| ETTm2 | 96 | prefix_token_decoder_multiprefix | 0.208979 | 24.68 | 28.44 | 29.64 | 34.89 |
| ETTm2 | 192 | prefix_token_decoder_multiprefix | 0.264139 | 20.33 | 22.22 | 22.54 | 25.64 |
| ETTm2 | 336 | prefix_token_decoder_multiprefix | 0.312366 | 15.99 | 17.21 | 17.29 | 18.63 |
| ETTm2 | 720 | prefix_token_decoder_multiprefix | 0.386988 | 12.64 | 13.44 | 13.17 | 12.64 |
| Weather | 96 | target_set_prefix_head_multiprefix | 0.166147 | 16.05 | 17.48 | 17.34 | 18.32 |
| Weather | 192 | target_set_prefix_head_multiprefix | 0.211518 | 14.42 | 15.84 | 15.95 | 16.07 |
| Weather | 336 | target_set_prefix_head_multiprefix | 0.259823 | 10.79 | 12.08 | 12.07 | 11.68 |
| Weather | 720 | target_set_prefix_head_multiprefix | 0.326487 | 6.43 | 7.53 | 7.55 | 6.43 |
| Weather | 96 | prefix_token_decoder_multiprefix | 0.199608 | 39.42 | 41.14 | 40.97 | 42.14 |
| Weather | 192 | prefix_token_decoder_multiprefix | 0.237260 | 28.34 | 29.94 | 30.06 | 30.20 |
| Weather | 336 | prefix_token_decoder_multiprefix | 0.279017 | 18.97 | 20.36 | 20.35 | 19.93 |
| Weather | 720 | prefix_token_decoder_multiprefix | 0.335850 | 9.48 | 10.61 | 10.63 | 9.48 |
