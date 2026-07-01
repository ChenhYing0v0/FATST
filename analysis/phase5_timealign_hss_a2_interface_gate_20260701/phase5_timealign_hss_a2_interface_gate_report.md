# Phase5 TimeAlign-HSS A2 Interface Gate

## Summary

| dataset | arm | settings | vs_h0_full | vs_h1_target_set | wins_vs_h1c_row_gated | vs_h1c_row_gated | wins_vs_fixed | vs_fixed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | dense_row_initialized_prefix_decoder_multiprefix | 4 | 6.30 | 12.03 | 0 | 10.44 | 3 | -2.08 |
| ETTh2 | nested_segment_decoder_multiprefix | 4 | -3.29 | 1.94 | 1 | 0.48 | 4 | -11.00 |
| ETTm2 | dense_row_initialized_prefix_decoder_multiprefix | 4 | 0.85 | 2.69 | 0 | 2.86 | 0 | 4.53 |
| ETTm2 | nested_segment_decoder_multiprefix | 4 | -1.82 | -0.01 | 1 | 0.15 | 1 | 1.81 |
| Weather | dense_row_initialized_prefix_decoder_multiprefix | 4 | 0.29 | 1.46 | 0 | 1.45 | 0 | 1.34 |
| Weather | nested_segment_decoder_multiprefix | 4 | -1.23 | -0.08 | 4 | -0.09 | 2 | -0.20 |
| ALL | dense_row_initialized_prefix_decoder_multiprefix | 12 | 2.48 | 5.39 | 0 | 4.92 | 3 | 1.27 |
| ALL | nested_segment_decoder_multiprefix | 12 | -2.12 | 0.61 | 6 | 0.18 | 7 | -3.13 |

## Per-Horizon Rows

| dataset | horizon | arm | mse | vs_h1_target_set | vs_h1c_row_gated | vs_fixed |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | dense_row_initialized_prefix_decoder_multiprefix | 0.264641 | 9.82 | 9.29 | -1.88 |
| ETTh2 | 192 | dense_row_initialized_prefix_decoder_multiprefix | 0.310069 | 11.84 | 10.01 | -7.33 |
| ETTh2 | 336 | dense_row_initialized_prefix_decoder_multiprefix | 0.338752 | 11.60 | 8.94 | -9.64 |
| ETTh2 | 720 | dense_row_initialized_prefix_decoder_multiprefix | 0.445742 | 14.86 | 13.53 | 10.54 |
| ETTh2 | 96 | nested_segment_decoder_multiprefix | 0.243723 | 1.14 | 0.65 | -9.64 |
| ETTh2 | 192 | nested_segment_decoder_multiprefix | 0.284350 | 2.57 | 0.88 | -15.02 |
| ETTh2 | 336 | nested_segment_decoder_multiprefix | 0.312516 | 2.96 | 0.50 | -16.64 |
| ETTh2 | 720 | nested_segment_decoder_multiprefix | 0.392251 | 1.08 | -0.09 | -2.73 |
| ETTm2 | 96 | dense_row_initialized_prefix_decoder_multiprefix | 0.162424 | 0.76 | 0.62 | 4.84 |
| ETTm2 | 192 | dense_row_initialized_prefix_decoder_multiprefix | 0.221279 | 2.65 | 2.85 | 5.25 |
| ETTm2 | 336 | dense_row_initialized_prefix_decoder_multiprefix | 0.276217 | 3.72 | 3.95 | 4.91 |
| ETTm2 | 720 | dense_row_initialized_prefix_decoder_multiprefix | 0.354348 | 3.62 | 4.02 | 3.14 |
| ETTm2 | 96 | nested_segment_decoder_multiprefix | 0.161988 | 0.49 | 0.35 | 4.56 |
| ETTm2 | 192 | nested_segment_decoder_multiprefix | 0.215539 | -0.01 | 0.18 | 2.52 |
| ETTm2 | 336 | nested_segment_decoder_multiprefix | 0.265965 | -0.13 | 0.10 | 1.01 |
| ETTm2 | 720 | nested_segment_decoder_multiprefix | 0.340582 | -0.40 | -0.03 | -0.87 |
| Weather | 96 | dense_row_initialized_prefix_decoder_multiprefix | 0.143386 | 1.27 | 1.39 | 2.11 |
| Weather | 192 | dense_row_initialized_prefix_decoder_multiprefix | 0.185570 | 1.73 | 1.63 | 1.83 |
| Weather | 336 | dense_row_initialized_prefix_decoder_multiprefix | 0.235411 | 1.54 | 1.55 | 1.18 |
| Weather | 720 | dense_row_initialized_prefix_decoder_multiprefix | 0.307515 | 1.30 | 1.23 | 0.24 |
| Weather | 96 | nested_segment_decoder_multiprefix | 0.141294 | -0.21 | -0.09 | 0.62 |
| Weather | 192 | nested_segment_decoder_multiprefix | 0.182481 | 0.03 | -0.06 | 0.14 |
| Weather | 336 | nested_segment_decoder_multiprefix | 0.231667 | -0.07 | -0.06 | -0.42 |
| Weather | 720 | nested_segment_decoder_multiprefix | 0.303337 | -0.08 | -0.15 | -1.12 |
