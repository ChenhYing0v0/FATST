# Phase5 TimeAlign-HSS A3 Interface Repair

## Summary

| dataset | arm | settings | wins_vs_a2_nested | vs_a2_nested | wins_vs_h1_target_set | vs_h1_target_set | wins_vs_h1c_row_gated | vs_h1c_row_gated | wins_vs_fixed | vs_fixed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | dense_initialized_nested_segment_decoder_multiprefix | 4 | 3 | -0.06 | 0 | 1.88 | 0 | 0.43 | 4 | -11.05 |
| ETTm2 | dense_initialized_nested_segment_decoder_multiprefix | 4 | 4 | -0.15 | 3 | -0.16 | 1 | 0.00 | 1 | 1.65 |
| Weather | dense_initialized_nested_segment_decoder_multiprefix | 4 | 0 | 0.03 | 2 | -0.06 | 4 | -0.06 | 2 | -0.17 |
| ALL | dense_initialized_nested_segment_decoder_multiprefix | 12 | 7 | -0.06 | 5 | 0.55 | 5 | 0.12 | 7 | -3.19 |

## Per-Horizon Rows

| dataset | horizon | arm | mse | vs_a2_nested | vs_h1_target_set | vs_h1c_row_gated | vs_fixed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | dense_initialized_nested_segment_decoder_multiprefix | 0.243428 | -0.12 | 1.02 | 0.53 | -9.75 |
| ETTh2 | 192 | dense_initialized_nested_segment_decoder_multiprefix | 0.283750 | -0.21 | 2.35 | 0.67 | -15.20 |
| ETTh2 | 336 | dense_initialized_nested_segment_decoder_multiprefix | 0.312314 | -0.06 | 2.89 | 0.43 | -16.69 |
| ETTh2 | 720 | dense_initialized_nested_segment_decoder_multiprefix | 0.392927 | 0.17 | 1.25 | 0.08 | -2.56 |
| ETTm2 | 96 | dense_initialized_nested_segment_decoder_multiprefix | 0.161483 | -0.31 | 0.18 | 0.04 | 4.23 |
| ETTm2 | 192 | dense_initialized_nested_segment_decoder_multiprefix | 0.215144 | -0.18 | -0.19 | 0.00 | 2.33 |
| ETTm2 | 336 | dense_initialized_nested_segment_decoder_multiprefix | 0.265752 | -0.08 | -0.21 | 0.02 | 0.93 |
| ETTm2 | 720 | dense_initialized_nested_segment_decoder_multiprefix | 0.340526 | -0.02 | -0.42 | -0.04 | -0.88 |
| Weather | 96 | dense_initialized_nested_segment_decoder_multiprefix | 0.141296 | 0.00 | -0.21 | -0.09 | 0.62 |
| Weather | 192 | dense_initialized_nested_segment_decoder_multiprefix | 0.182485 | 0.00 | 0.03 | -0.06 | 0.14 |
| Weather | 336 | dense_initialized_nested_segment_decoder_multiprefix | 0.231717 | 0.02 | -0.05 | -0.04 | -0.40 |
| Weather | 720 | dense_initialized_nested_segment_decoder_multiprefix | 0.303599 | 0.09 | 0.01 | -0.06 | -1.04 |
