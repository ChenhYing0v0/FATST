# Phase5 TimeAlign-HSS H1C Capacity-Preserving Gate

## Summary

| dataset | arm | settings | vs_h0_full | vs_h0b_stochastic_k2 | vs_h1_target_set | wins_vs_fixed | vs_fixed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | dense_prefix_residual_adapter_multiprefix | 4 | 5.37 | 9.04 | 11.04 | 3 | -2.97 |
| ETTh2 | row_gated_dense_head_multiprefix | 4 | -3.76 | -0.39 | 1.44 | 4 | -11.42 |
| ETTh2 | prefix_adapter_shared_dense_multiprefix | 4 | 17.52 | 21.61 | 23.85 | 0 | 8.30 |
| ETTm2 | dense_prefix_residual_adapter_multiprefix | 4 | 0.86 | 2.44 | 2.69 | 0 | 4.54 |
| ETTm2 | row_gated_dense_head_multiprefix | 4 | -1.97 | -0.42 | -0.17 | 1 | 1.65 |
| ETTm2 | prefix_adapter_shared_dense_multiprefix | 4 | 4.19 | 5.82 | 6.08 | 0 | 7.98 |
| Weather | dense_prefix_residual_adapter_multiprefix | 4 | 0.21 | 1.38 | 1.37 | 0 | 1.26 |
| Weather | row_gated_dense_head_multiprefix | 4 | -1.14 | 0.01 | 0.01 | 2 | -0.11 |
| Weather | prefix_adapter_shared_dense_multiprefix | 4 | 1.54 | 2.73 | 2.73 | 0 | 2.61 |
| ALL | dense_prefix_residual_adapter_multiprefix | 12 | 2.14 | 4.28 | 5.04 | 3 | 0.94 |
| ALL | row_gated_dense_head_multiprefix | 12 | -2.29 | -0.27 | 0.43 | 7 | -3.29 |
| ALL | prefix_adapter_shared_dense_multiprefix | 12 | 7.75 | 10.05 | 10.89 | 0 | 6.30 |

## Per-Horizon Rows

| dataset | horizon | arm | mse | vs_h0_full | vs_h0b_stochastic_k2 | vs_h1_target_set | vs_fixed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | dense_prefix_residual_adapter_multiprefix | 0.267430 | 7.35 | 9.94 | 10.98 | -0.85 |
| ETTh2 | 192 | dense_prefix_residual_adapter_multiprefix | 0.307594 | 3.93 | 8.72 | 10.95 | -8.07 |
| ETTh2 | 336 | dense_prefix_residual_adapter_multiprefix | 0.334293 | 2.31 | 7.14 | 10.14 | -10.83 |
| ETTh2 | 720 | dense_prefix_residual_adapter_multiprefix | 0.435029 | 7.88 | 10.36 | 12.10 | 7.88 |
| ETTh2 | 96 | row_gated_dense_head_multiprefix | 0.242149 | -2.79 | -0.45 | 0.48 | -10.22 |
| ETTh2 | 192 | row_gated_dense_head_multiprefix | 0.281868 | -4.76 | -0.38 | 1.67 | -15.76 |
| ETTh2 | 336 | row_gated_dense_head_multiprefix | 0.310967 | -4.83 | -0.34 | 2.45 | -17.05 |
| ETTh2 | 720 | row_gated_dense_head_multiprefix | 0.392610 | -2.64 | -0.40 | 1.17 | -2.64 |
| ETTh2 | 96 | prefix_adapter_shared_dense_multiprefix | 0.287767 | 15.52 | 18.31 | 19.41 | 6.69 |
| ETTh2 | 192 | prefix_adapter_shared_dense_multiprefix | 0.340296 | 14.98 | 20.28 | 22.75 | 1.70 |
| ETTh2 | 336 | prefix_adapter_shared_dense_multiprefix | 0.375596 | 14.95 | 20.37 | 23.74 | 0.19 |
| ETTh2 | 720 | prefix_adapter_shared_dense_multiprefix | 0.502548 | 24.62 | 27.49 | 29.50 | 24.62 |
| ETTm2 | 96 | dense_prefix_residual_adapter_multiprefix | 0.162432 | -3.09 | -0.17 | 0.76 | 4.84 |
| ETTm2 | 192 | dense_prefix_residual_adapter_multiprefix | 0.221300 | 0.81 | 2.40 | 2.66 | 5.26 |
| ETTm2 | 336 | dense_prefix_residual_adapter_multiprefix | 0.276228 | 2.57 | 3.65 | 3.72 | 4.91 |
| ETTm2 | 720 | dense_prefix_residual_adapter_multiprefix | 0.354340 | 3.14 | 3.87 | 3.62 | 3.14 |
| ETTm2 | 96 | row_gated_dense_head_multiprefix | 0.161417 | -3.70 | -0.79 | 0.13 | 4.19 |
| ETTm2 | 192 | row_gated_dense_head_multiprefix | 0.215143 | -1.99 | -0.45 | -0.19 | 2.33 |
| ETTm2 | 336 | row_gated_dense_head_multiprefix | 0.265710 | -1.34 | -0.30 | -0.23 | 0.91 |
| ETTm2 | 720 | row_gated_dense_head_multiprefix | 0.340670 | -0.84 | -0.14 | -0.38 | -0.84 |
| ETTm2 | 96 | prefix_adapter_shared_dense_multiprefix | 0.165873 | -1.04 | 1.95 | 2.90 | 7.07 |
| ETTm2 | 192 | prefix_adapter_shared_dense_multiprefix | 0.231340 | 5.38 | 7.05 | 7.32 | 10.04 |
| ETTm2 | 336 | prefix_adapter_shared_dense_multiprefix | 0.286653 | 6.44 | 7.56 | 7.64 | 8.87 |
| ETTm2 | 720 | prefix_adapter_shared_dense_multiprefix | 0.364026 | 5.96 | 6.71 | 6.45 | 5.96 |
| Weather | 96 | dense_prefix_residual_adapter_multiprefix | 0.143264 | 0.07 | 1.30 | 1.18 | 2.02 |
| Weather | 192 | dense_prefix_residual_adapter_multiprefix | 0.185350 | 0.26 | 1.51 | 1.60 | 1.71 |
| Weather | 336 | dense_prefix_residual_adapter_multiprefix | 0.235228 | 0.30 | 1.47 | 1.46 | 1.11 |
| Weather | 720 | dense_prefix_residual_adapter_multiprefix | 0.307374 | 0.19 | 1.23 | 1.25 | 0.19 |
| Weather | 96 | row_gated_dense_head_multiprefix | 0.141425 | -1.22 | -0.00 | -0.12 | 0.71 |
| Weather | 192 | row_gated_dense_head_multiprefix | 0.182589 | -1.23 | -0.00 | 0.09 | 0.19 |
| Weather | 336 | row_gated_dense_head_multiprefix | 0.231814 | -1.15 | -0.01 | -0.01 | -0.36 |
| Weather | 720 | row_gated_dense_head_multiprefix | 0.303783 | -0.98 | 0.05 | 0.07 | -0.98 |
| Weather | 96 | prefix_adapter_shared_dense_multiprefix | 0.145212 | 1.43 | 2.67 | 2.55 | 3.41 |
| Weather | 192 | prefix_adapter_shared_dense_multiprefix | 0.188445 | 1.94 | 3.20 | 3.30 | 3.41 |
| Weather | 336 | prefix_adapter_shared_dense_multiprefix | 0.238409 | 1.66 | 2.84 | 2.84 | 2.47 |
| Weather | 720 | prefix_adapter_shared_dense_multiprefix | 0.310300 | 1.15 | 2.19 | 2.22 | 1.15 |
