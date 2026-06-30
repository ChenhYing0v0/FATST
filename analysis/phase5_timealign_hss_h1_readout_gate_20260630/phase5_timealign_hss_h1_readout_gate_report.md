# Phase5 TimeAlign-HSS H1 Readout Gate

## Summary

| dataset | arm | settings | vs_full | vs_multi_prefix | vs_h0b_stochastic_k2 | wins_vs_fixed | vs_fixed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | prefix_conditioned_stochastic_k2 | 4 | -4.67 | -1.36 | -1.34 | 4 | -12.25 |
| ETTh2 | target_set_decoder_multiprefix | 4 | -5.12 | -1.83 | -1.80 | 4 | -12.65 |
| ETTm2 | prefix_conditioned_stochastic_k2 | 4 | -1.63 | -0.06 | -0.07 | 1 | 2.00 |
| ETTm2 | target_set_decoder_multiprefix | 4 | -1.80 | -0.24 | -0.25 | 1 | 1.81 |
| Weather | prefix_conditioned_stochastic_k2 | 4 | -1.04 | 0.13 | 0.11 | 2 | -0.00 |
| Weather | target_set_decoder_multiprefix | 4 | -1.15 | 0.02 | 0.00 | 2 | -0.12 |
| ALL | prefix_conditioned_stochastic_k2 | 12 | -2.45 | -0.43 | -0.43 | 7 | -3.42 |
| ALL | target_set_decoder_multiprefix | 12 | -2.69 | -0.68 | -0.69 | 7 | -3.65 |

## Per-Horizon Rows

| dataset | horizon | arm | mse | vs_full | vs_multi_prefix | vs_h0b_stochastic_k2 | vs_fixed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | prefix_conditioned_stochastic_k2 | 0.240252 | -3.56 | -1.04 | -1.23 | -10.92 |
| ETTh2 | 192 | prefix_conditioned_stochastic_k2 | 0.278606 | -5.86 | -1.57 | -1.53 | -16.73 |
| ETTh2 | 336 | prefix_conditioned_stochastic_k2 | 0.307038 | -6.03 | -1.86 | -1.60 | -18.10 |
| ETTh2 | 720 | prefix_conditioned_stochastic_k2 | 0.390180 | -3.24 | -0.99 | -1.01 | -3.24 |
| ETTh2 | 96 | target_set_decoder_multiprefix | 0.240982 | -3.26 | -0.74 | -0.93 | -10.65 |
| ETTh2 | 192 | target_set_decoder_multiprefix | 0.277231 | -6.33 | -2.06 | -2.01 | -17.15 |
| ETTh2 | 336 | target_set_decoder_multiprefix | 0.303530 | -7.10 | -2.98 | -2.72 | -19.04 |
| ETTh2 | 720 | target_set_decoder_multiprefix | 0.388059 | -3.77 | -1.53 | -1.55 | -3.77 |
| ETTm2 | 96 | prefix_conditioned_stochastic_k2 | 0.161864 | -3.43 | -0.56 | -0.51 | 4.48 |
| ETTm2 | 192 | prefix_conditioned_stochastic_k2 | 0.216134 | -1.54 | 0.01 | 0.01 | 2.80 |
| ETTm2 | 336 | prefix_conditioned_stochastic_k2 | 0.266621 | -1.00 | 0.09 | 0.04 | 1.26 |
| ETTm2 | 720 | prefix_conditioned_stochastic_k2 | 0.341705 | -0.54 | 0.23 | 0.16 | -0.54 |
| ETTm2 | 96 | target_set_decoder_multiprefix | 0.161200 | -3.83 | -0.97 | -0.92 | 4.05 |
| ETTm2 | 192 | target_set_decoder_multiprefix | 0.215558 | -1.80 | -0.26 | -0.26 | 2.53 |
| ETTm2 | 336 | target_set_decoder_multiprefix | 0.266311 | -1.11 | -0.02 | -0.07 | 1.14 |
| ETTm2 | 720 | target_set_decoder_multiprefix | 0.341962 | -0.47 | 0.31 | 0.24 | -0.47 |
| Weather | 96 | prefix_conditioned_stochastic_k2 | 0.141668 | -1.05 | 0.17 | 0.17 | 0.88 |
| Weather | 192 | prefix_conditioned_stochastic_k2 | 0.182622 | -1.21 | 0.03 | 0.01 | 0.21 |
| Weather | 336 | prefix_conditioned_stochastic_k2 | 0.232074 | -1.04 | 0.15 | 0.11 | -0.25 |
| Weather | 720 | prefix_conditioned_stochastic_k2 | 0.304131 | -0.86 | 0.17 | 0.16 | -0.86 |
| Weather | 96 | target_set_decoder_multiprefix | 0.141595 | -1.10 | 0.12 | 0.12 | 0.83 |
| Weather | 192 | target_set_decoder_multiprefix | 0.182423 | -1.32 | -0.07 | -0.09 | 0.10 |
| Weather | 336 | target_set_decoder_multiprefix | 0.231836 | -1.14 | 0.04 | 0.00 | -0.35 |
| Weather | 720 | target_set_decoder_multiprefix | 0.303574 | -1.04 | -0.01 | -0.02 | -1.04 |
