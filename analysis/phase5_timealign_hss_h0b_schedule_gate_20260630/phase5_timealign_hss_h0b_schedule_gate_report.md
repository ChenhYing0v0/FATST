# Phase5 TimeAlign-HSS H0B Schedule Gate

## Summary

| dataset | arm | settings | vs_full | vs_multi_prefix | vs_stochastic_prefix | vs_continuous_prefix | wins_vs_fixed | vs_fixed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | stochastic_prefix_k2 | 4 | -3.38 | -0.02 | 0.02 | -0.38 | 4 | -11.07 |
| ETTh2 | continuous_prefix_k2 | 4 | -3.13 | 0.24 | 0.28 | -0.12 | 4 | -10.86 |
| ETTh2 | continuous_prefix_pool96 | 4 | -2.63 | 0.75 | 0.79 | 0.39 | 4 | -10.40 |
| ETTm2 | stochastic_prefix_k2 | 4 | -1.56 | 0.02 | -0.31 | -0.49 | 1 | 2.08 |
| ETTm2 | continuous_prefix_k2 | 4 | -1.26 | 0.32 | -0.01 | -0.19 | 1 | 2.40 |
| ETTm2 | continuous_prefix_pool96 | 4 | -1.04 | 0.55 | 0.22 | 0.04 | 1 | 2.63 |
| Weather | stochastic_prefix_k2 | 4 | -1.15 | 0.02 | -0.08 | -0.22 | 2 | -0.12 |
| Weather | continuous_prefix_k2 | 4 | -0.96 | 0.21 | 0.11 | -0.02 | 2 | 0.08 |
| Weather | continuous_prefix_pool96 | 4 | -0.76 | 0.42 | 0.32 | 0.19 | 1 | 0.28 |
| ALL | stochastic_prefix_k2 | 12 | -2.03 | 0.00 | -0.13 | -0.36 | 7 | -3.04 |
| ALL | continuous_prefix_k2 | 12 | -1.78 | 0.26 | 0.13 | -0.11 | 7 | -2.79 |
| ALL | continuous_prefix_pool96 | 12 | -1.48 | 0.57 | 0.44 | 0.20 | 6 | -2.49 |

## Reading Guide

- `stochastic_prefix_k2` passes if it improves ETTm2 or overall mean over H0 `stochastic-prefix` and approaches or beats `multi-prefix`.
- `continuous_prefix_k2` passes if increasing sample count lets continuous scheduling approach `multi-prefix`.
- `continuous_prefix_pool96` passes if removing very short prefixes improves over H0 `continuous-prefix`.

## Per-Horizon Rows

| dataset | horizon | arm | mse | vs_full | vs_multi_prefix | vs_stochastic_prefix | vs_continuous_prefix | vs_fixed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | stochastic_prefix_k2 | 0.243241 | -2.36 | 0.19 | 0.25 | -0.28 | -9.81 |
| ETTh2 | 192 | stochastic_prefix_k2 | 0.282930 | -4.40 | -0.04 | 0.10 | -0.64 | -15.44 |
| ETTh2 | 336 | stochastic_prefix_k2 | 0.312028 | -4.50 | -0.26 | 0.12 | -0.68 | -16.77 |
| ETTh2 | 720 | stochastic_prefix_k2 | 0.394180 | -2.25 | 0.03 | -0.40 | 0.09 | -2.25 |
| ETTh2 | 96 | continuous_prefix_k2 | 0.243434 | -2.28 | 0.27 | 0.33 | -0.20 | -9.74 |
| ETTh2 | 192 | continuous_prefix_k2 | 0.284775 | -3.78 | 0.61 | 0.76 | 0.01 | -14.89 |
| ETTh2 | 336 | continuous_prefix_k2 | 0.313801 | -3.96 | 0.30 | 0.69 | -0.12 | -16.30 |
| ETTh2 | 720 | continuous_prefix_k2 | 0.393187 | -2.50 | -0.22 | -0.65 | -0.16 | -2.50 |
| ETTh2 | 96 | continuous_prefix_pool96 | 0.244063 | -2.03 | 0.53 | 0.59 | 0.06 | -9.51 |
| ETTh2 | 192 | continuous_prefix_pool96 | 0.286226 | -3.29 | 1.12 | 1.27 | 0.52 | -14.46 |
| ETTh2 | 336 | continuous_prefix_pool96 | 0.315849 | -3.33 | 0.96 | 1.34 | 0.53 | -15.75 |
| ETTh2 | 720 | continuous_prefix_pool96 | 0.395657 | -1.88 | 0.40 | -0.03 | 0.46 | -1.88 |
| ETTm2 | 96 | stochastic_prefix_k2 | 0.162700 | -2.93 | -0.05 | -0.30 | -0.81 | 5.02 |
| ETTm2 | 192 | stochastic_prefix_k2 | 0.216111 | -1.55 | -0.00 | -0.26 | -0.47 | 2.79 |
| ETTm2 | 336 | stochastic_prefix_k2 | 0.266501 | -1.04 | 0.05 | -0.25 | -0.28 | 1.22 |
| ETTm2 | 720 | stochastic_prefix_k2 | 0.341146 | -0.70 | 0.07 | -0.44 | -0.41 | -0.70 |
| ETTm2 | 96 | continuous_prefix_k2 | 0.163613 | -2.39 | 0.51 | 0.26 | -0.26 | 5.61 |
| ETTm2 | 192 | continuous_prefix_k2 | 0.216836 | -1.22 | 0.33 | 0.07 | -0.14 | 3.14 |
| ETTm2 | 336 | continuous_prefix_k2 | 0.266918 | -0.89 | 0.21 | -0.09 | -0.13 | 1.37 |
| ETTm2 | 720 | continuous_prefix_k2 | 0.341747 | -0.53 | 0.24 | -0.26 | -0.24 | -0.53 |
| ETTm2 | 96 | continuous_prefix_pool96 | 0.164652 | -1.77 | 1.15 | 0.90 | 0.38 | 6.28 |
| ETTm2 | 192 | continuous_prefix_pool96 | 0.217470 | -0.93 | 0.63 | 0.37 | 0.16 | 3.44 |
| ETTm2 | 336 | continuous_prefix_pool96 | 0.267277 | -0.75 | 0.34 | 0.04 | 0.01 | 1.51 |
| ETTm2 | 720 | continuous_prefix_pool96 | 0.341204 | -0.69 | 0.08 | -0.42 | -0.39 | -0.69 |
| Weather | 96 | stochastic_prefix_k2 | 0.141429 | -1.22 | 0.00 | -0.08 | -0.23 | 0.71 |
| Weather | 192 | stochastic_prefix_k2 | 0.182595 | -1.23 | 0.02 | -0.16 | -0.30 | 0.20 |
| Weather | 336 | stochastic_prefix_k2 | 0.231826 | -1.15 | 0.04 | -0.15 | -0.27 | -0.36 |
| Weather | 720 | stochastic_prefix_k2 | 0.303636 | -1.02 | 0.01 | 0.06 | -0.06 | -1.02 |
| Weather | 96 | continuous_prefix_k2 | 0.141693 | -1.03 | 0.19 | 0.10 | -0.04 | 0.90 |
| Weather | 192 | continuous_prefix_k2 | 0.182951 | -1.04 | 0.21 | 0.03 | -0.11 | 0.39 |
| Weather | 336 | continuous_prefix_k2 | 0.232204 | -0.99 | 0.20 | 0.01 | -0.11 | -0.19 |
| Weather | 720 | continuous_prefix_k2 | 0.304342 | -0.79 | 0.24 | 0.29 | 0.18 | -0.79 |
| Weather | 96 | continuous_prefix_pool96 | 0.142016 | -0.81 | 0.42 | 0.33 | 0.18 | 1.13 |
| Weather | 192 | continuous_prefix_pool96 | 0.183412 | -0.79 | 0.47 | 0.28 | 0.14 | 0.65 |
| Weather | 336 | continuous_prefix_pool96 | 0.232719 | -0.77 | 0.42 | 0.24 | 0.11 | 0.03 |
| Weather | 720 | continuous_prefix_pool96 | 0.304734 | -0.67 | 0.37 | 0.42 | 0.30 | -0.67 |
