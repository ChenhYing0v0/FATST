# Phase5 StageB B12-STBO Small Gate Report

## Scope

Required arms: `a6_clean`, `stbo_shared`, `stbo_bank4`, `stbo_dct`, `stbo_independent`.
Datasets: ETTh2, ETTm1, Weather. Horizons: 96, 192, 336, 720.

## Summary

| comparison | dataset | settings | mse_wins | mean_relative_mse_pct | mean_relative_mae_pct |
| --- | --- | --- | --- | --- | --- |
| stbo_bank4_vs_a6_clean | ETTh2 | 4 | 0 | +4.59% | +2.50% |
| stbo_bank4_vs_a6_clean | ETTm1 | 4 | 0 | +1.36% | +0.88% |
| stbo_bank4_vs_a6_clean | Weather | 4 | 3 | -0.0054% | +0.05% |
| stbo_bank4_vs_a6_clean | ALL | 12 | 3 | +1.98% | +1.14% |
| stbo_bank4_vs_stbo_dct | ETTh2 | 4 | 0 | +0.78% | +0.62% |
| stbo_bank4_vs_stbo_dct | ETTm1 | 4 | 0 | +0.44% | +0.67% |
| stbo_bank4_vs_stbo_dct | Weather | 4 | 4 | -0.04% | +0.08% |
| stbo_bank4_vs_stbo_dct | ALL | 12 | 4 | +0.40% | +0.46% |
| stbo_bank4_vs_stbo_independent | ETTh2 | 4 | 0 | +1.24% | +0.54% |
| stbo_bank4_vs_stbo_independent | ETTm1 | 4 | 4 | -0.48% | -0.67% |
| stbo_bank4_vs_stbo_independent | Weather | 4 | 4 | -0.19% | -0.22% |
| stbo_bank4_vs_stbo_independent | ALL | 12 | 8 | +0.19% | -0.12% |
| stbo_bank4_vs_stbo_shared | ETTh2 | 4 | 0 | +1.19% | +0.33% |
| stbo_bank4_vs_stbo_shared | ETTm1 | 4 | 0 | +0.09% | -0.09% |
| stbo_bank4_vs_stbo_shared | Weather | 4 | 4 | -0.16% | -0.18% |
| stbo_bank4_vs_stbo_shared | ALL | 12 | 4 | +0.37% | +0.02% |
| stbo_dct_vs_a6_clean | ETTh2 | 4 | 0 | +3.77% | +1.86% |
| stbo_dct_vs_a6_clean | ETTm1 | 4 | 0 | +0.91% | +0.20% |
| stbo_dct_vs_a6_clean | Weather | 4 | 2 | +0.03% | -0.03% |
| stbo_dct_vs_a6_clean | ALL | 12 | 2 | +1.57% | +0.68% |
| stbo_independent_vs_a6_clean | ETTh2 | 4 | 0 | +3.30% | +1.94% |
| stbo_independent_vs_a6_clean | ETTm1 | 4 | 0 | +1.84% | +1.56% |
| stbo_independent_vs_a6_clean | Weather | 4 | 0 | +0.19% | +0.27% |
| stbo_independent_vs_a6_clean | ALL | 12 | 0 | +1.78% | +1.26% |
| stbo_shared_vs_a6_clean | ETTh2 | 4 | 0 | +3.36% | +2.16% |
| stbo_shared_vs_a6_clean | ETTm1 | 4 | 0 | +1.27% | +0.97% |
| stbo_shared_vs_a6_clean | Weather | 4 | 0 | +0.15% | +0.23% |
| stbo_shared_vs_a6_clean | ALL | 12 | 0 | +1.59% | +1.12% |
| stbo_shared_vs_stbo_dct | ETTh2 | 4 | 2 | -0.40% | +0.29% |
| stbo_shared_vs_stbo_dct | ETTm1 | 4 | 0 | +0.36% | +0.77% |
| stbo_shared_vs_stbo_dct | Weather | 4 | 0 | +0.12% | +0.26% |
| stbo_shared_vs_stbo_dct | ALL | 12 | 2 | +0.03% | +0.44% |
| stbo_shared_vs_stbo_independent | ETTh2 | 4 | 2 | +0.05% | +0.21% |
| stbo_shared_vs_stbo_independent | ETTm1 | 4 | 4 | -0.56% | -0.58% |
| stbo_shared_vs_stbo_independent | Weather | 4 | 3 | -0.04% | -0.04% |
| stbo_shared_vs_stbo_independent | ALL | 12 | 9 | -0.18% | -0.13% |

## Gate Reading

[Decision] `generic_local_basis_control_explains`: learned STBO does not beat the fixed local DCT control.

- STBO-shared vs A6: mean MSE +1.59%, wins 0/12.
- STBO-bank4 vs A6: mean MSE +1.98%, wins 3/12.
- STBO-DCT vs A6: mean MSE +1.57%, wins 2/12.
- STBO-independent vs A6: mean MSE +1.78%, wins 0/12.
- STBO-shared vs DCT: mean MSE +0.03%, wins 2/12.
- STBO-bank4 vs DCT: mean MSE +0.40%, wins 4/12.
- STBO-shared vs independent: mean MSE -0.18%, wins 9/12.
- STBO-bank4 vs independent: mean MSE +0.19%, wins 8/12.

## Failure Attribution Rule

This report may reject only the tested B12-STBO implementation unless learned shared/bank STBO is stable and still fails the required DCT and independent controls.
If DCT matches learned STBO, classify the result as `generic_local_basis_control_explains`, not as a rejection of all native multi-horizon operators.
If only independent tile wins, classify the result as `independent_tile_capacity_explains`, not as a shared subspace method.

