# Phase2-C Region-Balanced Objective Gate Report

## Decision

[Decision] `PatchEncoderRegionBalanced` fails the Phase2-C region-balanced objective gate.

[Inference] Coverage balance alone is not sufficient as a paper-core objective. If it fails while prefix consistency remains intact, the next repair must add target covariance/novelty evidence or stop the objective-only path.

## Main Metrics vs R.3

- MSE wins vs R.3: `2/12`.
- MAE wins vs R.3: `0/12`.
- Mean relative MSE vs R.3: `+1.53%`.
- ETTh2 mean relative MSE vs R.3: `-0.29%`.
- ETTm1 mean relative MSE vs R.3: `+3.19%`.
- Weather mean relative MSE vs R.3: `+1.70%`.

## Secondary Metrics

- MSE wins vs uniform target-set: `4/12`.
- Mean relative MSE vs uniform target-set: `+0.47%`.
- MSE wins vs FixedHead: `3/12`.
- Mean relative MSE vs FixedHead: `+1.10%`.

## Gate

- mean_mse_vs_r3_improves: `False`
- mse_wins_vs_r3_at_least_7: `False`
- no_dataset_degrades_over_0_3pct: `False`
- specialist_gap_wins_at_least_2: `False`
- h720_stability_regions_not_worse: `False`
- prefix_consistency_pass: `True`
- pass: `False`

## Per-Horizon Metrics vs R.3

| Dataset | Horizon | Specialist gap | Relative MSE vs R.3 | Candidate MSE | R.3 MSE |
| --- | ---: | --- | ---: | ---: | ---: |
| ETTh2 | 96 | False | +1.23% | 0.308534 | 0.304796 |
| ETTh2 | 192 | False | -1.66% | 0.362906 | 0.369043 |
| ETTh2 | 336 | False | -1.38% | 0.377618 | 0.382910 |
| ETTh2 | 720 | True | +0.66% | 0.413167 | 0.410473 |
| ETTm1 | 96 | True | +5.36% | 0.314699 | 0.298685 |
| ETTm1 | 192 | False | +3.89% | 0.342502 | 0.329662 |
| ETTm1 | 336 | False | +2.46% | 0.369616 | 0.360729 |
| ETTm1 | 720 | True | +1.04% | 0.421644 | 0.417293 |
| Weather | 96 | True | +2.48% | 0.151694 | 0.148026 |
| Weather | 192 | False | +1.75% | 0.195784 | 0.192409 |
| Weather | 336 | False | +1.49% | 0.248448 | 0.244793 |
| Weather | 720 | False | +1.08% | 0.324314 | 0.320847 |

## H720 Region Stability vs R.3

| Dataset | Segment | Stability region | Relative MSE vs R.3 | Candidate MSE | R.3 MSE |
| --- | --- | --- | ---: | ---: | ---: |
| ETTh2 | 1-96 | False | +0.65% | 0.251386 | 0.249752 |
| ETTh2 | 97-192 | False | -1.54% | 0.332923 | 0.338119 |
| ETTh2 | 193-336 | True | -0.71% | 0.367050 | 0.369671 |
| ETTh2 | 337-720 | True | +1.43% | 0.490967 | 0.484043 |
| ETTm1 | 1-96 | False | +6.29% | 0.302061 | 0.284174 |
| ETTm1 | 97-192 | False | +2.98% | 0.362546 | 0.352049 |
| ETTm1 | 193-336 | False | +1.17% | 0.410380 | 0.405651 |
| ETTm1 | 337-720 | True | -0.15% | 0.470539 | 0.471249 |
| Weather | 1-96 | False | +2.08% | 0.150537 | 0.147463 |
| Weather | 97-192 | False | +0.84% | 0.234638 | 0.232687 |
| Weather | 193-336 | False | +0.87% | 0.315831 | 0.313093 |
| Weather | 337-720 | True | +1.08% | 0.393358 | 0.389141 |

## Objective Weights

| Dataset | Scope | Mean weight | Weighted pressure share | Shift vs uniform |
| --- | --- | ---: | ---: | ---: |
| ETTh2 | 1-96 | 0.339974 | 0.2500 | -47.89% |
| ETTh2 | 97-192 | 0.709895 | 0.2500 | +8.81% |
| ETTh2 | 193-336 | 1.037952 | 0.2500 | +59.09% |
| ETTh2 | 337-720 | 1.223301 | 0.2500 | +87.50% |
| ETTm1 | 1-96 | 0.339974 | 0.2500 | -47.89% |
| ETTm1 | 97-192 | 0.709895 | 0.2500 | +8.81% |
| ETTm1 | 193-336 | 1.037952 | 0.2500 | +59.09% |
| ETTm1 | 337-720 | 1.223301 | 0.2500 | +87.50% |
| Weather | 1-96 | 0.339974 | 0.2500 | -47.89% |
| Weather | 97-192 | 0.709895 | 0.2500 | +8.81% |
| Weather | 193-336 | 1.037952 | 0.2500 | +59.09% |
| Weather | 337-720 | 1.223301 | 0.2500 | +87.50% |

## Rollback Rule

If region-balanced fails, do not tune region multipliers by hand. Either add a source-grounded covariance/novelty prior and test it as a distinct Phase2-C repair, or stop the objective-only path and return to base architecture selection.
