# Phase2-C Region-Balanced Objective Gate Report

## Decision

[Decision] `PatchEncoderStepCovarianceBalanced` fails the Phase2-C region-balanced objective gate.

[Inference] Coverage balance alone is not sufficient as a paper-core objective. If it fails while prefix consistency remains intact, the next repair must add target covariance/novelty evidence or stop the objective-only path.

## Main Metrics vs R.3

- MSE wins vs R.3: `2/12`.
- MAE wins vs R.3: `0/12`.
- Mean relative MSE vs R.3: `+0.76%`.
- ETTh2 mean relative MSE vs R.3: `-0.09%`.
- ETTm1 mean relative MSE vs R.3: `+1.35%`.
- Weather mean relative MSE vs R.3: `+1.03%`.

## Secondary Metrics

- MSE wins vs uniform target-set: `12/12`.
- Mean relative MSE vs uniform target-set: `-0.28%`.
- MSE wins vs FixedHead: `6/12`.
- Mean relative MSE vs FixedHead: `+0.33%`.

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
| ETTh2 | 96 | False | +0.64% | 0.306755 | 0.304796 |
| ETTh2 | 192 | False | -0.66% | 0.366609 | 0.369043 |
| ETTh2 | 336 | False | -0.95% | 0.379263 | 0.382910 |
| ETTh2 | 720 | True | +0.62% | 0.413026 | 0.410473 |
| ETTm1 | 96 | True | +2.38% | 0.305783 | 0.298685 |
| ETTm1 | 192 | False | +1.67% | 0.335178 | 0.329662 |
| ETTm1 | 336 | False | +1.00% | 0.364319 | 0.360729 |
| ETTm1 | 720 | True | +0.34% | 0.418718 | 0.417293 |
| Weather | 96 | True | +1.34% | 0.150009 | 0.148026 |
| Weather | 192 | False | +1.01% | 0.194351 | 0.192409 |
| Weather | 336 | False | +0.94% | 0.247098 | 0.244793 |
| Weather | 720 | False | +0.83% | 0.323500 | 0.320847 |

## H720 Region Stability vs R.3

| Dataset | Segment | Stability region | Relative MSE vs R.3 | Candidate MSE | R.3 MSE |
| --- | --- | --- | ---: | ---: | ---: |
| ETTh2 | 1-96 | False | +1.79% | 0.254231 | 0.249752 |
| ETTh2 | 97-192 | False | -0.75% | 0.335595 | 0.338119 |
| ETTh2 | 193-336 | True | -1.25% | 0.365061 | 0.369671 |
| ETTh2 | 337-720 | True | +1.24% | 0.490069 | 0.484043 |
| ETTm1 | 1-96 | False | +2.53% | 0.291367 | 0.284174 |
| ETTm1 | 97-192 | False | +1.20% | 0.356276 | 0.352049 |
| ETTm1 | 193-336 | False | +0.37% | 0.407142 | 0.405651 |
| ETTm1 | 337-720 | True | -0.16% | 0.470508 | 0.471249 |
| Weather | 1-96 | False | +1.13% | 0.149129 | 0.147463 |
| Weather | 97-192 | False | +0.56% | 0.233988 | 0.232687 |
| Weather | 193-336 | False | +0.69% | 0.315253 | 0.313093 |
| Weather | 337-720 | True | +0.88% | 0.392564 | 0.389141 |

## Objective Weights

| Dataset | Scope | Mean weight | Weighted pressure share | Shift vs uniform |
| --- | --- | ---: | ---: | ---: |
| ETTh2 | 1-96 | 0.913598 | 0.4807 | +0.19% |
| ETTh2 | 97-192 | 0.774251 | 0.1951 | -15.09% |
| ETTh2 | 193-336 | 0.951528 | 0.1640 | +4.35% |
| ETTh2 | 337-720 | 1.096215 | 0.1603 | +20.21% |
| ETTm1 | 1-96 | 1.184000 | 0.5501 | +14.66% |
| ETTm1 | 97-192 | 0.781905 | 0.1740 | -24.28% |
| ETTm1 | 193-336 | 0.937577 | 0.1427 | -9.20% |
| ETTm1 | 337-720 | 1.031932 | 0.1332 | -0.07% |
| Weather | 1-96 | 0.956828 | 0.4813 | +0.32% |
| Weather | 97-192 | 0.882781 | 0.2127 | -7.44% |
| Weather | 193-336 | 0.963392 | 0.1587 | +1.01% |
| Weather | 337-720 | 1.053825 | 0.1473 | +10.49% |

## Rollback Rule

If region-balanced fails, do not tune region multipliers by hand. Either add a source-grounded covariance/novelty prior and test it as a distinct Phase2-C repair, or stop the objective-only path and return to base architecture selection.
