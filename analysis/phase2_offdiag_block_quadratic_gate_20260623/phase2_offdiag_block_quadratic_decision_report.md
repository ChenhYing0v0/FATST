# Phase2-C Region-Balanced Objective Gate Report

## Decision

[Decision] `PatchEncoderOffdiagBlockQuadratic` fails the Phase2-C region-balanced objective gate.

[Inference] Coverage balance alone is not sufficient as a paper-core objective. If it fails while prefix consistency remains intact, the next repair must add target covariance/novelty evidence or stop the objective-only path.

## Main Metrics vs R.3

- MSE wins vs R.3: `1/12`.
- MAE wins vs R.3: `0/12`.
- Mean relative MSE vs R.3: `+0.05%`.
- ETTh2 mean relative MSE vs R.3: `+0.06%`.
- ETTm1 mean relative MSE vs R.3: `+0.07%`.
- Weather mean relative MSE vs R.3: `+0.01%`.

## Secondary Metrics

- MSE wins vs uniform target-set: `10/12`.
- Mean relative MSE vs uniform target-set: `-0.99%`.
- MSE wins vs FixedHead: `8/12`.
- Mean relative MSE vs FixedHead: `-0.39%`.

## Gate

- mean_mse_vs_r3_improves: `False`
- mse_wins_vs_r3_at_least_7: `False`
- no_dataset_degrades_over_0_3pct: `True`
- specialist_gap_wins_at_least_2: `False`
- h720_stability_regions_not_worse: `True`
- prefix_consistency_pass: `True`
- pass: `False`

## Per-Horizon Metrics vs R.3

| Dataset | Horizon | Specialist gap | Relative MSE vs R.3 | Candidate MSE | R.3 MSE |
| --- | ---: | --- | ---: | ---: | ---: |
| ETTh2 | 96 | False | +0.05% | 0.304961 | 0.304796 |
| ETTh2 | 192 | False | +0.05% | 0.369235 | 0.369043 |
| ETTh2 | 336 | False | +0.04% | 0.383061 | 0.382910 |
| ETTh2 | 720 | True | +0.08% | 0.410807 | 0.410473 |
| ETTm1 | 96 | True | +0.10% | 0.298982 | 0.298685 |
| ETTm1 | 192 | False | +0.07% | 0.329898 | 0.329662 |
| ETTm1 | 336 | False | +0.07% | 0.360991 | 0.360729 |
| ETTm1 | 720 | True | +0.05% | 0.417522 | 0.417293 |
| Weather | 96 | True | -0.00% | 0.148024 | 0.148026 |
| Weather | 192 | False | +0.00% | 0.192415 | 0.192409 |
| Weather | 336 | False | +0.01% | 0.244808 | 0.244793 |
| Weather | 720 | False | +0.02% | 0.320924 | 0.320847 |

## H720 Region Stability vs R.3

| Dataset | Segment | Stability region | Relative MSE vs R.3 | Candidate MSE | R.3 MSE |
| --- | --- | --- | ---: | ---: | ---: |
| ETTh2 | 1-96 | False | +0.01% | 0.249771 | 0.249752 |
| ETTh2 | 97-192 | False | +0.05% | 0.338273 | 0.338119 |
| ETTh2 | 193-336 | True | +0.08% | 0.369957 | 0.369671 |
| ETTh2 | 337-720 | True | +0.10% | 0.484519 | 0.484043 |
| ETTm1 | 1-96 | False | +0.16% | 0.284626 | 0.284174 |
| ETTm1 | 97-192 | False | +0.07% | 0.352299 | 0.352049 |
| ETTm1 | 193-336 | False | +0.08% | 0.405991 | 0.405651 |
| ETTm1 | 337-720 | True | +0.03% | 0.471376 | 0.471249 |
| Weather | 1-96 | False | -0.00% | 0.147457 | 0.147463 |
| Weather | 97-192 | False | +0.01% | 0.232702 | 0.232687 |
| Weather | 193-336 | False | +0.01% | 0.313133 | 0.313093 |
| Weather | 337-720 | True | +0.03% | 0.389267 | 0.389141 |

## Objective Weights

| Dataset | Scope | Mean weight | Weighted pressure share | Shift vs uniform |
| --- | --- | ---: | ---: | ---: |
| ETTh2 | 1-96 | 2.611814 | 0.7217 | +50.43% |
| ETTh2 | 97-192 | 1.163544 | 0.1540 | -32.98% |
| ETTh2 | 193-336 | 0.855834 | 0.0775 | -50.71% |
| ETTh2 | 337-720 | 0.610223 | 0.0469 | -64.85% |
| ETTm1 | 1-96 | 2.611814 | 0.7217 | +50.43% |
| ETTm1 | 97-192 | 1.163544 | 0.1540 | -32.98% |
| ETTm1 | 193-336 | 0.855834 | 0.0775 | -50.71% |
| ETTm1 | 337-720 | 0.610223 | 0.0469 | -64.85% |
| Weather | 1-96 | 2.611814 | 0.7217 | +50.43% |
| Weather | 97-192 | 1.163544 | 0.1540 | -32.98% |
| Weather | 193-336 | 0.855834 | 0.0775 | -50.71% |
| Weather | 337-720 | 0.610223 | 0.0469 | -64.85% |

## Rollback Rule

If region-balanced fails, do not tune region multipliers by hand. Either add a source-grounded covariance/novelty prior and test it as a distinct Phase2-C repair, or stop the objective-only path and return to base architecture selection.
