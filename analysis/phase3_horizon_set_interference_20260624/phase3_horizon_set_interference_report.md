# Phase3 Horizon-Set Interference Analysis

## What This Tests

[Fact] This report analyzes `PatchEncoderPrefixRiskWeightedH96H720`, an R.3 carrier trained only on horizons `96,720` with no Phase3-C operator and no `window_index_norm`.

[Question] Does the earlier positive `h96,h720` result come from the Phase3-C operator, or from removing intermediate horizons `192/336` from the training objective?

## Statistic Definitions

- `relative_mse_vs_r3_full_pct`: `(control_mse / R3_full_mse - 1) * 100`; negative is better than full-horizon R.3.
- `operator_increment_vs_control_pct`: `(operator_reduced_mse / control_reduced_mse - 1) * 100`; negative means the Phase3-C operator improves over the reduced-set carrier.
- `full_operator_vs_reduced_operator_pct`: `(operator_full_mse / operator_reduced_mse - 1) * 100`; positive means the operator degrades when `192/336` are restored.
- `effective_pressure_delta_pct`: `(control_effective_pressure / R3_full_effective_pressure - 1) * 100`; positive means the reduced horizon set gives that segment more objective pressure.
- `observed gap`: previously flagged aggregate or H720-segment settings where R.3 still had a notable gap.

## Decision

[Decision] The new mainline should be `horizon-set interference`, not a pure operator story.

- reduced horizon set alone sufficient: `False`
- reduced horizon set is material factor: `True`
- operator-only claim rejected: `True`
- horizon-set interference mainline supported: `True`

## Key Results

- Control wins vs full-horizon R.3: `3/6`.
- Control mean relative MSE vs full-horizon R.3: `+0.49%`.
- H96 mean relative MSE: `+1.39%`.
- H720 mean relative MSE: `-0.41%`.
- Observed aggregate-gap wins: `1/2`.
- Observed H720 segment-gap wins: `3/3`.
- Operator reduced-set wins over this control: `3/6`.
- Operator reduced-set mean increment vs control: `-0.97%`.
- Full-set operator degradation vs reduced-set operator: `+2.49%`.
- Mean 337-720 effective-pressure delta from removing 192/336: `+92.28%`.
- Max prefix mismatch MSE: `4.823e-14`.
- Epochs recorded: `{'ETTh2': 12, 'ETTm1': 13, 'Weather': 13}`.

## Interpretation

[Fact] The reduced-set carrier improves the H720 average but hurts the H96 average. Therefore, removing `192/336` is not sufficient by itself to reproduce the full Phase3-C `h96,h720` result.

[Fact] The Phase3-C history-only operator still beats this reduced-set carrier on part of the reduced-set matrix. Therefore, the operator has conditional value when the horizon set is sparse.

[Strong Evidence] The same operator fails after restoring `96,192,336,720`. This rejects a simple operator-only paper story and points to interaction between the operator and horizon-set/objective pressure.

[Inference] The paper-worthy problem should be reframed as: multi-horizon training creates conflicting objective pressure across future steps; useful horizon-specific adaptation must control that interference rather than merely add a target-conditioned module.

## Control vs Full-Horizon R.3

| Dataset | Horizon | Control MSE | R.3 full MSE | Rel MSE | Win | Observed gap |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| ETTh2 | 96 | 0.310178 | 0.304796 | +1.77% | False | False |
| ETTh2 | 720 | 0.408120 | 0.410473 | -0.57% | True | False |
| ETTm1 | 96 | 0.305987 | 0.298685 | +2.44% | False | True |
| ETTm1 | 720 | 0.413671 | 0.417293 | -0.87% | True | False |
| Weather | 96 | 0.147971 | 0.148026 | -0.04% | True | True |
| Weather | 720 | 0.321507 | 0.320847 | +0.21% | False | False |

## Operator Increment Under Reduced Horizon Set

| Dataset | Horizon | Control MSE | Operator reduced MSE | Operator increment | Full operator degradation |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | 0.310178 | 0.300827 | -3.01% | +3.04% |
| ETTh2 | 720 | 0.408120 | 0.410184 | +0.51% | +3.26% |
| ETTm1 | 96 | 0.305987 | 0.296043 | -3.25% | +1.55% |
| ETTm1 | 720 | 0.413671 | 0.414905 | +0.30% | +0.08% |
| Weather | 96 | 0.147971 | 0.148619 | +0.44% | +3.36% |
| Weather | 720 | 0.321507 | 0.318870 | -0.82% | +3.62% |

## H720 Segment Control vs R.3

| Dataset | Segment | Control MSE | R.3 full MSE | Rel MSE | Win | Observed gap |
| --- | --- | ---: | ---: | ---: | --- | --- |
| ETTh2 | 1-96 | 0.257801 | 0.249752 | +3.22% | False | False |
| ETTh2 | 97-192 | 0.339522 | 0.338119 | +0.42% | False | False |
| ETTh2 | 193-336 | 0.365311 | 0.369671 | -1.18% | True | True |
| ETTh2 | 337-720 | 0.478903 | 0.484043 | -1.06% | True | True |
| ETTm1 | 1-96 | 0.288449 | 0.284174 | +1.50% | False | False |
| ETTm1 | 97-192 | 0.353762 | 0.352049 | +0.49% | False | False |
| ETTm1 | 193-336 | 0.404358 | 0.405651 | -0.32% | True | False |
| ETTm1 | 337-720 | 0.463447 | 0.471249 | -1.66% | True | True |
| Weather | 1-96 | 0.147300 | 0.147463 | -0.11% | True | False |
| Weather | 97-192 | 0.232788 | 0.232687 | +0.04% | False | False |
| Weather | 193-336 | 0.311920 | 0.313093 | -0.37% | True | False |
| Weather | 337-720 | 0.390835 | 0.389141 | +0.44% | False | False |

## Objective Pressure Shift

| Dataset | Segment | Control exposure | R.3 full exposure | Control pressure | R.3 full pressure | Pressure delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | 1-96 | 0.566667 | 0.479762 | 0.819548 | 0.721703 | +13.56% |
| ETTh2 | 97-192 | 0.066667 | 0.229762 | 0.042953 | 0.153976 | -72.10% |
| ETTh2 | 193-336 | 0.100000 | 0.157143 | 0.047391 | 0.077459 | -38.82% |
| ETTh2 | 337-720 | 0.266667 | 0.133333 | 0.090108 | 0.046862 | +92.28% |
| ETTh2 | horizon_96 | 1.000000 | 1.000000 | 2.611814 | 2.611814 | +0.00% |
| ETTh2 | horizon_720 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | +0.00% |
| ETTm1 | 1-96 | 0.566667 | 0.479762 | 0.819548 | 0.721703 | +13.56% |
| ETTm1 | 97-192 | 0.066667 | 0.229762 | 0.042953 | 0.153976 | -72.10% |
| ETTm1 | 193-336 | 0.100000 | 0.157143 | 0.047391 | 0.077459 | -38.82% |
| ETTm1 | 337-720 | 0.266667 | 0.133333 | 0.090108 | 0.046862 | +92.28% |
| ETTm1 | horizon_96 | 1.000000 | 1.000000 | 2.611814 | 2.611814 | +0.00% |
| ETTm1 | horizon_720 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | +0.00% |
| Weather | 1-96 | 0.566667 | 0.479762 | 0.819548 | 0.721703 | +13.56% |
| Weather | 97-192 | 0.066667 | 0.229762 | 0.042953 | 0.153976 | -72.10% |
| Weather | 193-336 | 0.100000 | 0.157143 | 0.047391 | 0.077459 | -38.82% |
| Weather | 337-720 | 0.266667 | 0.133333 | 0.090108 | 0.046862 | +92.28% |
| Weather | horizon_96 | 1.000000 | 1.000000 | 2.611814 | 2.611814 | +0.00% |
| Weather | horizon_720 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | +0.00% |

## Next Research Direction

[Decision] Return to Step 2-3/6 of the 11-step loop: define and validate `horizon-set interference` as the problem before adding another complex architecture.

[Plan] The next minimal experiment should build a horizon-interference map with pair controls: `96,192`, `96,336`, `96,720`, `192,720`, and `336,720` on the R.3 carrier. The purpose is to identify which neighboring or distant horizons create the destructive pressure observed in the full four-horizon run.

[Plan] Only after that map is clear should we design a mechanism, likely a conflict-aware objective/sampler or horizon-clustered training schedule. A new MoE/router should be delayed until the interference source is measured.
