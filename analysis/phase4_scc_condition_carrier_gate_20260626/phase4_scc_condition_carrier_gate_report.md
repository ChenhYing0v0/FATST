# Phase4 SCC Condition Carrier Gate Report

## Decision

[Strong Evidence] SCC condition carrier is active and improves over the adapter carrier on Weather,
but it does not pass the core gate against R.3. The state-open variant gets one Weather h720 win,
yet remains worse than R.3 on mean MSE and most horizons.

[Decision] SCC-E1 fails as a core-route candidate. Do not continue local sweeps over aux weight,
top ratio, or condition-delta size. The next step should roll back to Step 2/3 and reassess whether
Phase4 should pivot from local supervision scheduling to future-aware representation or pretraining.

## Candidate vs R.3

| candidate_strategy | dataset | settings | mse_wins | mean_relative_mse_pct | mae_wins | mean_relative_mae_pct |
| --- | --- | --- | --- | --- | --- | --- |
| scc_condition_delta_detached | ETTh2 | 4 | 0 | +2.65% | 0 | +2.00% |
| scc_condition_delta_detached | Weather | 4 | 0 | +1.96% | 0 | +2.01% |
| scc_condition_delta_state_open | ETTh2 | 4 | 0 | +2.83% | 0 | +2.08% |
| scc_condition_delta_state_open | Weather | 4 | 1 | +0.68% | 0 | +2.01% |

## Training Summary

| strategy | dataset | epochs_ran | best_epoch | post_best_val_drift_pct | last_condition_delta_grad_norm |
| --- | --- | --- | --- | --- | --- |
| single_720_prefix_risk | ETTh2 | 14 | 4 | +9.43% | 0.000000 |
| single_720_prefix_risk | Weather | 14 | 4 | +8.01% | 0.000000 |
| r3_prefix_risk | ETTh2 | 11 | 1 | +19.19% | 0.000000 |
| r3_prefix_risk | Weather | 13 | 3 | +13.39% | 0.000000 |
| dynamic_residual_stability_routing | ETTh2 | 13 | 3 | +12.95% | 0.000000 |
| dynamic_residual_stability_routing | Weather | 12 | 2 | +5.88% | 0.000000 |
| scc_condition_delta_detached | ETTh2 | 14 | 4 | +9.32% | 0.062445 |
| scc_condition_delta_detached | Weather | 17 | 7 | +8.96% | 0.060231 |
| scc_condition_delta_state_open | ETTh2 | 14 | 4 | +7.47% | 0.057568 |
| scc_condition_delta_state_open | Weather | 12 | 2 | +5.16% | 0.054860 |

## Trace Summary

| strategy | dataset | trace_rows | mean_learnable_blocks | mean_noisy_blocks | mean_noisy_suppression_ratio | mean_condition_delta_abs_residual | last_condition_delta_abs_residual |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single_720_prefix_risk | ETTh2 | 826 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| single_720_prefix_risk | Weather | 2000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| r3_prefix_risk | ETTh2 | 2000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| r3_prefix_risk | Weather | 2000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| dynamic_residual_stability_routing | ETTh2 | 767 | 1.160365 | 1.443286 | 0.360821 | 0.000000 | 0.000000 |
| dynamic_residual_stability_routing | Weather | 2000 | 1.440500 | 1.866000 | 0.466500 | 0.000000 | 0.000000 |
| scc_condition_delta_detached | ETTh2 | 826 | 1.157385 | 1.443099 | 0.360775 | 0.050529 | 0.069285 |
| scc_condition_delta_detached | Weather | 2000 | 1.412500 | 1.861000 | 0.465250 | 0.112761 | 0.134620 |
| scc_condition_delta_state_open | ETTh2 | 826 | 1.157385 | 1.443099 | 0.360775 | 0.054470 | 0.074357 |
| scc_condition_delta_state_open | Weather | 2000 | 1.412500 | 1.861000 | 0.465250 | 0.120207 | 0.155570 |

## Checkpoint Selection Diagnostics

| strategy | dataset | selector | best_epoch | official_best_epoch | official_gap_to_selector_best_pct |
| --- | --- | --- | --- | --- | --- |
| single_720_prefix_risk | ETTh2 | long_mean | 4 | 4 | +0.00% |
| single_720_prefix_risk | ETTh2 | h720 | 4 | 4 | +0.00% |
| single_720_prefix_risk | Weather | long_mean | 4 | 4 | +0.00% |
| single_720_prefix_risk | Weather | h720 | 4 | 4 | +0.00% |
| r3_prefix_risk | ETTh2 | long_mean | 3 | 1 | +0.21% |
| r3_prefix_risk | ETTh2 | h720 | 3 | 1 | +0.49% |
| r3_prefix_risk | Weather | long_mean | 3 | 3 | +0.00% |
| r3_prefix_risk | Weather | h720 | 3 | 3 | +0.00% |
| dynamic_residual_stability_routing | ETTh2 | long_mean | 3 | 3 | +0.00% |
| dynamic_residual_stability_routing | ETTh2 | h720 | 3 | 3 | +0.00% |
| dynamic_residual_stability_routing | Weather | long_mean | 2 | 2 | +0.00% |
| dynamic_residual_stability_routing | Weather | h720 | 2 | 2 | +0.00% |
| scc_condition_delta_detached | ETTh2 | long_mean | 3 | 4 | +0.46% |
| scc_condition_delta_detached | ETTh2 | h720 | 3 | 4 | +0.22% |
| scc_condition_delta_detached | Weather | long_mean | 7 | 7 | +0.00% |
| scc_condition_delta_detached | Weather | h720 | 7 | 7 | +0.00% |
| scc_condition_delta_state_open | ETTh2 | long_mean | 3 | 4 | +0.50% |
| scc_condition_delta_state_open | ETTh2 | h720 | 3 | 4 | +0.52% |
| scc_condition_delta_state_open | Weather | long_mean | 7 | 2 | +0.07% |
| scc_condition_delta_state_open | Weather | h720 | 7 | 2 | +0.59% |

## Gate

Pass only if a SCC candidate is within `+0.5%` mean MSE vs R.3, wins at least `2/4` Weather horizons,
keeps ETTh2 h96/h192 within `+1.0%`, has drift below about `8%`, and shows non-collapsed carrier trace.
