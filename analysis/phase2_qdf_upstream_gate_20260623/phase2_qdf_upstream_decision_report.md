# Phase2-D QDF Upstream Reproduction Gate Report

## Decision

[Decision] QDF upstream reproduction gate is incomplete or fails.

- meta_types_present: `all`
- completed_metric_rows: `12`
- all_runs_completed: `12/12`

## Gate

- all_12_runs_complete: `True`
- diag_control_available: `False`
- off_diag_control_available: `False`
- all_vs_diag_mean_mse_improves: `False`
- all_vs_diag_wins_at_least_7: `False`
- specialist_gap_wins_at_least_2: `False`
- covariance_artifacts_present: `True`
- pass: `False`

## All Meta-Type Metrics

| Dataset | Horizon | MSE | MAE | Cov loss | Cov artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| ETTh2 | 96 | 0.285880 | 0.337921 | 0.077645 | True |
| ETTh2 | 192 | 0.361037 | 0.388220 | 0.296859 | True |
| ETTh2 | 336 | 0.407588 | 0.422399 | 0.122215 | True |
| ETTh2 | 720 | 0.419218 | 0.438822 | 0.173580 | True |
| ETTm1 | 96 | 0.306606 | 0.348975 | 0.121592 | True |
| ETTm1 | 192 | 0.352415 | 0.376267 | 0.287310 | True |
| ETTm1 | 336 | 0.382601 | 0.397518 | 0.169737 | True |
| ETTm1 | 720 | 0.441164 | 0.434478 | 0.222438 | True |
| Weather | 96 | 0.159555 | 0.202416 | 0.055837 | True |
| Weather | 192 | 0.209021 | 0.246954 | 0.131674 | True |
| Weather | 336 | 0.264798 | 0.288555 | 0.127803 | True |
| Weather | 720 | 0.342472 | 0.339333 | 0.118382 | True |

## Meta-Type Comparisons

[Fact] No meta-type control comparison is available yet. Run `META_TYPES="diag off_diag"` to complete controls.

## Interpretation

[Fact] This report parses native QDF upstream outputs. It does not compare directly against FATST R.3 because the upstream model and training protocol are different.

[Decision Rule] QDF should only be localized into FATST if `meta_type=all` beats its own diagonal control and learned covariance artifacts are present. If only `all` has run, this gate remains incomplete even when metrics exist.
