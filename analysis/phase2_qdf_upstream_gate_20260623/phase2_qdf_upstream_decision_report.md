# Phase2-D QDF Upstream Reproduction Gate Report

## Decision

[Decision] QDF upstream reproduction gate passes.

- meta_types_present: `all, diag, off_diag`
- completed_metric_rows: `36`
- all_runs_completed: `12/12`

## Gate

- all_12_runs_complete: `True`
- diag_control_available: `True`
- off_diag_control_available: `True`
- all_vs_diag_mean_mse_improves: `True`
- all_vs_diag_wins_at_least_7: `True`
- specialist_gap_wins_at_least_2: `True`
- covariance_artifacts_present: `True`
- pass: `True`

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

| Candidate | Baseline | Dataset | Horizon | Specialist gap | Relative MSE | Candidate MSE | Baseline MSE |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| all | diag | ETTh2 | 96 | False | -2.39% | 0.285880 | 0.292882 |
| all | diag | ETTh2 | 192 | False | -0.07% | 0.361037 | 0.361283 |
| all | diag | ETTh2 | 336 | False | -1.89% | 0.407588 | 0.415434 |
| all | diag | ETTh2 | 720 | True | -3.22% | 0.419218 | 0.433160 |
| all | diag | ETTm1 | 96 | True | -0.97% | 0.306606 | 0.309600 |
| all | diag | ETTm1 | 192 | False | -0.43% | 0.352415 | 0.353926 |
| all | diag | ETTm1 | 336 | False | -1.37% | 0.382601 | 0.387927 |
| all | diag | ETTm1 | 720 | True | -1.71% | 0.441164 | 0.448851 |
| all | diag | Weather | 96 | True | -0.42% | 0.159555 | 0.160221 |
| all | diag | Weather | 192 | False | +0.09% | 0.209021 | 0.208827 |
| all | diag | Weather | 336 | False | -0.14% | 0.264798 | 0.265168 |
| all | diag | Weather | 720 | False | -0.45% | 0.342472 | 0.344013 |
| all | off_diag | ETTh2 | 96 | False | +0.42% | 0.285880 | 0.284695 |
| all | off_diag | ETTh2 | 192 | False | +0.00% | 0.361037 | 0.361033 |
| all | off_diag | ETTh2 | 336 | False | +0.12% | 0.407588 | 0.407107 |
| all | off_diag | ETTh2 | 720 | True | +0.47% | 0.419218 | 0.417271 |
| all | off_diag | ETTm1 | 96 | True | -1.49% | 0.306606 | 0.311258 |
| all | off_diag | ETTm1 | 192 | False | +0.00% | 0.352415 | 0.352410 |
| all | off_diag | ETTm1 | 336 | False | +0.07% | 0.382601 | 0.382341 |
| all | off_diag | ETTm1 | 720 | True | +0.01% | 0.441164 | 0.441138 |
| all | off_diag | Weather | 96 | True | +0.48% | 0.159555 | 0.158786 |
| all | off_diag | Weather | 192 | False | +0.47% | 0.209021 | 0.208048 |
| all | off_diag | Weather | 336 | False | +0.55% | 0.264798 | 0.263345 |
| all | off_diag | Weather | 720 | False | -0.35% | 0.342472 | 0.343690 |

## Interpretation

[Fact] This report parses native QDF upstream outputs. It does not compare directly against FATST R.3 because the upstream model and training protocol are different.

[Decision Rule] QDF should only be localized into FATST if `meta_type=all` beats its own diagonal control and learned covariance artifacts are present. If only `all` has run, this gate remains incomplete even when metrics exist.
