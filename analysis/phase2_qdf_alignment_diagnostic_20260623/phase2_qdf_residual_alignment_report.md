# Phase2-E2 QDF-to-FATST Residual Alignment Diagnostic

## Decision Status

[Decision] residual-level artifacts are complete; use the tables below to judge whether QDF learned matrices align with FATST R.3 residuals.

## Gate

- prediction_artifacts_complete: `True`
- qdf_matrices_complete: `True`
- ready_for_alignment_decision: `True`

- missing_count: `0`

## Matrix Family Summary

| Matrix family | Count | Mean ratio | Std ratio | Pearson loss~MSE | Spearman loss~MSE | Specialist ratio | Non-specialist ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 12 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| prefix_risk | 12 | 1.460518 | 0.504141 | 0.254231 | 0.244755 | 1.528278 | 1.426638 |
| qdf_all_precision | 12 | 0.545758 | 0.153360 | 0.629508 | 0.706294 | 0.502381 | 0.567446 |
| qdf_diag_precision | 12 | 0.998441 | 0.002543 | 0.999952 | 1.000000 | 0.999521 | 0.997902 |
| qdf_off_diag_precision | 12 | 0.531174 | 0.159856 | 0.614531 | 0.748252 | 0.481565 | 0.555978 |
| static_train_target_offdiag | 12 | 0.156607 | 0.079932 | 0.873014 | 0.958042 | 0.168521 | 0.150650 |

## Interpretation Rule

[Fact] `ratio_to_residual_mse` normalizes each matrix loss by the plain R.3 residual MSE in the same dataset-horizon setting.

[Decision Rule] If QDF `off_diag/all` ratios separate specialist gaps or hard horizons better than `static_train_target_offdiag`, then the next local mechanism should be learned or validation-informed. If they do not, stop the objective route.
