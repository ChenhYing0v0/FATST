# Phase2-E2 QDF-to-FATST Residual Alignment Diagnostic

## Decision Status

[Decision] diagnostic tooling is ready, but residual-level artifacts are incomplete. Do not make an alignment decision yet.

## Gate

- prediction_artifacts_complete: `False`
- qdf_matrices_complete: `True`
- ready_for_alignment_decision: `False`

- missing_count: `12`

## Missing Artifacts

- `r3_prediction:ETTh2:h96`
- `r3_prediction:ETTh2:h192`
- `r3_prediction:ETTh2:h336`
- `r3_prediction:ETTh2:h720`
- `r3_prediction:ETTm1:h96`
- `r3_prediction:ETTm1:h192`
- `r3_prediction:ETTm1:h336`
- `r3_prediction:ETTm1:h720`
- `r3_prediction:Weather:h96`
- `r3_prediction:Weather:h192`
- `r3_prediction:Weather:h336`
- `r3_prediction:Weather:h720`

## Matrix Family Summary

| Matrix family | Count | Mean ratio | Std ratio | Pearson loss~MSE | Spearman loss~MSE | Specialist ratio | Non-specialist ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## Interpretation Rule

[Fact] `ratio_to_residual_mse` normalizes each matrix loss by the plain R.3 residual MSE in the same dataset-horizon setting.

[Decision Rule] If QDF `off_diag/all` ratios separate specialist gaps or hard horizons better than `static_train_target_offdiag`, then the next local mechanism should be learned or validation-informed. If they do not, stop the objective route.
