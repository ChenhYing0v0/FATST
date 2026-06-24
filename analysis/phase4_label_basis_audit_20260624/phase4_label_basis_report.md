# Phase4 Label Basis Audit

## Purpose

[Fact] This diagnostic uses train split future labels only. It does not train a model and does not inspect validation/test labels.

[Question] Is there enough nontrivial future-label covariance / low-rank component structure to justify horizon-agnostic component-space supervision before implementing a training loss?

## Metrics

- `effective_rank`: entropy rank of the step covariance eigenvalue distribution; smaller than `pred_len` means label variation concentrates in fewer components.
- `topK_variance_ratio`: cumulative variance captured by the top K eigen-components.
- `mean_abs_offdiag_corr`: average absolute step-to-step label correlation outside the diagonal.
- `share_abs_corr_gt_0_25`: fraction of off-diagonal step correlations whose absolute value exceeds `0.25`.

## Summary

| Dataset | Effective rank | Top16 var | Top32 var | Top64 var | Mean abs offdiag corr | Share abs corr > 0.25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | 11.47 | 0.873 | 0.916 | 0.948 | 0.532 | 1.000 |
| ETTm1 | 9.74 | 0.881 | 0.940 | 0.967 | 0.554 | 1.000 |
| Weather | 24.18 | 0.790 | 0.833 | 0.865 | 0.411 | 0.864 |

## Decision Rule

[Decision] Candidate C remains viable if top components capture meaningful variance and off-diagonal correlations are nontrivial. If the covariance is close to diagonal/full-rank, component supervision should be deprioritized in favor of random interval supervision.

## Current Interpretation

[Strong Evidence] The train-label covariance is structured enough to justify the next residual-projection diagnostic for component-space supervision.
