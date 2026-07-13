# Phase4 Label Basis Audit Code Explanation

## Purpose

`scripts/analyze_phase4_label_basis_audit.py` is a Step 1-3 diagnostic for the
horizon-agnostic supervision reset. It does not train a model. It reads only
train split labels and estimates whether future labels have enough covariance /
low-rank structure to justify component-space supervision.

## Data Flow

1. Reuse `ForecastDataset` from `baselines/patch_encoder_target_set_decoder`.
2. For each dataset, load the train split with `seq_len=336` and `pred_len=720`.
3. Build future-label rows without materializing the full matrix:
   each chunk has target tensor `[B, T, D]`, then reshapes to `[B*D, T]`.
4. Accumulate:
   - step sum: $\sum Y_t$;
   - step cross product: $Y^\top Y$.
5. Estimate covariance:

$$
\operatorname{Cov}(Y)=\frac{Y^\top Y}{N}-\mu^\top\mu.
$$

6. Run symmetric eigendecomposition on the `720 x 720` covariance matrix.

## Statistics

- `effective_rank`: entropy rank of normalized eigenvalues. Smaller values mean
  label variation concentrates in fewer future components.
- `topK_variance_ratio`: cumulative variance explained by the top `K`
  eigen-components.
- `mean_abs_offdiag_corr`: average absolute non-diagonal step correlation.
- `share_abs_corr_gt_0_25`: share of non-diagonal correlations with absolute
  value above `0.25`.
- `component_region_energy`: squared eigenvector loading mass inside
  `1-96`, `97-192`, `193-336`, and `337-720`.

## Artifact Effects

The script writes:

- `phase4_label_basis_summary.csv`;
- `phase4_label_basis_components.csv`;
- `phase4_label_basis_region_contribution.csv`;
- `phase4_label_basis_summary.json`;
- `phase4_label_basis_report.md`.

## Theory Consistency

[Theory] If future-label steps are highly correlated and low-rank, evaluation
horizons are likely not the most natural training supervision units.

[Code Realization] The script tests this by estimating the train-label step
covariance directly and analyzing its eigen-spectrum.

[Proxy Boundary] This is label-side structure, not model residual structure.
It supports the next diagnostic, but does not prove component-space training
will improve forecasting.

[Falsification] If residual projection later shows known model gaps do not
align with component groups, component-space supervision should not proceed to
remote training.
