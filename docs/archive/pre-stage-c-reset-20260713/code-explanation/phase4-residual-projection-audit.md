# Phase4 Residual Projection Audit Code Explanation

## Purpose

`scripts/analyze_phase4_residual_projection_audit.py` is the second diagnostic
for the horizon-agnostic supervision reset. It uses existing R.3 prediction
artifacts and train-label component bases to test whether current forecast
errors have meaningful component-space structure.

It does not train a model and does not require new remote experiments.

## Data Flow

1. Load R.3 predictions from:
   `analysis/phase2_qdf_alignment_diagnostic_20260623/raw`.
2. For each `dataset` and `horizon`, estimate the train-label covariance basis
   using the same logic as `analyze_phase4_label_basis_audit.py`.
3. Load `predictions_test.npz`:

$$
\hat{Y},Y \in \mathbb{R}^{B \times H \times D}
$$

4. Build residual vectors:

$$
E = \hat{Y}-Y,\qquad
E' \in \mathbb{R}^{(B D) \times H}
$$

5. Project residuals onto train-label eigenvectors:

$$
C = E'P
$$

6. Compare residual component energy with label component variance.

## Statistics

- `residual_topK_energy_share`: fraction of residual squared energy in the top
  `K` train-label components.
- `label_topK_variance_ratio`: fraction of train-label variance explained by
  those same components.
- `residual_topK_over_label_ratio`: residual concentration normalized by label
  concentration.
- `topK_reconstruction_mse_share`: H720 segment residual energy reconstructed
  from top `K` components divided by full segment residual MSE.

## Result Meaning

[Fact] Existing R.3 residuals are structured in component space: top16 components
explain most residual energy across both gap and non-gap rows.

[Inference] Known gap rows are not more concentrated in dominant top components.
Their top16-over-label ratio is slightly lower than non-gap rows, and H720
segment gaps are almost indistinguishable by top16 reconstruction share.

## Design Consequence

[Decision] A top-only TransDF-style loss is not supported as the first local
candidate. The next training objective should keep time-domain MSE and add a
hybrid or variance-balanced component loss so lower-variance detail components
are not ignored.

## Falsification Boundary

This diagnostic does not prove that component-supervised training will improve
forecasting. It only shows that component-space residual structure is real
enough to justify a carefully constrained Step 4-6 objective design.
