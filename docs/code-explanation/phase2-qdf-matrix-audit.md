# Phase2 QDF Learned Matrix Audit Code Explanation

## Purpose

`scripts/analyze_phase2_qdf_matrix_audit.py` reads the native QDF `A.pth`
artifacts from the Phase2-D upstream reproduction gate and audits the learned
matrix structure before any local FATST implementation.

The goal is to determine whether the useful signal is a full learned matrix, a
fixed-diagonal off-diagonal coupling, a banded future-step structure, or no
stable localizable structure.

## Loading Boundary

QDF saves a pickled `CovarianceMatrix` module. FATST does not vendor QDF source,
so the script registers a tiny stub module at the original pickle path:

`exp.exp_long_term_forecasting_meta_ml3.CovarianceMatrix`

The stub is only used to unpickle the stored parameters. It does not import QDF
code, call QDF training logic, or reuse upstream APIs.

## Matrix Reconstruction

For each `A.pth`, the script reconstructs the same lower-triangular matrix `L`
used by QDF:

- `meta_type=all`: use `torch.tril(L_param)` and add `eps` to the diagonal.
- `meta_type=diag`: use `sqrt(abs(diag_param) + eps)` as the diagonal.
- `meta_type=off_diag`: use the strict lower triangle of `L_param`, fix the
  diagonal to `1`, then row-normalize.

QDF's stored forward matrix is:

$$
\Sigma = LL^\top.
$$

But QDF's loss solves:

$$
Lx = E,
$$

which means the effective residual weighting is:

$$
W = \Sigma^{-1}.
$$

Therefore the audit records both `covariance_*` statistics and
`precision_*` statistics, and the local objective decision should primarily use
the `precision_*` fields.

## Output Columns

`phase2_qdf_matrix_audit_metrics.csv` contains one row per
`meta_type/dataset/horizon`:

- `covariance_trace`, `precision_trace`: matrix traces.
- `*_diag_mean/std/min/max`: diagonal statistics.
- `*_offdiag_abs_mean/max`: absolute off-diagonal magnitudes.
- `*_offdiag_fro_share`: squared Frobenius energy share outside the diagonal.
- `*_weighted_bandwidth`: average `|i-j|` weighted by absolute off-diagonal
  magnitude.
- `*_weighted_bandwidth_ratio`: weighted bandwidth divided by `pred_len - 1`,
  making horizons comparable.
- `*_condition_number`: matrix condition number.

`phase2_qdf_matrix_audit_region_blocks.csv` aggregates each matrix into the
project's future regions:

- `1-96`
- `97-192`
- `193-336`
- `337-720`

Each row reports one source-region to destination-region block.

## Code-Theory Consistency

Intended theory:

- Phase2-D showed `all` beats `diag`, so diagonal-only weighting is insufficient.
- Phase2-D also showed `off_diag` often beats `all`, so fixed-diagonal residual
  coupling may be the most stable localizable mechanism.

How the code realizes it:

- it extracts learned matrices from all 36 QDF runs;
- it distinguishes saved covariance from effective loss precision;
- it summarizes off-diagonal energy and bandwidth at step and region levels.

Remaining proxy:

- the audit is artifact-level and does not train FATST;
- matrix structure may be useful in QDF's bilevel loop but still fail in a
  static local objective;
- high off-diagonal energy is not by itself performance proof.

Falsification evidence for the next stage:

- no stable off-diagonal precision structure;
- structure exists only in runs where `off_diag` does not help;
- a local `offdiag_region_quadratic` objective fails against R.3.
