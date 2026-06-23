# Phase2 Off-Diagonal Block Quadratic Objective Code Explanation

## Purpose

`offdiag_block_quadratic` is the first local FATST candidate after the QDF
upstream reproduction and learned matrix audit.

It tests whether future-step residual interaction can help the existing R.3
target-set carrier without copying QDF's full bilevel/meta-learning procedure.

## Design Boundary

The candidate keeps the existing prediction path unchanged:

$$
\hat{Y}_T = D_\theta(E_\theta(X), Q_T, M_T).
$$

It also keeps R.3's `prefix_risk` weighted MSE as the base prediction loss. The
new term is only an additional train-split-derived residual interaction penalty.

No QDF module is imported. No inference-time computation changes.

## Train-Split Block Matrix

The script divides the H720 future span into fixed-size blocks. The default is:

```text
offdiag_block_size = 48
```

This gives 15 blocks for H720 and still gives 2 blocks for H96, so short-horizon
specialist gaps can receive an off-diagonal signal.

For each training target window, each block is averaged. Across all windows and
channels, the code computes a standardized block correlation matrix, inverts it
with ridge regularization, removes the diagonal, and normalizes the off-diagonal
matrix by spectral norm:

$$
O = \frac{\operatorname{offdiag}((C + \epsilon I)^{-1})}
{\|\operatorname{offdiag}((C + \epsilon I)^{-1})\|_2}.
$$

## Loss Flow

For each batch, residuals are averaged into active blocks:

$$
e_b \in \mathbb{R}^{R}.
$$

The penalty is:

$$
\mathcal{L}_{offdiag}
= \operatorname{mean}\left((e_b O_R^\top)^2\right),
$$

where $O_R$ is the active top-left block matrix for the current horizon.

The final prediction loss is:

$$
\mathcal{L}
= \mathcal{L}_{prefix\_risk}
+ \lambda \mathcal{L}_{offdiag}.
$$

Default:

```text
offdiag_quadratic_weight = 0.05
offdiag_ridge_eps = 1e-3
```

## Artifact Effects

Each run writes:

- `offdiag_block_matrix.csv`: the normalized train-split off-diagonal matrix;
- `effective_config.json`: `offdiag_block_stats`;
- existing target horizon metrics, segment metrics, prefix consistency, and
  objective weight stats.

## Code-Theory Consistency

Intended theory:

- QDF `all` beats `diag`, proving diagonal-only loss is insufficient.
- QDF `off_diag` is often stronger than `all`, pointing to fixed-diagonal
  residual coupling as a more stable local mechanism.

How the code realizes it:

- it preserves R.3's diagonal pressure;
- it adds a positive residual interaction penalty derived from train targets;
- it gives H96 at least two blocks, unlike a 4-region-only design.

Remaining proxy:

- the matrix is static, not bilevel learned;
- block means compress within-block dynamics;
- the penalty uses a stable squared projection rather than QDF's exact
  triangular solve loss.

Falsification evidence:

- remote gate fails against R.3;
- gains appear only against uniform target-set;
- prefix consistency is damaged;
- specialist gaps do not improve.
