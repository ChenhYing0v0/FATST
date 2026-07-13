# Phase2 Step-Covariance Balanced Objective Code Explanation

## Purpose

`step_covariance_balanced` is the first training candidate after the Phase2-C.1
offline novelty diagnostic. It is inspired by QDF's objective-side claim that
future steps should not be treated as independent equal-weight tasks, but it is
not a full QDF implementation.

The local candidate only tests a static diagonal objective:

- no off-diagonal quadratic matrix;
- no bilevel / meta-learning update;
- no architecture or inference-path change;
- train-split target novelty only affects the training loss.

## QDF Boundary

QDF motivates two ideas:

1. label autocorrelation across future steps matters;
2. different future steps need heterogeneous task weights.

This implementation adopts only the second idea at region level. The full QDF
matrix is intentionally deferred because the current research loop needs to test
whether objective pressure can beat R.3 before adding a more expensive learned
quadratic objective.

## Loss Flow

The model prediction path remains unchanged:

$$
\hat{Y}_T=D_\theta(E_\theta(X),Q_T,M_T).
$$

Only the MSE loss receives a precomputed step-weight vector:

$$
\mathcal{L}
=
\frac{1}{BHC}
\sum_{b,t,c}
w_t(\hat{y}_{b,t,c}-y_{b,t,c})^2.
$$

For each region $r$, the raw multiplier is:

$$
m_r
=
\left(p_r^{uniform}\right)^{-\beta}
\left(u_r+\epsilon\right)^\eta,
$$

where:

- $p_r^{uniform}$ is the expected pressure share induced by the current
  mixed-horizon sampler;
- $u_r$ is train-split region novelty from normalized target windows;
- defaults are `beta=0.5`, `eta=0.5`, `eps=1e-6`.

The full step-weight vector is normalized to mean `1.0` over H720 so the
candidate changes allocation rather than simply scaling the loss.

## Train-Split Novelty

The implementation reuses `ForecastDataset`:

1. load the train split with the same borders and scaler;
2. construct H720 target windows after `seq_len=336`;
3. compute each region mean $\bar{Y}_r$;
4. compute:

$$
u_r
=
\operatorname{std}(\bar{Y}_r)
\left(1-\max_{s<r}\rho^2(\bar{Y}_r,\bar{Y}_s)\right).
$$

The dataset-local novelty shares are written to `effective_config.json` under
`step_covariance_novelty_share`.

## Smoke Result

Local smoke:

- command mode: `--step-loss-weighting step_covariance_balanced`;
- output:
  `artifacts/runs/smoke_phase2_step_covariance_balanced/PatchEncoderStepCovarianceBalanced/ETTh2/mixed_h96_h192_h336_h720/seed2021`;
- scope: `ETTh2`, `{96,192,336,720}`, `epochs=1`, `steps_per_epoch=2`,
  `max_eval_batches=1`, CPU.

Key checks:

- required artifacts were written;
- prefix mismatch remains numerical-zero level:
  `96/720 = 8.455293790696292e-15`,
  `192/720 = 8.434740536231167e-15`,
  `336/720 = 3.5504944524786947e-15`;
- ETTh2 weighted pressure share becomes:
  `1-96 = 0.4807`,
  `97-192 = 0.1951`,
  `193-336 = 0.1640`,
  `337-720 = 0.1603`.

This is intentionally milder than `region_balanced`: it preserves early-region
pressure near uniform while giving middle / late regions a novelty-based
increase.

## Code-Theory Consistency

Intended theory:

- R.3 works partly because early future regions have high novelty;
- `region_balanced` fails because it cuts early pressure too aggressively;
- a diagonal novelty-aware objective may preserve early gains while repairing
  underweighted middle / late regions.

How the code realizes it:

- computes static novelty from train targets only;
- precomputes one step-weight vector before training;
- passes the vector into `weighted_mse_loss`;
- records pressure and novelty in `objective_weight_stats.csv`.

Remaining proxy:

- no off-diagonal future-step covariance is modeled;
- novelty is dataset-level static, not sample-adaptive;
- region means compress within-region temporal structure.

Falsification evidence:

- remote gate cannot beat R.3;
- gains appear only against uniform target-set;
- h96 loses R.3's early-prefix advantage;
- dataset mean degradation exceeds the `+0.3%` gate.
