# Phase2-C Step-Covariance Balanced Objective Design

## Current Step

`current_step`: step 1-6 of the long research loop.

Phase2-A, Phase2-R.1, and Phase2-B all passed implementation-safety checks but
failed to convert their state/residual mechanisms into stable forecast gains.
The rollback point is therefore step 2-3: verify whether the next real problem
is the mixed-horizon objective itself.

## Problem

[Fact] R.3 `PatchEncoderPrefixRiskWeighted` is the only Phase1/Phase2 candidate
that clearly improves the uniform target-set model without changing the
architecture.

[Strong Evidence] Phase2-C objective-pressure diagnostic shows:

- R.3 wins against uniform `PatchEncoderTargetSetDecoder` in `11/12` settings;
- mean relative MSE vs uniform is `-1.03%`;
- mean h96 relative MSE vs uniform is `-1.81%`;
- H720-prefix h96/h192 mean relative MSE vs uniform is `-1.62%`;
- segment-level wins are `24/30`.

[Fact] The same diagnostic also shows that naive prefix-risk weighting heavily
concentrates pressure on the first 96 steps:

| Region | Uniform Pressure Share | Prefix-Risk Pressure Share | Share Delta |
| --- | ---: | ---: | ---: |
| `1-96` | `0.4798` | `0.7217` | `+50.43%` |
| `97-192` | `0.2298` | `0.1540` | `-32.98%` |
| `193-336` | `0.1571` | `0.0775` | `-50.71%` |
| `337-720` | `0.1333` | `0.0469` | `-64.85%` |

[Inference] The objective bottleneck is real, but a single monotone
prefix-risk curve is too crude. It improves early prefixes and most settings,
yet still leaves fixed-specialist gaps such as `ETTm1/h96`, `ETTm1/h720`, and
`ETTh2/h720`.

## Idea

Design a `Step-Covariance Balanced Objective` that treats future prediction as
a region-structured trajectory rather than independent point errors.

Partition the target horizon into regions:

$$
\mathcal{R}=\{[1,96],[97,192],[193,336],[337,720]\}.
$$

For a sampled horizon $H$, the prediction loss is written as a weighted sum of
region-normalized losses:

$$
\mathcal{L}_{scb}
=
\sum_{r\in\mathcal{R}_H}
\lambda_r
\frac{1}{|r\cap[1,H]|C}
\sum_{t\in r\cap[1,H]}\sum_{c=1}^{C}
(\hat{y}_{t,c}-y_{t,c})^2.
$$

The key is not merely changing $\lambda_r$ by hand. The weights should encode
two pressures:

1. `coverage balance`: early regions appear in more sampled horizons and
   otherwise dominate gradient pressure;
2. `future covariance / novelty`: regions whose normalized future labels are
   weakly explained by earlier regions should receive enough pressure, because
   they represent genuinely new future-state information.

A practical static prior is:

$$
\lambda_r
\propto
\left(p_r^{uniform}\right)^{-\beta}
\left(u_r+\epsilon\right)^{\eta},
$$

where $p_r^{uniform}$ is the expected region pressure under the actual
mixed-horizon sampler, and $u_r$ is a target-derived region novelty score.

One possible novelty score is computed from training targets after the same
normalization used by the model:

$$
u_r
=
\sqrt{\operatorname{tr}(\Sigma_{r,r})}
\left(
1-
\max_{s<r}
\rho^2(\bar{Y}_r,\bar{Y}_s)
\right).
$$

Here $\bar{Y}_r$ is the region mean or pooled normalized target vector, and
$\rho$ is the cross-region correlation. This is intentionally a training-time
objective prior; it does not introduce future labels into inference.

## Theory Check

[Hypothesis] If mixed-horizon one-model training fails because early prefixes,
middle transitions, and long-tail regions impose different optimization
pressures, then balancing region pressure with covariance/novelty should
preserve R.3's h96 gains while repairing the mid/late gaps that monotone
prefix-risk underweights.

[Counterargument] If R.3's gain is merely a regularization accident or dataset
noise, covariance-balanced weighting will not consistently improve over R.3.
It may also reduce h96 gains by taking pressure away from the early region.

[Boundary] This is still an objective-level mechanism, not a new architecture.
It can become paper-relevant only if it does more than improve an average:
the evidence must show that future-step covariance or region novelty explains
where the one-model decoder needs pressure.

## Minimal Design

Implementation candidate:

- model: `PatchEncoderTargetSetDecoder`;
- carrier: R.3-compatible target-set decoder;
- new `step_loss_weighting` mode:
  `step_covariance_balanced`;
- static region boundaries: `96,192,336,720`;
- no future labels in inference;
- region weights computed from training targets or a documented static
  diagnostic prior;
- compare against both:
  `PatchEncoderTargetSetDecoder` and `PatchEncoderPrefixRiskWeighted`.

First implementation should prefer a conservative two-stage design:

1. `region_balanced`: use only coverage balance, to test whether pressure
   equalization alone helps.
2. `step_covariance_balanced`: add target covariance/novelty once the region
   path is verified.

## Gate

Local smoke:

- dataset: `ETTh2`;
- target horizons: `{96,192,336,720}`;
- epoch: `1`;
- verify `metrics_by_target_horizon.csv`, `metrics_by_segment.csv`,
  `prefix_consistency.csv`, and saved objective-weight diagnostics.

Remote gate:

- datasets: `ETTh2`, `ETTm1`, `Weather`;
- target horizons: `{96,192,336,720}`;
- seed: `2021`;
- compare against `PatchEncoderTargetSetDecoder`,
  `PatchEncoderPrefixRiskWeighted`, and `PatchEncoderFixedHead`.

Pass conditions:

1. vs R.3 mean relative MSE < `0`;
2. vs R.3 wins at least `7/12`;
3. no dataset mean degrades more than `+0.3%` vs R.3;
4. at least two current fixed-specialist gaps improve:
   `ETTm1/h96`, `ETTm1/h720`, `ETTh2/h720`, `Weather/h96`;
5. H720 middle/late regions do not regress while h96 remains competitive;
6. prefix mismatch stays numerical-zero level.

Paper-core conditions:

1. region/covariance weights must explain the observed gain pattern;
2. coverage-only and covariance-aware variants must be separable;
3. the method must beat R.3, not merely uniform target-set;
4. the story must remain about forecasting-process optimization, not
   hyperparameter tuning.

## Artifacts

Current diagnostic artifacts:

- report:
  `analysis/phase2_objective_pressure_diagnostic_20260623/phase2_objective_pressure_diagnostic_report.md`;
- pressure summary:
  `analysis/phase2_objective_pressure_diagnostic_20260623/objective_pressure_summary.csv`;
- R.3 vs uniform:
  `analysis/phase2_objective_pressure_diagnostic_20260623/r3_vs_uniform_main.csv`;
- figures:
  `analysis/phase2_objective_pressure_diagnostic_20260623/objective_pressure_curve.png`,
  `analysis/phase2_objective_pressure_diagnostic_20260623/segment_effect_vs_pressure.png`.

## Decision

[Decision] Proceed to implementation only after preserving this diagnostic
boundary: Phase2-C is not another residual/future-state mechanism. It tests
whether the training objective should model target-step covariance and
region-specific pressure for one-model multi-horizon forecasting.
