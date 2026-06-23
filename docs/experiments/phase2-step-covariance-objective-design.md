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

First implementation uses a conservative two-stage design:

1. `region_balanced`: implemented as a new `step_loss_weighting` mode; use only
   coverage balance, to test whether pressure equalization alone helps.
2. `step_covariance_balanced`: future candidate; add target covariance/novelty once the region
   path is verified.

`region_balanced` computes the uniform expected pressure induced by the current
mixed-horizon sampler and assigns region multipliers:

$$
m_r
\propto
\frac{1/|\mathcal{R}|}{\sum_{t\in r}p_t}.
$$

The multiplier vector is normalized by its full-horizon mean, so the objective
changes pressure allocation rather than merely scaling the loss.

## Gate

Local smoke:

- dataset: `ETTh2`;
- target horizons: `{96,192,336,720}`;
- epoch: `1`;
- verify `metrics_by_target_horizon.csv`, `metrics_by_segment.csv`,
  `prefix_consistency.csv`, `objective_weight_stats.csv`, and saved config.

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

Implementation artifacts:

- train switch:
  `--step-loss-weighting region_balanced`;
- code:
  `baselines/patch_encoder_target_set_decoder/train.py`;
- analyzer:
  `scripts/analyze_phase2_region_balanced_gate.py`;
- remote runner:
  `scripts/remote/run_phase2_region_balanced_gate.sh`;
- progress checker:
  `scripts/remote/check_phase2_region_balanced_progress.sh`;
- sync wrapper:
  `scripts/sync_phase2_region_balanced_results.sh`;
- code explanation:
  `docs/code-explanation/phase1-target-set-decoder.md`.

Local smoke artifacts:

- output:
  `artifacts/runs/smoke_phase2_region_balanced/PatchEncoderRegionBalanced/ETTh2/mixed_h96_h192_h336_h720/seed2021`;
- command mode:
  `--step-loss-weighting region_balanced`;
- smoke scope:
  `ETTh2`, `{96,192,336,720}`, `epoch=1`, `steps_per_epoch=2`,
  `max_eval_batches=1`;
- required artifacts written:
  `metrics_by_target_horizon.csv`, `prefix_consistency.csv`,
  `objective_weight_stats.csv`, `effective_config.json`;
- prefix mismatch MSE:
  `96/720 = 8.284057610079363e-15`,
  `192/720 = 8.386538432301766e-15`,
  `336/720 = 3.621962433886803e-15`;
- objective-weight audit:
  weighted pressure share is `0.25` for each of
  `1-96`, `97-192`, `193-336`, and `337-720`.

## Decision

[Decision] Proceed to implementation only after preserving this diagnostic
boundary: Phase2-C is not another residual/future-state mechanism. It tests
whether the training objective should model target-step covariance and
region-specific pressure for one-model multi-horizon forecasting.

[Decision Update: 2026-06-23] The first implementation is limited to
`region_balanced`. It is an objective-only coverage-balance test; it should be
judged against R.3 before adding covariance/novelty priors.

[Decision Update: 2026-06-23] Local smoke passed. The next evidence must be the
full remote gate against R.3; smoke metrics are not interpreted as performance
evidence because only two training steps and one eval batch were used.

[Decision Update: 2026-06-23] The full remote `region_balanced` gate failed.
The synchronized report is:

- `analysis/phase2_region_balanced_gate_20260623/phase2_region_balanced_decision_report.md`.

Key metrics:

- MSE wins vs R.3: `2/12`;
- MAE wins vs R.3: `0/12`;
- mean relative MSE vs R.3: `+1.53%`;
- dataset mean relative MSE vs R.3:
  `ETTh2 -0.29%`, `ETTm1 +3.19%`, `Weather +1.70%`;
- mean relative MSE vs uniform target-set: `+0.47%`;
- prefix consistency remains numerical-zero level.

[Decision] Coverage balance alone is falsified. It should not be tuned by
hand, and it should not be used as the carrier for future-aware or MoE
mechanisms.

[Rollback] Return to step 2-3. The only objective-side continuation that remains
defensible is a separate offline covariance/novelty diagnostic. If that
diagnostic cannot explain why R.3 helps while equal-region coverage hurts, stop
the objective-only path and reconsider base architecture or external baseline
selection.

[Decision Update: 2026-06-23] The offline covariance/novelty diagnostic passed
the continuation gate. The synchronized artifacts are:

- report:
  `analysis/phase2_covariance_novelty_diagnostic_20260623/phase2_covariance_novelty_diagnostic_report.md`;
- script:
  `scripts/analyze_phase2_covariance_novelty.py`;
- code explanation:
  `docs/code-explanation/phase2-covariance-novelty-diagnostic.md`.

Key evidence:

- R.3 segment delta vs novelty share Pearson: `-0.7219`;
- R.3 segment delta vs prefix pressure share Pearson: `-0.6909`;
- `region_balanced` delta vs novelty deficit Pearson: `+0.6253`;
- aggregate R.3 delta vs novelty share Pearson: `-0.6714`;
- aggregate `region_balanced` delta vs novelty deficit Pearson: `+0.6253`.

[Decision] Proceed to step 4-6 for `step_covariance_balanced`, but keep the
scope narrow:

1. do not change the model architecture or inference path;
2. compute static novelty from the train split only;
3. use one fixed hyperparameter setting first, not a sweep;
4. compare primarily against R.3;
5. stop the objective-only path if the candidate cannot beat R.3.

[Caveat] This is not yet a paper-core pass. The `region_balanced` failure has a
positive Pearson relationship with novelty deficit, but its Spearman correlation
is only `0.1538`; the next training gate must prove forecast gains directly.
