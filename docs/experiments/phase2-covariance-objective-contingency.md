# Phase2-C Contingency: Covariance-Aware Objective

## Current Step

`current_step`: step 2-6 of the long research loop.

This document is a contingency plan. It does not replace the active
`region_balanced` remote gate. It defines the next rollback path if
coverage-only balancing cannot beat R.3 `PatchEncoderPrefixRiskWeighted`.

## Problem

[Fact] Phase2-C diagnostic proved that objective pressure matters: R.3 beats
uniform target-set training in `11/12` settings without changing the model
architecture.

[Fact] The active `region_balanced` candidate tests only coverage balance. It
forces the four future regions `[1,96]`, `[97,192]`, `[193,336]`, and
`[337,720]` to receive equal expected pressure under mixed-horizon sampling.

[Hypothesis] If `region_balanced` fails, the failure should not immediately
falsify the objective route. It may only show that equal coverage is too crude:
different future regions do not merely need equal pressure; they need pressure
according to how much new future information they carry beyond earlier regions.

## Existence Evidence

[Strong Evidence] Existing artifacts already show two separate facts:

1. uniform mixed-horizon training overweights early prefix steps because early
   steps appear in every sampled horizon;
2. R.3 improves many settings by changing this pressure, but its monotone
   prefix-risk curve over-concentrates pressure on `[1,96]` and underweights
   middle/late regions.

[Inference] This leaves a precise unresolved question:

> Can one-model multi-horizon training allocate objective pressure by
> future-region dependency or novelty, rather than by coverage count alone?

The question is worth one more objective-level test only if `region_balanced`
is not a clear pass but still preserves the R.3 compatibility carrier.

## Idea

Use a `step_covariance_balanced` objective:

$$
\mathcal{L}
=\sum_{r\in\mathcal{R}_H}
\lambda_r
\frac{1}{|r\cap[1,H]|C}
\sum_{t\in r\cap[1,H]}\sum_{c=1}^{C}
(\hat{y}_{t,c}-y_{t,c})^2.
$$

Region weights combine two terms:

$$
\lambda_r
\propto
\left(p_r^{uniform}\right)^{-\beta}
\left(u_r+\epsilon\right)^\eta.
$$

Here $p_r^{uniform}$ is the expected pressure share induced by the current
mixed-horizon sampler, and $u_r$ is a training-set novelty score for region
$r$.

One conservative novelty score is:

$$
u_r =
\sqrt{\operatorname{tr}(\Sigma_{r,r})}
\left(1-\max_{s<r}\rho^2(\bar{Y}_r,\bar{Y}_s)\right).
$$

$\bar{Y}_r$ is a pooled normalized target vector for region $r$, computed on
the training split only. The score is static and never uses future labels at
inference.

## Theory Check

[Hypothesis] Coverage balance repairs sampling-frequency bias, while novelty
balance repairs dependency bias. A later region whose normalized future values
are weakly explained by earlier regions should receive enough gradient pressure
because it represents a distinct future state, not just a longer copy of the
prefix.

[Counterargument] A static novelty prior may be too weak or too dataset-level:
it can overfit the training distribution summary and ignore sample-specific
forecast difficulty. If so, it may improve one dataset but degrade another,
which would not support a strong paper claim.

[Boundary] This remains an objective-side mechanism. It can become paper
material only if its weights explain the observed error pattern better than
both uniform and prefix-risk weighting. Otherwise it is just another loss
tuning variant.

## Minimal Design

Implementation should be attempted only after the active `region_balanced`
gate is evaluated.

Minimal implementation plan:

1. Add `--step-loss-weighting step_covariance_balanced`.
2. Precompute static region novelty from the training split after the same
   normalization used by `ForecastDataset`.
3. Write `objective_weight_stats.csv` with both coverage and novelty terms.
4. Keep the model architecture and inference path identical to
   `PatchEncoderTargetSetDecoder`.
5. Run the same `ETTh2`, `ETTm1`, `Weather` gate against R.3.

The first version should use fixed hyperparameters, for example
`beta=1.0`, `eta=0.5`, and `epsilon=1e-6`, to avoid turning the stage into a
wide tuning sweep.

## Gate

Primary gate is identical to Phase2-C:

1. mean relative MSE vs R.3 < `0`;
2. vs R.3 wins at least `7/12`;
3. no dataset mean degrades more than `+0.3%` vs R.3;
4. at least two fixed-specialist gaps improve:
   `ETTm1/h96`, `ETTm1/h720`, `ETTh2/h720`, `Weather/h96`;
5. H720 middle/late regions do not regress while h96 remains competitive;
6. prefix mismatch stays numerical-zero level.

Additional paper-story gate:

1. novelty weights must correlate with segment-level improvements better than
   coverage-only weights;
2. the result must separate `coverage balance` from `covariance/novelty
   balance`;
3. if gains are dataset-specific, the report must explain why the mechanism
   should still be considered a forecasting-process claim rather than an
   empirical accident.

## Decision

[Decision] Do not implement this contingency before the active
`region_balanced` remote gate is analyzed.

[Rollback] If `region_balanced` clearly fails against R.3 and also loses the
R.3 compatibility carrier, rollback to step 2-3 and stop the objective route.
The next problem should be base architecture or external baseline selection,
not another objective patch.

[Rollback] If `region_balanced` is partial, meaning it preserves compatibility
but cannot beat R.3, this contingency is the next legitimate step 4-6
candidate.
