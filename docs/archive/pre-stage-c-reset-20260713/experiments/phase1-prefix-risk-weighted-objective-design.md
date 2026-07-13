# Phase1-R.3 Prefix-Risk Weighted Objective Design

## Current Step

`current_step`: step 3-6 of the long research loop.

Phase1-R.2 rejected deeper target interaction as a stable carrier. We therefore
return to the objective-level bottleneck before adding more architecture.

## Problem

[Fact] `PatchEncoderTargetSetDecoder` solves prefix consistency mechanically but
does not reach compatibility pass:

- mean relative MSE vs `PatchEncoderFixedHead`: `+0.62%`;
- h96 mean relative MSE: `+2.74%`;
- H720-prefix strict no-degradation still fails on some settings.

[Fact] Two architecture-level repairs failed:

- `PatchEncoderTargetSetPrefixResidual`: mean relative MSE `+2.03%`;
- `PatchEncoderCausalTargetInteraction`: mean relative MSE `+1.40%`, with
  Weather degrading by `+4.68%`.

[Inference] The next plausible bottleneck is not readout capacity or target
interaction depth. It may be the mixed-horizon objective: standard MSE treats
each observed target point as an independent squared error and does not
explicitly encode that early future prefixes are shared by all requested
horizons and are the main consistency-sensitive region.

## Existence Evidence

[Strong Evidence] The current evidence points to prefix-sensitive risk:

| Artifact | Evidence |
| --- | --- |
| `PatchEncoderTargetSetDecoder` | h96 relative MSE `+2.74%` despite prefix mismatch `5.1e-14`. |
| `PatchEncoderTargetSetPrefixResidual` | output residual worsens h96 to `+4.52%`. |
| `PatchEncoderCausalTargetInteraction` | interaction helps ETTm1 average but h96 remains `+3.62%` and Weather degrades. |

[Strong Evidence] Paper notes support objective-level diagnosis:

- `QDF` argues standard multi-step MSE ignores future-step covariance and
  heterogeneous task weights.
- `SRP++` argues different future steps can require different representations,
  which means a uniform loss may under-specify the optimization pressure needed
  for step-sensitive outputs.
- `ElasTST` motivates varied-horizon prefix invariance, so early prefix risk is
  not merely a metric artifact.

## Idea

`PatchEncoderPrefixRiskWeighted` keeps the first target-set decoder architecture
unchanged and changes only the training objective. Instead of uniform MSE,
future step $t$ receives a prefix-risk weight:

$$
w_t =
\frac{(t / H_{max})^{-\alpha}}
{
\frac{1}{H_{max}}\sum_{s=1}^{H_{max}}(s / H_{max})^{-\alpha}
},
\quad
\alpha \ge 0.
$$

The training loss for requested horizon $H$ is:

$$
\mathcal{L}_{prefix}
=
\frac{1}{BHC}
\sum_{b=1}^{B}\sum_{t=1}^{H}\sum_{c=1}^{C}
w_t(\hat{y}_{b,t,c}-y_{b,t,c})^2.
$$

When $\alpha=0$, this becomes the original uniform MSE. For $\alpha>0$, earlier
future steps and shorter requested horizons receive stronger optimization
pressure, but the model architecture and inference path remain unchanged.

## Theory Check

[Hypothesis] If the near-miss comes from objective mismatch, prefix-risk
weighting should improve h96 and H720-prefix h96/h192 without harming the
target-set interface or collapsing target states.

[Counterargument] If the near-miss is caused by model capacity or dataset-specific
dynamics, prefix-risk weighting will only trade long-horizon or Weather accuracy
for h96. In that case it is a useful diagnostic but not a paper-core mechanism.

[Boundary] This is not a final QDF implementation. It is a low-cost diagnostic
inspired by QDF's claim that direct multi-step objectives need future-step
weighting. A successful result would justify a more principled covariance-aware
objective; a failed result would push us away from decoder/objective patching.

## Design

Model candidate:

- name: `PatchEncoderPrefixRiskWeighted`;
- implementation path: `baselines/patch_encoder_target_set_decoder`;
- architecture: first `PatchEncoderTargetSetDecoder`;
- `target_interaction_layers=0`;
- `prefix_residual_segments=0`;
- `step_loss_weighting=prefix_risk`;
- first gate alpha: `0.5`.

The candidate intentionally changes only the loss. This isolates whether the
mixed-horizon objective, rather than decoder architecture, is the active
bottleneck.

## Gate

Local smoke:

- `ETTh2`, target horizons `{96,192,336,720}`;
- `epoch=1`, local `r2026-fsa`;
- verify required artifacts and prefix consistency.

Remote gate:

- datasets: `ETTh2`, `ETTm1`, `Weather`;
- horizons: `{96,192,336,720}`;
- seed: `2021`;
- compare against `PatchEncoderFixedHead`, `PatchEncoderTargetSetDecoder`,
  `PatchEncoderTargetSetPrefixResidual`, and
  `PatchEncoderCausalTargetInteraction`.

Compatibility pass:

1. mean relative MSE vs `PatchEncoderFixedHead` <= `+0.62%`;
2. h96 mean relative MSE < `+2.74%`;
3. Weather mean relative MSE does not exceed the first target-set decoder's
   `+0.13%` by more than `+1.0%`;
4. H720-prefix h96/h192 comparison does not regress from `-0.85%`;
5. prefix mismatch remains numerical zero-level.

Paper-story pass:

1. compatibility pass holds; and
2. improvement pattern supports the claim that multi-horizon target-set
   forecasting needs prefix-risk aware optimization rather than only decoder
   architecture.

Rollback condition:

- If h96 improves but Weather or h720 degrades strongly, treat this as a
  risk-tradeoff diagnostic and move to a principled covariance-aware objective
  or a new backbone.
- If h96 does not improve, stop objective patching and reconsider the base
  architecture or external baseline reproduction.

## Artifacts

Expected artifacts:

- local smoke:
  `artifacts/runs/smoke_phase1_prefix_risk_weighted`;
- remote output:
  `/home/yingch/exp_outputs/r-2026-fatst/phase1_target_set_decoder/PatchEncoderPrefixRiskWeighted`;
- analysis:
  `analysis/phase1_prefix_risk_weighted_gate_20260622`.
