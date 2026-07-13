# Phase1-R.2 Causal Target Interaction Design

## Current Step

`current_step`: step 3-6 of the long research loop.

Phase1-R.1 failed after remote training. We therefore do not continue from
implementation by adding more residual capacity. The current task is to
re-evaluate the real bottleneck and define a falsifiable decoder idea.

## Problem

[Fact] `PatchEncoderTargetSetDecoder` makes requested future segments explicit model
inputs and achieves numerical prefix consistency. However, it does not reach strict
compatibility pass:

- mean relative MSE vs horizon-specific `PatchEncoderFixedHead`: `+0.62%`;
- h96 mean relative MSE: `+2.74%`;
- strict H720-prefix no-degradation fails on `ETTh2/h96` and `ETTm1/h96`.

[Fact] `PatchEncoderTargetSetPrefixResidual` worsens this result:

- mean relative MSE: `+2.03%`;
- h96 mean relative MSE: `+4.52%`;
- H720-prefix h96/h192 mean relative MSE: `+1.20%`;
- mean target state cosine rises to `0.836250`.

[Inference] The failure is unlikely to be fixed by adding an uncontrolled
output residual. The residual path preserved prefix consistency but reduced
target-state separability and worsened short-horizon accuracy.

The next problem is therefore:

> How can a target-set decoder preserve prefix stability while modeling
> dependency among future target segments?

The first target-set decoder used independent target queries for prefix stability.
This is safe, but it treats target segments as conditionally independent after
they read from history. A horizon-specific fixed head does not have this
constraint: its output rows can implicitly encode cross-step covariance and
future-step coupling.

## Existence Evidence

[Strong Evidence] Project artifacts show the problem is real:

| Evidence | Meaning |
| --- | --- |
| `max prefix mismatch MSE = 5.112619e-14` for `PatchEncoderTargetSetDecoder` | Prefix-stable interface works mechanically. |
| `mean relative MSE = +0.62%` vs specialists | Interface alone is not enough for performance. |
| `h96 mean relative MSE = +2.74%` | Short-prefix amortization gap remains. |
| Prefix residual mean relative MSE `+2.03%` | Direct residual capacity repair is not the right bottleneck. |
| Prefix residual target-state cosine `0.836250` | Residual repair makes target states less discriminative. |

[Strong Evidence] Literature notes support the same diagnosis:

- `ElasTST` motivates horizon-invariant / prefix-stable varied-horizon
  forecasting, but its main value here is the masking principle, not a full
  architecture to copy.
- `TimePerceiver` supports target timestamp / target-query forecasting
  interfaces.
- `SRP++` argues that multi-step forecasting needs step-specific
  representations; a single shared latent plus output rows can be an
  expressiveness bottleneck.
- `QDF` argues that future steps are not independent equal-weight tasks; the
  training objective should account for future-step dependency / covariance.
- `TimeAlign` supports training-only future-side representation anchors, but it
  needs a stable target-side carrier before being added.

## Idea

`PatchEncoderCausalTargetInteraction` adds causal target self-attention after
target-to-history cross-attention:

$$
Z = E_\theta(X),
$$

$$
U_T = \operatorname{CrossAttn}(Q_T, Z, Z),
$$

$$
V_j =
\operatorname{CausalTargetAttn}
(U_j, U_{\le j}),
$$

$$
\hat{Y}_{a_j:b_j}
=
O_\theta
\left(
r \odot (1+\gamma(V_j)) + \beta(V_j)
\right),
$$

where

$$
r = R_\theta(\operatorname{Flatten}(Z)).
$$

The causal mask is over target segments, not over ground-truth future values.
No future labels or future covariates enter the prediction path.

## Theory Check

[Hypothesis] Independent target queries solve prefix consistency but under-model
future-step dependency. Causal target interaction keeps prefix stability because
the state of segment $j$ depends only on $U_{\le j}$:

$$
V_j(T_{1:J}) = V_j(T_{1:K}), \quad j \le K \le J.
$$

Therefore, asking the same model for `H=720` cannot change the internal state of
the first `H=96` segment, as long as the mask blocks attention from earlier
segments to later target queries.

[Inference] This is not autoregression. The model does not feed predicted values
back into the decoder. It models the process-level dependency among target
positions through latent target states.

[Counterargument] If the bottleneck is mixed-horizon optimization rather than
target interaction, causal target attention may preserve prefix consistency but
not improve MSE. If the additional interaction over-smooths target states, it may
repeat the prefix-residual failure and increase target-state cosine.

## Design

Model candidate:

- name: `PatchEncoderCausalTargetInteraction`;
- implementation path: `baselines/patch_encoder_target_set_decoder`;
- base: `PatchEncoderTargetSetDecoder`;
- added mechanism: `target_interaction_layers > 0`;
- default gate setting: `target_interaction_layers=1`;
- `target_interaction_heads=target_heads`;
- `target_interaction_d_ff=target_d_ff`;
- `prefix_residual_segments=0`.

Forward path:

1. encode history patches into $Z$;
2. build target segment queries $Q_T$;
3. cross-attend target queries to history: $U_T$;
4. apply causal self-attention over target states: $V_T$;
5. use $V_T$ to FiLM-condition dense history readout;
6. output segment values with shared segment head.

## Gate

Local smoke:

- `ETTh2`, target horizons `{96,192,336,720}`;
- `epoch=1`, CPU or local `r2026-fsa`;
- verify `metrics_by_target_horizon.csv`, `prefix_consistency.csv`,
  `target_state_similarity.csv`, and `effective_config.json`;
- prefix mismatch must remain near zero.

Remote gate:

- datasets: `ETTh2`, `ETTm1`, `Weather`;
- horizons: `{96,192,336,720}`;
- seed: `2021`;
- compare with horizon-specific `PatchEncoderFixedHead`;
- compare with `PatchEncoderTargetSetDecoder` and
  `PatchEncoderTargetSetPrefixResidual`.

Compatibility pass:

1. mean relative MSE vs `PatchEncoderFixedHead` <= `+0.62%`, i.e. no worse than
   the first target-set decoder;
2. h96 mean relative MSE < `+2.74%`;
3. H720-prefix h96/h192 mean relative MSE <= `-0.85%` or strict
   no-degradation improves in per-setting count;
4. max prefix mismatch remains numerical zero-level;
5. mean target-state cosine does not collapse toward the prefix residual value
   `0.836250`.

Paper-core pass:

1. mean relative MSE vs `PatchEncoderFixedHead` < `0`; or
2. compatibility pass holds and causal interaction gives a clear segment-level
   gain pattern that can support future-aware state or MoE as a next stage.

Rollback condition:

- If MSE worsens while prefix consistency remains solved, rollback to step 3-5
  and test mixed-horizon objective / task weighting instead of architecture.
- If target states become homogeneous, reject target interaction as a carrier
  and do not add MoE on top.

## Artifacts

Expected artifacts:

- local smoke: `artifacts/runs/smoke_phase1_causal_target_interaction`;
- remote output:
  `/home/yingch/exp_outputs/r-2026-fatst/phase1_target_set_decoder/PatchEncoderCausalTargetInteraction`;
- analysis:
  `analysis/phase1_causal_target_interaction_gate_20260622`.
