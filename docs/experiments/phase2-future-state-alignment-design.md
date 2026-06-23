# Phase2 Future-State Alignment Design

## Current Step

`current_step`: step 1-6 of the long research loop.

Phase1-R.3 already gives a compatible one-model carrier:
`PatchEncoderPrefixRiskWeighted` reaches mean relative MSE `-0.43%` versus
horizon-specific `PatchEncoderFixedHead`, keeps prefix mismatch at numerical
zero level, and improves H720-prefix h96/h192 by `-2.46%`.

The current task is no longer to prove that one-model target-set forecasting is
possible. The task is to decide whether a future-aware mechanism can turn the
target-side state into a paper-core forecasting-process contribution.

## Research Analysis

[Strong Evidence] Prior local evidence rejects three weak routes:

| Route | Evidence | Decision |
| --- | --- | --- |
| output / adapter patching | Phase1-A.1 to A.6 gives only partial local wins | not paper-core |
| uncontrolled prefix residual | R.1 worsens mean relative MSE to `+2.03%` | fail |
| deeper target interaction | R.2 worsens mean relative MSE to `+1.40%`, Weather `+4.68%` | fail |

[Strong Evidence] R.3 shows the target-set carrier is useful but incomplete:

| Remaining weak setting | Relative MSE vs `PatchEncoderFixedHead` |
| --- | ---: |
| `ETTm1 / h96` | `+2.83%` |
| `ETTm1 / h720` | `+1.09%` |
| `ETTh2 / h720` | `+0.75%` |
| `Weather / h96` | `+0.64%` |

[Strong Evidence] R.3 target-state statistics show dataset-dependent target
state geometry rather than a collapsed representation:

| Dataset / horizon | mean target-state cosine |
| --- | ---: |
| `ETTh2 / h720` | `+0.918` |
| `ETTm1 / h96` | `-0.559` |
| `Weather / h96` | `-0.108` |

[Inference] The target states $U_T$ are active enough to serve as a carrier, but
they are only inferred from history $X$ and target segment features $Q_T$.
They are not explicitly calibrated to the realized future distribution during
training.

## Paper Evidence

[Strong Evidence] `TimeAlign` supports the general principle that a forecasting
branch hidden state can suffer from past/future distribution mismatch and that a
training-only future reconstruction branch can provide a future-side anchor.

[Strong Evidence] `TimePerceiver` supports target-query forecasting: target
positions should enter the decoder as explicit query tokens rather than only as
output-head rows.

[Strong Evidence] `SRP++` supports step/segment-specific representation: future
steps may require different representations, so aligning a single global
history state is weaker than aligning target-segment states.

[Strong Evidence] `QDF` supports future-step dependency in the objective:
uniform MSE can under-specify the covariance and heterogeneous risk of future
steps. R.3 already confirms that this issue exists in our target-set carrier.

[Inference] Combining these papers suggests a more precise gap:

> A target-set decoder exposes future positions as model inputs, but its
> target-side states are still history-implied states. Training should teach
> those states to approximate the distributional state of the future segment
> they will predict, without using future values at inference.

## Problem

[Problem] In `PatchEncoderPrefixRiskWeighted`, the prediction for target segment
$j$ is:

$$
Z = E_\theta(X),
$$

$$
U_j = D_\theta(Q_j, Z),
$$

$$
\hat{Y}_{a_j:b_j}
=
O_\theta(r \odot (1+\gamma(U_j)) + \beta(U_j)).
$$

Here $U_j$ is target-aware, but it is not future-distribution-aware. It is
optimized only through prediction loss, so it may learn whatever FiLM features
reduce MSE, even if those features do not align with the latent structure of
the realized future segment.

[Problem Worth] This is different from the old Phase1-A future-aware adapter.
The old adapter tried to align or modulate a fixed-head path that had no stable
target-set state and no one-model compatibility carrier. Phase2 will align the
segment-level state $U_j$ that directly conditions the decoder output.

## Idea

Candidate name: `PatchEncoderFutureStateAlignment`.

Use a training-only future teacher to encode each ground-truth future segment:

$$
S_j^Y = T_\psi(Y_{a_j:b_j}, q_j),
$$

and a student projection from the inference-time target state:

$$
S_j^X = P_\theta(U_j).
$$

The prediction path remains inference-safe:

$$
\hat{Y}_{a_j:b_j}
=
O_\theta(r \odot (1+\gamma(U_j)) + \beta(U_j)).
$$

Training adds a stop-gradient future-state alignment loss:

$$
\mathcal{L}
=
\mathcal{L}_{prefix\_risk}
+
\lambda_{local}
\sum_j
\left(1-\cos(S_j^X, \operatorname{sg}(S_j^Y))\right)
+
\lambda_{rel}
\left\|
\operatorname{sim}(S^X) -
\operatorname{sg}(\operatorname{sim}(S^Y))
\right\|_F^2.
$$

The local term teaches each target state to match the realized future segment.
The relation term teaches the geometry among target segments, without allowing
later future values to enter inference.

## Theory Check

[Hypothesis] If R.3's remaining errors come from a mismatch between
history-implied target states and realized future segment states, aligning
$U_j$ to a future teacher should improve the weak settings without sacrificing
prefix consistency.

[Why it can work]

1. $U_j$ already controls $\gamma_j,\beta_j$, so alignment gradients can change
   the actual decoder conditioning path.
2. The teacher branch only exists in training. At inference, predictions still
   depend only on $X$ and target segment queries $Q_T$.
3. Segment-level alignment matches the target-set decoder contract. It does not
   force one global history representation to explain every future step.
4. Relation alignment is closer to QDF/SRP evidence than simple per-segment
   cosine, because it supervises future-step dependency without using
   autoregressive rollouts.

[Counterargument] Future teacher states may become an autoencoding shortcut:
they can be easy to reconstruct but irrelevant to forecasting from history. In
that case alignment distance will improve while MSE/MAE do not.

[Counterargument] Strong alignment may over-constrain $U_T$ and damage the R.3
carrier, especially on Weather where R.3 already performs well.

[Boundary] This candidate is not MoE, not autoregression, and not a direct copy
of `TimeAlign`. It is a target-set version of future-state alignment whose
minimal claim is:

> target-side decoder states should be calibrated to future segment states
> during training.

## Design

Base:

- `PatchEncoderPrefixRiskWeighted`;
- target-set horizons `{96,192,336,720}`;
- `target_interaction_layers=0`;
- `prefix_residual_segments=0`;
- `step_loss_weighting=prefix_risk`;
- `step_loss_alpha=0.5`.

Future teacher:

- input: normalized ground-truth future segment
  $Y_{a_j:b_j} \in \mathbb{R}^{B \times S \times C}$;
- shape contract after channel flattening:
  $(B C) \times J \times S$;
- segment encoder: small shared MLP or one-layer temporal Conv/Linear;
- add the same deterministic target feature $q_j$ used by the student side;
- output:
  $S^Y \in \mathbb{R}^{(B C) \times J \times d_s}$.

Student projection:

- input:
  $U_T \in \mathbb{R}^{(B C) \times J \times d}$;
- projection:
  $P_\theta(U_T)$;
- output:
  $S^X \in \mathbb{R}^{(B C) \times J \times d_s}$.

Loss:

- keep R.3 prefix-risk prediction loss as the primary objective;
- add local cosine alignment with stop-gradient teacher;
- add relation alignment among target segments;
- start with small weights, e.g. `future_align_weight=0.02` and
  `future_relation_weight=0.01`;
- use a low reconstruction weight, e.g. `future_recon_weight=0.001`, to train
  the teacher state without making future autoencoding the paper claim.

Leakage boundary:

- teacher branch may read $Y$ only inside training loss construction;
- evaluation and saved predictions must call the model without `future_y`;
- a leakage audit must compare predictions with true future, shuffled future,
  and zero future teacher input. All three predictions must be identical up to
  numerical noise when `model.eval()` is used for forecasting.

## Gate

Local smoke:

- dataset: `ETTh2`;
- target horizons `{96,192,336,720}`;
- epoch: `1`;
- verify required artifacts:
  `metrics_by_target_horizon.csv`, `prefix_consistency.csv`,
  `target_state_similarity.csv`, `future_alignment_stats.csv`,
  `future_leakage_audit.json`, `effective_config.json`.

Remote gate:

- datasets: `ETTh2`, `ETTm1`, `Weather`;
- horizons: `{96,192,336,720}`;
- seed: `2021`;
- compare against both `PatchEncoderFixedHead` and
  `PatchEncoderPrefixRiskWeighted`.

Compatibility-preserving pass:

1. mean relative MSE vs `PatchEncoderFixedHead` remains <= `-0.43%`;
2. no dataset average is worse than R.3 by more than `+0.3%`;
3. prefix mismatch remains numerical zero-level;
4. prediction leakage max abs <= `1e-7`;
5. mean teacher/student cosine improves during training and is reported.

Paper-core candidate pass:

1. wins vs R.3 on at least `7/12` dataset-horizon settings; or
2. mean relative MSE vs R.3 improves by at least `0.5%`; and
3. at least two R.3 weak settings improve:
   `ETTm1/h96`, `ETTm1/h720`, `ETTh2/h720`, `Weather/h96`; and
4. improvement can be localized to horizon/segment/frequency or target-state
   alignment diagnostics.

Rollback:

- If alignment metrics improve but MSE/MAE do not, future teacher is an
  auxiliary proxy, not a paper-core mechanism. Return to step 2-5 and consider
  covariance-aware objective or base architecture change.
- If Weather degrades while ETT improves, alignment is dataset-sensitive; try
  gating or uncertainty weighting only after diagnosing alignment conflict.
- If leakage audit fails, discard the design until the training/evaluation
  boundary is repaired.

## Expected Artifacts

- design doc:
  `docs/experiments/phase2-future-state-alignment-design.md`;
- code explanation after implementation:
  `docs/code-explanation/phase2-future-state-alignment.md`;
- local smoke:
  `artifacts/runs/smoke_phase2_future_state_alignment`;
- remote output:
  `/home/yingch/exp_outputs/r-2026-fatst/phase2_future_state_alignment`;
- analysis:
  `analysis/phase2_future_state_alignment_gate_20260622`.

## Decision

[Decision] Phase2 should begin with `PatchEncoderFutureStateAlignment`, not MoE.
MoE needs a stable and semantically meaningful future-side state; otherwise the
router would only route noisy decoder features.

[Decision] The first implementation should be alignment-only and
compatibility-preserving. If it cannot beat R.3, it still gives useful evidence:
the paper direction should then move from future-aware architecture to either a
principled covariance-aware objective or a stronger base architecture.

## Phase2-R.1 Repair Candidate

[Fact] Phase2-A remote gate failed because `ETTh2` degraded across all horizons,
while `ETTm1` improved across all horizons and `Weather` mostly improved. The
failure is therefore alignment conflict, not leakage or prefix inconsistency.

[Problem] Uniform future-state alignment assumes every teacher segment is a
reliable target for the student decoder state. This assumption is too strong:
when the teacher cannot reconstruct a future segment after normalization, its
state should be treated as a weak anchor rather than an equally weighted
semantic target.

[Idea] `PatchEncoderFutureStateAlignmentConfWeighted` keeps the Phase2-A
prediction path unchanged, but adds:

- `future_recon_normalization=target_energy`, so reconstruction loss is
  comparable across datasets;
- `future_align_weighting=reconstruction_confidence`, where segment confidence
  is computed from normalized teacher reconstruction error;
- pairwise relation weights $\sqrt{c_i c_j}$, so low-confidence segments do not
  dominate relation alignment.

[Gate] The repair candidate must first prove it can remove the `ETTh2`
regression without losing the useful `ETTm1/Weather` signal. If it only improves
alignment metrics but not MSE/MAE, the correct rollback is step 2-3: redefine the
decoder problem instead of stacking MoE on the same future teacher state.

[Decision Update: 2026-06-23] The Phase2-R.1 remote gate failed:

- mean relative MSE vs R.3: `+1.28%`;
- `ETTh2` mean relative MSE vs R.3: `+5.08%`;
- `ETTm1` mean relative MSE vs R.3: `-1.28%`;
- `Weather` mean relative MSE vs R.3: `+0.04%`;
- leakage: `0`;
- max prefix mismatch MSE: `4.7318994e-14`.

This confirms that the issue is not prediction leakage or prefix inconsistency.
The current future-teacher alignment is a dataset-dependent auxiliary proxy and
should not be used as the carrier for MoE. The rollback point is step 2-3:
redefine the decoder problem around output/error-process modeling.
