# Phase2-C Objective Pressure Diagnostic

## Decision

[Decision] `mixed-horizon objective / step-region covariance` is a real Phase2-C problem.

[Inference] However, naive monotone prefix-risk weighting is insufficient as a paper-core mechanism.

## What Was Tested

[Fact] This diagnostic compares R.3 `PatchEncoderPrefixRiskWeighted` against the uniform `PatchEncoderTargetSetDecoder`, not only against `PatchEncoderFixedHead`.

[Fact] The expected step pressure follows the actual training loop: one horizon is sampled uniformly from `{96,192,336,720}`, and the loss is averaged over the selected horizon. For prefix-risk, the per-step weight is normalized by the full `Hmax=720` weight mean.

For a step $t$, the expected pressure is:

$$
p_t = \frac{1}{|\mathcal{H}|}\sum_{H\in\mathcal{H}, t\le H}\frac{w_t}{H}.
$$

Here `alpha=0.5` for R.3.

## Main Evidence

- R.3 wins vs uniform target-set: `11/12`.
- Mean relative MSE vs uniform: `-1.03%`.
- Mean h96 relative MSE vs uniform: `-1.81%`.
- Mean h720 relative MSE vs uniform: `-0.70%`.
- H720-prefix h96/h192 mean relative MSE vs uniform: `-1.62%`.
- Segment wins vs uniform: `24/30`.
- Segment relative MSE vs uniform: `-1.00%`.

## Objective Pressure Shift

| Region | Uniform pressure share | Prefix-risk pressure share | Share delta | Raw pressure ratio |
| --- | ---: | ---: | ---: | ---: |
| 1-96 | 0.4798 | 0.7217 | +50.43% | 2.612 |
| 97-192 | 0.2298 | 0.1540 | -32.98% | 1.164 |
| 193-336 | 0.1571 | 0.0775 | -50.71% | 0.856 |
| 337-720 | 0.1333 | 0.0469 | -64.85% | 0.610 |

## Horizon Loss Multipliers

| Horizon | Mean prefix-risk weight | Interpretation |
| ---: | ---: | --- |
| 96 | 2.612 | amplified relative to uniform |
| 192 | 1.888 | amplified relative to uniform |
| 336 | 1.445 | amplified relative to uniform |
| 720 | 1.000 | same as uniform |

## R.3 vs Uniform Target-Set

| Dataset | Horizon | R.3 vs uniform | Uniform vs FixedHead | R.3 vs FixedHead |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | -1.33% | +0.48% | -0.86% |
| ETTh2 | 192 | -0.03% | -2.17% | -2.20% |
| ETTh2 | 336 | +0.55% | -0.86% | -0.31% |
| ETTh2 | 720 | -0.71% | +1.47% | +0.75% |
| ETTm1 | 96 | -2.65% | +5.63% | +2.83% |
| ETTm1 | 192 | -2.06% | -0.33% | -2.38% |
| ETTm1 | 336 | -1.25% | +1.03% | -0.22% |
| ETTm1 | 720 | -0.52% | +1.62% | +1.09% |
| Weather | 96 | -1.45% | +2.12% | +0.64% |
| Weather | 192 | -1.09% | -0.35% | -1.43% |
| Weather | 336 | -0.99% | -1.41% | -2.39% |
| Weather | 720 | -0.87% | +0.16% | -0.71% |

## Alignment Between Pressure And Effect

- Pearson r between horizon loss multiplier and R.3 main-horizon delta: `-0.5530`.
- Pearson r between segment pressure-share delta and segment-level R.3 delta: `-0.6804`.

[Inference] A useful objective-level direction should not be another hand-tuned monotone prefix emphasis. The evidence supports a more structured objective that can distinguish early prefixes, middle regions, and long-tail regions rather than assigning all steps a single decreasing curve.

## Gate

- r3_improves_uniform_mean_mse: `True`
- r3_improves_all_h96_vs_uniform: `True`
- r3_improves_h720_prefix_short_vs_uniform: `True`
- r3_still_has_fixed_specialist_gap: `True`
- segment_effect_varies_by_pressure: `True`
- objective_problem_exists: `True`
- naive_prefix_risk_is_insufficient: `True`

## Decision Impact

[Decision] The next implementable candidate should be an objective-level mechanism, not a new target-state interaction or MoE layer. It should explicitly model step-region covariance or horizon-region balance, then be evaluated against both uniform target-set and R.3.

[Candidate] Phase2-C can test a `Step-Covariance Balanced Objective`: estimate a fixed region covariance/importance prior from training targets or validation residual structure, then use it to balance loss pressure across early prefix, middle transition, and long-tail regions. The pass condition must require improvement over R.3, not merely over uniform target-set.

## Artifacts

- `objective_pressure_summary.csv`
- `r3_vs_uniform_main.csv`
- `r3_vs_uniform_segments.csv`
- `r3_vs_uniform_h720_prefix.csv`
- `objective_pressure_summary.json`
- `objective_pressure_curve.png`
- `segment_effect_vs_pressure.png`
