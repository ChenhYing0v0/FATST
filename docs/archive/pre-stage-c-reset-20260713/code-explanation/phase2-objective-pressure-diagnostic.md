# Phase2 Objective Pressure Diagnostic Code Explanation

## Purpose

`scripts/analyze_phase2_objective_pressure.py` audits whether the current
mixed-horizon training objective is a real bottleneck after Phase2-A/R/B failed.
It does not train a model and does not change model code. It compares completed
artifacts from:

- uniform `PatchEncoderTargetSetDecoder`;
- R.3 `PatchEncoderPrefixRiskWeighted`.

## Inputs

Default inputs:

- `analysis/phase1_target_set_decoder_gate_20260622`;
- `analysis/phase1_prefix_risk_weighted_gate_20260622`.

The script reads:

- `*_vs_fixed.csv` for dataset-horizon MSE/MAE;
- `*_vs_fixed_segments.csv` for region-level MSE/MAE;
- `*_h720_prefix_reference.csv` for H720 prefix comparisons.

## Objective Pressure Computation

The script mirrors the training loop in
`baselines/patch_encoder_target_set_decoder/train.py`:

1. sample one horizon uniformly from `{96,192,336,720}`;
2. average loss over the selected horizon;
3. optionally apply prefix-risk step weights.

For forecast step $t$, expected pressure is:

$$
p_t
=
\frac{1}{|\mathcal{H}|}
\sum_{H\in\mathcal{H},t\le H}
\frac{w_t}{H}.
$$

For `uniform`, $w_t=1$. For `prefix_risk`, $w_t$ is the same normalized
power-law weight used by training:

$$
w_t
=
\frac{(t/H_{max})^{-\alpha}}
{\frac{1}{H_{max}}\sum_{s=1}^{H_{max}}(s/H_{max})^{-\alpha}}.
$$

The output `objective_pressure_summary.csv` reports both normalized pressure
share and raw pressure ratio for four step regions:

- `1-96`;
- `97-192`;
- `193-336`;
- `337-720`.

## R.3 vs Uniform Comparison

The script reconstructs R.3's effect against uniform target-set by comparing
their actual `target_mse` values:

$$
\Delta_{R3/uniform}
=
\frac{\operatorname{MSE}_{R3}}{\operatorname{MSE}_{uniform}}-1.
$$

This is separate from the already existing comparison against
`PatchEncoderFixedHead`. The distinction matters because Phase2-C is asking
whether objective weighting is real, not whether it already solves the full
specialist gap.

## Correlation Diagnostics

Two correlations are written to `objective_pressure_summary.json`:

- horizon loss multiplier vs main-horizon R.3 delta;
- segment pressure-share delta vs segment-level R.3 delta.

A negative correlation means stronger pressure is associated with lower MSE.
This does not prove causality, but it is useful evidence that R.3's behavior is
not random with respect to objective pressure.

## Outputs

The script writes:

- `phase2_objective_pressure_diagnostic_report.md`;
- `objective_pressure_summary.csv`;
- `objective_pressure_summary.json`;
- `objective_pressure_curves.json`;
- `r3_vs_uniform_main.csv`;
- `r3_vs_uniform_segments.csv`;
- `r3_vs_uniform_h720_prefix.csv`;
- `objective_pressure_curve.png`;
- `segment_effect_vs_pressure.png`.

## Code-Theory Consistency

Intended theory:

- mixed-horizon one-model training creates nonuniform step pressure;
- R.3's gains should align with changed pressure if the objective bottleneck is
  real;
- if R.3 still leaves fixed-specialist gaps, monotone prefix weighting is only
  a diagnostic and should be replaced by a structured region/covariance
  objective.

How the code realizes it:

- computes the exact expected pressure induced by the current horizon sampler
  and weighted loss;
- compares R.3 directly to uniform target-set artifacts;
- summarizes whether the pressure-effect relationship supports moving to
  Phase2-C.

Remaining proxy:

- the script uses evaluation artifacts, not training gradients;
- covariance is proposed in the follow-up design doc but not yet implemented;
- correlations are diagnostic evidence, not proof that a new objective will
  improve performance.

Falsification evidence:

- R.3 does not beat uniform on mean MSE;
- h96 and H720-prefix h96/h192 do not improve;
- pressure-effect correlations are near zero and gains appear dataset-random;
- a later covariance-balanced objective cannot beat R.3.
