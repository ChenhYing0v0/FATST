# Phase2 Covariance / Novelty Diagnostic Code Explanation

## Purpose

`scripts/analyze_phase2_covariance_novelty.py` is the Phase2-C.1 offline
diagnostic after the full `PatchEncoderRegionBalanced` gate failed. It checks
whether a training-target covariance / novelty prior can explain two observed
facts:

- R.3 `PatchEncoderPrefixRiskWeighted` improves the uniform target-set model;
- equal-region `region_balanced` hurts most settings despite preserving prefix
  consistency.

The script does not train a model and does not change inference behavior.

## Inputs

Default inputs:

- datasets under `/Users/river/PaperResearch/Project/datasets`;
- `analysis/phase2_objective_pressure_diagnostic_20260623`;
- `analysis/phase2_region_balanced_gate_20260623`.

Dataset loading follows `ForecastDataset`: the train split is selected with the
same borders, the scaler is fit on the train split, and target regions are
analyzed in normalized space.

## Tensor And Statistic Flow

For each dataset, the script forms H720 train target windows after the same
`seq_len=336` history context:

$$
Y_{1:720}\in\mathbb{R}^{720\times C}.
$$

It partitions each target into:

- `1-96`;
- `97-192`;
- `193-336`;
- `337-720`.

For each region $r$, the script computes the per-window region mean
$\bar{Y}_r\in\mathbb{R}^C$, then computes:

$$
u_r
=
\operatorname{std}(\bar{Y}_r)
\left(1-\max_{s<r}\rho^2(\bar{Y}_r,\bar{Y}_s)\right).
$$

In code this is implemented as a product, not a sum:

$$
u_r
=
\operatorname{std}(\bar{Y}_r)
\cdot
\left(1-\max_{s<r}\rho^2(\bar{Y}_r,\bar{Y}_s)\right).
$$

The dataset-local normalized score is:

$$
\tilde{u}_r
=
\frac{u_r}{\sum_k u_k}.
$$

## Effect Alignment

The script aligns novelty with existing completed artifacts:

- `r3_vs_uniform_segments.csv` for R.3's segment-level effect against uniform;
- `phase2_region_balanced_h720_regions_vs_r3.csv` for the failed
  `region_balanced` H720 region effect against R.3.

The key correlations are:

- R.3 segment delta vs novelty share;
- R.3 segment delta vs prefix pressure share;
- `region_balanced` delta vs `novelty_share - 0.25`;
- aggregate region versions of the same checks.

Negative R.3 correlation means higher novelty is associated with larger MSE
reduction. Positive `region_balanced` correlation means equal-region weighting
fails more where it underweights novelty.

## Outputs

The script writes:

- `phase2_covariance_novelty_diagnostic_report.md`;
- `region_novelty_stats.csv`;
- `r3_novelty_effect_alignment.csv`;
- `region_balanced_novelty_effect_alignment.csv`;
- `aggregate_novelty_effect_alignment.csv`;
- `covariance_novelty_summary.json`;
- `novelty_effect_alignment.png`.

## Code-Theory Consistency

Intended theory:

- one-model multi-horizon training may need objective pressure based on
  future-region dependency, not only coverage count or monotone prefix emphasis;
- if the novelty prior explains both R.3 gains and `region_balanced` losses, it
  can justify designing `step_covariance_balanced`.

How the code realizes it:

- uses train targets only;
- reuses the current dataset split and scaling contract;
- compares novelty against already completed evaluation artifacts;
- makes the next-step decision before remote training.

Remaining proxy:

- novelty is static and dataset-level, not sample-adaptive;
- region means compress temporal structure inside each region;
- correlations are diagnostic evidence, not causal proof.

Falsification evidence:

- novelty does not align with R.3 gains;
- equal-region failures do not align with novelty underweighting;
- the effect is driven by one dataset only;
- a later covariance-aware objective cannot beat R.3.
