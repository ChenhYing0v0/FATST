# Phase2-C.1 Covariance / Novelty Diagnostic

## Decision

[Decision] Offline covariance/novelty diagnostic supports entering `step_covariance_balanced` step 4-6.

[Fact] This diagnostic uses only train-split target windows and completed
Phase2 analysis artifacts. It does not train a model and does not use future
labels at inference.

## What Was Computed

- Dataset split and scaling follow `ForecastDataset`: the scaler is fit on
  the train split, then train targets are analyzed in normalized space.
- Regions are `1-96`, `97-192`, `193-336`, and `337-720` for H720 target
  windows with `seq_len=336`.
- `pooled_region_mean_std`: RMS variation of each window's region mean
  after channel-wise centering.
- `max_prev_region_r2`: largest squared correlation between the current
  region mean and any earlier region mean.
- `novelty_score = pooled_region_mean_std * (1 - max_prev_region_r2)`.
- `novelty_share`: dataset-local normalized novelty across the four regions.

## Dataset Novelty Summary

| Dataset | Max novelty region | Max share | Early share | Late share | Early prefix minus novelty | Late prefix minus novelty |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | 1-96 | 0.4763 | 0.4763 | 0.1906 | +0.2454 | -0.1437 |
| ETTm1 | 1-96 | 0.6152 | 0.6152 | 0.1299 | +0.1065 | -0.0830 |
| Weather | 1-96 | 0.4815 | 0.4815 | 0.1623 | +0.2402 | -0.1155 |

## Effect Alignment

| Correlation | Value | Interpretation |
| --- | ---: | --- |
| R.3 segment delta vs novelty share | -0.7219 | negative means higher novelty aligns with larger R.3 MSE reduction |
| R.3 segment delta vs prefix pressure share | -0.6909 | negative means R.3's prefix pressure aligns with MSE reduction |
| region-balanced delta vs novelty deficit | 0.6253 | positive means underweighting high-novelty regions aligns with loss |
| aggregate R.3 delta vs novelty share | -0.6714 | region-averaged version of the first test |
| aggregate region-balanced delta vs novelty deficit | 0.6253 | region-averaged version of the failure test |

## Region-Level Alignment Table

| Dataset | Segment | Novelty share | R.3 vs uniform mean MSE | Region-balanced vs R.3 H720 MSE |
| --- | --- | ---: | ---: | ---: |
| ETTh2 | 1-96 | 0.4763 | -1.74% | +0.65% |
| ETTh2 | 193-336 | 0.1692 | +1.24% | -0.71% |
| ETTh2 | 337-720 | 0.1906 | -1.19% | +1.43% |
| ETTh2 | 97-192 | 0.1638 | +0.74% | -1.54% |
| ETTm1 | 1-96 | 0.6152 | -2.68% | +6.29% |
| ETTm1 | 193-336 | 0.1264 | -0.45% | +1.17% |
| ETTm1 | 337-720 | 0.1299 | +0.01% | -0.15% |
| ETTm1 | 97-192 | 0.1285 | -1.51% | +2.98% |
| Weather | 1-96 | 0.4815 | -1.39% | +2.08% |
| Weather | 193-336 | 0.1599 | -0.84% | +0.87% |
| Weather | 337-720 | 0.1623 | -0.90% | +1.08% |
| Weather | 97-192 | 0.1963 | -0.78% | +0.84% |

## Gate

- r3_gain_aligns_with_novelty: `True`
- region_balanced_loss_aligns_with_novelty_deficit: `True`
- aggregate_patterns_align: `True`
- supports_step_covariance_balanced: `True`

## Decision Impact

[Inference] The next step can move to loop step 4-6 and define a
`step_covariance_balanced` objective. The design must keep the
diagnostic separation between coverage balance and novelty balance,
then test against R.3 rather than only against uniform target-set.

## Artifacts

- `region_novelty_stats.csv`
- `r3_novelty_effect_alignment.csv`
- `region_balanced_novelty_effect_alignment.csv`
- `aggregate_novelty_effect_alignment.csv`
- `covariance_novelty_summary.json`
- `novelty_effect_alignment.png`
