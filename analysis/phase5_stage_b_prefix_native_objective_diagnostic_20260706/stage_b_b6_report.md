# Phase5 StageB B6 Prefix-Native Objective Diagnostic Report

`current_step`: StageB Step 2/3 problem-existence diagnostic.

## Scope

[Fact] This diagnostic uses train split labels and existing A6-LBF-r256 prediction artifacts.

[Boundary] It does not implement a new objective and does not use validation/test labels to build the label basis.

## 11-step Record

| Field | Content |
| --- | --- |
| `current_step` | StageB Step 2/3 prefix-native objective diagnostic |
| `problem` | Does clean A6-LBF need an objective matched to prefix-native label/basis structure? |
| `existence_evidence` | Train-label PCA/DCT comparison, prefix subspace stability, residual projection into train-label basis |
| `idea` | A basis-native forecast operator may need a basis-native objective instead of generic time-domain point loss |
| `theory_check` | Evidence is positive only if label/residual structure is stable and not just generic low-frequency smoothness |
| `design` | Offline diagnostic; no model training |
| `narrative_gate` | diagnostic_not_enough_pause_b6 |
| `effectiveness_gate` | not applicable before method implementation |
| `artifacts` | this directory |
| `decision` | diagnostic_not_enough_pause_b6 |

## Label Basis Summary

| Dataset | PCA top32 | DCT top32 | A6 basis top32 | PCA-DCT | A6-DCT | Lag1 | Lag24 | Lag96 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | `0.917` | `0.889` | `0.675` | `0.028` | `-0.214` | `0.958` | `0.815` | `0.636` |
| ETTm1 | `0.939` | `0.930` | `0.690` | `0.009` | `-0.240` | `0.976` | `0.610` | `0.761` |
| Weather | `0.832` | `0.831` | `0.251` | `0.001` | `-0.580` | `0.888` | `0.627` | `0.365` |

## Prefix Stability

| Dataset | H96 top16 overlap | H192 top16 overlap | H336 top16 overlap |
| --- | ---: | ---: | ---: |
| ETTh2 | `0.605` | `0.685` | `0.754` |
| ETTm1 | `0.709` | `0.790` | `0.810` |
| Weather | `0.550` | `0.624` | `0.745` |

## Residual Basis Summary

| Dataset | Residual PCA top32 | Residual DCT top32 | Residual A6 top32 | PCA-DCT | A6-DCT | Step Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | `0.351` | `0.353` | `0.287` | `-0.002` | `-0.066` | `0.536` |
| ETTm1 | `0.287` | `0.288` | `0.110` | `-0.001` | `-0.178` | `0.307` |
| Weather | `0.118` | `0.118` | `0.081` | `0.000` | `-0.037` | `0.112` |

## Gate Evaluation

- Label compressibility / non-generic basis evidence passes on: `none`.
- Prefix subspace stability passes on: `ETTm1`.
- Residual basis structure passes on: `ETTh2, ETTm1`.
- A6-specific learned-basis advantage passes on: `none`.
- Non-distance residual check passes on: `ETTh2, ETTm1, Weather`.

## Interpretation

[Fact] `label_pca_top32_energy` measures train-label temporal energy captured by the top 32 train-only PCA components.

[Fact] `label_dct_top32_energy` is a generic low-frequency control. A small PCA-DCT gap weakens the novelty of a learned label-basis objective.

[Fact] `label_a6_basis_top32_energy` and `residual_a6_basis_top32_energy` use the learned temporal basis from the pure no-align/no-recon A6 checkpoint when available.

[Fact] `residual_label_pca_top32_energy` measures how much A6-LBF residual energy lies in train-label PCA directions.

[Fact] `residual_step_spearman` measures whether residual energy is still mostly a forecast-distance effect.

[Decision] `diagnostic_not_enough_pause_b6`.

[Next] Do not implement a B6 objective. Pause StageB or redefine the problem.

## Output Files

- `stage_b_b6_label_basis_summary.csv`
- `stage_b_b6_prefix_stability.csv`
- `stage_b_b6_residual_basis_summary.csv`
- `stage_b_b6_report.md`
