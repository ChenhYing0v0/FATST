# Phase5 StageB TimeAlign Dependency Ablation Report

`current_step`: StageB Step 9/10 dependency ablation result analysis.

## Scope

[Fact] This report analyzes the completed 12-run remote matrix under `official-last` policy.

[Boundary] These runs test dependency on inherited `w_align` and `w_recon`. They do not implement basis-aware alignment.

## 11-step Record

| Field | Content |
| --- | --- |
| `current_step` | StageB Step 9/10 dependency ablation analysis |
| `problem` | Does A6-LBF depend on inherited TimeAlign alignment/reconstruction enough to require PhaseB align innovation? |
| `existence_evidence` | 4-arm no-align/no-recon matrix on ETTh2/ETTm1/Weather with horizons 96/192/336/720 |
| `idea` | Compare removing `w_align` and/or `w_recon` against current A6-LBF |
| `theory_check` | If no-align/no-recon is competitive, full architecture independence is less risky; if removing align collapses performance, basis-aware align is better motivated |
| `design` | Remote official-last ablation; no new method trained |
| `narrative_gate` | dependency_ablation_pass_for_head_contribution_but_not_for_b5 |
| `effectiveness_gate` | not applicable for new method; this is diagnostic evidence |
| `artifacts` | this directory |
| `decision` | A6-LBF is not heavily dependent on inherited alignment; do not prioritize B5 basis-aware alignment as the next method |

## Overall Summary

| Arm | Mean MSE | Mean MSE vs Current | Wins vs Current | Max Regression | Max Gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| `current_align_recon` | `0.2857` | `0.00%` | 0/12 | `0.00%` | `0.00%` |
| `no_align_recon` | `0.2859` | `0.07%` | 7/12 | `0.67%` | `-0.34%` |
| `align_no_recon` | `0.2856` | `-0.04%` | 8/12 | `0.04%` | `-0.32%` |
| `no_align_no_recon` | `0.2859` | `0.07%` | 7/12 | `0.67%` | `-0.34%` |

## Dataset Summary

| Arm | ETTh2 vs Current | ETTm1 vs Current | Weather vs Current |
| --- | ---: | ---: | ---: |
| `current_align_recon` | `0.00%` | `0.00%` | `0.00%` |
| `no_align_recon` | `0.39%` | `-0.14%` | `-0.05%` |
| `align_no_recon` | `0.00%` | `-0.11%` | `-0.00%` |
| `no_align_no_recon` | `0.39%` | `-0.14%` | `-0.05%` |

## Horizon-Level Deltas

| Arm | Dataset | H96 | H192 | H336 | H720 |
| --- | --- | ---: | ---: | ---: | ---: |
| `current_align_recon` | ETTh2 | `0.00%` | `0.00%` | `0.00%` | `0.00%` |
| `current_align_recon` | ETTm1 | `0.00%` | `0.00%` | `0.00%` | `0.00%` |
| `current_align_recon` | Weather | `0.00%` | `0.00%` | `0.00%` | `0.00%` |
| `no_align_recon` | ETTh2 | `0.27%` | `0.67%` | `0.64%` | `0.07%` |
| `no_align_recon` | ETTm1 | `-0.34%` | `-0.20%` | `-0.17%` | `0.06%` |
| `no_align_recon` | Weather | `-0.07%` | `-0.08%` | `-0.03%` | `-0.03%` |
| `align_no_recon` | ETTh2 | `0.03%` | `0.04%` | `0.02%` | `-0.05%` |
| `align_no_recon` | ETTm1 | `-0.32%` | `-0.09%` | `-0.14%` | `0.04%` |
| `align_no_recon` | Weather | `-0.01%` | `-0.01%` | `-0.00%` | `-0.00%` |
| `no_align_no_recon` | ETTh2 | `0.27%` | `0.67%` | `0.64%` | `0.07%` |
| `no_align_no_recon` | ETTm1 | `-0.34%` | `-0.20%` | `-0.17%` | `0.06%` |
| `no_align_no_recon` | Weather | `-0.07%` | `-0.08%` | `-0.03%` | `-0.03%` |

## Training Component Shares

| Arm | Dataset | Selector | Epoch | Pred Share | Recon Share | Align Share | Val MSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `align_no_recon` | ETTh2 | `last` | 10 | `0.80` | `0.00` | `0.20` | `0.4586` |
| `align_no_recon` | ETTm1 | `last` | 10 | `0.91` | `0.00` | `0.09` | `0.6007` |
| `align_no_recon` | Weather | `last` | 10 | `0.88` | `0.00` | `0.12` | `0.4903` |
| `current_align_recon` | ETTh2 | `last` | 10 | `0.78` | `0.03` | `0.19` | `0.4587` |
| `current_align_recon` | ETTm1 | `last` | 10 | `0.75` | `0.16` | `0.08` | `0.6019` |
| `current_align_recon` | Weather | `last` | 10 | `0.87` | `0.00` | `0.12` | `0.4904` |
| `no_align_no_recon` | ETTh2 | `last` | 10 | `1.00` | `0.00` | `0.00` | `0.4672` |
| `no_align_no_recon` | ETTm1 | `last` | 10 | `1.00` | `0.00` | `0.00` | `0.6007` |
| `no_align_no_recon` | Weather | `last` | 10 | `1.00` | `0.00` | `0.00` | `0.4903` |
| `no_align_recon` | ETTh2 | `last` | 10 | `0.96` | `0.04` | `0.00` | `0.4672` |
| `no_align_recon` | ETTm1 | `last` | 10 | `0.82` | `0.18` | `0.00` | `0.6007` |
| `no_align_recon` | Weather | `last` | 10 | `1.00` | `0.00` | `0.00` | `0.4903` |

## Interpretation

[Fact] Removing alignment while keeping reconstruction (`no_align_recon`) changes mean MSE by `0.07%` and wins `7/12` settings against current.

[Fact] Removing reconstruction while keeping alignment (`align_no_recon`) changes mean MSE by `-0.04%` and wins `8/12` settings against current.

[Fact] Removing both inherited losses (`no_align_no_recon`) changes mean MSE by `0.07%` and wins `7/12` settings against current.

[Strong Evidence] A6-LBF keeps most of its performance without inherited TimeAlign alignment/reconstruction. The pure head/operator arm is not collapsing.

[Strong Evidence] The full inherited-auxiliary setting is not clearly better than ablated settings. `align_no_recon` is slightly better on mean MSE, and `no_align_no_recon` wins more than half of the horizon settings despite removing both inherited losses.

[Mechanism Note] `no_align_recon` and `no_align_no_recon` are effectively identical in the returned metrics. This is expected from the current code path: when `w_align=0`, the future reconstruction branch has no active path into the history-derived prediction head, so reconstruction alone mostly trains the future branch/proj_y rather than the forecast operator.

## Decision

[Decision] `dependency_ablation_pass_for_head_contribution_but_not_for_b5`.

[Narrative Consequence] The paper can defend A6-LBF as more than a TimeAlign-alignment artifact: the learned-basis head/operator remains competitive even without `w_align` and `w_recon`. This reduces the urgency of modifying TimeAlign's align mechanism just to claim independence.

[StageB Consequence] B5 basis-aware future alignment is not strongly motivated as the next paper-core mechanism. If we implement it now, it risks being a small auxiliary-loss variant rather than a necessary architectural innovation.

[Next Research Direction] Prefer returning to architecture-aware objective design around prefix-native learned basis / label-autocorrelation, with `no_align_no_recon` and `current_align_recon` as controls.

## Output Files

- `stage_b_dependency_ablation_horizon_metrics.csv`
- `stage_b_dependency_ablation_dataset_summary.csv`
- `stage_b_dependency_ablation_overall_summary.csv`
- `stage_b_dependency_ablation_training_summary.csv`
- `stage_b_dependency_ablation_report.md`
