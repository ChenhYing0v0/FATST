# Phase5 StageB B9-FSN-SCF Small Gate Report

## Scope

Arms: `a6_clean`, `b9_fsn_scf`, `b9_no_stage`.
Datasets: ETTh2, ETTm1, Weather. Horizons: 96, 192, 336, 720.

## Summary

| comparison | dataset | settings | mse_wins | mean_relative_mse_pct | mean_relative_mae_pct |
| --- | --- | --- | --- | --- | --- |
| b9_fsn_scf_vs_a6_clean | ETTh2 | 4 | 4 | -0.29% | -0.20% |
| b9_fsn_scf_vs_a6_clean | ETTm1 | 4 | 4 | -0.07% | -0.05% |
| b9_fsn_scf_vs_a6_clean | Weather | 4 | 4 | -0.02% | -0.06% |
| b9_fsn_scf_vs_a6_clean | ALL | 12 | 12 | -0.13% | -0.11% |
| b9_fsn_scf_vs_b9_no_stage | ETTh2 | 4 | 2 | +0.0046% | +0.0013% |
| b9_fsn_scf_vs_b9_no_stage | ETTm1 | 4 | 0 | +0.0059% | -0.0002% |
| b9_fsn_scf_vs_b9_no_stage | Weather | 4 | 0 | +0.0003% | -0.0002% |
| b9_fsn_scf_vs_b9_no_stage | ALL | 12 | 2 | +0.0036% | +0.0003% |
| b9_no_stage_vs_a6_clean | ETTh2 | 4 | 4 | -0.30% | -0.20% |
| b9_no_stage_vs_a6_clean | ETTm1 | 4 | 4 | -0.07% | -0.05% |
| b9_no_stage_vs_a6_clean | Weather | 4 | 4 | -0.02% | -0.06% |
| b9_no_stage_vs_a6_clean | ALL | 12 | 12 | -0.13% | -0.11% |

## Gate Reading

[Decision] `blocked_by_no_stage_control`: B9 does not beat the no-stage capacity control overall.

- B9 vs A6 mean relative MSE: `-0.13%`, MSE wins `12/12`.
- B9 vs no-stage mean relative MSE: `+0.0036%`, MSE wins `2/12`.

## Artifacts

- `b9_fsn_scf_small_gate_comparison.csv`
- `b9_fsn_scf_small_gate_summary.csv`
- `b9_fsn_scf_model_diagnostics.csv`
