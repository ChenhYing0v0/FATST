# Phase5 StageB TimeAlign Dependency Audit Code Explanation

This document explains the local dependency audit, the training adapter change required for ablations, and the
returned ablation analyzer.

## Touched Code

| File | Role |
| --- | --- |
| `scripts/analyze_phase5_stage_b_timealign_dependency_audit.py` | reads existing artifacts and writes dependency audit tables/report |
| `scripts/analyze_phase5_stage_b_timealign_dependency_ablation.py` | reads returned no-align/no-recon artifacts and writes causal dependency summaries/report |
| `baselines/timealign_official/train_repo.py` | adds `--w-align` override for no-align ablations |
| `scripts/remote/run_phase5_stage_b_timealign_dependency_ablation.sh` | remote ablation matrix runner |

## Analyzer Flow

`analyze_phase5_stage_b_timealign_dependency_audit.py` reads:

- A6-LBF-r256 official-last artifacts from
  `analysis/phase5_timealign_hss_a6_capacity_native_gate_20260703/raw/official-last`;
- official unified TimeAlign artifacts from
  `analysis/phase5_timealign_hss_a3e_ettm1_replacement_gate_20260701/raw/official/official-last`.

It writes:

| Output | Meaning |
| --- | --- |
| `stage_b_timealign_dependency_metric_comparison.csv` | A6 vs official unified MSE under same align/recon setting |
| `stage_b_timealign_dependency_training_components.csv` | prediction/recon/alignment shares at best-val and last epoch |
| `stage_b_timealign_dependency_report.md` | diagnostic decision |

## Training Adapter Change

`train_repo.py` now accepts:

```text
--w-align <float>
```

If omitted, it keeps the official dataset preset value. This preserves current behavior for existing runners.

The effective argument is:

```text
w_align = 0.0 if A6-LBF else (preset.w_align if args.w_align is None else args.w_align)
w_recon = 0.0 if A6-LBF else args.w_recon
```

This originally enabled no-align ablations because `w_align` was fixed inside `OFFICIAL_PRESETS`. After the B4
decision, the A6-LBF path always disables these auxiliary weights; the official TimeAlign path still keeps them for
baseline reproduction.

## Ablation Analyzer Flow

`analyze_phase5_stage_b_timealign_dependency_ablation.py` reads returned remote artifacts under:

```text
analysis/phase5_stage_b_timealign_dependency_ablation_20260706/raw/official-last/
```

Expected matrix:

| Arm | Meaning |
| --- | --- |
| `current_align_recon` | current A6-LBF with inherited `w_recon=1.0`, `w_align=0.1` |
| `no_align_recon` | remove alignment only |
| `align_no_recon` | remove reconstruction only |
| `no_align_no_recon` | pure A6-LBF head/operator objective |

The script parses every `metrics_by_target_horizon.csv` and `training_log.csv` for ETTh2, ETTm1, and Weather. It
fails fast if the expected `4 arms * 3 datasets = 12` metric files are missing.

It writes:

| Output | Meaning |
| --- | --- |
| `stage_b_dependency_ablation_horizon_metrics.csv` | horizon-level MSE/MAE and deltas vs `current_align_recon` |
| `stage_b_dependency_ablation_dataset_summary.csv` | dataset-level mean MSE/MAE and wins vs current |
| `stage_b_dependency_ablation_overall_summary.csv` | overall mean deltas, wins, max regression, and max gain |
| `stage_b_dependency_ablation_training_summary.csv` | best-val/last loss shares for prediction/recon/align |
| `stage_b_dependency_ablation_report.md` | Step 9/10 diagnostic report and decision |

## Code-Theory Consistency Evaluation

[Intended Theory] The audit separates readout/operator evidence from inherited TimeAlign alignment dependence.

[Code Realization] The artifact analyzer only compares existing same-align runs and training component shares. It does
not claim causal attribution.

[Remaining Proxy] Loss component share is not performance attribution. It only shows that inherited alignment remains
active in optimization.

[Falsification] A future `no_align_no_recon` arm matching current A6-LBF would weaken the need for PhaseB align
innovation. A large collapse after removing `w_align` would support basis-aware alignment as a meaningful PhaseB target.

## Returned Code-Theory Evaluation

[Observed] `no_align_no_recon` matches current A6-LBF closely: mean MSE is only `+0.07%` and it wins `7/12`
horizon settings.

[Interpretation] This falsifies the strong version of the dependency concern: A6-LBF performance is not primarily
carried by inherited TimeAlign alignment/reconstruction.

[Boundary] The analyzer does not prove that alignment is useless in all settings. It only shows that, in the current
ETTh2/ETTm1/Weather official-last matrix, alignment dependence is too small to justify B5 as the immediate paper-core
method.

[Code Consequence] The active A6-LBF path now removes the future reconstruction/alignment branch and forces
`w_recon=w_align=0.0`. The historical dependency ablation runner is therefore deprecated for new runs; its returned
artifacts remain the evidence for this cleanup decision.
