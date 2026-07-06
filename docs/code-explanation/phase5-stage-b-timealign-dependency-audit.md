# Phase5 StageB TimeAlign Dependency Audit Code Explanation

This document explains the local dependency audit and the training adapter change required for future ablations.

## Touched Code

| File | Role |
| --- | --- |
| `scripts/analyze_phase5_stage_b_timealign_dependency_audit.py` | reads existing artifacts and writes dependency audit tables/report |
| `baselines/timealign_official/train_repo.py` | adds `--w-align` override for no-align ablations |
| `scripts/remote/run_phase5_stage_b_timealign_dependency_ablation.sh` | prepared remote ablation matrix; not launched |

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
w_align = preset.w_align if args.w_align is None else args.w_align
```

This is required for no-align ablations because `w_align` was previously fixed inside `OFFICIAL_PRESETS`.

## Code-Theory Consistency Evaluation

[Intended Theory] The audit separates readout/operator evidence from inherited TimeAlign alignment dependence.

[Code Realization] The artifact analyzer only compares existing same-align runs and training component shares. It does
not claim causal attribution.

[Remaining Proxy] Loss component share is not performance attribution. It only shows that inherited alignment remains
active in optimization.

[Falsification] A future `no_align_no_recon` arm matching current A6-LBF would weaken the need for PhaseB align
innovation. A large collapse after removing `w_align` would support basis-aware alignment as a meaningful PhaseB target.
