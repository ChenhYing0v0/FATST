# Phase5 StageB: TimeAlign Dependency And Basis-Aware Align Diagnostic

`current_step`: StageB Step 9/10 dependency diagnostic decision completed.

本文档处理当前论文叙事风险：A6-LBF 已改造 decoder/head，但仍继承 TimeAlign encoder 与
future alignment。若不证明 A6-LBF 的独立贡献边界，论文会被视为 TimeAlign variant。

## 11-step Record

| Field | Content |
| --- | --- |
| `current_step` | StageB Step 9/10：TimeAlign dependency ablation decision |
| `problem` | A6-LBF 的收益是否主要来自 inherited TimeAlign align/recon，而不是 learned-basis operator |
| `existence_evidence` | artifact audit plus returned no-align/no-recon dependency ablation |
| `idea` | 先做 attribution diagnostic，再决定是否设计 basis-aware future alignment |
| `theory_check` | same-align improvement 证明 head/operator contribution；no-align/no-recon competitive 则说明 inherited align/recon 不是必要性能来源 |
| `design` | artifact-only audit completed；12-run remote ablation completed |
| `narrative_gate` | `dependency_ablation_pass_for_head_contribution_but_not_for_b5` |
| `effectiveness_gate` | diagnostic-only；未训练新方法 |
| `artifacts` | `analysis/phase5_stage_b_timealign_dependency_audit_20260706/`; `analysis/phase5_stage_b_timealign_dependency_ablation_20260706/` |
| `decision` | Contribution 1 attribution strengthened；B5 basis-aware alignment deferred |

## What We Tested

[Fact] Existing artifact audit compares A6-LBF-r256 against official unified TimeAlign while keeping the inherited
TimeAlign alignment/reconstruction setting.

[Boundary] This is not a causal ablation. It cannot answer what happens when `w_align` or `w_recon` is removed.

## Current Evidence

[Strong Evidence] A6-LBF-r256 beats official unified TimeAlign under the same inherited align/recon mechanism:
`11/12` MSE wins, mean MSE change `-1.94%`.

[Risk Evidence] The training objective still includes inherited terms. At last epoch, A6-LBF weighted alignment
share is about ETTh2 `0.19`, ETTm1 `0.08`, Weather `0.12`. Therefore the paper cannot claim the full model is
independent from TimeAlign.

[Returned Evidence] The causal ablation weakens that risk. Removing both inherited auxiliary losses
(`no_align_no_recon`) changes mean MSE by only `+0.07%` and wins `7/12` horizon settings against current A6-LBF.
Removing reconstruction while keeping alignment (`align_no_recon`) is slightly better on mean MSE (`-0.04%`) and
wins `8/12`, but the margin is too small to motivate a paper-core alignment method by itself.

## Required Remote Ablation

The completed remote matrix was:

| Arm | `w_recon` | `w_align` | Purpose |
| --- | ---: | ---: | --- |
| `current_align_recon` | `1.0` | `0.1` | reproduce current A6-LBF |
| `no_align_recon` | `1.0` | `0.0` | isolate alignment dependence |
| `align_no_recon` | `0.0` | `0.1` | isolate reconstruction dependence |
| `no_align_no_recon` | `0.0` | `0.0` | pure A6-LBF head/operator |

Runner:

```text
scripts/remote/run_phase5_stage_b_timealign_dependency_ablation.sh
```

The runner was launched after commit/push and GPU preflight. Results are analyzed in
`analysis/phase5_stage_b_timealign_dependency_ablation_20260706/stage_b_dependency_ablation_report.md`.

## Deferred Basis-Aware Align Hypothesis

If future diagnostics show that alignment is important, the PhaseB innovation should not be a generic `glocal_align`
variant. It should be A6-LBF-specific:

> align history-derived and future-derived representations in learned basis / coefficient space.

Candidate tensor path:

```text
history path: hidden_hist -> learned_basis_coeff -> c_hist
future path:  hidden_future/reconstruct path -> future_basis_coeff -> c_future
alignment:    align(c_hist, stopgrad_or_detached(c_future))
inference:    use only c_hist and learned_temporal_basis[:H]
```

This would connect to Contribution 1 because the alignment target is the same prefix-native basis space used by the
forecast operator.

## Narrative Gate

Basis-aware align can re-enter Step 4-6 only if:

1. A6-LBF still has a measurable pure-head/operator contribution in `no_align_no_recon`.
2. A new diagnostic, unlike the returned B4 ablation, shows a material alignment-specific failure mode.
3. A checkpoint-based diagnostic shows `c_hist` and `c_future` have non-trivial, stable alignment in basis space.
4. The method does not use future labels at inference and does not become teacher/EMA distillation.

## Decision

[Decision] Current result is `dependency_ablation_pass_for_head_contribution_but_not_for_b5`.

[Next] Do not implement basis-aware alignment now. Roll StageB to `B6-PLO`: prefix-native label/basis objective
diagnostic.
