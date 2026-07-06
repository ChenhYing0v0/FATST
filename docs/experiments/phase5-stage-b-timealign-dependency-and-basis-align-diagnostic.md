# Phase5 StageB: TimeAlign Dependency And Basis-Aware Align Diagnostic

`current_step`: StageB Step 2/3 dependency diagnostic before any encoder/align innovation.

本文档处理当前论文叙事风险：A6-LBF 已改造 decoder/head，但仍继承 TimeAlign encoder 与
future alignment。若不证明 A6-LBF 的独立贡献边界，论文会被视为 TimeAlign variant。

## 11-step Record

| Field | Content |
| --- | --- |
| `current_step` | StageB Step 2/3：TimeAlign dependency audit |
| `problem` | A6-LBF 的收益是否主要来自 inherited TimeAlign align/recon，而不是 learned-basis operator |
| `existence_evidence` | existing artifact audit: same-align A6 vs official unified; loss component shares |
| `idea` | 先做 attribution diagnostic，再决定是否设计 basis-aware future alignment |
| `theory_check` | same-align improvement 只能证明 head/operator contribution；要成为新 architecture，还需 no-align/no-recon 和 basis-space alignability 证据 |
| `design` | artifact-only audit completed；remote ablation runner prepared but not launched |
| `narrative_gate` | `partial_dependency_risk_confirmed` |
| `effectiveness_gate` | not applicable；未训练新方法 |
| `artifacts` | `analysis/phase5_stage_b_timealign_dependency_audit_20260706/` |
| `decision` | 当前不应宣称 full new architecture；需要补 TimeAlign dependency ablation 和 basis-space diagnostic |

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

## Required Remote Ablation

Before any PhaseB align mechanism is designed as a paper-core method, run:

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

The runner is prepared only. Per project rule, do not launch it before commit/push and GPU preflight.

## Basis-Aware Align Hypothesis

If dependency ablation shows that inherited TimeAlign alignment is important, the next PhaseB innovation should not
be a generic `glocal_align` variant. It should be A6-LBF-specific:

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

Basis-aware align can enter Step 4-6 only if:

1. A6-LBF still has a measurable pure-head/operator contribution in `no_align_no_recon`.
2. Removing `w_align` materially changes performance or stability, proving alignment is a real dependency.
3. A checkpoint-based diagnostic shows `c_hist` and `c_future` have non-trivial, stable alignment in basis space.
4. The method does not use future labels at inference and does not become teacher/EMA distillation.

## Decision

[Decision] Current result is `partial_dependency_risk_confirmed`.

[Next] Run the remote dependency ablation matrix before implementing basis-aware alignment.
