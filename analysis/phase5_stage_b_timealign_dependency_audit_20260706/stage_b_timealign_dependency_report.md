# Phase5 StageB TimeAlign Dependency Audit

`current_step`: StageB Step 2/3 dependency diagnostic before any align/encoder modification.

## Scope

[Fact] This audit uses existing official-last artifacts only. It does not retrain any no-align/no-recon ablation.

[Boundary] The audit can show whether A6-LBF improves under the same inherited TimeAlign alignment/reconstruction setting. It cannot causally isolate the alignment mechanism without new ablation runs.

## 11-step Record

| Field | Content |
| --- | --- |
| `current_step` | StageB Step 2/3 TimeAlign dependency diagnostic |
| `problem` | Is A6-LBF's apparent advantage merely inherited from TimeAlign's future alignment mechanism? |
| `existence_evidence` | A6 vs official unified metrics under the same align/recon setting; training loss component shares |
| `idea` | Separate same-backbone readout evidence from missing causal alignment ablations |
| `theory_check` | Same-align improvement supports a head/operator contribution, but does not make the whole architecture independent from TimeAlign |
| `design` | Artifact-only audit; no code or remote training change |
| `narrative_gate` | partial_dependency_risk_confirmed |
| `effectiveness_gate` | not applicable; no new method trained |
| `artifacts` | this directory |
| `decision` | A6-LBF has same-align readout evidence, but paper still needs no-align/no-recon and basis-aware align diagnostics |

## Same-Alignment Metric Comparison

| Dataset | Horizon | A6 MSE | Official Unified MSE | A6 vs Official |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | `0.2421` | `0.2491` | `-2.83%` |
| ETTh2 | 192 | `0.2828` | `0.2960` | `-4.45%` |
| ETTh2 | 336 | `0.3128` | `0.3267` | `-4.25%` |
| ETTh2 | 720 | `0.3942` | `0.4033` | `-2.25%` |
| ETTm1 | 96 | `0.2731` | `0.2812` | `-2.88%` |
| ETTm1 | 192 | `0.3094` | `0.3134` | `-1.28%` |
| ETTm1 | 336 | `0.3472` | `0.3500` | `-0.80%` |
| ETTm1 | 720 | `0.4079` | `0.4067` | `0.29%` |
| Weather | 96 | `0.1414` | `0.1432` | `-1.27%` |
| Weather | 192 | `0.1824` | `0.1849` | `-1.31%` |
| Weather | 336 | `0.2318` | `0.2345` | `-1.18%` |
| Weather | 720 | `0.3034` | `0.3068` | `-1.10%` |

[Evidence] A6-LBF wins `11/12` settings against official unified TimeAlign under the same inherited align/recon setting; mean MSE change is `-1.94%`.

## Last-Epoch Loss Component Shares

| Variant | Dataset | Pred Share | Recon Share | Align Share | Val MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| `a6_lbf_r256` | ETTh2 | `0.78` | `0.03` | `0.19` | `0.4587` |
| `a6_lbf_r256` | ETTm1 | `0.75` | `0.16` | `0.08` | `0.6019` |
| `a6_lbf_r256` | Weather | `0.87` | `0.00` | `0.12` | `0.4904` |
| `official_unified` | ETTh2 | `0.80` | `0.04` | `0.16` | `0.4913` |
| `official_unified` | ETTm1 | `0.78` | `0.15` | `0.07` | `0.6160` |
| `official_unified` | Weather | `0.89` | `0.00` | `0.11` | `0.4907` |

## Decision

[Decision] `partial_dependency_risk_confirmed`.

[Interpretation] A6-LBF has real same-alignment evidence because it improves over official unified while keeping TimeAlign's align/recon mechanism. However, this is not enough to claim a fully independent architecture. The training objective still contains inherited `w_recon * recon_loss + w_align * align_loss`, and no no-align/no-recon ablation exists in the current artifact set.

[Next Required Diagnostic] Run a minimal TimeAlign dependency ablation matrix before any PhaseB align innovation claim:

- A6-LBF with current `w_recon=1.0,w_align=0.1`;
- A6-LBF with `w_align=0.0,w_recon=1.0`;
- A6-LBF with `w_align=0.1,w_recon=0.0`;
- A6-LBF with `w_align=0.0,w_recon=0.0`;
- official unified TimeAlign under the same protocol as control.

[Basis-Align Precondition] A future PhaseB align mechanism should only proceed if a separate checkpoint-based diagnostic shows that history-derived and future-derived coefficients are alignable in A6-LBF basis space.

## Output Files

- `stage_b_timealign_dependency_metric_comparison.csv`
- `stage_b_timealign_dependency_training_components.csv`
- `stage_b_timealign_dependency_report.md`
