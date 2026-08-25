# HoriScope Paper-Facing Naming Migration Audit

## Scope

This audit records the author-approved migration from the former paper-facing dual acronym to `HoriScope`. The change applies to the title, Abstract, Sections 1--7, visible figure text, main and Appendix tables, captions, and Appendices A--C. It does not rename frozen experiment candidates, checkpoints, configuration keys, artifact paths, or historical ledgers.

## Terminology ledger

| Role | Canonical paper-facing term | Usage contract |
| --- | --- | --- |
| Forecasting task | Unified Varied-Horizon Forecasting (UVHF) | Define once, then use `UVHF` |
| Model/framework | HoriScope | Use for the complete paper-facing method |
| Architecture | HoriScope decoder; adaptive multi-scope decoder | Use when the decoder architecture is the subject |
| Training strategy | Balanced Scope Co-Adaptation (BSCA) | Define once; keep separate from the inference architecture |
| Structural property | Cross-Horizon Prefix Consistency (CHPC) | Architectural property of the horizon-agnostic trajectory |
| Frozen provenance | `ISCF-BSCA-v1`; `ISCF-BSCA-MAIN-v1` | Internal implementation and checkpoint identifiers only |

The approved title is:

> HoriScope: Adaptive Multi-Scope Decoding for Unified Varied-Horizon Time-Series Forecasting

## Context-sensitive migration decisions

- Former references to the complete method are rewritten as `HoriScope`.
- Former references to the inference architecture are rewritten as `HoriScope decoder` or `HoriScope`, depending on sentence subject.
- BSCA remains an independently named training strategy and is not absorbed into an acronym expansion for HoriScope.
- Ablation labels use `Full HoriScope`, while `w/o BSCA` continues to identify the optimization control.
- Backbone studies compare `Original Decoder` with `HoriScope`; the prose states that the composite models are trained end to end using BSCA.
- Table assembly maps the frozen experiment row label to `HoriScope` without changing source values or provenance.
- Figures 6, 7, and C1 were regenerated from their canonical scripts so that visible labels use `HoriScope`.

## Verification

The manuscript was regenerated with `python manuscript/build_manuscript.py` and compiled with `latexmk`. The final PDF contains 40 pages and no undefined references or citations. Text extraction from the compiled PDF reports:

| Query | Count |
| --- | ---: |
| `HoriScope` | 79 |
| `ISCF-BSCA` | 0 |
| `Independent Scope-Conditioned Forecasting` | 0 |
| standalone `ISCF` | 0 |

Visual inspection covered the title page, the accuracy--storage figure, the backbone-transfer figure, and Appendix Figure C1. The remaining float-size and bibliography URL warnings predate the naming migration and do not indicate a terminology or reference failure.

## Claim boundary

This migration changes presentation only. It does not alter the architecture, forward computation, loss definitions, experiment results, figure data, table values, evidence roles, or authorization state.
