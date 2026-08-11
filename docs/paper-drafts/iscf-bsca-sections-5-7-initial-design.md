# ISCF-BSCA Sections 5--7: Initial Structural Design

## Design status

| Field | Content |
| --- | --- |
| `document_role` | Provisional structural design for the manuscript sections after Method |
| `version` | `v0.1-initial-structure` |
| `date` | `2026-08-11` |
| `review_status` | `pending_author_discussion` |
| `upstream_dependency` | Introduction v0.9, Related Work v0.2, Section 3 v0.7 and Section 4 v0.7 remain temporarily frozen and unchanged |
| `scope` | Subsection functions, evidence order, table/figure placement, claim boundaries and appendix routing only |
| `manuscript_prose` | Not drafted |
| `experiment_change` | None; this document does not authorize implementation, remote training or formal test |
| `evidence_snapshot` | Main-I and Main-II complete; Efficiency, Core-Ablation and Decoder-Transfer remain pending under the current table registry |
| `structure_decision` | A standalone Discussion is recommended for author review but is not yet frozen |

This document designs the argumentative architecture of the remaining manuscript. It does not fill result values, write result paragraphs or promote pending experiments to completed evidence.

## 1. Manuscript contract

### 1.1 One-sentence argument

In varied-horizon forecasting, ISCF-BSCA constructs one prefix-consistent prediction trajectory with target-adaptive output-side sharing, and its value must be established through system-level comparisons, matched ablations, internal-behavior analysis, efficiency measurements and end-to-end backbone transfer within explicitly stated protocol boundaries.

### 1.2 Primary reader and evidence order

The primary reader is a time-series forecasting reviewer who will ask, in order: whether the unified system is accurate, whether one model genuinely serves different horizons, which components account for its behavior, what the system costs, whether the decoder transfers and where the conclusions stop. Sections 5--7 should answer these questions in the same order.

### 1.3 Terminology ledger for later sections

| Canonical term | Use in later sections | Avoid |
| --- | --- | --- |
| Comparison with horizon-specific forecasters | Main-I system-level comparison between one unified model and separately optimized fixed-$H$ models | matched architecture attribution |
| One-model-all-horizons evaluation | Main-II comparison in which each system uses one H720-trained model and prefix forecasts | generic multi-horizon benchmark without protocol definition |
| Efficiency and system cost | model count, storage, training cost, latency, memory and CHPC capability | efficiency claim before the profiler contract is complete |
| Component and training-objective ablations | matched end-to-end controls for ISCF components and BSCA | interpreting native baselines as ablations |
| Forecast consistency | CHPC/CHPD behavior across shared future targets | forecast accuracy |
| Scope-allocation behavior | Scope Probabilities, scope-wise forecasts and regional behavior | oracle selection or universal specialization |
| Backbone transferability | end-to-end decoder evaluation on preregistered backbone families | frozen-consumer replacement as effectiveness evidence |
| Test-informed evaluation | disclosed project protocol for paper-facing model/profile comparison | untouched-holdout or strictly confirmatory evaluation |

The current ablation registry contains labels inherited from earlier experiment design that do not fully match the frozen Method terminology. Before Table 4 is drafted, the control identities should remain unchanged while their manuscript-facing names are reconciled:

| Current registry label | Manuscript-consistent candidate | Status |
| --- | --- | --- |
| w/o Independent Fields | w/o Scope-Specific Projections | wording proposal only; verify exact control implementation |
| w/o Target-Wise Fusion | w/o Target-Adaptive Allocation | wording proposal only; preserve the frozen fusion control |
| w/o Multiple Coupling Scopes | Single-Scope Forecasting | wording proposal only; verify whether the control removes both scope multiplicity and allocation |

## 2. Recommended manuscript structure

```text
5. Experiments
   5.1 Experimental setup
   5.2 Comparison with horizon-specific forecasters
   5.3 One-model-all-horizons evaluation
   5.4 Efficiency and system cost
   5.5 Component and training-objective ablations
   5.6 Forecast consistency and scope-allocation behavior
   5.7 Backbone transferability
   5.8 Qualitative analysis and failure cases

6. Discussion
   6.1 From horizon-specific predictions to a unified forecasting system
   6.2 Output-side sharing as a forecasting design dimension
   6.3 Limitations and future scope

7. Conclusion

Appendices
   A. Detailed experimental settings and source-role disclosures
   B. Full dataset-horizon results and additional comparisons
   C. Additional ablations, sensitivity and mechanism diagnostics
   D. Reproducibility, selection protocol and artifact provenance
```

The principal structural change relative to the earlier six-section outline is the proposed standalone Discussion. This change is recommended but remains subject to author approval.

## 3. Section 5: Experiments

Section 5 should follow an evidence ladder rather than the implementation order: `evaluation contract -> system effectiveness -> one-model capability -> cost -> matched attribution -> internal behavior -> transfer -> failure-aware examples`.

### 3.1 Subsection contracts

| Subsection | Scientific question | Required content blocks | Primary artifact | Permitted conclusion | Current evidence status |
| --- | --- | --- | --- | --- | --- |
| 5.1 Experimental setup | Are the comparisons reproducible and are their roles distinguishable? | datasets and splits; horizons and metrics; baseline families and source roles; main model versus ablation anchor; checkpoint/profile selection; seeds; test-informed disclosure; implementation and hardware | experiment protocol + table registry | Defines the evaluation contract only | Partially frozen; final wording waits for all experiment blocks |
| 5.2 Comparison with horizon-specific forecasters | Can one unified ISCF-BSCA model compete with separately optimized fixed-horizon systems? | explain one-versus-four-model protocol; introduce Main-I; report aggregate pattern, dataset/horizon exceptions and source-role caveat | Table 1 / `Main-I` | System-level accuracy competitiveness under the audited mixed-source comparison | Complete and hash-frozen |
| 5.3 One-model-all-horizons evaluation | Is ISCF-BSCA competitive when every system must serve all horizons from one trained model? | define H720-prefix protocol; explain how it differs from Main-I; introduce Main-II; report dominant pattern and negative cells; retain unmatched-protocol caveat | Table 2 / `Main-II` | One-model-all-horizons system effectiveness, not decoder or BSCA attribution | Complete; H5A remains a separate active selection block and cannot be presumed successful |
| 5.4 Efficiency and system cost | What practical cost changes when one model replaces a horizon-specific family? | trained-model count; stored parameters; training GPU-hours; single-request and all-horizon latency; peak memory; CHPC capability; profiler protocol | Table 3 / `Efficiency` | Cost and deployment trade-offs only after matched measurement | Measurement and baseline subset pending |
| 5.5 Component and training-objective ablations | Which architectural and optimization components contribute within the frozen design family? | Full; w/o BSCA; scope-projection control; allocation control; single-scope control; matched budgets and end-to-end training | Table 4 / `Core-Ablation` | Component utility within the tested design; stronger mechanism claims require internal diagnostics | Partially reusable; prelaunch and missing cells pending |
| 5.6 Forecast consistency and scope-allocation behavior | Does the trained system behave in the manner motivated by Section 3? | CHPC/CHPD verification; Scope Probability maps; scope-wise and fused errors; prediction diversity; regional behavior; allocation/gradient balance; oracle headroom as diagnostic only | Figure 5 / mechanism-analysis bundle | Behavior consistent with prefix consistency and heterogeneous sharing; no universal specialization or oracle-allocation claim | Figure/statistic contract not yet frozen |
| 5.7 Backbone transferability | Does the decoder remain useful beyond its current Encoder realization? | DLinear-style and PatchTST-style backbones; Original Decoder, +ISCF and +ISCF-BSCA; backbone-specific matched profiles; end-to-end training | Table 5 / `Decoder-Transfer` | Decoder portability only if both matched transfer blocks support it | Source patch, retraining and formal evaluation pending |
| 5.8 Qualitative analysis and failure cases | What does the system produce on representative and difficult trajectories? | one full trajectory with nested prefixes; Scope Probability evolution; scope-conditioned forecasts; at least one failure or negative case; selection rule in caption | Figure 6 or Appendix figure | Illustration of observed behavior, not prevalence or mechanism proof | Not frozen; default routing to Appendix unless it carries unique explanatory value |

### 3.2 Why Main-I and Main-II must remain separate

Main-I and Main-II answer different reviewer questions and should not be merged into one overloaded table. Main-I evaluates the practical replacement of a horizon-specific model family by one unified system. Main-II asks whether different forecasting systems remain competitive when each must use one trained model to serve all horizons. Their baseline contracts and permissible claims therefore differ, even when both report MSE and MAE over the same displayed horizons.

### 3.3 Recommended paragraph logic inside result subsections

Each result subsection should use the same internal order: `question -> comparison protocol -> table/figure scope -> aggregate pattern -> important exceptions -> bounded conclusion`. Numeric values and positive verbs should be inserted only after the corresponding artifact is frozen.

## 4. Section 6: Discussion

### 4.1 Why a standalone Discussion is recommended

The paper makes both a system contribution and a forecasting-design argument. A standalone Discussion can interpret CHPC, output-side sharing and varied-horizon system design without mixing these claims into result reporting. It also provides a visible place for source-protocol differences, test-informed evaluation and negative cases that would otherwise overload the Conclusion.

### 4.2 Subsection contracts

| Subsection | Main job | Evidence it may interpret | Boundary |
| --- | --- | --- | --- |
| 6.1 From horizon-specific predictions to a unified forecasting system | Explain what changes when forecasts at different endpoints are treated as nested views of one trajectory | CHPC construction, Main-I, Main-II and system-cost evidence | Do not infer accuracy from CHPC or claim all prior flexible-horizon models violate it |
| 6.2 Output-side sharing as a forecasting design dimension | Connect future-region sharing-demand heterogeneity to scope-indexed forecast generation and allocation | Figure 3 motivation, Core-Ablation, mechanism diagnostics and transfer results | Do not promote active allocation, diversity or oracle headroom to causal specialization |
| 6.3 Limitations and future scope | State where the evaluated claims stop and which extensions are scientifically meaningful | negative cells, unmatched baseline roles, seed coverage, profiler results and pending/negative controls | Limitations must be artifact-specific rather than generic future-work language |

### 4.3 Limitation inventory to audit before prose drafting

- deterministic point forecasting rather than probabilistic forecasting;
- the frozen benchmark horizon set and current dataset domains;
- external baseline protocol heterogeneity in Main-I and Main-II;
- explicit test-informed/test-tuned profile selection;
- incomplete cross-seed evidence unless a full non-selective extension is completed;
- the unverified necessity of canonical contiguous grouping;
- current full-domain materialization versus architecture-supported prefix-bounded execution;
- dataset- or horizon-specific negative cells that constrain uniform-superiority wording;
- the distinction between observed scope behavior and learned semantic specialization.

Only limitations still supported by the final artifacts should enter the manuscript. This list is an audit inventory, not prewritten Discussion content.

## 5. Section 7: Conclusion

The Conclusion should remain compact and should not use subsections. A two-paragraph structure is recommended if Section 6 is retained:

1. restate the varied-horizon forecasting problem, CHPC requirement and ISCF-BSCA response at principle level;
2. summarize only evidence-backed effectiveness, attribution, efficiency and transfer outcomes, then close with the narrowest material boundary.

The Conclusion must not introduce new metrics, citations, mechanisms or future claims. If a standalone Discussion is rejected, the Conclusion should expand to three paragraphs by adding one explicit limitations paragraph.

## 6. Main-text table and figure plan

| Provisional number | Artifact | Main narrative role | Status |
| --- | --- | --- | --- |
| Table 1 | Main-I | one unified model versus separately optimized horizon-specific systems | complete/hash-frozen |
| Table 2 | Main-II | one-model-all-horizons system comparison | complete/presentation-aligned; active H5A cannot be anticipated |
| Table 3 | Efficiency | deployment and computation trade-offs | pending |
| Table 4 | Core-Ablation | component and objective attribution | pending |
| Figure 5 | Forecast consistency and scope-allocation behavior | connect Section 3 problems to trained-system behavior | pending design/evidence freeze |
| Table 5 | Decoder-Transfer | end-to-end portability across backbone families | pending |
| Figure 6 | Qualitative behavior and failure cases | reader-facing explanation of nested trajectories and allocation | optional main-text item; Appendix by default |

The main text is already table-heavy. Figure 6 should remain supplementary unless it provides a failure mode or mechanism distinction that cannot be communicated by Figure 5.

## 7. Appendix routing

| Appendix | Content role | Examples |
| --- | --- | --- |
| A. Detailed experimental settings and source-role disclosures | reproduce the evaluated protocols and make native/published/matched roles explicit | dataset statistics, splits, look-back windows, baseline sources, hardware, profiler contract |
| B. Full dataset-horizon results and additional comparisons | preserve complete cells without overloading the main argument | full-precision Main-I/Main-II results, Exchange companion, per-dataset breakdowns, negative cells |
| C. Additional ablations, sensitivity and mechanism diagnostics | retain secondary controls and robustness evidence | random partition, scope count, loss weights, extended allocation/gradient plots, extra case studies |
| D. Reproducibility, selection protocol and artifact provenance | disclose the test-informed workflow and artifact identity | full HPO trial records, checkpoint hashes, selector definitions, seed/environment details |

## 8. Claim-evidence map

| Planned claim | Evidence route | Current status | Drafting rule |
| --- | --- | --- | --- |
| One unified model is competitive with horizon-specific systems | Main-I | supported at system level | retain source/protocol caveat; no mechanism attribution |
| One model can serve all evaluated horizons competitively | Main-II | supported at system level | report negative cells and unmatched external contracts |
| ISCF-BSCA reduces practical multi-horizon system cost | Efficiency | needs evidence | no positive efficiency wording before measurement |
| ISCF components and BSCA are effective | Core-Ablation | needs complete matched evidence | Introduction P6 remains provisional until closure |
| Allocation adapts useful sharing information across future steps | Core-Ablation + Figure 5 diagnostics | needs matched attribution and frozen diagnostics | active probabilities alone are insufficient |
| The decoder transfers across forecasting backbones | Decoder-Transfer | needs evidence | no portability verb before both end-to-end blocks close |
| CHPC holds for shared targets | architecture + implementation verification | construction fact under the stated graph | keep separate from forecasting accuracy |

## 9. Recommended writing order

1. freeze the final Section 5.1 evaluation contract and evidence-role vocabulary;
2. draft 5.2 and 5.3 from the already complete Main-I/Main-II artifacts without anticipating H5A;
3. wait for Efficiency, Core-Ablation and Decoder-Transfer before drafting 5.4, 5.5 and 5.7;
4. freeze a compact Figure 5 evidence contract before writing 5.6;
5. decide whether Figure 6 remains in the main text before drafting 5.8;
6. write Discussion only after all positive and negative evidence is known;
7. write Conclusion and then revisit the provisional result sentence in Introduction P6.

## 10. Author decisions requested before prose drafting

1. Whether to adopt the recommended standalone Section 6 Discussion or retain the earlier six-section paper with limitations inside Conclusion.
2. Whether qualitative case studies deserve a main-text subsection/figure or should default to Appendix C.
3. Whether Section 5.6 should remain one integrated consistency-and-allocation analysis or be split into two subsections after its evidence contract is frozen.

`Decision=sections_5_7_v0_1_structural_design_pending_author_discussion`.
