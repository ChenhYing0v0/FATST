# ISCF-BSCA Sections 5--7: Frozen Structural Design

## Design status

| Field | Content |
| --- | --- |
| `document_role` | Temporarily frozen structural design for the manuscript sections after Method |
| `version` | `v0.14-section5-6-narrative-first-reassessment` |
| `date` | `2026-08-21 Section 5.6 narrative-first reassessment` |
| `review_status` | `temporarily_frozen_usable` |
| `upstream_dependency` | Introduction v0.9, Related Work v0.2, Section 3 v0.7 and Section 4 v0.7 remain temporarily frozen and unchanged |
| `scope` | Subsection functions, evidence order, table/figure placement, claim boundaries and appendix routing only |
| `manuscript_prose` | Section 5 v0.8 drafted at `docs/paper-drafts/iscf-bsca-experiments-initial-draft.md`; opening、5.1--5.5与5.7 are author-refined；Section 5.6正文仍留空；narrative-first proposal complete，Panel a framing revision pending；Sections 6--7 remain structural only |
| `experiment_change` | None; this document does not authorize implementation, remote training or formal test |
| `evidence_snapshot` | Main-I, corrected Main-II, Efficiency, Core-Ablation, rejected historical Figure 5 diagnostics, Figure 5 v3 author-review draft and Decoder-Transfer complete |
| `structure_decision` | Sections 5--7 use `Experiments -> Discussion -> Conclusion`; Section 5.6 should focus on jointly trained regional scope competence、adaptive integration and BSCA；CHPC remains a construction fact without a main-text visualization |

This document remains the frozen argumentative architecture of the remaining manuscript. Section 5 has now been instantiated as an initial evidence-complete draft, while Sections 6--7 remain structural designs; neither document promotes evidence beyond the boundaries recorded below.

## 1. Manuscript contract

### 1.1 One-sentence argument

In varied-horizon forecasting, ISCF-BSCA constructs one prefix-consistent prediction trajectory with target-adaptive output-side sharing, and its value must be established through system-level comparisons, matched ablations, internal-behavior analysis, efficiency measurements and end-to-end backbone transfer within explicitly stated protocol boundaries.

### 1.2 Primary reader and evidence order

The primary reader is a time-series forecasting reviewer who will ask, in order: whether the unified system is accurate, whether one model genuinely serves different horizons, which components account for its behavior, what the system costs, whether the decoder transfers and where the conclusions stop. Sections 5--7 should answer these questions in the same order.

### 1.3 Terminology ledger for later sections

| Canonical term | Use in later sections | Avoid |
| --- | --- | --- |
| Comparison with horizon-specific forecasters | Main-I system-level comparison between one unified model and separately optimized fixed-$H$ models | matched architecture attribution |
| One-model-all-horizons evaluation | Main-II comparison in which each method uses one unified model trained for the maximum horizon and returns prefix forecasts | generic multi-horizon benchmark without protocol definition |
| Efficiency and system cost | Main-I accuracy, peak inference memory and four-horizon checkpoint storage | collapsing one-checkpoint consolidation into a uniform resource-efficiency claim |
| Component and training-objective ablations | Full, w/o BSCA, w/o Target-Adaptive Allocation, Shared Scope Projection and Fixed Scope ($s=144$) under matched end-to-end training | balance-only controls or interpreting native baselines as ablations |
| Forecast consistency | CHPC/CHPD behavior across shared future targets | forecast accuracy |
| Scope-allocation behavior | Scope Probabilities, aggregate scope utilization and regional preference behavior | realized allocation value, oracle selection or universal specialization |
| Generalization studies | end-to-end decoder evaluation on the reported backbone families | universal architecture-agnostic effectiveness or frozen-consumer replacement evidence |
| Test-informed evaluation | disclosed project protocol for paper-facing model/profile comparison | untouched-holdout or strictly confirmatory evaluation |

The author-fixed Core-Ablation contract contains exactly five variants:

| Variant | Frozen intervention | Attribution role |
| --- | --- | --- |
| Full ISCF-BSCA | complete frozen architecture and objective | reference |
| w/o BSCA | retain the architecture and Uniform-Prefix Forecasting Loss; remove both Scope-Wise Forecasting Loss and Allocation-Balance Regularizer | joint contribution of the BSCA optimization terms |
| w/o Target-Adaptive Allocation | replace learned target-wise Scope Probabilities with a non-adaptive matched fusion rule; exact rule must be frozen before launch | utility of target-adaptive scope allocation |
| Shared Scope Projection | use one shared projection in place of scope-specific projections while preserving the remaining graph | utility of scope-specific history information pools |
| Fixed Scope ($s=144$) | use the preregistered middle scope only, without validation search over scope sizes | utility of multi-scope generation and allocation relative to a budget-aware fixed control |

No standalone ablation is allocated to the Allocation-Balance Regularizer. The Fixed Scope value $s=144$ is a pragmatic preregistered middle value, not a validation-selected best scope and not evidence that $s=144$ is optimal.

## 2. Recommended manuscript structure

```text
5. Experiments
   5.1 Experimental setup
   5.2 Comparison with horizon-specific forecasters
   5.3 One-model-all-horizons evaluation
   5.4 Efficiency and system cost
   5.5 Component and training-objective ablations
   5.6 Regional scope behavior and adaptive integration
   5.7 Generalization studies
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

The standalone Discussion remains frozen as the working manuscript structure. Section 5.6 retains its position in the outline but is reframed around method-internal scope behavior. Panel a remains a main-text candidate when explicitly presented as an aggregate diagnostic of the jointly trained ISCF scope field; Panels b/c retain the Target-Adaptive Allocation and BSCA gains. CHPC is stated as an architectural property and is not assigned a result panel.

## 3. Section 5: Experiments

Section 5 should follow an evidence ladder rather than the implementation order: `evaluation contract -> system effectiveness -> one-model capability -> cost -> matched attribution -> consistency and allocation behavior -> transfer`.

### 3.1 Subsection contracts

| Subsection | Scientific question | Required content blocks | Primary artifact | Permitted conclusion | Current evidence status |
| --- | --- | --- | --- | --- | --- |
| 5.1 Experimental setup | Are the comparisons reproducible and are their roles distinguishable? | datasets and splits; horizons and metrics; baseline families and source roles; main model versus ablation anchor; checkpoint/profile selection; seeds; test-informed disclosure; implementation and hardware | experiment protocol + table registry | Defines the evaluation contract only | Evidence inputs complete; ready for final prose consolidation |
| 5.2 Comparison with horizon-specific forecasters | Can one unified ISCF-BSCA model compete with separately optimized fixed-horizon baselines? | explain one-versus-four-model protocol; report seven dataset-level four-horizon means, overall best/second counts and low-/high-dimensional dataset-group gains; route complete per-H results to Appendix A | Table 1 / `Main-I` | System-level accuracy competitiveness under the audited mixed-source comparison | Complete and hash-frozen |
| 5.3 One-model-all-horizons evaluation | Is ISCF-BSCA competitive when every baseline must serve all horizons from one trained model? | define H720-prefix protocol; explain how it differs from Main-I; report dataset-level four-horizon means; route complete per-H results to Appendix A; retain unmatched-protocol caveat | Table 2 / `Main-II` | One-model-all-horizons system effectiveness, not decoder or BSCA attribution | Complete and horizon-loader re-audited |
| 5.4 Efficiency and system cost | What accuracy and deployment cost changes when one model replaces a four-model horizon-specific family? | Main-I macro MSE; peak inference memory; four-horizon checkpoint storage | Figure 6 / `Efficiency` | Lowest macro MSE among eight displayed methods and one-checkpoint consolidation, with DLinear/QDF retained as visible resource counterexamples | Complete; Figure 6 is the sole main-text presentation and the table artifact remains a numerical source |
| 5.5 Component and training-objective ablations | Which architectural and optimization components contribute within the frozen design family? | Full; w/o BSCA; w/o Target-Adaptive Allocation; Shared Scope Projection; Fixed Scope ($s=144$); matched budgets and end-to-end training | Table 3 / `Core-Ablation` | Author-corrected aggregate table supports all four interventions; per-horizon rerun provenance remains unsynchronized | Complete at dataset-aggregate table level; Full best in 12/12 metric columns |
| 5.6 Regional scope behavior and adaptive integration | Does the jointly trained ISCF scope field retain region-dependent competence, and do adaptive integration and BSCA improve aggregate accuracy? | Panel a: lowest-error scope and error separation across all 5 × 8 × 5 validation aggregates；Panel b: Full versus equal fusion；Panel c: Full versus prefix-only training；state near-uniform/8-of-40 boundary | Figure 5 v3 method-internal reframe pending; not yet canonical | Region-dependent competence within the jointly trained scope field plus aggregate allocation/BSCA utility; no hard selection、oracle recovery或sparse specialization claim | Paragraph contract complete；Panels b/c pass；Panel a title/caption revision and visible prose pending author approval |
| 5.7 Generalization studies | Does the complete framework remain effective beyond its current Encoder realization? | Weather、ETTm1、ETTm2；DLinear-style与PatchTST-style；Original Decoder versus complete ISCF-BSCA；end-to-end training | Figure 7 / `Decoder-Transfer` | 两类backbones在Figure 7所示三数据集four-horizon mean MSE上均取得3/3 wins；conclusion restricted to evaluated-scope compatibility | Figure 7 complete；aggregate LaTeX table retained as numerical/audit source but not inserted into the main text；per-H/hash rerun provenance unsynchronized；no additional HPO required |

### 3.2 Why Main-I and Main-II must remain separate

Main-I and Main-II answer different reviewer questions and should not be merged into one overloaded table. Main-I evaluates the practical replacement of a horizon-specific model family by one unified system. Main-II asks whether different forecasting systems remain competitive when each must use one trained model to serve all horizons. Their baseline contracts and permissible claims therefore differ, even when both report MSE and MAE over the same displayed horizons.

### 3.3 Recommended paragraph logic inside result subsections

Each result subsection should use the same internal order: `question -> comparison protocol -> table/figure scope -> aggregate pattern -> important exceptions -> bounded conclusion`. Numeric values and positive verbs should be inserted only after the corresponding artifact is frozen.

## 4. Section 6: Discussion

### 4.1 Why a standalone Discussion is recommended

The paper makes both a system contribution and a forecasting-design argument. The standalone Discussion interprets CHPC, output-side sharing and varied-horizon system design without mixing these claims into result reporting. It also provides a visible place for source-protocol differences, test-informed evaluation and negative cells that would otherwise overload the Conclusion.

### 4.2 Subsection contracts

| Subsection | Main job | Evidence it may interpret | Boundary |
| --- | --- | --- | --- |
| 6.1 From horizon-specific predictions to a unified forecasting system | Explain what changes when forecasts at different endpoints are treated as nested views of one trajectory | CHPC construction, Main-I, Main-II and system-cost evidence | Do not infer accuracy from CHPC or claim all prior flexible-horizon models violate it |
| 6.2 Output-side sharing as a forecasting design dimension | Connect future-region sharing-demand heterogeneity to scope-indexed forecast generation and allocation | Figure 3 motivation, Core-Ablation, allocation diagnostics and transfer results | Do not promote active probabilities or regional variation to causal specialization |
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
| Table 1 | Main-I | main text: dataset-level four-horizon means；Appendix A: complete one unified model versus separately optimized horizon-specific baselines | complete/hash-frozen |
| Table 2 | Main-II | main text: dataset-level four-horizon means；Appendix A: complete one-model-all-horizons comparison | complete/presentation-aligned; active H5A cannot be anticipated |
| Table 3 | Core-Ablation | component and objective attribution | author-corrected aggregate table complete; 12/12 metric columns best for Full; historical 100-cell audit retained |
| Figure 5 | Forecast consistency, allocation behavior and an illustrative improved trajectory | evidence reserved for the redesign of Section 5.6 | complete but temporarily deferred from the visible manuscript body; exact CHPC, near-uniform utilization, regional scope-error differences and disclosed selected trajectory |
| Figure 6 | Efficiency | main-text accuracy, peak-memory and checkpoint-storage trade-off | complete; sole main-text evidence carrier for Section 5.4 |
| Figure 7 | Generalization studies | sole main-text comparison of Original Decoder and ISCF-BSCA on two reported backbone families | complete; MSE axis begins at 0.20；the Decoder-Transfer table remains a numerical/audit source and is not inserted into the main text；claim remains bounded to the displayed backbones |

The qualitative example remains integrated into the completed Figure 5 asset rather than assigned a separate figure or subsection, but Figure 5 is temporarily deferred from the visible manuscript body together with Section 5.6. If reinstated, the comparator, split and selection rule must remain disclosed, and the performance-selected example must be described as illustrative rather than representative. A dedicated failure-case panel is not required; negative aggregate cells remain reported in Sections 5.2--5.3 and interpreted in Section 6.3.

## 7. Appendix routing

| Appendix | Content role | Examples |
| --- | --- | --- |
| A. Detailed experimental settings and source-role disclosures | reproduce the evaluated protocols and make native/published/matched roles explicit | dataset statistics, splits, look-back windows, baseline sources, hardware, profiler contract |
| B. Full dataset-horizon results and additional comparisons | preserve complete cells without overloading the main argument | full-precision Main-I/Main-II results, Exchange companion, per-dataset breakdowns, negative cells |
| C. Additional ablations, sensitivity and mechanism diagnostics | retain secondary controls and robustness evidence | random partition, scope count, loss weights, extended allocation/gradient plots, optional extra case studies |
| D. Reproducibility, selection protocol and artifact provenance | disclose the test-informed workflow and artifact identity | full HPO trial records, checkpoint hashes, selector definitions, seed/environment details |

## 8. Claim-evidence map

| Planned claim | Evidence route | Current status | Drafting rule |
| --- | --- | --- | --- |
| One unified model is competitive with horizon-specific systems | Main-I | supported at system level | retain source/protocol caveat; no mechanism attribution |
| One model can serve all evaluated horizons competitively | Main-II | supported at system level | report negative cells and unmatched external contracts |
| ISCF-BSCA consolidates a four-horizon service while improving Main-I macro accuracy | Efficiency | supported as a resource trade-off | report memory/storage advantages over TimeAlign, AMD, iTransformer, PatchTST and TimeMixer together with DLinear/SimpleTM/QDF counterexamples; disclose architecture-equivalent resource rows and do not claim uniform efficiency |
| BSCA objective, Target-Adaptive Allocation, scope-specific projections and multi-scope design improve aggregate accuracy in the matched setting | Author-corrected Core-Ablation | supported at dataset-aggregate level | report all five variants and disclose that new per-horizon/checkpoint rerun provenance is not synchronized |
| Learned Target-Adaptive Allocation improves aggregate accuracy over equal fusion | Author-corrected Core-Ablation + Figure 5 diagnostics | supported at dataset-aggregate accuracy level; internal routing interpretation remains mixed | Full macro MSE/MAE `.305/.344` versus equal fusion `.310/.349`; probabilities remain near-uniform and highest utilization matches lowest-error scope in only 8/40 dataset-region cells, so do not claim reliable region-best routing or causal specialization |
| The complete ISCF-BSCA framework transfers across the evaluated forecasting backbones | Figure 7 backed by the author-corrected Decoder-Transfer source | supported for four-horizon mean MSE on the author-refined three-dataset scope | DLinear-style与PatchTST-style均3/3 dataset MSE wins；正文只陈述Figure 7可见MSE并保持evaluated-scope portability边界；unsynchronized per-H provenance、test-tuned history与five-dataset negative evidence继续保留在audit/limitations；不写universal或architecture-agnostic |
| CHPC holds for shared targets | architecture + implementation verification | supported numerically in 20/20 dataset-horizon cells with maximum absolute CHPD=0 | keep separate from forecasting accuracy |

## 9. Recommended writing order

1. freeze the final Section 5.1 evaluation contract and evidence-role vocabulary;
2. draft 5.2 and 5.3 from the already complete Main-I/Main-II artifacts without anticipating H5A;
3. draft 5.4, 5.5 and 5.7 from the complete Efficiency, Core-Ablation and Decoder-Transfer artifacts, retaining their negative boundaries;
4. redesign 5.6 from the complete mixed Figure 5 evidence, retaining the near-uniform allocation and 8/40 agreement boundary before restoring any visible prose;
5. write Discussion now that all positive and negative evidence is known;
6. write Conclusion and then revisit the provisional result sentence in Introduction P6.

## 10. Temporarily frozen author decisions

1. Retain a standalone Section 6 Discussion.
2. Retain the Section 5.6 heading but keep its visible body blank until the author approves a redesigned consistency-and-allocation analysis; preserve the complete mixed Figure 5 evidence in editorial records.
3. Omit a standalone failure-case subsection and figure.
4. Restrict Core-Ablation to the five author-fixed variants above; do not add a balance-only control.
5. Fix the single-scope control to $s=144$ without a best-scope search.
6. Exclude realized allocation value from the current mechanism-analysis contract.

These decisions freeze the writing and experiment-design reference only. They do not authorize implementation, remote training or formal test.

`Decision=sections_5_7_v0_9_section5_v0_7_evidence_integrated_5_6_pending_sections6_7_pending`.
