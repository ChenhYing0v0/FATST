# ISCF-BSCA Sections 5--7: Frozen Structural Design

## Design status

| Field | Content |
| --- | --- |
| `document_role` | Temporarily frozen structural design for the manuscript sections after Method |
| `version` | `v0.7-section5-v0.5-integrated` |
| `date` | `2026-08-19 Section 5 Main-I author refinement` |
| `review_status` | `temporarily_frozen_usable` |
| `upstream_dependency` | Introduction v0.9, Related Work v0.2, Section 3 v0.7 and Section 4 v0.7 remain temporarily frozen and unchanged |
| `scope` | Subsection functions, evidence order, table/figure placement, claim boundaries and appendix routing only |
| `manuscript_prose` | Section 5 v0.5 drafted at `docs/paper-drafts/iscf-bsca-experiments-initial-draft.md`; opening、5.1与5.2 are author-refined, while Sections 5.3--5.7 remain under review；Sections 6--7 remain structural only |
| `experiment_change` | None; this document does not authorize implementation, remote training or formal test |
| `evidence_snapshot` | Main-I, corrected Main-II, Efficiency, Core-Ablation, Figure 5 mechanism diagnostics and Decoder-Transfer complete |
| `structure_decision` | Sections 5--7 use `Experiments -> Discussion -> Conclusion`; qualitative evidence is integrated into Section 5.6 rather than assigned a standalone subsection |

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
| Backbone transferability | end-to-end decoder evaluation on preregistered backbone families | frozen-consumer replacement as effectiveness evidence |
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
   5.6 Forecast consistency and scope-allocation behavior
   5.7 Backbone transferability
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

The standalone Discussion and the integrated Section 5.6 qualitative analysis are temporarily frozen as the working manuscript structure.

## 3. Section 5: Experiments

Section 5 should follow an evidence ladder rather than the implementation order: `evaluation contract -> system effectiveness -> one-model capability -> cost -> matched attribution -> consistency and allocation behavior -> transfer`.

### 3.1 Subsection contracts

| Subsection | Scientific question | Required content blocks | Primary artifact | Permitted conclusion | Current evidence status |
| --- | --- | --- | --- | --- | --- |
| 5.1 Experimental setup | Are the comparisons reproducible and are their roles distinguishable? | datasets and splits; horizons and metrics; baseline families and source roles; main model versus ablation anchor; checkpoint/profile selection; seeds; test-informed disclosure; implementation and hardware | experiment protocol + table registry | Defines the evaluation contract only | Evidence inputs complete; ready for final prose consolidation |
| 5.2 Comparison with horizon-specific forecasters | Can one unified ISCF-BSCA model compete with separately optimized fixed-horizon baselines? | explain one-versus-four-model protocol; report seven dataset-level four-horizon means, overall best/second counts and low-/high-dimensional dataset-group gains; route complete per-H results to Appendix A | Table 1 / `Main-I` | System-level accuracy competitiveness under the audited mixed-source comparison | Complete and hash-frozen |
| 5.3 One-model-all-horizons evaluation | Is ISCF-BSCA competitive when every baseline must serve all horizons from one trained model? | define H720-prefix protocol; explain how it differs from Main-I; report dataset-level four-horizon means; route complete per-H results to Appendix A; retain unmatched-protocol caveat | Table 2 / `Main-II` | One-model-all-horizons system effectiveness, not decoder or BSCA attribution | Complete and horizon-loader re-audited |
| 5.4 Efficiency and system cost | What accuracy and deployment cost changes when one model replaces a four-model horizon-specific family? | Main-I macro MSE/MAE; peak inference memory; four-horizon checkpoint storage | Table 3 / `Efficiency` | Best nine-system macro accuracy and one-checkpoint consolidation, with DLinear/SimpleTM/QDF retained as resource counterexamples | Complete; 252/252 accuracy cells, 63/63 service units and 231 table-role objects |
| 5.5 Component and training-objective ablations | Which architectural and optimization components contribute within the frozen design family? | Full; w/o BSCA; w/o Target-Adaptive Allocation; Shared Scope Projection; Fixed Scope ($s=144$); matched budgets and end-to-end training | Table 4 / `Core-Ablation` | Author-corrected aggregate table supports all four interventions; per-horizon rerun provenance remains unsynchronized | Complete at dataset-aggregate table level; Full best in 12/12 metric columns |
| 5.6 Forecast consistency and scope-allocation behavior | Does the trained system behave in the manner motivated by Section 3? | exact CHPC/CHPD verification; Scope Probability map; aggregate scope utilization across future regions; scope-wise regional preference/error analysis; one performance-selected qualitative trajectory with nested prefixes | Figure 5 / mechanism-analysis bundle | Exact CHPC and scope-arm regional error heterogeneity; probabilities remain near-uniform and do not reliably select the lowest-error scope; selected trajectory is illustrative only | Complete mixed evidence; 20/20 CHPC cells, 8/40 utilization-error agreements; realized allocation value excluded |
| 5.7 Backbone transferability | Does the complete framework remain effective beyond its current Encoder realization? | Weather、ETTm1、ETTm2；DLinear-style与PatchTST-style；Original Decoder versus complete ISCF-BSCA；end-to-end training | Table 5 / `Decoder-Transfer` | 作者修正aggregate表中两类backbones均取得3/3 dataset MSE/MAE wins与16/16 comparator columns正向；five-dataset negative audit保留 | author-corrected aggregate complete；per-H/hash rerun provenance unsynchronized；no additional HPO required |

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
| Table 3 | Efficiency | accuracy, peak memory and four-horizon checkpoint-storage trade-off | complete; accuracy advantage with mixed resource ranking |
| Table 4 | Core-Ablation | component and objective attribution | author-corrected aggregate table complete; 12/12 metric columns best for Full; historical 100-cell audit retained |
| Figure 5 | Forecast consistency, allocation behavior and an illustrative improved trajectory | connect Section 3 problems to trained-system behavior and show the resulting forecast concretely | complete; exact CHPC, near-uniform utilization, regional scope-error differences and disclosed selected trajectory |
| Table 5 | Decoder-Transfer | end-to-end portability of the complete framework across two backbone families | author-corrected three-dataset aggregate complete；Original versus ISCF-BSCA only；historical 48-cell audit retained |

The qualitative example is integrated into Figure 5 rather than assigned a separate figure or subsection. It should be selected from one of the clearest per-origin improvements of Full ISCF-BSCA over a frozen matched control, with the comparator, split and selection rule disclosed in the caption. This performance-selected example is intentionally illustrative and must not be described as representative or used to estimate prevalence. A dedicated failure-case panel is not required; negative aggregate cells remain reported in Sections 5.2--5.3 and interpreted in Section 6.3.

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
| The complete ISCF-BSCA framework transfers across the evaluated forecasting backbones | Author-corrected Decoder-Transfer | supported on the author-refined three-dataset aggregate scope | DLinear-style与PatchTST-style均3/3 dataset MSE/MAE wins；只写evaluated-scope portability并披露unsynchronized per-H provenance、test-tuned history与five-dataset negative boundary，不写universal或architecture-agnostic |
| CHPC holds for shared targets | architecture + implementation verification | supported numerically in 20/20 dataset-horizon cells with maximum absolute CHPD=0 | keep separate from forecasting accuracy |

## 9. Recommended writing order

1. freeze the final Section 5.1 evaluation contract and evidence-role vocabulary;
2. draft 5.2 and 5.3 from the already complete Main-I/Main-II artifacts without anticipating H5A;
3. draft 5.4, 5.5 and 5.7 from the complete Efficiency, Core-Ablation and Decoder-Transfer artifacts, retaining their negative boundaries;
4. draft 5.6 from the complete mixed Figure 5 evidence, retaining the near-uniform allocation and 8/40 agreement boundary;
5. write Discussion now that all positive and negative evidence is known;
6. write Conclusion and then revisit the provisional result sentence in Introduction P6.

## 10. Temporarily frozen author decisions

1. Retain a standalone Section 6 Discussion.
2. Keep Section 5.6 as one integrated consistency-and-allocation analysis and incorporate one deliberately clear improved example into Figure 5.
3. Omit a standalone failure-case subsection and figure.
4. Restrict Core-Ablation to the five author-fixed variants above; do not add a balance-only control.
5. Fix the single-scope control to $s=144$ without a best-scope search.
6. Exclude realized allocation value from the current mechanism-analysis contract.

These decisions freeze the writing and experiment-design reference only. They do not authorize implementation, remote training or formal test.

`Decision=sections_5_7_v0_6_section5_v0_4_setup_refined_sections6_7_pending`.
