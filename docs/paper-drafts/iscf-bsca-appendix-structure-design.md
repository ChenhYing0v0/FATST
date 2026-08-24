# ISCF-BSCA Appendix Structure Design

**Version:** v0.1-minimal-appendix-design
**Date:** 2026-08-24
**Scope:** appendix routing for the frozen Sections 1--7 manuscript

## 1. Design principle

The Appendix should preserve material that is necessary for reproducibility or
for checking complete result coverage, while leaving the main narrative and
its evidence carriers unchanged. The current recommendation is therefore to
use two appendices only. Secondary diagnostics, historical audits and
exploratory sensitivity analyses remain in `analysis/` unless a later author
decision promotes one of them into the paper.

## 2. Recommended appendix map

### Appendix A. Experimental protocol and reproducibility details

This appendix consolidates the information needed to reproduce the evaluated
comparison without duplicating the main text:

1. dataset names, dimensions, sampling frequency, split lengths and concise
   descriptive statistics;
2. look-back window, prediction horizons, metrics, normalization and
   dataset-specific preprocessing;
3. Encoder--decoder configuration, baseline families, source roles
   (official-code reproduction, published-context entry or matched local
   comparison), and protocol differences relevant to interpretation;
4. software and hardware environment, optimizer and learning-rate schedule,
   seed/checkpoint conventions, and the frozen profile-selection rule;
5. a short artifact/provenance note linking the complete machine-readable
   tables and audit records.

This keeps the implementation and source-role disclosures requested in
Section 5.1 in one place rather than creating separate settings,
reproducibility and provenance appendices.

### Appendix B. Complete horizon-wise benchmark results

This appendix contains the full-precision cells omitted from the compact
dataset-average tables in Sections 5.2 and 5.3:

- **Table B1:** complete Main-I results for the unified ISCF-BSCA model and
  horizon-specific baselines, for every dataset, horizon and metric;
- **Table B2:** complete Main-II results when every method serves all horizons
  with one unified model, using the same dataset--horizon--metric coverage.

Negative cells and the source/protocol labels needed to interpret mixed-source
comparisons should remain visible. The main-text averages are not replaced;
the appendix tables provide the complete audit surface behind Tables 1 and 2.

## 3. Section-by-section routing audit

| Section | Appendix requirement | Routing decision |
| --- | --- | --- |
| Sections 1--4 | No explicit appendix dependency; definitions, method equations and Figures 1--4 are self-contained. | No appendix item. |
| 5.1 Experimental setup | Dataset details, statistics, splits, preprocessing, implementation and source roles. | Appendix A. |
| 5.2 Horizon-specific comparison | Complete dataset--horizon cells behind compact Table 1. | Appendix B, Table B1. |
| 5.3 One-model-all-horizons evaluation | Complete dataset--horizon cells behind compact Table 2. | Appendix B, Table B2. |
| 5.4 Accuracy and system cost | Figure 6 is the only main-text evidence carrier; numerical resource sources remain auditable artifacts. | No additional appendix table; provenance is summarized in Appendix A. |
| 5.5 Ablations | Table 3 is the required aggregate attribution surface. | No duplicate appendix table. |
| 5.6 Scope diversity and allocation behavior | Figure 5 is a bounded, sample-specific validation diagnostic. | No duplicate figure or expanded prevalence table. |
| 5.7 Generalization studies | Figure 7 is the sole main-text carrier; the transfer table remains its numerical/audit source. | No appendix table under the minimal plan; source provenance is covered in Appendix A. |
| Sections 6--7 | Interpretation, limitations and conclusion are self-contained. | No appendix item. |

## 4. Material deliberately excluded

The minimal plan does not create standalone appendices for the following
items:

- repeated copies of Figures 2, 3, 5, 6 or 7;
- a second ablation table for the Allocation-Balance Regularizer, random
  partitions, scope-count or loss-weight sensitivity, which are not part of
  the frozen core evidence;
- historical profiler outputs, old HPO trials or superseded diagnostics;
- a separate transfer-results table, unless the target venue later requires
  numerical supplements for Figure 7;
- failure cases or extra qualitative examples.

The preferred-scope diagnostic discussed in earlier governance records remains
an analysis artifact and is not promoted automatically to the Appendix,
because its reader-facing message overlaps with Figure 3.

## 5. Open editorial decisions

Before final submission, confirm only the following venue-dependent details:

1. whether the journal requires machine-readable source data for Figures 6 and
   7 in addition to the repository artifacts;
2. whether the Appendix tables should be labeled A1/A2 or B1/B2 after the
   journal template fixes appendix numbering;
3. whether the source/provenance note in Appendix A should link to a separate
   Supplementary Data file.

No new training, formal test, or sensitivity experiment is implied by this
design.
