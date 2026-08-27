# HoriScope Section 7: Conclusion Structure Design

## Design status

| Field | Content |
| --- | --- |
| `document_role` | Initial structural design for the manuscript Conclusion |
| `version` | `v0.4-author-refined-structure` |
| `date` | `2026-08-27` |
| `review_status` | Two-paragraph structure retained; paragraph functions refined to avoid Discussion overlap and sequential summary |
| `upstream_dependency` | Introduction v0.9, Related Work v0.2, Section 3 v0.7, Section 4 v0.7, Section 5 v0.15 and Section 6 v0.5 remain temporarily frozen and unchanged |
| `section_format` | No subsections; two compact paragraphs |
| `target_length` | Approximately 140--180 words in the eventual English manuscript |
| `evidence_scope` | CHPC construction, Sections 5.2--5.7 and Figure 6 deployment evidence |
| `experiment_change` | None; no new implementation, training or formal test |
| `claim_boundary` | The Conclusion closes the established argument and introduces no new results, citations, mechanisms or generality claims |

This file records the author-confirmed argumentative structure. The corresponding English draft is available at `docs/paper-drafts/iscf-bsca-conclusion-initial-draft.md`.

## 1. Section-level role

Section 7 should close the paper by returning to the central unified varied-horizon forecasting (UVHF) problem and stating what the complete evidence establishes. Section 6 already interprets output-side multi-scope forecasting and records the methodological limitations, so the Conclusion should neither repeat that discussion nor enumerate every experiment. Its role is to compress the paper into the chain `problem -> principle -> method -> evidence-backed outcome -> final implication`.

## 2. One-sentence argument

For UVHF, a horizon-agnostic predictor can combine prefix-consistent serving with adaptive forecast generation by treating output-side sharing granularity as an explicit decoder design axis.

## 3. Recommended paragraph structure

### Paragraph 1: Conceptual contribution

**Main job:** state the conceptual change introduced by the paper without replaying the Introduction or Method section.

Recommended sentence functions:

1. Reframe multiple request horizons as nested views of one coherent future trajectory.
2. Identify CHPC and output-side sharing granularity as the two principles that organize this formulation.
3. Compress HoriScope and BSCA into their functional roles: adaptive multi-scope forecast generation and balanced joint learning.
4. Close on the capability of the complete architecture, not on a module inventory.

This paragraph should avoid repeating tensor notation, module-by-module computation, loss formulas or the complete contribution list from the Introduction.

### Paragraph 2: Selective evidence and closing implication

**Main job:** state the strongest empirical outcome and close on the capability established by the paper.

Recommended sentence functions:

1. State the principal result: one HoriScope model achieves state-of-the-art aggregate accuracy among the evaluated methods while serving every supported horizon.
2. Retain only complementary evidence needed for trust, namely the unified-workflow comparison, controlled ablations and two-backbone studies.
3. Avoid enumerating every experiment, figure or component already discussed in Section 5.
4. End with the broader implication that output-side sharing granularity is a productive design axis for unified time-series forecasting.

The final sentence should close on the established capability rather than introduce a new future-work agenda. Methodological extensions have already been stated in Section 6.

## 4. Claim-evidence map

| Planned conclusion statement | Evidence route | Permitted strength | Boundary |
| --- | --- | --- | --- |
| One predictor can serve multiple horizons through nested prefixes | Sections 3.1 and 4.4; implementation CHPD audit | Architectural property under the stated inference construction | Do not infer accuracy from CHPC alone |
| The unified model is accurate relative to evaluated baselines | Sections 5.2 and 5.3 | Paper-facing system-level effectiveness across seven datasets and four horizons | Do not claim superiority over every possible forecasting model or protocol |
| Unified serving provides a favorable accuracy--cost trade-off | Section 5.4 and Figure 6 | Favorable combined accuracy, checkpoint-storage and inference-memory profile among the displayed methods | Do not claim lowest cost on every resource dimension |
| Proposed components contribute to aggregate performance | Section 5.5 | Supported within the matched five-variant ablation design | Do not elevate ablation drops into unique causal proof |
| Scope signals and allocation follow the intended non-collapse behavior | Section 5.6 and Figure 5 | Supported as sample-specific validation diagnostics | Do not claim population prevalence, oracle routing or causal specialization |
| The decoder is compatible with different Encoder families | Section 5.7 and Figure 7 | Supported for the evaluated DLinear and PatchTST backbones on three datasets | Do not claim universal Encoder-agnostic generalization |

## 5. Content exclusions

The final Conclusion should not:

- introduce citations, equations, new metrics or new numerical comparisons;
- repeat the methodological limitations already stated in Section 6;
- list every dataset, baseline, ablation variant or figure individually;
- describe CHPC as an accuracy guarantee;
- describe Scope Probabilities as oracle, sparse or causally specialized routing;
- claim universal superiority, universal transferability or uniform resource efficiency;
- end with generic future-work language such as `more work is needed`.

## 6. Terminology lock

Use the established canonical forms: **unified varied-horizon forecasting (UVHF)**, **horizon-specific forecasting**, **cross-horizon prefix consistency (CHPC)**, **HoriScope**, **BSCA**, **sharing scope**, **scope-indexed forecast field**, **Target-Adaptive Allocation**, **Scope-conditioned Forecast**, **prefix-consistent trajectory** and **Encoder**. Use `forecast generation`, not `forecast synthesis`, and describe HoriScope as an output-side decoder framework instead of a collection of independently trained forecasters.
