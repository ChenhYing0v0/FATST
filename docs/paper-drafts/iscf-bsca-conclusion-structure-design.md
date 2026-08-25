# ISCF-BSCA Section 7: Conclusion Structure Design

## Design status

| Field | Content |
| --- | --- |
| `document_role` | Initial structural design for the manuscript Conclusion |
| `version` | `v0.3-author-fixed-structure` |
| `date` | `2026-08-24` |
| `review_status` | Two-paragraph structure and corresponding prose temporarily fixed usable |
| `upstream_dependency` | Introduction v0.9, Related Work v0.2, Section 3 v0.7, Section 4 v0.7, Section 5 v0.13 and Section 6 v0.4 remain temporarily frozen and unchanged |
| `section_format` | No subsections; two compact paragraphs |
| `target_length` | Approximately 180--230 words in the eventual English manuscript |
| `evidence_scope` | CHPC construction, Sections 5.2--5.7 and Figure 6 deployment evidence |
| `experiment_change` | None; no new implementation, training or formal test |
| `claim_boundary` | The Conclusion closes the established argument and introduces no new results, citations, mechanisms or generality claims |

This file records the author-confirmed argumentative structure. The corresponding English draft is available at `docs/paper-drafts/iscf-bsca-conclusion-initial-draft.md`.

## 1. Section-level role

Section 7 should close the paper by returning to the central unified varied-horizon forecasting (UVHF) problem and stating what the complete evidence establishes. Section 6 already interprets output-side multi-scope forecasting and records the methodological limitations, so the Conclusion should neither repeat that discussion nor enumerate every experiment. Its role is to compress the paper into the chain `problem -> principle -> method -> evidence-backed outcome -> final implication`.

## 2. One-sentence argument

For UVHF, one horizon-agnostic predictor can generate accurate and prefix-consistent forecasts by organizing forecast generation over multiple output-side sharing scopes and jointly optimizing their target-adaptive integration.

## 3. Recommended paragraph structure

### Paragraph 1: Problem and technical contribution

**Main job:** restate the problem solved and the technical principle introduced.

Recommended sentence functions:

1. Return to the limitation of horizon-specific forecasting: separate predictors fragment multiple request endpoints and do not enforce one coherent trajectory.
2. State the paper's formulation: UVHF should use one future-step-indexed predictor whose outputs satisfy CHPC across requests.
3. Introduce ISCF at principle level: it constructs one scope-indexed forecast field through region-wise forecast generation under multiple output-side sharing extents and performs target-adaptive integration.
4. State the role of BSCA in one clause or sentence: it supports joint learning of the fused trajectory and the individual scope lines.

This paragraph should avoid repeating tensor notation, module-by-module computation, loss formulas or the complete contribution list from the Introduction.

### Paragraph 2: Evidence-backed outcome and closing implication

**Main job:** state the strongest empirical outcome and close on the capability established by the paper.

Recommended sentence functions:

1. Summarize the principal system result: across seven forecasting benchmarks and four horizons, one ISCF-BSCA model achieves stronger aggregate forecasting accuracy than the evaluated horizon-specific baselines and remains strongest under the one-model-all-horizons comparison.
2. Add the practical result: unified serving combines competitive accuracy with a favorable checkpoint-storage and inference-memory profile.
3. Compress the supporting evidence into one sentence: controlled ablations support the contribution of the multi-scope architecture, Target-Adaptive Allocation and BSCA, while internal diagnostics and two-backbone studies support the intended scope behavior and decoder compatibility within the evaluated settings.
4. End with the conceptual implication: output-side sharing granularity is a useful design axis for building unified forecasters that serve multiple horizons through one prefix-consistent trajectory.

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

Use the established canonical forms: **unified varied-horizon forecasting (UVHF)**, **horizon-specific forecasting**, **cross-horizon prefix consistency (CHPC)**, **ISCF**, **BSCA**, **sharing scope**, **scope-indexed forecast field**, **Target-Adaptive Allocation**, **Scope-conditioned Forecast**, **prefix-consistent trajectory** and **Encoder**. Use `forecast generation`, not `forecast synthesis`, and describe the method as an output-side decoder framework rather than a collection of independently trained forecasters.
