# ISCF-BSCA Section 4 Style and Figure Calibration

## Scope

| Field | Content |
| --- | --- |
| `search_date` | 2026-08-05 |
| `purpose` | Calibrate the organization and visual grammar of the ISCF-BSCA Method section |
| `source_policy` | Primary conference proceedings, OpenReview or author preprints only |
| `coverage_claim` | None; this is a style audit rather than a literature-completeness or novelty search |
| `zotero_checked` | false |

## Audited papers

| Paper | Venue | Primary source | Reusable presentation pattern |
| --- | --- | --- | --- |
| iTransformer | ICLR 2024 | https://openreview.net/forum?id=JePfAI8fah | Opens the method with an overall architecture and uses component callouts around a central computation path before giving the compact forward equations. |
| TimeXer | NeurIPS 2024 | https://proceedings.neurips.cc/paper_files/paper/2024/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html | Uses a dominant end-to-end stack with separate insets for the two interactions that distinguish the model. |
| TimeMixer++ | ICLR 2025 | https://proceedings.iclr.cc/paper_files/paper/2025/hash/2b187165e28fdfdc0ffb34d1bfff2b0c-Abstract-Conference.html | Presents input transformation, the repeated block and output projection in one visual hierarchy, then follows the figure with a structure overview and component equations. |
| ROSE | ICML 2025 | https://proceedings.mlr.press/v267/wang25ci.html | Separates the shared architecture from stage-specific behavior and distinguishes trainable, frozen and backward-only paths through line style and compact symbols. |

## Adopted writing pattern

Section 4 follows the field-specific sequence `architecture overview -> tensor interface -> scope-field construction -> target-conditioned integration -> training-only objective -> structural properties and complexity`. Each subsection begins with the architectural role of the component, then defines its tensors and operations, and closes with the exact property it provides. Empirical superiority, ablation effectiveness and transferability remain outside the Method section.

The prose uses direct declarative transitions and medium-length sentences. Equations are introduced only after the reader knows what operation they formalize. The wording distinguishes construction properties from hypotheses: CHPC and the absence of an inference-time BSCA path are structural facts, whereas adaptive use of scopes and forecasting gains require later experiments.

## Adopted figure grammar

Figure 4 uses an asymmetric schematic-led composite rather than an equal grid. Panel b is the hero panel because the scope-indexed forecast field is the principal architectural contribution. Panel a supplies the single-scope contrast, panel c isolates the target-conditioned contraction and prefix-consistent output, and panel d places BSCA on a visually separate dashed training-only layer.

The figure inherits only general presentation principles from the audited papers: a dominant computation path, component callouts, consistent tensor direction and explicit stage boundaries. It does not copy their visual assets, mechanism diagrams, wording or color mappings.

## Rejected patterns

- No decorative three-dimensional tokens or perspective effects, because they obscure the scope-by-future-step tensor semantics.
- No scope-to-horizon color mapping, because a sharing scope is not a requested horizon.
- No empirical-looking allocation heatmap with performance labels, because Figure 4 explains the architecture rather than reporting learned behavior.
- No generic router, expert-specialization or load-balancing language, because the frozen contribution is target-conditioned scope allocation with ISCF-specific balanced co-adaptation.
