# ISCF Architecture Figure v2 Concept Contract and QA

## Status

| Field | Content |
| --- | --- |
| `figure_id` | `figure_iscf_architecture_concept_v2` |
| `status` | `concept_draft_for_information_hierarchy_review` |
| `manuscript_replacement` | false |
| `backend` | Python/matplotlib only |
| `archetype` | single asymmetric mechanism schematic |
| `data_role` | architecture schematic; no empirical observations |
| `implementation_change` | none |
| `paper_text_change` | none |

## Figure contract

**Core conclusion.** ISCF converts one encoded history and one explicit future-coordinate field into a scope-indexed forecast field, while a parallel target-conditioned allocation path contracts that field into one trajectory.

**Visual grammar.** Named module boxes are restricted to the `Encoder` and `Allocation MLP`. Computational objects are shown directly as curves, matrices, tensor stacks, repeated vectors, segmented future regions, probability fields and forecast trajectories.

**Single-canvas flow.**

1. history curves pass through the encoder and become a compact history-state vector;
2. a parallel four-channel coordinate plot represents the future-coordinate field;
3. the upper forecasting path maps history state to independent scope matrices, pools coordinates into scope-region descriptors, constructs region representations and applies shared step-specific synthesis to obtain scope-wise global forecasts;
4. the lower allocation path combines history state and target coordinates into condition vectors, maps them through an allocation MLP and obtains a target-wise scope-probability field;
5. scope forecasts and probabilities meet at weighted contraction and produce one final trajectory.

**Evidence hierarchy.** The forecasting path is the hero path because it carries the scope-indexed output-field construction. The allocation path is visually subordinate but continuous. The final contraction is the only merge point.

**Reviewer risks and controls.**

- The five scope rows must not look like five complete forecasting models. They begin after one shared encoder and merge into one field.
- Sharing scope must not be confused with requested horizon. Scope rows use direct extent labels; requested-horizon markers are omitted from this concept draft.
- Region sharing must not imply identical step predictions. A distinct step-specific synthesis tensor appears before the scope-wise forecast curves.
- Allocation must not appear label-conditioned. Its inputs are only the history state and future coordinates.
- The probability field is schematic and carries no empirical specialization claim.

## Export contract

- concept review size: 183 × 116 mm;
- SVG/PDF: preliminary vector review copies;
- PNG: 300 dpi;
- TIFF: 600 dpi with LZW compression;
- manuscript-grade editable refinement: intentionally deferred until the information hierarchy is approved;
- no source data file is required because the figure contains no empirical observations.

## QA

- Static preflight: 13 checks passed, 1 warning, 0 failures. The warning reports that the validator cannot statically resolve the final-width variable; the rendered PDF media box verifies the specified size.
- Vector outputs: the PDF contains one 518.740 × 328.819 pt page, corresponding to 183 × 116 mm; the SVG retains editable text.
- Raster outputs: the PNG is 2,161 × 1,370 pixels at 300 dpi; the TIFF is 4,322 × 2,740 pixels at 600 dpi with LZW compression.
- Visual review: no clipping or unintended overlap was observed at the concept-review size. The two computation paths, their single merge point and the distinction between scope-indexed forecasts and allocation probabilities remain legible.
- Claim-boundary review: all tensors and probabilities are schematic; the figure does not report empirical values or claim learned scope specialization, effectiveness or superiority.
- Remaining author review: confirm the five-row scope layout, the four-channel coordinate depiction, and the amount of tensor detail before manuscript-grade refinement.

The concept draft does not replace the current manuscript Figure 4 until author approval.
