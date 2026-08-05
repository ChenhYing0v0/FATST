# ISCF Framed Coordinate Regions: Design Contract and QA

## Status

| Field | Content |
| --- | --- |
| component_id | iscf_framed_coordinate_regions_v1 |
| status | local_design_draft_for_author_review |
| backend | Python/matplotlib only |
| scope | history and future-coordinate visual grammar |
| method_change | none |
| main_figure_replacement | false |
| empirical_data | none |

## Core conclusion

A history tensor and the fixed future-coordinate field can share one framed, channel-wise visual grammar, while region segmentation and region-wise coordinate averaging remain explicit.

## Semantic mapping

| Visual mark | Exact meaning |
| --- | --- |
| one history row | one representative input variable/channel |
| one coordinate row | one DCT coordinate channel $d$ |
| coordinate curve | $\phi_{\tau,d}$ over future step $\tau$ |
| vertical partition | a contiguous region $\mathcal G_g^{(s)}$ under one representative scope |
| coordinate cell fill | signed region mean $\overline{\phi}_{g,d}^{(s)}$ |
| highlighted region column | the selected region $g^\star$ |
| four extracted squares | $\overline{\boldsymbol\phi}^{(s)}_{g^\star}\in\mathbb R^{D_q}$ with $D_q=4$ |

The provided reference images are used only for style-level inspiration: framed channel rows, shared borders, and vertical region partitioning. All mathematical content, values, labels and color semantics are rebuilt from the ISCF coordinate contract.

## Exact coordinate construction

The prototype uses the manuscript definition

$$
\widetilde\phi_{\tau,d}
=
\cos\!\left(
\frac{\pi(\tau-\tfrac12)d}{T}
\right),
$$

with the constant channel $\phi_{\tau,0}=1$ and centered/scaled channels for $d\geq1$. The displayed region means are computed directly from these analytic curves. Five equal regions are shown because they form a compact illustrative partition; the visual does not assert that every scope has five regions.

## Color contract

Coordinate values are signed. A single-hue opacity scale would preserve magnitude but lose sign, so the prototype uses a restrained diverging map:

- muted blue: negative region mean;
- near-white: value near zero;
- muted terracotta: positive region mean;
- saturation: absolute magnitude.

The same normalization and colors are used for field cells and the extracted descriptor squares. Curves remain visible above the translucent fills, so the field and its region averages can be read simultaneously.

## Reviewer-risk controls

- The coordinate background is an analytic region summary, not an attention map or learned activation.
- History curves are schematic and do not depict a selected dataset sample.
- Three history rows are representative channels rather than the exact value of $C$.
- Five regions illustrate one partition only; they are not requested horizons.
- The extracted four-square object is a region descriptor, not four separate predictions.
- The current prototype shows the region-pooling use of $\boldsymbol\Phi$. A later main-figure integration may add a thin target-step cursor for the allocation input $\boldsymbol\phi_\tau$ without changing the framed grammar.

## Figure contract

- archetype: local schematic component study;
- hero object: framed future-coordinate field with signed region means;
- support object: framed multivariate history demonstrating the shared visual language;
- export: editable SVG/PDF, 300 dpi PNG and 600 dpi TIFF;
- statistics/source data: not applicable; all coordinate values are analytic and history curves are explicitly schematic.

## Design decision

The framed coordinate field is preferred over the previous standalone harmonic ribbon as the **base visual language** for the architecture figure because it unifies history and coordinate representations and makes region averaging spatially explicit. The harmonic sampling idea should not be discarded completely: when the allocation path is integrated later, one thin target-step cursor can be placed inside the same frame to extract $\boldsymbol\phi_\tau$. This yields one shared object with two readouts:

- region column $\rightarrow\overline{\boldsymbol\phi}^{(s)}_g$ for region construction;
- target cursor $\rightarrow\boldsymbol\phi_\tau$ for allocation.

This combined interpretation is more faithful than using either a region-only heatmap or a target-only harmonic ribbon in isolation.

## QA

- `nature-figure` source preflight: **14 PASS, 0 WARN, 0 FAIL**;
- combined PDF: one page, 518.4 × 234.72 pt (182.88 × 82.80 mm);
- standalone coordinate PDF: one page, 339.84 × 175.68 pt (119.89 × 61.98 mm);
- combined raster exports: 2160 × 977 pixels at 300 dpi and 4320 × 1955 pixels at 600 dpi;
- standalone coordinate raster exports: 1416 × 732 pixels at 300 dpi and 2832 × 1464 pixels at 600 dpi;
- editable text retained: 27 SVG text nodes in the combined study and 17 in the standalone component;
- visual inspection: no detected clipping, label collision, broken region boundary or descriptor-order ambiguity at native review size;
- color inspection: the diverging fill preserves sign as well as magnitude and is repeated exactly in the extracted descriptor;
- source-data/statistical checks: not applicable because the coordinate quantities are analytic and the history curves are explicitly schematic;
- the manuscript and existing Figure 4 remain unchanged.
