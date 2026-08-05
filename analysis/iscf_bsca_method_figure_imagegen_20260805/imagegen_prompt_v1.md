# ISCF Main Figure ImageGen Prompt v1

## Role and boundary

- Purpose: explore the visual hierarchy and composition of the ISCF inference architecture.
- Status: raster concept draft; not a manuscript-ready replacement.
- Included: shared history encoding, future-coordinate field, scope-conditioned forecasting, target-conditioned allocation and weighted contraction.
- Excluded: single-scope forecasting comparison, BSCA, losses, training-only paths, requested-horizon markers and empirical results.

## Prompt

```text
Use case: scientific-educational
Asset type: landscape architecture figure for a top-tier time-series forecasting paper

Primary request: Create a highly polished, visually ordered scientific architecture diagram for Independent Scope-Conditioned Forecasting (ISCF). The figure must explain how one encoded history and an explicit future-coordinate field produce multiple scope-conditioned forecasts, and how target-conditioned scope probabilities fuse them into one final trajectory. This is an inference-only architecture diagram. Do not show BSCA, loss functions, training paths, single-scope baselines, requested horizons, performance numbers, or empirical results.

Scene/backdrop: pure white background, wide landscape canvas, generous margins, subtle publication-grade vector appearance. Use a strict left-to-right column grid. Keep all stage headings aligned. No decorative background, shadows, gradients, perspective, 3D objects, or floating cards.

Overall composition: one integrated figure with two clean horizontal lanes that begin from the same history state and future coordinates and merge exactly once near the right edge. The upper lane is the larger hero lane, labelled "Scope-conditioned forecasting". The lower lane is slimmer and labelled "Target-conditioned allocation". Separate the lanes with whitespace rather than enclosing boxes. Use thin consistent arrows, short orthogonal connectors, and aligned columns. No crossing arrows, diagonal cables, tangled curves, or duplicated final outputs.

Left input area: show three smooth multivariate history curves labelled "History". Feed them into one compact rounded rectangle labelled "Encoder"; this is one of only two allowed named module boxes. The encoder outputs a compact blue stacked vector or tensor labelled "History state". In parallel below, show four clean gold coordinate curves over future step tau, including a constant channel and progressively varying smooth basis curves, labelled "Future coordinates". Depict the coordinate field as an explicit multidimensional object, not as a generic box.

Upper hero lane: arrange five perfectly aligned scope rows labelled exactly "s=1", "s=48", "s=144", "s=360", and "s=720". The five rows are slices of one shared architecture, not five independent models, so use one shared encoder and one common column structure. In each row, depict:
1. a small blue matrix generated from the history state, under the column heading "Scope matrix";
2. a gold segmented horizontal ribbon generated from the future coordinates, under "Region descriptor"; use progressively wider contiguous segments for larger scopes;
3. combine the matrix and descriptor visually with a small multiplication symbol, producing a cyan segmented tensor under "Region representation";
4. pass the tensor through one shared narrow synthesis ribbon with step-varying colored cells, under "Step-specific synthesis";
5. output one teal forecast curve under "Scope-wise forecasts".
Align every row precisely. Use direct visual objects—matrices, segmented ribbons, tensor cells and curves—not additional named boxes. The segmentation should become coarser from s=1 to s=720 while every forecast remains step-specific and smooth.

Lower allocation lane: route the blue history-state vector and gold future-coordinate curves into a compact two-tone blue-and-gold tensor labelled "Condition vector". Feed it into one rounded rectangle labelled "Allocation MLP"; this is the second and final allowed named module box. Output a clean five-row probability heatmap labelled "Scope probabilities", aligned to the same five scope colors/order as the upper lane. Beside the heatmap, show one selected future step as a five-element probability vector using five vertically stacked circles or short bars of different sizes. Do not imply ground-truth or label conditioning.

Right merge: collect the five scope-wise forecast curves as a coherent stacked forecast field. Connect the selected five-element probability vector to one green circular weighted-sum symbol. The forecast field and the probability vector must merge only at this symbol, labelled "Weighted contraction". From it, draw one prominent dark-teal final forecast curve labelled "Final trajectory". Make this single trajectory the visual endpoint and strongest focal element.

Visual style: refined flat vector-like scientific infographic, rigorous geometry, restrained contemporary journal aesthetic, crisp sans-serif typography, consistent stroke weights, ample whitespace and clear reading order. Primary colors: indigo/blue for history-conditioned objects, warm gold for future coordinates and region descriptors, cyan/teal for forecasts, violet for allocation, dark teal for fusion and final output. Use colorblind-safe muted tones with high contrast on white.

Text (verbatim): "History", "Encoder", "History state", "Future coordinates", "Scope-conditioned forecasting", "Scope matrix", "Region descriptor", "Region representation", "Step-specific synthesis", "Scope-wise forecasts", "Target-conditioned allocation", "Condition vector", "Allocation MLP", "Scope probabilities", "Weighted contraction", "Final trajectory", "s=1", "s=48", "s=144", "s=360", "s=720".

Constraints: render the listed text exactly once where applicable; no extra prose, equations, legend, caption, title banner, watermark, logo, panel letters, icons, people, gears, neural-network node clouds, generic server blocks, or black-box modules. Only "Encoder" and "Allocation MLP" may appear inside named rounded rectangles. Prioritize architectural correctness, strict alignment, whitespace, and immediate visual comprehension over decorative complexity.

Avoid: crowded layout; tiny text; inconsistent row spacing; overlapping labels; crossed connectors; large enclosing boxes; box-and-arrow flowchart appearance; five separate encoder-decoder models; horizon labels; BSCA; loss terms; gradients; training loops; empirical charts; pseudo-3D tensors; photorealism; dark background; excessive glow or gradients.
```

## Generated concept

| Field | Value |
| --- | --- |
| `mode` | built-in ImageGen |
| `output` | `figure_iscf_imagegen_concept_v1.png` |
| `dimensions` | 1,693 × 929 px |
| `sha256` | `14ecf4b3db00e2167c7f4486cdbcdeefd5a4432cc63483e9fb8d233a7b0b5e43` |
| `manuscript_replacement` | false |

## Initial visual audit

- The strict column grid, two-lane hierarchy and single merge point are substantially clearer than in the deterministic v2 concept.
- The visual grammar follows the prompt: only `Encoder` and `Allocation MLP` are named module boxes; matrices, ribbons, tensors, probability fields and curves carry the remaining computation.
- Scope rows read as aligned slices of one forecasting field rather than five complete models.
- All requested major labels are legible, and no BSCA, loss or requested-horizon path is introduced.
- Remaining structural issue: the coarsest region descriptor is not rendered as a single region, so the scope-to-segmentation relation is approximate rather than diagrammatically exact.
- Raster generation is suitable for composition exploration but not yet for final mathematical fidelity, editable typography or publication export.
