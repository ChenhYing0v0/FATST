# ISCF Future-Coordinate Component: Design Research and Decision

## Status and boundary

| Field | Content |
| --- | --- |
| `component_id` | `iscf_future_coordinate_component_v1` |
| `status` | `local_design_study_for_author_review` |
| `scope` | Future-coordinate glyph only |
| `backend` | Python/matplotlib only |
| `method_change` | none |
| `main_figure_replacement` | false |
| `empirical_data` | none |

This study does not redesign the global architecture figure. It asks only how the fixed future-step coordinate field should be represented locally so that its mathematical role is immediately legible and visually compatible with a top-conference method figure.

## Exact semantic contract

The method defines a fixed coordinate field

$$
\boldsymbol\Phi
=
[\boldsymbol\phi_1,\ldots,\boldsymbol\phi_T]^\top
\in\mathbb R^{T\times D_q},
$$

with

$$
\widetilde\phi_{\tau,d}
=
\cos\!\left(\frac{\pi(\tau-\tfrac12)d}{T}\right),
$$

a constant channel for $d=0$, and centered/scaled nonconstant channels. It is parameter-free and label-free. A single future target is identified by the column vector $\boldsymbol\phi_\tau$. The same field is reused in two places: contiguous coordinate pooling constructs scope-region descriptors, and $\boldsymbol\phi_\tau$ conditions target-wise scope allocation.

The component must therefore communicate five facts:

1. $\tau$ is an ordered position on one future axis;
2. one position is evaluated through several cosine basis channels;
3. the values at one position form the vector $\boldsymbol\phi_\tau$;
4. the full field is shared across all future targets;
5. the representation is fixed and schematic, not learned data or an empirical heatmap.

## Figure contract

**Core conclusion.** A future index $\tau$ is mapped by a fixed multiscale harmonic basis into the coordinate vector $\boldsymbol\phi_\tau$, while the same continuous coordinate field remains available for region pooling.

**Archetype.** Local schematic component study with three side-by-side design candidates.

**Evidence hierarchy.** Candidate A is the recommended paper glyph; Candidates B and C expose the trade-off between visual depth and compactness.

**Export contract.** Review sheet at double-column width, editable SVG/PDF, 300 dpi PNG and 600 dpi TIFF. No statistics or source data are required because all marks are deterministic schematic encodings of the analytic coordinate basis.

**Reviewer risks.** A heatmap can be mistaken for learned activations; a helix or phase wheel can imply paired sine/cosine rotation; a 3D manifold can imply learned geometry; a collection of tokens can hide the ordered future axis.

## Targeted primary-source review

Search date: 2026-08-05. The review covered positional encoding, coordinate networks, multiscale Fourier representations, and recent time-series visual grammar. Sources were used for representation analysis only; no external method or result is imported.

| Source | Relevant representation lesson | Local implication |
| --- | --- | --- |
| [Attention Is All You Need, NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) | A scalar position is expanded through sinusoidal functions with different frequencies. | Show the position axis and frequency family explicitly. |
| [Time2Vec](https://arxiv.org/abs/1907.05321) | Time is represented as a vector containing an aperiodic component and periodic components. | The output should visibly terminate in a coordinate vector, not only a surface. |
| [Fourier Features, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html) | A low-dimensional coordinate is mapped to a bank of sinusoidal features before downstream computation. | The clean visual primitive is `coordinate → harmonic family → feature vector`. |
| [Learnable Fourier Features, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/84c2d4860a0fc27bcf854c444fb8b400-Abstract.html) | Positional similarity heatmaps and coordinate-to-vector mappings serve different explanatory purposes. | A heatmap is suitable for pairwise similarity, but visually indirect for the present mapping. |
| [NeRF, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460392.pdf) | Multiresolution sinusoidal encoding is communicated as a functional expansion of a query coordinate. | A compact function-bank glyph is defensible when space is limited. |
| [SIREN, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/53c04118df112c13a8c34b38343b9c10-Abstract.html) | Periodic functions provide an interpretable frequency-domain language for coordinate-based signals. | Curves are semantically stronger than colored cells for showing harmonic structure. |
| [BACON, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Lindell_BACON_Band-Limited_Coordinate_Networks_for_Multiscale_Scene_Representation_CVPR_2022_paper.html) | Explicit frequency bands make multiscale coordinate processing interpretable. | Order the displayed channels from constant/low to higher frequency and retain frequency hierarchy. |
| [TimesNet, ICLR 2023](https://arxiv.org/abs/2210.02186) | A 2D tensor is effective when two spatial axes carry distinct temporal semantics. | The current atlas is mathematically valid, but its heatmap grammar overstates a data-like 2D object. |
| [RoFormer / RoPE](https://arxiv.org/abs/2104.09864) | Circular rotation is exact for paired-coordinate rotations used by RoPE. | Phase wheels and helices should not represent the present cosine-only DCT field. |
| [Instant-NGP, SIGGRAPH 2022](https://research.nvidia.com/publication/2022-07_instant-neural-graphics-primitives-multiresolution-hash-encoding) | Nested grids communicate learned multiresolution feature lookup. | Grid pyramids would falsely imply trainable lookup tables and are rejected. |

## Diagnosis of the current atlas

The current heatmap-based coordinate atlas is mathematically faithful but visually weak for a method overview:

- colored cells are normally read as measured activations or attention weights;
- the harmonic origin of the coordinates is invisible without reading the formula;
- the large ochre block has greater visual mass than its architectural importance;
- the selected column and the detached vector repeat the same color encoding without clarifying the mapping;
- region pooling is not naturally suggested by the dense cell grid;
- the component consumes too much area relative to `Encoder`, scope construction and allocation.

The problem is therefore not that the coordinate field is high-dimensional. The problem is that the current visual grammar explains its **storage layout** rather than its **functional meaning**.

## Candidate comparison

| Candidate | Main glyph | Fidelity | Elegance | Compactness | Reviewer risk | Decision |
| --- | --- | ---: | ---: | ---: | --- | --- |
| A. Sampled harmonic ribbon | aligned cosine lanes + one sampling cursor + extracted vector | 5/5 | 5/5 | 4/5 | low | **recommended** |
| B. Spectral fan | perspective-separated harmonic curves + sampling blade | 4/5 | 5/5 | 3/5 | medium: perspective may imply an extra geometric axis | optional for talks, not preferred for paper |
| C. Coordinate capsule | normalized future index + compact DCT function bank + output vector | 4/5 | 4/5 | 5/5 | medium: hides the full shared field and pooling relation | compact fallback |
| D. Heatmap atlas | channel-by-step colored cells | 5/5 | 2/5 | 4/5 | medium: resembles empirical activation data | reject for hero glyph |
| E. Phase wheel or helix | circular/spiral phase geometry | 2/5 | 5/5 | 3/5 | high: implies sine/cosine pairs or rotational encoding | reject |
| F. Multiresolution grid pyramid | nested learned grids | 1/5 | 4/5 | 2/5 | high: implies trainable spatial lookup | reject |

## Recommended design: sampled harmonic ribbon

Candidate A uses four aligned, axis-free harmonic lanes, ordered from the constant channel to increasing DCT frequency. A thin violet cursor at future step $\tau$ intersects every lane. Small filled nodes at the intersections are gathered into the vertical vector $\boldsymbol\phi_\tau$. The shared horizontal baseline is labelled only at $1$, $\tau$ and $T$.

This design is preferred because it explains the operator in one glance:

$$
\tau
\quad\longmapsto\quad
\{\phi_{\tau,d}\}_{d=0}^{D_q-1}
\quad=\quad
\boldsymbol\phi_\tau.
$$

For insertion into the main figure:

- display four representative lanes: $d=0,1,2,D_q-1$;
- use a cool indigo-to-teal family for basis curves and reserve violet for the selected $\tau$;
- draw the coordinate vector as four signed nodes on a neutral vertical spine, not as another heatmap;
- show one subtle bracket under a contiguous interval only when explaining coordinate pooling into a region descriptor;
- use the label `fixed future-step coordinates` rather than a large `Future coordinates` heading;
- keep the formula $\boldsymbol\Phi\in\mathbb R^{T\times D_q}$ in the caption or nearby text, not inside the main glyph;
- add `fixed, parameter-free` as a small qualifier if space permits.

## Claim boundary

The curves in the comparison sheet are analytic basis functions, not dataset observations. The component does not claim that coordinate channels are learned, that a Euclidean distance in coordinate space has a specific forecasting meaning, or that the coordinate field alone improves accuracy. Its purpose is only to communicate future-target identity and its shared use in the ISCF computation graph.

## Delivered prototypes

- `figure_iscf_future_coordinate_component_concepts_v1.*`: three-candidate review sheet;
- `figure_iscf_future_coordinate_component_recommended_v1.*`: standalone sampled-harmonic-ribbon glyph at a realistic architecture-panel size;
- `plot_iscf_future_coordinate_concepts.py`: deterministic Python renderer;
- `figure_manifest.json`: machine-readable status, boundary and export inventory.

The existing Figure 4 and Method manuscript remain unchanged. These assets are local design candidates for author review, not a manuscript replacement.

## QA

- `nature-figure` source preflight: **14 PASS, 0 WARN, 0 FAIL**;
- comparison PDF: one page, 518.4 × 221.04 pt (182.88 × 77.98 mm);
- recommended-component PDF: one page, 231.84 × 124.56 pt (81.79 × 43.94 mm);
- comparison raster exports: 2160 × 921 pixels at 300 dpi and 4320 × 1842 pixels at 600 dpi;
- recommended-component raster exports: 966 × 519 pixels at 300 dpi and 1932 × 1038 pixels at 600 dpi;
- editable text retained: 35 SVG text nodes in the review sheet and 13 in the recommended component;
- visual inspection: no detected clipping, collision or ambiguous path merge at native review size;
- color and typography: restrained cool basis family, one violet sampling accent, one ochre pooling accent, and a 5 pt minimum text size;
- statistical/source-data checks: not applicable because the curves are analytic DCT basis functions and carry no empirical values.
