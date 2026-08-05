# ISCF Vector Main Figure v4: Contract, Research Notes and QA

## Status

| Field | Content |
| --- | --- |
| `figure_id` | `figure_iscf_architecture_vector_v4` |
| `status` | `vector_draft_for_author_review` |
| `backend` | Python/matplotlib only |
| `archetype` | single-canvas schematic-led mechanism figure |
| `data_role` | architecture schematic; no empirical observations |
| `manuscript_replacement` | false |
| `method_change` | none |

## Figure contract

**Core conclusion.** ISCF uses one history state and one fixed future-step coordinate field to construct a scope-indexed forecast field, then contracts that field with target-conditioned scope probabilities to obtain one trajectory.

**Evidence hierarchy.** The upper scope-conditioned forecasting path is the hero path. The lower target-conditioned allocation path is subordinate but continuous. Their only merge occurs at weighted contraction.

**Scope abstraction.** The drawing shows three representative branches, labelled $s_0$, $s_1$ and $s_2$, to communicate fine, intermediate and broad state sharing. These are visual scope indices rather than the exact frozen scope values or requested horizons. The complete implementation may contain more scope branches.

**Named-box restriction.** Only `Encoder` and `Allocation MLP` are named module boxes. Curves, matrices, coordinate atlases, vectors, segmented region states, probability fields and forecast trajectories directly represent the remaining objects.

**Export contract.** Double-column width 183 mm; target height 112 mm; editable SVG and PDF; PNG at 300 dpi; TIFF at 600 dpi with LZW compression.

## Targeted primary-source review

Search date: 2026-08-05. Scope: recent top-conference time-series architecture figures and primary coordinate/Fourier-feature representations. Sources were used only to calibrate visual grammar; no external mechanism, result or implementation was imported.

| Primary source | Visual or conceptual lesson used here |
| --- | --- |
| [iTransformer, ICLR 2024](https://arxiv.org/abs/2310.06625) | Make tensor orientation and token semantics visually explicit instead of hiding them inside generic blocks. |
| [TimeMixer, ICLR 2024](https://arxiv.org/abs/2405.14616) | Use aligned parallel rows for related resolutions/scales and a single visually dominant mixing endpoint. |
| [TimeXer, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html) | Distinguish information roles by representation granularity and preserve a direct information path through the architecture figure. |
| [Timer-XL, ICLR 2025](https://arxiv.org/abs/2410.04803) | Matrix views can communicate positional or dependency structure more precisely than decorative geometric embeddings. |
| [Fourier Features, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html) | A coordinate is mapped into a high-dimensional bank of sinusoidal features; depicting the feature vector and its basis family is more faithful than drawing an unsupported latent manifold. |
| [CFPT, ICML 2025](https://proceedings.mlr.press/v267/kou25b.html) | Structured 2D tensors can expose periodic/time-coordinate organization more clearly than several detached scalar curves. |

## Future-coordinate decision

The exact method defines

$$
\boldsymbol\Phi
=
[\boldsymbol\phi_1,\ldots,\boldsymbol\phi_T]^\top
\in\mathbb R^{T\times D_q}
$$

from a constant channel and centered DCT basis channels. A 3D manifold projection is therefore visually attractive but mathematically secondary and potentially misleading. The v4 figure instead uses three coupled glyphs:

1. a **coordinate basis atlas**, with future step $\tau$ on the horizontal axis and coordinate channel $d$ on the vertical axis;
2. a highlighted atlas column pulled out as the high-dimensional target coordinate $\boldsymbol\phi_\tau$;
3. contiguous pooling guides that connect coordinate columns to scope-dependent region descriptors $\overline{\boldsymbol\phi}^{(s)}_g$.

This representation directly answers: what the coordinates contain, how one future target is indexed, and how the same field supports both region construction and allocation.

## Visual grammar

- **History and forecasts:** deterministic schematic multiscale waveforms with one strong trajectory and faint close companion frequencies; no dots, uncertainty bands or heavy axes.
- **History state:** a slim variable-wise latent vector with grouped feature bands.
- **Scope matrix:** low-rank stripe matrices, preserving the $D_q\times K$ interpretation.
- **Region descriptor:** scope-dependent contiguous coordinate summaries; segment count decreases from $s_0$ to $s_2$.
- **Region representation:** segmented latent-state ribbons with internal $K$-mode strokes.
- **Step-specific synthesis:** a consistent indigo-to-aqua basis bank; no rainbow map.
- **Scope probabilities:** a smooth $3\times T$ violet probability field and one highlighted probability vector $\boldsymbol\pi_\tau$.
- **Final trajectory:** one strong dark-teal curve after a single weighted contraction.

## Claim and reviewer-risk controls

- The three drawn rows are representative scope paths, not three separately trained models and not an exact implementation count.
- Scope labels do not encode requested horizon.
- Coordinate values and probability intensities are schematic and carry no empirical claim.
- Close companion waveform lines indicate multiscale visual structure only; they are not uncertainty intervals or multiple predictions.
- Region-state sharing does not imply identical step predictions because step-specific synthesis remains explicit.
- The figure contains no BSCA, loss, training-only path or performance claim.

## QA

The deterministic Python renderer and the exported assets passed the following minimal honest verification:

- `nature-figure` preflight: **14 PASS, 0 WARN, 0 FAIL**;
- PDF: one page at 518.4 × 316.8 pt (182.88 × 111.76 mm), within the nominal 183 × 112 mm double-column contract;
- PNG: 2160 × 1320 pixels at approximately 300 dpi;
- TIFF: 4320 × 2640 pixels at 600 dpi with LZW compression;
- SVG: 57 live `<text>` elements, preserving editable labels instead of outlining all text;
- visual inspection: no detected text collision, clipping, broken connector, or ambiguous merge point at review scale;
- semantic inspection: coordinate-channel orientation is explicit, the highlighted atlas column matches $\boldsymbol\phi_\tau$, and scope probabilities contract only the scope-indexed forecasts;
- source-data/statistical checks: not applicable because the figure is a deterministic architecture schematic without empirical observations.

The current manuscript draft and its existing Figure 4 reference remain unchanged. Before manuscript insertion, author review should confirm the final scope-row abstraction and terminology; the production version can then be regenerated from the same script without changing the method contract.
