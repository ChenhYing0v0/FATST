# ISCF scope allocation and varied-horizon output design

## Figure contract

- **Core conclusion:** target-conditioned scope probabilities fuse the scope-conditioned forecast field into one maximum-length trajectory, and varied-horizon requests return nested prefixes of that same trajectory.
- **Figure archetype:** schematic-led method-figure continuation.
- **Backend:** Python / Matplotlib only.
- **Reuse level:** style-only inheritance from the current author-designed main figure; exact tensor semantics follow the frozen ISCF-BSCA-v1 method draft.
- **Target output:** editable SVG/PDF plus 600 dpi TIFF and PNG review assets.
- **Final extension size:** 183 × 84 mm; standalone components use modular widths for manual composition.
- **Empirical data:** none. Curves and probabilities are deterministic schematic glyphs.

## Evidence hierarchy

1. **Scope-probability field:** three violet probability rows extend over a compact future-step axis, while scope-colored row labels preserve correspondence with $s_0,s_1,s_2$. Color intensity encodes probability, and one highlighted future step exposes its normalized probability vector.
2. **Target-wise weighted fusion:** the three forecast slices and the probability vector converge at a compact scope-axis contraction operator.
3. **One prediction trajectory:** fusion produces one trajectory defined over the maximum future domain.
4. **Varied-horizon forecasting:** four start-aligned cropped views terminate at different request endpoints, making the nested-prefix relation explicit.

## Visual-language decision

- Scope identities use the existing light-to-dark blue family.
- The allocation path retains the existing muted violet outline only for routing and tensor boundaries; probability values inherit the corresponding scope color.
- Future-step selection and request endpoints use the existing orange accent.
- The final trajectory uses dark teal with a restrained pale-teal support fill.
- Scope indices are written as $s_0,s_1,s_2$; horizon endpoints are written as $H_1,H_2,H_3,H_4$ so the two concepts cannot be visually conflated.

## V2 revision

- The long blue probability field is replaced by a compact violet tensor. Violet denotes the allocation path, while the blue scope family is retained only in the $s_0,s_1,s_2$ row labels and the forecast slices.
- The component title is shortened from **Target-wise Scope Probability** to **Scope Probability**. Step dependence remains explicit through the future-step cue $\tau$, the tensor shape $T\times3$ and the selected vector $\boldsymbol\pi_{b,c,\tau}$.
- The trajectory and horizon-prefix views are merged into one **Varied-horizon Prediction** panel. Four start-aligned prefix bars and their endpoint guides sit inside the trajectory frame instead of forming a second external section.
- The full schematic probability sequence is aggregated into ten equal-coverage display columns rather than sampled or truncated. This is a visual compression only; every displayed column remains normalized across scopes.

## Claim boundary

The displayed scope probabilities are schematic and are not learned routing statistics. The fused curve is a deterministic target-wise convex mixture used only to explain the computation. The varied-horizon panel illustrates the architectural prefix contract; it is not accuracy evidence.

## Reviewer risks addressed

- A static probability vector would hide target conditioning; the full probability field avoids this.
- Independent horizon curves could imply separate predictions; cropped views of one common trajectory avoid this.
- Similar blue encodings could confuse scope and horizon; orange endpoint markers and symbolic $H_i$ labels separate the concepts.
- A large named fusion box would duplicate the current box-heavy grammar; a compact $\sum_s$ contraction glyph is used instead.

## QA

- [x] Python source passes `nature-figure` static preflight with 14 passes and no warnings or failures.
- [x] Probability columns sum to one within floating-point precision (maximum error $2.22\times10^{-16}$).
- [x] The displayed fused trajectory equals the recorded target-wise convex mixture (maximum error $2.22\times10^{-16}$).
- [x] The four horizon views are exact indexed prefixes of the displayed full trajectory by construction from the same recorded array.
- [x] SVG text remains editable and vector paths remain present in all three SVG outputs.
- [x] Integrated and standalone renders are visually inspected at final size; endpoint labels are not clipped.
- [x] No unrelated working-tree files are modified or staged.
- [x] V2 uses a violet probability tensor distinct from the blue scope matrices and removes the redundant `Target-wise` title text.
- [x] V2 places the trajectory, endpoint guides and all four nested-prefix bars within one framed output panel.
- [x] The redundant upper-right `one trajectory · nested prefixes` annotation is removed; the trajectory and prefix bars carry the relation directly.
