# ISCF region-to-forecast synthesis design

## Status

- Status: `local_design_draft_for_author_review`
- Backend: Python / Matplotlib
- Reuse level: style-only inheritance from the current author draft
- Main-figure replacement: no
- Method or manuscript change: no
- Empirical data: none

## Naming decision

`Scope Projection` is not reused after the region representations because it already denotes the upstream history-to-scope-matrix operation. The recommended input name is **Scope-region states**, matching $\mathbf z_g^{(s)}$ and avoiding ambiguity with the earlier history state. The recommended operation name is **Shared step-specific synthesis**, matching the manuscript parameters $\mathbf a_\tau$, $\mathbf n_\tau$ and $\beta_\tau$. The recommended row output is **Scope-conditioned forecasts**, because every row is a slice of the shared forecast field rather than an independently trained forecasting model. Stacking all rows produces the **Scope-indexed forecast field**.

## Visual contract

The continuation reads as:

`region-indexed latent states -> one shared step-specific synthesis module -> region-aware scope-conditioned forecast ribbons`.

- Three rows preserve the current $s_0,s_1,s_2$ presentation.
- One vertically shared module makes parameter sharing across scopes explicit.
- Thick separators in each output ribbon retain the region partition of the input representation.
- Fine separators denote future steps within a region.
- A continuous curve shows that shared region state does not imply identical predictions.
- One highlighted middle-row region states the key relation: one shared $\mathbf z_g^{(s)}$, distinct step predictions.

## Claim boundary

The curves and tensor colors are deterministic visual glyphs. They do not show learned forecasts, specialization, performance or empirical scope allocation.

## QA

- [x] Python source passes `nature-figure` preflight with 14 passes and no warnings or failures.
- [x] SVG text remains editable through explicit `<text>` elements.
- [x] SVG/PDF/PNG/TIFF exports exist.
- [x] Three input region granularities remain distinguishable.
- [x] Region and step separators remain distinguishable at the 183 × 72 mm review size.
- [x] No unrelated working-tree changes are staged or modified.
