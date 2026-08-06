# ISCF standalone forecast-curve assets

## Status

- Status: `local_design_assets_for_author_review`
- Backend: Python / Matplotlib
- Main-figure replacement: no
- Method or manuscript change: no
- Empirical data: none

## Visual contract

The set contains three scope-conditioned forecasts and one final fused forecast. All four assets use the same dimensions, vertical scale, line weight and cool palette as the current ISCF main-figure draft. The three scope curves combine shared trajectory structure with distinct fine-, mid- and broad-scale variations. The final curve is computed as a smooth target-dependent convex mixture of the three scope curves, matching the intended allocation semantics without depicting learned values.

Primary assets use a transparent background and contain no frame, axes, labels or legend. White-background PNG/PDF/TIFF variants are provided for direct review and manuscript assembly.

## Palette

- $s_0$: `#8AB9CA`
- $s_1$: `#68A3B8`
- $s_2$: `#4685A0`
- final fused forecast: `#2D7068`

## Claim boundary

All curve values and fusion weights are deterministic visual glyphs. They do not represent learned forecasts, empirical routing, performance or specialization evidence.

## QA

- [x] Python source passes `nature-figure` preflight with 12 passes, two expected warnings and no failures. The warnings are non-applicable text-size guidance for text-free assets and intentional use of a 78 mm component width rather than a full journal column.
- [x] All four transparent PNGs contain a non-opaque alpha channel (`0–255`).
- [x] Transparent SVG and white-background PNG/PDF/TIFF exports exist for every curve.
- [x] All standalone assets share identical 78 × 24 mm dimensions and plotting limits; the 600 dpi PNGs are 1,842 × 566 px.
- [x] The final curve equals the recorded target-wise convex mixture of the three scope curves within floating-point precision (maximum error $2.22\times10^{-16}$).
- [x] Rendered assets are visually inspected at final size.
