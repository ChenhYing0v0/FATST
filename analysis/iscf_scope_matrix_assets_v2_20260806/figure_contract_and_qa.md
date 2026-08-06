# ISCF v2 scope-matrix standalone assets

## Contract

- Core purpose: extract the five scope-matrix glyphs from the ISCF architecture concept v2 as independent composition assets.
- Reuse level: exact visual-logic reuse from the existing Python source, not raster cropping.
- Scope set: $\{1,48,144,360,720\}$.
- Matrix glyph: 4 rows × 6 columns, scope-specific blue family, white cell separators and scope-specific outline.
- Labels: omitted inside the assets; scope identity is carried by the filename and manifest.
- Data role: schematic architecture object only. Cell intensities are deterministic design values, not learned parameters or empirical observations.
- Primary export: transparent-background PNG at 600 dpi; SVG, PDF and white-background TIFF are provided from the same source.
- Method, manuscript and main-figure changes: none.

## QA

- [x] plotting source passes `nature-figure` preflight with 13 passes and no failures;
- [x] exactly five PNG assets exist and are visually distinct;
- [x] transparent PNG alpha channels are present, with alpha extrema $(0,255)$;
- [x] all PNG assets have identical dimensions of 1,086 × 708 pixels;
- [x] SVG/PDF/TIFF companion exports exist for every scope;
- [x] the main v2 concept figure and its source script remain unchanged;
- [x] unrelated working-tree changes remain unstaged and unmodified.

The preflight width warning is accepted because 46 mm is the deliberate size of an independent composition asset, not a submitted single- or double-column figure.
