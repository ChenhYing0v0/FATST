# ISCF varied-horizon registered component layers

## Contract

- Backend: Python / Matplotlib.
- Source: the deterministic trajectory used by the current Varied-horizon Prediction design.
- Canvas: 112 × 54 mm for every layer.
- Registration: all assets share identical coordinate limits and can be overlaid without resizing or repositioning.
- Background: transparent outside the requested element.
- Empirical data: none.

## Delivered layers

1. trajectory curve only;
2. rounded background only;
3. $H_1$ prefix bar only;
4. $H_2$ prefix bar only;
5. $H_3$ prefix bar only.

$H_4$ is represented by the complete trajectory and is intentionally omitted from the lower-bar export. Titles, labels, endpoint markers and dashed guides are also omitted.

## QA

- [x] Python source passes `nature-figure` static preflight: 13 pass, 1 expected width warning and 0 fail. The 112 mm width is retained intentionally because all layers must register to the existing V2 panel rather than a full journal column.
- [x] Exactly five registered layer bundles are generated, each in SVG, PDF, PNG and TIFF.
- [x] Every SVG uses the same physical dimensions and `viewBox` (`317.480315 pt × 153.070866 pt`; `0 0 317.480315 153.070866`).
- [x] Every PNG is RGBA on the same 2645 × 1275 px canvas and contains transparent pixels. Component opacity is preserved: the background and three prefix bars are intentionally semi-transparent.
- [x] The curve layer loads the same recorded `fused` array and applies the exact V2 coordinate transform and line styling.
- [x] The overlay preview was visually inspected and contains only the rounded background, trajectory curve and three requested prefix bars.
