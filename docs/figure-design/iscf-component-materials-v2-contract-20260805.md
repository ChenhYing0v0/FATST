# ISCF component materials v2: design contract and QA

## Status and scope

- Status: `local_design_draft_for_author_review`
- Backend: Python / Matplotlib
- Method change: no
- Manuscript Figure 4 replacement: no
- Empirical data: none
- Purpose: provide reusable local components for the future ISCF architecture figure.

## Deliverables

1. A three-channel history glyph on white and transparent backgrounds. The curves use a larger within-row amplitude than the previous prototype so that the frame is visually occupied without touching the row boundaries.
2. Two four-channel future-coordinate curve sets:
   - `exact`: the frozen $D_q=4$ coordinate definition in Section 4.2;
   - `design`: a frequency-separated schematic used only to test the visual language of the architecture figure.
3. A revised coordinate-region component with 15 narrow regions, one base hue per coordinate row, opacity encoding of region values, and a compact four-square region descriptor.

## Coordinate fidelity boundary

The exact coordinate curves are

$$
\widetilde\phi_{\tau,d}=\cos\!\left(\frac{\pi(\tau-\tfrac12)d}{T}\right),
$$

with $\phi_{\tau,0}=1$ and centered, $\sqrt{2}$-scaled nonconstant channels. Their relative frequencies are fixed by the frozen implementation and must not be visually altered in a method-faithful panel.

The frequency-separated design curves use one constant channel followed by 0.75, 2, and 5 cycles across the displayed future domain. They are explicitly schematic and must not be described as the implemented coordinate basis. The revised region component currently uses this schematic set because its purpose is author-side visual evaluation.

## Visual encoding

- Coordinate-row hues: indigo, blue, teal, and muted ochre.
- Region fill: one hue per row; value $v\in[-1,1]$ maps to opacity $0.10+0.78(v+1)/2$.
- Curve overlay: a darker tone of the corresponding row hue.
- Regions: 15 equal-width cells, matching the valid $s=48$ partition count when $T=720$.
- Selected descriptor: four compact squares placed immediately to the right of the field; each square inherits the hue and opacity of its source row.
- No rainbow map, no red--green contrast, and no empirical or learned quantity is implied.

## Intended reading path

`future-step coordinate curves -> contiguous region partition -> row-wise region averaging -> one four-dimensional region descriptor`.

## Reviewer risks and controls

- Risk: readers may interpret the frequency-separated design curves as the implemented basis.
  - Control: retain the exact curve asset, mark the design asset as schematic in its filename and manifest, and do not promote it to the manuscript without an explicit fidelity decision.
- Risk: opacity alone may obscure the signed-value meaning.
  - Control: include a compact $-1,0,+1$ opacity key in the full coordinate component.
- Risk: the constant row remains uniformly dark.
  - Control: this is intentional and communicates the intercept-like coordinate; the other three rows carry the varying regional pattern.

## QA checklist

- [x] Python source passes the `nature-figure` validator with no `FAIL`.
- [x] White-background PNGs are visually inspected at full size and as a compact contact sheet.
- [x] Transparent PNG alpha channel is present.
- [x] SVG text remains editable through `svg.fonttype = none` and `<text>` elements.
- [x] PDF, SVG, PNG, and TIFF exports exist for white-background assets.
- [x] Exact and schematic coordinate roles are correctly recorded in the manifest.
- [x] No tracked manuscript text or frozen method definition is changed.

The validator's width warning is accepted because these are modular composition assets at 118 mm, not a submitted single- or double-column figure. Their final dimensions will be set when the complete Figure 4 layout is assembled.
