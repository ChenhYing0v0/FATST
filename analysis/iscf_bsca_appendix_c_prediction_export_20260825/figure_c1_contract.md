# Figure C1 contract

- **Core conclusion:** The frozen unified ISCF-BSCA model follows the ground-truth
  trajectory on representative validation samples while exposing the nested
  prefixes used by the four supported forecast horizons.
- **Figure archetype:** quantitative grid.
- **Backend:** Python/matplotlib; SVG/PDF text remains editable.
- **Final size:** 183 mm wide, approximately 188 mm high.
- **Panel map:** seven dataset rows × two deterministic validation samples per
  row. Each panel contains only ground truth and ISCF-BSCA prediction; a shared
  nested-prefix ruler above the grid identifies the four requested horizons,
  while faint vertical guides align their endpoints across panels.
- **Prefix encoding:** the ruler draws four nested horizontal segments ending
  at $H\in\{96,192,336,720\}$, with endpoint labels and a neutral $H=720$
  terminus. This makes the varied-horizon relation explicit without repeating
  prominent dashed lines in every trace panel.
- **Evidence role:** qualitative validation-only illustration, not a population
  prevalence estimate and not a replacement for Section 5 aggregate metrics.
- **Source data:** `figure_c1_source_data.csv`, generated from the frozen
  `appendix_c_predictions.npz` arrays and their provenance metadata.
- **Reviewer risk:** samples are selected by a deterministic low-error rule using
  validation labels; this is disclosed in the source audit and should remain
  explicit in the Appendix C caption.
