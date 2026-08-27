# Figure C1 contract

- **Core conclusion:** The frozen unified ISCF-BSCA model follows the ground-truth
  trajectory on representative validation samples while exposing the nested
  prefixes used by the four supported forecast horizons.
- **Figure archetype:** quantitative grid.
- **Backend:** Python/matplotlib; SVG/PDF text remains editable.
- **Final size:** 183 mm wide, approximately 160 mm high, allowing the
  Appendix C introduction and the full-width figure to remain on one portrait
  manuscript page.
- **Panel map:** seven dataset rows × two deterministic validation samples per
  row. Each panel contains only ground truth and ISCF-BSCA prediction; a shared
  nested-prefix ruler above the grid identifies the four requested horizons,
  while faint vertical guides align their endpoints across panels.
- **Space allocation:** dataset names are rotated vertically and offset to the
  left of the first-column y tick labels. The redundant common `Value` label is
  omitted so that the two trajectory columns occupy the full 183-mm figure
  width.
- **Title policy:** the manuscript provides the `C. VISUALIZATION` hierarchy;
  the image title is `Representative validation trajectories` and does not
  repeat the appendix identifier.
- **Prefix encoding:** the ruler draws four nested horizontal segments ending
  at $H\in\{96,192,336,720\}$, with endpoint labels and a neutral $H=720$
  terminus. One ruler is aligned with each sample column, so the segment lengths
  share the same horizontal geometry as the traces below. This makes the
  varied-horizon relation explicit without repeating prominent dashed lines in
  every trace panel.
- **Channel policy:** one validation-audited channel is fixed per dataset before
  sample ranking; the selected channel identifiers are recorded in the source
  metadata and visual-fidelity audit.
- **Evidence role:** qualitative validation-only illustration, not a population
  prevalence estimate and not a replacement for Section 5 aggregate metrics.
- **Source data:** `figure_c1_source_data.csv`, generated from the frozen
  `appendix_c_predictions.npz` arrays and their provenance metadata.
- **Reviewer risk:** the channel and samples are selected by a deterministic
  visual-fidelity rule using validation labels. This remains documented in the
  source audit, while the manuscript caption identifies the panels as validation
  examples without reproducing the full selection formula.
