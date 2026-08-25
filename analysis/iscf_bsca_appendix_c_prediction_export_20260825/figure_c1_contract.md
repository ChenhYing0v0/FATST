# Figure C1 contract

- **Core conclusion:** The frozen unified ISCF-BSCA model follows the ground-truth
  trajectory on representative validation samples while exposing the nested
  prefixes used by the four supported forecast horizons.
- **Figure archetype:** quantitative grid.
- **Backend:** Python/matplotlib; SVG/PDF text remains editable.
- **Final size:** 183 mm wide, approximately 188 mm high.
- **Panel map:** seven dataset rows × two deterministic validation samples per
  row. Each panel contains only ground truth, ISCF-BSCA prediction and vertical
  horizon markers.
- **Evidence role:** qualitative validation-only illustration, not a population
  prevalence estimate and not a replacement for Section 5 aggregate metrics.
- **Source data:** `figure_c1_source_data.csv`, generated from the frozen
  `appendix_c_predictions.npz` arrays and their provenance metadata.
- **Reviewer risk:** samples are selected by a deterministic low-error rule using
  validation labels; this is disclosed in the source audit and should remain
  explicit in the Appendix C caption.
