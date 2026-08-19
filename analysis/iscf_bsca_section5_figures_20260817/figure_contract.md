# Section 5 Figures: Design and Evidence Contract

## Scope

This bundle creates two manuscript-facing figures from already frozen Section 5 artifacts. It does not launch training, access labels, select checkpoints or modify canonical result tables.

## Figure 6: accuracy and system cost

- **Question:** What practical trade-off follows when one ISCF-BSCA model replaces a four-model horizon-specific service?
- **Conclusion:** In the eight-system display, ISCF-BSCA combines the lowest Main-I macro MSE with moderate checkpoint storage and peak memory, while DLinear remains the lightweight resource counterexample.
- **Evidence:** Main-I MSE over seven datasets and four horizons; four-horizon service checkpoint storage and fresh-process RTX 3090 peak allocated memory from the frozen Efficiency audit.
- **Visual encoding:** horizontal position = four-horizon checkpoint storage on a disclosed log scale; vertical position = Main-I macro MSE; bubble area is directly proportional to peak inference memory; direct labels report exact storage and memory values.
- **Archetype and size:** single comparison hero panel at 7.0 × 3.65 inches for double-column placement.
- **Evidence roles:** ISCF-BSCA, TimeAlign, QDF and AMD use actual trained checkpoint artifacts. DLinear, iTransformer, PatchTST and TimeMixer use official-configuration architecture-equivalent resource footprints; this boundary is disclosed in the caption rather than annotated inside the plot.
- **Display-scope boundary:** The frozen source has nine systems. SimpleTM is excluded from this figure at the author's explicit request, giving an 8/9 display, but remains in the complete Table 3 and audit. The figure therefore cannot be used to claim a uniform resource advantage.

## Figure 7: decoder transfer

- **Question:** Does the complete ISCF-BSCA decoder improve two evaluated backbone realizations within the author-refined three-dataset scope?
- **Conclusion:** The complete framework lowers four-horizon mean MSE for DLinear-style and PatchTST-style backbones on Weather, ETTm1 and ETTm2, as well as their macro average.
- **Evidence:** Author-corrected aggregate transfer table; each dataset bar is the mean over $H\in\{96,192,336,720\}$.
- **Visual encoding:** paired bars compare complete ISCF-BSCA (muted coral, cross hatch) with the Original Decoder (muted blue, diagonal hatch); curved arrows and annotations report relative MSE reduction; panels separate the two backbone families. This is style-only inheritance from the author-provided reference image, with no change to data, ordering or derived statistics.
- **Boundary:** The shared MSE axis begins at 0.20 and carries an explicit break mark so that small paired differences remain legible without implying a zero baseline. No error bars are drawn because the corrected artifact contains aggregate point estimates rather than repeated-run uncertainty. The figure does not attribute gains separately to ISCF or BSCA, and it does not establish universal or architecture-agnostic transfer.

## Export and QA

- Backend: Python/matplotlib, following the saved `nature-figure` backend preference.
- Outputs: editable SVG, vector PDF, 600 dpi TIFF and review PNG.
- Style: restrained print-safe colors, redundant color-and-hatch encoding, direct labels and no decorative gradients or shadows. Figure 6 retains its log-scale storage axis and proportional bubble areas; Figure 7 uses a disclosed truncated MSE axis.
- Source-data bundle: the plotting script exports the exact plotted values and derived relative reductions.
