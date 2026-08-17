# Section 5 Figures: Design and Evidence Contract

## Scope

This bundle creates two manuscript-facing figures from already frozen Section 5 artifacts. It does not launch training, access labels, select checkpoints or modify canonical result tables.

## Figure 6: accuracy and system cost

- **Question:** What practical trade-off follows when one ISCF-BSCA model replaces a four-model horizon-specific service?
- **Conclusion:** Among the three systems with audited full four-horizon service artifacts, ISCF-BSCA combines the lowest Main-I macro MSE with fewer deployed parameters, while requiring a longer one-epoch cycle.
- **Evidence:** Main-I MSE over seven datasets and four horizons; Efficiency audit totals for deployed parameters and median native one-epoch cycles.
- **Visual encoding:** horizontal position = deployed parameters per dataset after summing four horizon-specific checkpoints for TimeAlign and QDF; vertical position = Main-I macro MSE; bubble area = seven-dataset macro one-epoch cycle seconds; label = number of deployed models and exact one-epoch cycle.
- **Boundary:** The plot is an accuracy--system-cost comparison, not evidence that ISCF-BSCA trains or infers faster. DLinear-$H720$-prefix and PatchTST-$H720$-prefix are excluded because they represent the one-model service protocol evaluated in Main-II rather than a four-model horizon-specific family.

## Figure 7: decoder transfer

- **Question:** Does the complete ISCF-BSCA decoder improve two evaluated backbone realizations within the author-refined three-dataset scope?
- **Conclusion:** The complete framework lowers four-horizon mean MSE for DLinear-style and PatchTST-style backbones on Weather, ETTm1 and ETTm2, as well as their macro average.
- **Evidence:** Author-corrected aggregate transfer table; each dataset bar is the mean over $H\in\{96,192,336,720\}$.
- **Visual encoding:** paired bars compare the Original Decoder and complete ISCF-BSCA; annotations report relative MSE reduction; panels separate the two backbone families.
- **Boundary:** No error bars are drawn because the corrected artifact contains aggregate point estimates rather than repeated-run uncertainty. The figure does not attribute gains separately to ISCF or BSCA, and it does not establish universal or architecture-agnostic transfer.

## Export and QA

- Backend: Python/matplotlib, following the saved `nature-figure` backend preference.
- Outputs: editable SVG, vector PDF, 600 dpi TIFF and review PNG.
- Style: white background, restrained colorblind-conscious palette, direct labels, no decorative gradients or shadows.
- Source-data bundle: the plotting script exports the exact plotted values and derived relative reductions.
