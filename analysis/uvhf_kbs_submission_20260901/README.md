# UVHF KBS submission refinement (2026-09-01)

## Scope

This package records the submission-facing refinement requested after the UVHF
naming review was promoted to the canonical KBS manuscript. It changes only
presentation and notation:

The current submission title is **UVHF: Unified Varied-Horizon Forecasting with
Multi-Scope Decoder and Balanced Co-Adaptation**.

- Figure 7 displays `Original Decoder` before `UVHF (MSD + BCA)` within every
  group;
- all numerical values, relative reductions, panels, colours and hatches are
  preserved;
- the manuscript numbers every display equation;
- the BCA balance schedule uses symbolic hyperparameters in the method section,
  while the frozen values remain disclosed in Appendix A.

## Figure 7 artifacts

- Plot source: `plot_uvhf_section5_figures.py`
- Source data: `source_data/figure7_decoder_transfer.csv`
- Vector exports: `outputs/figure_7_decoder_transfer.pdf` and `.svg`
- Raster exports: `outputs/figure_7_decoder_transfer.png` and `.tiff`

The canonical manuscript copy is
`UVHF_KBS_submission/figure_07_uvhf_generalization.pdf`.

## Evidence boundary

No experiment, metric, model output or claim boundary is changed. The plot
revision affects only the left-to-right order of paired bars and the matching
annotation direction.
