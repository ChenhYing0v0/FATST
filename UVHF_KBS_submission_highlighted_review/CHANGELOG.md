# UVHF naming revision review package

## Review use

- The source manuscript is `elsarticle-template-num.tex`.
- Blue text marks material introduced or rewritten in this naming revision.
- Set `\showrevisionsfalse` in the preamble to obtain a clean, uncoloured view.
- The compiled review PDF is `output/pdf/UVHF_KBS_submission_highlighted_review.pdf`.
- The original `HoriScope_KBS_submission/` package is preserved unchanged.

## Terminology contract

| Role | Canonical term |
|---|---|
| Research task | unified varied-horizon forecasting |
| Proposed framework/model | UVHF |
| Decoder architecture | Multi-Scope Decoder (MSD) |
| Training strategy | Balanced Co-Adaptation (BCA) |
| Structural property | cross-horizon prefix consistency (CHPC) |

The task name is deliberately written in full and is not abbreviated as UVHF.
This keeps `UVHF` unambiguous whenever it appears in the manuscript.

## Revised figures

| Manuscript role | Revised asset | Visible change |
|---|---|---|
| Accuracy and system cost | `figure_05_uvhf_accuracy_system_cost.pdf` | `UVHF` replaces the former model label |
| Generalization studies | `figure_07_uvhf_generalization.pdf` | treatment is identified as `UVHF (MSD + BCA)` |
| Appendix visualization | `figure_c1_uvhf_varied_horizon_forecasts.pdf` | prediction legend is identified as `UVHF` |
| Method overview | `figure_04_uvhf_msd_overview.png` | independent review asset; caption defines the displayed architecture as the MSD |
| Scope behavior | `figure_06_uvhf_scope_allocation_behavior.pdf` | independent review asset; caption identifies the complete framework as UVHF |

All revised quantitative figures retain the frozen source data, geometry,
metrics and statistical values of the original paper-facing assets.
