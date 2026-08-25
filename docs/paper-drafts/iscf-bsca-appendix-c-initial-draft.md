# Appendix C. Qualitative varied-horizon forecasts

Figure C1 provides qualitative validation-only examples of the unified
ISCF-BSCA forecaster across the seven paper-core datasets. Each row contains
two validation samples selected from the frozen dataset-level Main-I/II
profiles. The plots show the ground-truth future and the fused ISCF-BSCA
forecast over the maximum target length $T=720$. A shared nested-prefix ruler
above the grid identifies the four benchmark endpoints
$H\in\{96,192,336,720\}$, while faint vertical guides align those endpoints
with the traces.

The two samples for each dataset were selected deterministically by ranking
validation windows according to the mean scaled MSE over the four prefix
lengths on a fixed channel (channel 0), while enforcing a minimum separation
of 720 raw time steps between selected origins. The figure is therefore an
illustrative view of representative low-error validation trajectories, not an
estimate of population prevalence or an additional test-set result. The
underlying predictions, ground truth, selection scores, raw origins and
checkpoint identifiers are provided in the accompanying source-data artifact.

![Validation-only qualitative varied-horizon forecasts.](../../analysis/iscf_bsca_appendix_c_prediction_export_20260825/outputs/figure_c1_varied_horizon_forecasts.png)

**Figure C1 | Unified ISCF-BSCA forecasts across varied horizons.** Two
validation examples are shown for each paper-core dataset. The dark curve is
the ground truth and the teal curve is the prediction from the frozen unified
ISCF-BSCA model. The nested-prefix ruler marks the four supported horizons;
faint vertical guides align their endpoints in each trace panel. Samples were
selected using the deterministic validation-only rule described above; no
ablation checkpoint or test label was used.
