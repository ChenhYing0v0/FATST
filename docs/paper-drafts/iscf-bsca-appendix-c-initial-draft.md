<!-- No visible Appendix C heading is used by author request. -->

Figure C1 presents supplementary varied-horizon forecasting examples for the seven paper-core datasets. For each dataset, two validation samples show the ground-truth future and the fused ISCF-BSCA prediction over the maximum target length $T=720$. The paired rulers and vertical guides mark the four supported prediction lengths, showing how the same predicted trajectory provides the prefixes requested at $H\in\{96,192,336,720\}$.

![Validation-only qualitative varied-horizon forecasts.](../../analysis/iscf_bsca_appendix_c_prediction_export_20260825/outputs/figure_c1_varied_horizon_forecasts.png)

**Figure C1 | Visualization of varied-horizon forecasts across seven datasets.** The dark curve denotes the ground truth and the teal curve denotes the prediction from the frozen unified ISCF-BSCA model. Two examples are shown for each dataset. The examples are selected on the validation split using a deterministic visual-fidelity score that combines train-scale level RMSE (0.70), trajectory-correlation loss (0.15), first-difference-correlation loss (0.10) and relative amplitude error (0.05), after excluding the lowest-variance 20% of channels. The selected origins are separated by at least 720 raw time steps. The figure is an illustrative validation diagnostic; no ablation checkpoint or test label is used.
