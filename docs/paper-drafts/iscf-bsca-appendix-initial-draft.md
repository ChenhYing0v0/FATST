# Appendix

## Appendix A. Datasets, training protocol and ISCF configuration

This appendix records the dataset contracts and the settings used for the paper-facing ISCF-BSCA experiments. Raw series lengths and sliding-window counts are reported separately. The paper-facing ISCF-BSCA runs use the same four horizons, one seed and validation-based checkpoint selection; the frozen dataset-level profiles are test-informed, as disclosed in the main text.

### Table A1 | Dataset metadata and split construction

| Dataset | Variables | Sampling frequency | Raw observations | Train boundary | Validation boundary | Test boundary | Train windows | Validation windows |
| --- | ---: | --- | ---: | --- | --- | --- | ---: | ---: |
| ETTh1 | 7 | 1 h | 17,420 | 0:8,640 | 7,920:11,520 | 10,800:14,400 | 7,201 | 2,161 |
| ETTh2 | 7 | 1 h | 17,420 | 0:8,640 | 7,920:11,520 | 10,800:14,400 | 7,201 | 2,161 |
| ETTm1 | 7 | 15 min | 69,680 | 0:34,560 | 33,840:46,080 | 45,360:57,600 | 33,121 | 10,801 |
| ETTm2 | 7 | 15 min | 69,680 | 0:34,560 | 33,840:46,080 | 45,360:57,600 | 33,121 | 10,801 |
| Weather | 21 | 10 min | 52,696 | 0:36,887 | 36,167:42,157 | 41,437:52,696 | 35,448 | 4,551 |
| ECL | 321 | 1 h | 26,304 | 0:18,412 | 17,692:21,044 | 20,324:26,304 | 16,973 | 1,913 |
| Solar | 137 | source-declared 10 min | 52,560 | 0:36,792 | 36,072:42,048 | 41,328:52,560 | 35,353 | 4,537 |

The boundaries are row indices in the raw series. Validation and test windows include the preceding look-back context for history construction; the scaler is fitted on the training segment. The ETT datasets use the standard fixed chronological split, whereas Weather, ECL and Solar use chronological 70/10/20 splits. The evaluated horizons are $\mathcal H=\{96,192,336,720\}$, with maximum target length $T=720$.

### Table A2 | Shared training and evaluation settings

| Dataset | Look-back $L$ | Target length $T$ | Evaluated horizons | Optimizer | Initial LR | Schedule | Batch | Grad. accumulation | Max epochs | Patience |
| --- | ---: | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| ETTh1 | 720 | 720 | 96/192/336/720 | AdamW | $3\times10^{-4}$ | cosine | 32 | 1 | 45 | 10 |
| ETTh2 | 720 | 720 | 96/192/336/720 | AdamW | $5\times10^{-4}$ | cosine | 32 | 1 | 30 | 7 |
| ETTm1 | 720 | 720 | 96/192/336/720 | AdamW | $1\times10^{-4}$ | cosine | 16 | 2 | 30 | 7 |
| ETTm2 | 720 | 720 | 96/192/336/720 | AdamW | $5\times10^{-5}$ | cosine | 16 | 2 | 60 | 12 |
| Weather | 608 | 720 | 96/192/336/720 | AdamW | $2\times10^{-5}$ | cosine | 32 | 1 | 120 | 24 |
| ECL | 720 | 720 | 96/192/336/720 | AdamW | $5\times10^{-4}$ | cosine | 4 | 8 | 30 | 7 |
| Solar | 720 | 720 | 96/192/336/720 | AdamW | $3\times10^{-4}$ | cosine | 16 | 2 | 45 | 10 |

All models are trained from scratch with seed 2021. The checkpoint is selected by the lowest validation mean MSE over the four evaluated horizons, and the selected checkpoint serves all four requests. Weight decay is $0.01$ except for ETTm2, where it is $0.001$; early stopping uses zero minimum improvement. Dataset-level hyperparameter profiles were selected under the frozen test-informed protocol, which is distinct from validation-based checkpoint selection.

### Table A3 | Dataset-specific ISCF-BSCA configuration

| Dataset | Scope set | Mode rank | Patch number | $d_{\mathrm{model}}$ | $d_{\mathrm{ff}}$ | Dropout | Weight decay | $\lambda_{\mathrm{scope}}$ | $\lambda_{\mathrm{balance}}^{\max}$ / ramp |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ETTh1 | $\{1,48,144,360,720\}$ | 109 | 24 | 32 | 32 | 0.1 | 0.01 | 1.0 | 0.1 / 0.25 |
| ETTh2 | $\{1,48,144,360,720\}$ | 116 | 12 | 64 | 128 | 0.1 | 0.01 | 1.0 | 0.1 / 0.25 |
| ETTm1 | $\{1,48,144,360,720\}$ | 116 | 1 | 128 | 256 | 0.9 | 0.01 | 1.0 | 0.1 / 0.25 |
| ETTm2 | $\{1,48,144,360,720\}$ | 64 | 6 | 128 | 128 | 0.2 | 0.001 | 1.0 | 0.1 / 0.25 |
| Weather | $\{1,48,144,360,720\}$ | 116 | 19 | 64 | 128 | 0.0 | 0.01 | 1.0 | 0.1 / 0.25 |
| ECL | $\{1,48,144,360,720\}$ | 64 | 1 | 256 | 1,024 | 0.3 | 0.01 | 1.0 | 0.1 / 0.25 |
| Solar | $\{1,48,144,360,720\}$ | 128 | 4 | 256 | 256 | 0.3 | 0.01 | 1.0 | 0.1 / 0.25 |

The scope set, scope-wise loss weight and allocation-balance schedule are shared across datasets. Specifically, $\lambda_{\mathrm{scope}}=1$ and $\lambda_{\mathrm{balance}}(u)=0.1\min(u/0.25,1)$, where $u$ is normalized optimizer progress. Table A3 reports the dataset-specific representation-interface and decoder-rank settings required to reproduce the frozen ISCF-BSCA profiles.

## Appendix B. Complete horizon-wise benchmark results

The main text reports four-horizon dataset means to keep Tables 1 and 2 compact. Appendix B retains the complete displayed result cells for every paper-core dataset, benchmark horizon and metric. The tables preserve the frozen three-decimal values, the arithmetic four-horizon averages and all negative cells; they do not alter the source-role interpretation or the system-level claim boundary of the main comparisons.

### Table B1 | Complete horizon-specific comparison

Table B1 reports MSE and MAE for ISCF-BSCA and the horizon-specific baselines at $H\in\{96,192,336,720\}$, together with the four-horizon average for each of the seven datasets. The canonical manuscript table is available as [Table B1 LaTeX source](../../analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_tables_author_corrected_20260815/main_i/table_iscf_bsca_main_i_qdf.tex), with a standalone compilation at [Table B1 standalone source](../../analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_tables_author_corrected_20260815/main_i/table_iscf_bsca_main_i_standalone.tex).

<!-- Typeset insertion: include the complete contents of the canonical Table B1 LaTeX source at this position. -->

**Table B1 | Complete horizon-specific forecasting results.** Each dataset block contains the four evaluated horizons and an `Avg.` row computed as the arithmetic mean over those horizons. ISCF-BSCA uses one unified model per dataset, whereas the baseline entries follow the horizon-specific protocol encoded in the corresponding source contracts. Best and second-best values follow the common three-decimal ranking rule. This table is a complete horizon-wise system comparison rather than matched mechanism attribution.

### Table B2 | Complete one-model-all-horizons comparison

Table B2 reports MSE and MAE for the one-model-all-horizons evaluation at the same four horizons and for the same seven paper-core datasets. Each method contributes one fixed maximum-horizon checkpoint per dataset, and the requested shorter forecasts are evaluated from the corresponding output prefixes under the frozen H720-prefix protocol. The canonical manuscript table is available as [Table B2 LaTeX source](../../analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_tables_author_corrected_20260815/main_ii/table_iscf_bsca_main_ii.tex), with a standalone compilation at [Table B2 standalone source](../../analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_tables_author_corrected_20260815/main_ii/table_iscf_bsca_main_ii_standalone.tex).

<!-- Typeset insertion: include the complete contents of the canonical Table B2 LaTeX source at this position. -->

**Table B2 | Complete one-model-all-horizons forecasting results.** Each dataset block contains the four prefix evaluations and an `Avg.` row. The table preserves the frozen source-native evaluation contracts and is intended to expose the complete horizon-wise audit surface behind Table 2. Because the external source contracts are not matched architectural interventions, the table supports system-level unified-horizon comparison rather than component or decoder attribution.

## Appendix C. Qualitative varied-horizon forecasts

Figure C1 provides qualitative validation-only examples of the unified ISCF-BSCA forecaster across the seven paper-core datasets. Each row contains two validation samples selected from the frozen dataset-level profiles used for the main comparisons. The plots show the ground-truth future and the fused ISCF-BSCA forecast over the maximum target length $T=720$. A nested-prefix ruler above each sample column identifies the four benchmark endpoints $H\in\{96,192,336,720\}$, while faint vertical guides align those endpoints with the traces below.

To select visually faithful examples without relying on a single absolute error, one channel per dataset is fixed after excluding the lowest-variance 20% of channels. For each eligible validation window, we compute the four-horizon visual-fidelity score $S_{\mathrm{vis}}=0.70E_{\mathrm{level}}+0.15E_{\mathrm{corr}}+0.10E_{\Delta\mathrm{corr}}+0.05E_{\mathrm{amp}}$, where the terms are train-scale level RMSE, trajectory-correlation loss, first-difference-correlation loss and relative amplitude error, respectively, averaged over the four prefixes. Lower values indicate closer visual and numerical agreement. The two lowest-scoring windows are retained subject to a minimum separation of 720 raw time steps between selected forecast origins.

The resulting figure is an illustrative validation diagnostic rather than an estimate of population prevalence or an additional test-set result. The underlying predictions, ground truth, selected channel and sample scores, raw origins and checkpoint identifiers are provided in the accompanying source-data artifact.

![Validation-only qualitative varied-horizon forecasts.](../../analysis/iscf_bsca_appendix_c_prediction_export_20260825/outputs/figure_c1_varied_horizon_forecasts.png)

**Figure C1 | Unified ISCF-BSCA forecasts across varied horizons.** Two validation examples are shown for each paper-core dataset. The dark curve is the ground truth and the teal curve is the prediction from the frozen unified ISCF-BSCA model. The paired nested-prefix rulers mark the four supported horizons, and faint vertical guides align their endpoints in each trace panel. Samples were selected using the deterministic validation-only visual-fidelity rule described above; no ablation checkpoint or test label was used.

The editable vector figure and source data are available as [Figure C1 SVG](../../analysis/iscf_bsca_appendix_c_prediction_export_20260825/outputs/figure_c1_varied_horizon_forecasts.svg), [Figure C1 PDF](../../analysis/iscf_bsca_appendix_c_prediction_export_20260825/outputs/figure_c1_varied_horizon_forecasts.pdf) and [Figure C1 source-data CSV](../../analysis/iscf_bsca_appendix_c_prediction_export_20260825/outputs/figure_c1_source_data.csv).
