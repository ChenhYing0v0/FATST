# Appendix

## A. EXPERIMENT DETAILS

### A.1 METRIC DETAILS

We use mean squared error (MSE) and mean absolute error (MAE) to evaluate forecasting accuracy. Given the ground-truth values $y_i$ and their predictions $\widehat y_i$, the two metrics are defined as

$$
\operatorname{MSE}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left(y_i-\widehat y_i\right)^2,
\qquad
\operatorname{MAE}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left|y_i-\widehat y_i\right|,
$$

where $N$ is the total number of scalar predictions over all evaluation windows, forecast steps and variables for the requested horizon. Lower MSE and MAE indicate more accurate forecasts.

### A.2 DATASETS

We evaluate ISCF-BSCA on seven widely used multivariate time-series forecasting datasets covering electricity transformers, meteorology, electricity consumption and solar generation. Table A1 summarizes their dimensions, sampling frequencies, chronological splits and evaluation-window counts. Brief descriptions are provided below.

1. **ETT (Electricity Transformer Temperature)** comprises load and temperature measurements collected from electricity transformers between 2016 and 2018 \citep{zhou2021informer}. We use four standard subsets: ETTh1 and ETTh2 are sampled hourly, whereas ETTm1 and ETTm2 are sampled every 15 minutes. Each subset contains seven variables.

2. **Weather** contains 21 meteorological variables recorded at 10-minute intervals in Germany during 2020, including air temperature, humidity, pressure and visibility \citep{wu2023timesnet}.

3. **Electricity (ECL)** records hourly electricity consumption in kilowatt-hours for 321 clients from 2012 to 2014 \citep{wu2023timesnet}.

4. **Solar** contains solar-power measurements collected every 10 minutes from 137 photovoltaic plants in Alabama during 2006 \citep{liu2024itransformer}.

### Table A1 | Dataset Statistics and Split Construction

| Dataset | Variables | Prediction lengths | Raw observations | Sampling frequency | Train boundary | Validation boundary | Test boundary | Train windows | Validation windows |
| --- | ---: | --- | ---: | --- | --- | --- | --- | ---: | ---: |
| ETTh1 | 7 | 96/192/336/720 | 17,420 | 1 h | 0:8,640 | 7,920:11,520 | 10,800:14,400 | 7,201 | 2,161 |
| ETTh2 | 7 | 96/192/336/720 | 17,420 | 1 h | 0:8,640 | 7,920:11,520 | 10,800:14,400 | 7,201 | 2,161 |
| ETTm1 | 7 | 96/192/336/720 | 69,680 | 15 min | 0:34,560 | 33,840:46,080 | 45,360:57,600 | 33,121 | 10,801 |
| ETTm2 | 7 | 96/192/336/720 | 69,680 | 15 min | 0:34,560 | 33,840:46,080 | 45,360:57,600 | 33,121 | 10,801 |
| Weather | 21 | 96/192/336/720 | 52,696 | 10 min | 0:36,887 | 36,167:42,157 | 41,437:52,696 | 35,448 | 4,551 |
| ECL | 321 | 96/192/336/720 | 26,304 | 1 h | 0:18,412 | 17,692:21,044 | 20,324:26,304 | 16,973 | 1,913 |
| Solar | 137 | 96/192/336/720 | 52,560 | 10 min | 0:36,792 | 36,072:42,048 | 41,328:52,560 | 35,353 | 4,537 |

The split boundaries are row indices in the raw series. Validation and test segments include the preceding look-back context required to construct their first input window; this overlap is used only for history construction. The scaler is fitted exclusively on the training segment. The ETT datasets follow the standard fixed chronological splits, whereas Weather, ECL and Solar use chronological 70/10/20 splits.

### A.3 IMPLEMENTATION DETAILS

The local experiments are implemented in Python 3.12.13 and PyTorch 2.9.0 with CUDA 12.8. Each training run uses one NVIDIA GeForce RTX 3090 GPU. ISCF-BSCA is trained from scratch with AdamW and a cosine learning-rate schedule. Table A2 reports the dataset-specific optimization settings, and Table A3 lists the corresponding Encoder interface and ISCF-BSCA configuration.

### Table A2 | Training and Evaluation Settings

| Dataset | Look-back $L$ | Target length $T$ | Initial LR | Batch | Grad. accumulation | Max epochs | Patience |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 720 | 720 | $3\times10^{-4}$ | 32 | 1 | 45 | 10 |
| ETTh2 | 720 | 720 | $5\times10^{-4}$ | 32 | 1 | 30 | 7 |
| ETTm1 | 720 | 720 | $1\times10^{-4}$ | 16 | 2 | 30 | 7 |
| ETTm2 | 720 | 720 | $5\times10^{-5}$ | 16 | 2 | 60 | 12 |
| Weather | 608 | 720 | $2\times10^{-5}$ | 32 | 1 | 120 | 24 |
| ECL | 720 | 720 | $5\times10^{-4}$ | 4 | 8 | 30 | 7 |
| Solar | 720 | 720 | $3\times10^{-4}$ | 16 | 2 | 45 | 10 |

All runs use seed 2021. The checkpoint with the lowest validation mean MSE over $H\in\{96,192,336,720\}$ is retained and serves all four forecasting requests. Weight decay is $0.01$ except for ETTm2, where it is $0.001$, and early stopping uses zero minimum improvement. The dataset-level hyperparameter profiles used in the paper-facing comparison are test-informed; this profile-selection role is distinct from validation-based checkpoint selection.

### Table A3 | ISCF-BSCA Configuration

| Dataset | Scope set | Mode rank | Patch number | $d_{\mathrm{model}}$ | $d_{\mathrm{ff}}$ | Dropout | Weight decay | $\lambda_{\mathrm{scope}}$ | $\lambda_{\mathrm{balance}}^{\max}$ / ramp |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ETTh1 | $\{1,48,144,360,720\}$ | 109 | 24 | 32 | 32 | 0.1 | 0.01 | 1.0 | 0.1 / 0.25 |
| ETTh2 | $\{1,48,144,360,720\}$ | 116 | 12 | 64 | 128 | 0.1 | 0.01 | 1.0 | 0.1 / 0.25 |
| ETTm1 | $\{1,48,144,360,720\}$ | 116 | 1 | 128 | 256 | 0.9 | 0.01 | 1.0 | 0.1 / 0.25 |
| ETTm2 | $\{1,48,144,360,720\}$ | 64 | 6 | 128 | 128 | 0.2 | 0.001 | 1.0 | 0.1 / 0.25 |
| Weather | $\{1,48,144,360,720\}$ | 116 | 19 | 64 | 128 | 0.0 | 0.01 | 1.0 | 0.1 / 0.25 |
| ECL | $\{1,48,144,360,720\}$ | 64 | 1 | 256 | 1,024 | 0.3 | 0.01 | 1.0 | 0.1 / 0.25 |
| Solar | $\{1,48,144,360,720\}$ | 128 | 4 | 256 | 256 | 0.3 | 0.01 | 1.0 | 0.1 / 0.25 |

The scope set and BSCA coefficients are shared across datasets. Specifically, $\lambda_{\mathrm{scope}}=1$ and $\lambda_{\mathrm{balance}}(u)=0.1\min(u/0.25,1)$, where $u\in[0,1]$ denotes normalized optimizer progress.

## B. FULL RESULTS

### B.1 MAIN EXPERIMENTS

Tables B1 and B2 provide the complete results for the two forecasting comparisons in Sections 5.2 and 5.3. The main text reports the four-horizon average for each dataset, whereas the appendix retains MSE and MAE at every evaluated horizon. Table B1 compares one unified ISCF-BSCA model with baselines optimized separately for each requested horizon. Table B2 places every method under the same one-model-all-horizons workflow, where shorter requests are evaluated from the corresponding prefixes of one maximum-length forecast.

<!-- Typeset insertion: analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_tables_author_corrected_20260815/main_i/table_iscf_bsca_main_i_qdf.tex -->

**Table B1 | Full results for the horizon-specific comparison.** Results are reported as MSE and MAE for $H\in\{96,192,336,720\}$, and Avg. is the arithmetic mean over the four horizons. ISCF-BSCA uses one unified model per dataset, whereas the baselines follow their horizon-specific evaluation protocols. The best and second-best displayed values are highlighted in bold and underlined, respectively.

<!-- Typeset insertion: analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_tables_author_corrected_20260815/main_ii/table_iscf_bsca_main_ii.tex -->

**Table B2 | Full results for the one-model-all-horizons comparison.** Each method uses one maximum-horizon model per dataset, and the shorter horizons are evaluated from the corresponding output prefixes. Results are reported as MSE and MAE, and Avg. denotes the arithmetic mean over $H\in\{96,192,336,720\}$. The best and second-best displayed values are highlighted in bold and underlined, respectively.

<!-- Appendix C begins here. No visible section heading is used by author request. -->

Figure C1 presents supplementary varied-horizon forecasting examples for the seven paper-core datasets. For each dataset, two validation samples show the ground-truth future and the fused ISCF-BSCA prediction over the maximum target length $T=720$. The paired rulers and vertical guides mark the four supported prediction lengths, showing how the same predicted trajectory provides the prefixes requested at $H\in\{96,192,336,720\}$.

![Validation-only qualitative varied-horizon forecasts.](../../analysis/iscf_bsca_appendix_c_prediction_export_20260825/outputs/figure_c1_varied_horizon_forecasts.png)

**Figure C1 | Visualization of varied-horizon forecasts across seven datasets.** The dark curve denotes the ground truth and the teal curve denotes the prediction from the frozen unified ISCF-BSCA model. Two examples are shown for each dataset. The examples are selected on the validation split using a deterministic visual-fidelity score that combines train-scale level RMSE (0.70), trajectory-correlation loss (0.15), first-difference-correlation loss (0.10) and relative amplitude error (0.05), after excluding the lowest-variance 20% of channels. The selected origins are separated by at least 720 raw time steps. The figure is an illustrative validation diagnostic; no ablation checkpoint or test label is used.
