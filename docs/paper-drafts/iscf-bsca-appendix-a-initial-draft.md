# A. EXPERIMENT DETAILS

## A.1 METRIC DETAILS

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

## A.2 DATASETS

We evaluate ISCF-BSCA on seven widely used multivariate time-series forecasting datasets spanning electricity transformers, electricity consumption, meteorology and solar generation. These benchmarks differ substantially in variable dimension, sampling frequency and temporal dynamics. Table A1 summarizes their basic statistics.

1. **ETT (Electricity Transformer Temperature)** records seven load and temperature factors from two electricity transformers in two regions of China between 2016 and 2018 \citep{zhou2021informer}. We use its four standard subsets: ETTh1 and ETTh2 are sampled hourly, whereas ETTm1 and ETTm2 are sampled every 15 minutes.

2. **Electricity (ECL)** records the hourly electricity consumption of 321 clients from 2012 to 2014 and is provided by the UCI Machine Learning Repository \citep{wu2023timesnet}.

3. **Weather** is collected by the Beutenberg Weather Station at the Max Planck Institute for Biogeochemistry in Jena, Germany. It contains 21 meteorological indicators sampled every 10 minutes during 2020 \citep{wu2023timesnet}.

4. **Solar** contains solar-power measurements collected every 10 minutes from 137 photovoltaic plants in Alabama during 2006 \citep{liu2024itransformer}.

### Table A1 | Dataset Statistics

| Dataset | Variables | Sampling Frequency | Dataset Size | Domain |
| --- | ---: | --- | ---: | --- |
| ETTh1, ETTh2 | 7 | 1 h | 17,420 | Electricity |
| ETTm1, ETTm2 | 7 | 15 min | 69,680 | Electricity |
| ECL | 321 | 1 h | 26,304 | Electricity |
| Weather | 21 | 10 min | 52,696 | Weather |
| Solar | 137 | 10 min | 52,560 | Solar energy |

The ETT datasets follow their standard fixed chronological splits, whereas Weather, ECL and Solar use chronological 70/10/20 splits. The scaler is fitted exclusively on the training segment. All datasets are evaluated at prediction lengths $H\in\{96,192,336,720\}$.

## A.3 IMPLEMENTATION DETAILS

The local experiments are implemented in Python 3.12.13 and PyTorch 2.9.0 with CUDA 12.8. Each training run uses one NVIDIA GeForce RTX 3090 GPU. ISCF-BSCA is trained from scratch with AdamW and a cosine learning-rate schedule. Table A2 reports the dataset-specific optimization settings, and Table A3 lists the corresponding Encoder interface and ISCF-BSCA configuration.

### Table A2 | Training and Evaluation Settings

| Dataset | Max Length | Initial LR | Batch | Grad. accumulation | Max epochs | Patience |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 720 | $3\times10^{-4}$ | 32 | 1 | 45 | 10 |
| ETTh2 | 720 | $5\times10^{-4}$ | 32 | 1 | 30 | 7 |
| ETTm1 | 720 | $1\times10^{-4}$ | 16 | 2 | 30 | 7 |
| ETTm2 | 720 | $5\times10^{-5}$ | 16 | 2 | 60 | 12 |
| Weather | 720 | $2\times10^{-5}$ | 32 | 1 | 120 | 24 |
| ECL | 720 | $5\times10^{-4}$ | 4 | 8 | 30 | 7 |
| Solar | 720 | $3\times10^{-4}$ | 16 | 2 | 45 | 10 |

All runs use seed 2021. The checkpoint with the lowest validation mean MSE over $H\in\{96,192,336,720\}$ is retained and serves all four forecasting requests. Weight decay is $0.01$ except for ETTm2, where it is $0.001$, and early stopping uses zero minimum improvement.

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
