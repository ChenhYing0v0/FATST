# Appendix A. Datasets, Training Protocol and ISCF Configuration

This appendix records the dataset contracts and the settings used for the paper-facing ISCF-BSCA experiments. The raw series length and the number of sliding-window examples are reported separately. The paper-facing ISCF-BSCA runs use the same four horizons, a single seed and validation-based checkpoint selection; the frozen dataset-level profiles are test-informed, as disclosed in the main text.

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

The boundaries are row indices in the raw series. Validation and test windows include the preceding look-back context, which overlaps the preceding split only for history construction; the scaler is fitted on the training segment. The ETT datasets use the standard fixed chronological split, whereas Weather, ECL and Solar use chronological 70/10/20 splits. The evaluation horizons are $\{96,192,336,720\}$, with a maximum target length of $T=720$.

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

All models are trained from scratch with seed 2021. The checkpoint is selected by the lowest validation mean MSE over the four evaluated horizons; the resulting checkpoint serves all four requests. Weight decay is $0.01$ except for ETTm2, where it is $0.001$. Early stopping uses zero minimum improvement. The dataset-level hyperparameter profiles were selected under the frozen test-informed protocol; this selection role is distinct from validation-based checkpoint selection.

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

The scope set, scope-wise loss weight and allocation-balance schedule are shared across datasets. Specifically, $\lambda_{\mathrm{scope}}=1$ and $\lambda_{\mathrm{balance}}(u)=0.1\min(u/0.25,1)$, where $u$ is normalized optimizer progress. Table A3 reports the dataset-specific representation interface and decoder-rank settings required to reproduce the frozen ISCF-BSCA profiles.
