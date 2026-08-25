# Appendix A/C Data Audit

**Version:** v0.1-source-and-availability-audit
**Date:** 2026-08-25
**Role:** working note for the staged Appendix A and Appendix C build

## 1. Source boundary

The local `PDT_final.pdf` and the primary-source TimeAlign paper are reference materials for dataset-table structure and qualitative-figure conventions. They are not protocol authority for ISCF-BSCA. The authoritative values must come from the frozen FATST configs, selected-profile manifest, trial ledgers and paper-facing table registry.

The TimeAlign paper provides a useful compact pattern: a dataset-description table separated from an implementation/hyperparameter table, with complete benchmark cells placed later in the appendix. Its dataset-size column counts split windows, whereas the PDT table reports raw series length. Appendix A must preserve that distinction.

## 2. Paper-core dataset inventory and loader audit

| Dataset | Variables | Frequency | Raw observations | Train boundary | Validation boundary | Test boundary | Train windows | Validation windows |
| --- | ---: | --- | ---: | --- | --- | --- | ---: | ---: |
| ETTh1 | 7 | 1 h | 17,420 | 0:8,640 | 7,920:11,520 | 10,800:14,400 | 7,201 | 2,161 |
| ETTh2 | 7 | 1 h | 17,420 | 0:8,640 | 7,920:11,520 | 10,800:14,400 | 7,201 | 2,161 |
| ETTm1 | 7 | 15 min | 69,680 | 0:34,560 | 33,840:46,080 | 45,360:57,600 | 33,121 | 10,801 |
| ETTm2 | 7 | 15 min | 69,680 | 0:34,560 | 33,840:46,080 | 45,360:57,600 | 33,121 | 10,801 |
| Weather | 21 | 10 min | 52,696 | 0:36,887 | 36,167:42,157 | 41,437:52,696 | 35,448 | 4,551 |
| ECL | 321 | 1 h | 26,304 | 0:18,412 | 17,692:21,044 | 20,324:26,304 | 16,973 | 1,913 |
| Solar | 137 | source-declared 10 min | 52,560 | 0:36,792 | 36,072:42,048 | 41,328:52,560 | 35,353 | 4,537 |

The paper-facing dataset list excludes Exchange. Exchange remains a limited companion surface in the table registry and should not receive a Figure C1 panel unless the main-text dataset scope changes. The loader audit reconstructed train and validation windows with the paper-facing $L$ and $T$ settings, confirmed finite batches for all seven datasets and did not construct a test loader.

## 3. Frozen selected-profile parameters

The selected-profile manifest and trial ledgers provide the following dataset-level profiles. These values are suitable inputs to Tables A2--A3 after a final field-name audit against the effective training configs.

| Dataset | Selected profile | $L$ | Patch number | $d_{model}$ | $d_{ff}$ | Dropout | Learning rate | Weight decay | Batch | Grad. accumulation | Mode rank | Best epoch | Patience / max epochs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ETTh1 | `h4j_lr3e4` | 720 | 24 | 32 | 32 | 0.1 | $3\times10^{-4}$ | 0.01 | 32 | 1 | 109 | 2 | 10 / 45 |
| ETTh2 | `h2_lr5e4` | 720 | 12 | 64 | 128 | 0.1 | $5\times10^{-4}$ | 0.01 | 32 | 1 | 116 | 1 | 7 / 30 |
| ETTm1 | `h2_table5_capacity` | 720 | 1 | 128 | 256 | 0.9 | $1\times10^{-4}$ | 0.01 | 16 | 2 | 116 | 26 | 7 / 30 |
| ETTm2 | `h4m_p6_lr5e5` | 720 | 6 | 128 | 128 | 0.2 | $5\times10^{-5}$ | 0.001 | 16 | 2 | 64 | 8 | 12 / 60 |
| Weather | `h4n_seq608_p19_lr2e5` | 608 | 19 | 64 | 128 | 0.0 | $2\times10^{-5}$ | 0.01 | 32 | 1 | 116 | 37 | 24 / 120 |
| ECL | `h2_intermediate_capacity` | 720 | 1 | 256 | 1024 | 0.3 | $5\times10^{-4}$ | 0.01 | 4 | 8 | 64 | 29 | 7 / 30 |
| Solar | `h4j_patch4_lr3e4` | 720 | 4 | 256 | 256 | 0.3 | $3\times10^{-4}$ | 0.01 | 16 | 2 | 128 | 22 | 10 / 45 |

The effective patience and maximum-epoch values are taken from the phase configuration that produced each selected profile. The table above remains an audit aid; Appendix A presents the same values without trial identifiers or checkpoint hashes.

All selected profiles use scopes ${1,48,144,360,720}$, seed 2021, the four-horizon validation mean-MSE checkpoint selector and the shared ISCF-BSCA loss coefficients from the frozen protocol. The final table should record these shared decoder values once in a table note and repeat only dataset-specific fields in the body.

## 4. Appendix C prediction-data availability

The repository currently contains complete sample-level diagnostic arrays for several earlier or mechanism-specific probes, but it does not contain a single final-profile prediction export covering all seven paper-core datasets. In particular, no local artifact currently provides the final ISCF-BSCA fused prediction and target arrays needed to select two validation origins for ECL and Solar under the Appendix C rule.

Therefore, Figure C1 is not generated from synthetic or substituted predictions. The next authorized data step is an evaluation-only export from the frozen final profiles, with no retraining and no profile reselection. The export must include the fused $T=720$ forecast, ground truth, dataset, origin, channel and checkpoint identifier. After the export, the selection rule is frozen before ranking examples.

## 5. Remaining checks before manuscript tables

1. Confirm whether Appendix C should plot a fixed channel (recommended: channel 0) or the benchmark’s conventional target channel; record the choice in the selection manifest.
2. Obtain the final-profile prediction export for all seven datasets through an explicitly authorized evaluation-only route.
