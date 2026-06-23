# Phase2-E0 QDF Learned Matrix Audit

## Decision

[Decision] learned matrix audit supports continuing toward a local off-diagonal objective probe.

## Gate

- all_36_matrices_available: `True`
- diag_precision_offdiag_near_zero: `True`
- off_diag_precision_has_interaction: `True`
- all_precision_has_interaction: `True`
- supports_local_offdiag_audit: `True`

## Meta-Type Matrix Summary

| Meta type | Count | Precision offdiag fro share | Covariance offdiag fro share | Precision bandwidth | Covariance condition |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 12 | 0.011602 | 0.126068 | 0.201041 | 6.280086 |
| diag | 12 | 0.000000 | 0.000000 | 0.000000 | 1.055341 |
| off_diag | 12 | 0.013700 | 0.149504 | 0.202526 | 7.085517 |

## Per-Run Precision Off-Diagonal Share

| Meta type | Dataset | Horizon | Precision offdiag fro share | Precision bandwidth | Covariance condition |
| --- | --- | ---: | ---: | ---: | ---: |
| all | ETTh2 | 96 | 0.033635 | 0.181289 | 9.961423 |
| all | ETTh2 | 192 | 0.000633 | 0.227595 | 1.457773 |
| all | ETTh2 | 336 | 0.017420 | 0.180620 | 12.870449 |
| all | ETTh2 | 720 | 0.007884 | 0.198183 | 14.234012 |
| all | ETTm1 | 96 | 0.021802 | 0.171542 | 3.814959 |
| all | ETTm1 | 192 | 0.001449 | 0.236593 | 1.543368 |
| all | ETTm1 | 336 | 0.013087 | 0.208907 | 5.744887 |
| all | ETTm1 | 720 | 0.007435 | 0.222231 | 6.657037 |
| all | Weather | 96 | 0.019644 | 0.171754 | 5.043719 |
| all | Weather | 192 | 0.003776 | 0.218573 | 2.441341 |
| all | Weather | 336 | 0.005543 | 0.200406 | 3.786978 |
| all | Weather | 720 | 0.006916 | 0.194795 | 7.805084 |
| diag | ETTh2 | 96 | 0.000000 | 0.000000 | 1.169220 |
| diag | ETTh2 | 192 | 0.000000 | 0.000000 | 1.002308 |
| diag | ETTh2 | 336 | 0.000000 | 0.000000 | 1.131592 |
| diag | ETTh2 | 720 | 0.000000 | 0.000000 | 1.060420 |
| diag | ETTm1 | 96 | 0.000000 | 0.000000 | 1.062668 |
| diag | ETTm1 | 192 | 0.000000 | 0.000000 | 1.006037 |
| diag | ETTm1 | 336 | 0.000000 | 0.000000 | 1.054870 |
| diag | ETTm1 | 720 | 0.000000 | 0.000000 | 1.031588 |
| diag | Weather | 96 | 0.000000 | 0.000000 | 1.077520 |
| diag | Weather | 192 | 0.000000 | 0.000000 | 1.012325 |
| diag | Weather | 336 | 0.000000 | 0.000000 | 1.018870 |
| diag | Weather | 720 | 0.000000 | 0.000000 | 1.036670 |
| off_diag | ETTh2 | 96 | 0.043866 | 0.190694 | 13.374866 |
| off_diag | ETTh2 | 192 | 0.000645 | 0.227321 | 1.462396 |
| off_diag | ETTh2 | 336 | 0.019900 | 0.185995 | 14.786290 |
| off_diag | ETTh2 | 720 | 0.008512 | 0.199790 | 15.102009 |
| off_diag | ETTm1 | 96 | 0.025695 | 0.173846 | 4.424065 |
| off_diag | ETTm1 | 192 | 0.001496 | 0.236359 | 1.558963 |
| off_diag | ETTm1 | 336 | 0.014587 | 0.208372 | 6.330262 |
| off_diag | ETTm1 | 720 | 0.007953 | 0.221592 | 7.032873 |
| off_diag | Weather | 96 | 0.024357 | 0.176570 | 6.155900 |
| off_diag | Weather | 192 | 0.003967 | 0.217119 | 2.503589 |
| off_diag | Weather | 336 | 0.005961 | 0.198364 | 3.989098 |
| off_diag | Weather | 720 | 0.007457 | 0.194294 | 8.305890 |

## Interpretation

[Fact] QDF stores a `CovarianceMatrix` module. Its loss solves `L x = E`, so the effective residual weighting matrix is the inverse of `Sigma = L L^T`.

[Decision] Local FATST experiments should audit and use the precision/off-diagonal structure, not only the saved covariance visualization.
