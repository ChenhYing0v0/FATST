# SC1-D4 Structured-Basis Diagnostic Report

- `decision`: `standard_structured_basis_explains_gain_return_step2`
- `method_training_authorized`: `false`

## Eight-Horizon Balanced-vs-Control Effects

Positive means balanced interval has lower error.

| Dataset | Control | MSE reduction | MAE reduction | H720 reduction | Positive checkpoints |
| --- | --- | ---: | ---: | ---: | ---: |
| ETTh1 | dct2 | -0.3521% | -0.1309% | -0.1788% | 0/3 |
| ETTh1 | identity | 2.2148% | 1.3444% | 4.5445% | 3/3 |
| ETTh1 | pca_fit | -2.0495% | -1.3172% | -1.8681% | 0/3 |
| ETTh1 | permuted_interval | 0.6267% | 0.6141% | -0.2530% | 3/3 |
| ETTh1 | random_interval_tree | 0.6764% | 0.4091% | 0.4339% | 3/3 |
| ETTh1 | random_orthogonal | 2.1579% | 1.2939% | 3.8433% | 3/3 |
| ETTh2 | dct2 | -1.5377% | -0.7548% | -0.1938% | 0/3 |
| ETTh2 | identity | 5.6656% | 3.6262% | 7.3948% | 3/3 |
| ETTh2 | pca_fit | -2.6858% | -1.4266% | 0.3246% | 0/3 |
| ETTh2 | permuted_interval | 0.3421% | 0.5298% | 0.8647% | 1/3 |
| ETTh2 | random_interval_tree | 0.8958% | 0.4380% | 0.6620% | 3/3 |
| ETTh2 | random_orthogonal | 4.3828% | 2.7418% | 5.6326% | 3/3 |
| ETTm1 | dct2 | -1.7680% | -1.0620% | -0.8100% | 0/3 |
| ETTm1 | identity | 2.2100% | 1.5277% | 0.5021% | 3/3 |
| ETTm1 | pca_fit | -1.4041% | -1.1491% | -0.6173% | 0/3 |
| ETTm1 | permuted_interval | 0.7082% | 0.4515% | -0.0334% | 3/3 |
| ETTm1 | random_interval_tree | -0.1171% | -0.0551% | -0.1536% | 1/3 |
| ETTm1 | random_orthogonal | 1.4252% | 1.1110% | 0.5910% | 3/3 |
| ETTm2 | dct2 | -0.6139% | -0.6489% | -0.2988% | 0/3 |
| ETTm2 | identity | 5.0758% | 3.8081% | 3.0338% | 3/3 |
| ETTm2 | pca_fit | -2.1621% | -1.1308% | -0.3462% | 0/3 |
| ETTm2 | permuted_interval | 5.7395% | 2.7768% | 1.7542% | 3/3 |
| ETTm2 | random_interval_tree | -0.2122% | -0.4050% | -0.1559% | 1/3 |
| ETTm2 | random_orthogonal | 5.1087% | 3.5103% | 2.7284% | 3/3 |
| Weather | dct2 | -0.0443% | 0.0793% | 0.0202% | 2/3 |
| Weather | identity | 2.4193% | 2.9113% | 0.9114% | 3/3 |
| Weather | pca_fit | 0.7410% | 0.7611% | -0.0543% | 3/3 |
| Weather | permuted_interval | 0.6346% | 0.7612% | 0.2516% | 3/3 |
| Weather | random_interval_tree | 0.1235% | 0.2007% | 0.1005% | 2/3 |
| Weather | random_orthogonal | 1.5370% | 1.8954% | 0.7008% | 3/3 |

## Gates

- random replication: `{'h720_reduction': 0.027181410015398755, 'positive_datasets': 5, 'pass': True}`
- global noninferiority: `{'details': {'identity': {'macro_reduction': 0.03529237297436094, 'noninferior_datasets': 5, 'noninferior_horizons': 8, 'pass': True}, 'dct2': {'macro_reduction': -0.008609405298306116, 'noninferior_datasets': 3, 'noninferior_horizons': 0, 'pass': False}, 'pca_fit': {'macro_reduction': -0.015049925950062715, 'noninferior_datasets': 1, 'noninferior_horizons': 0, 'pass': False}}, 'pass': False}`
- locality: `{'mse_macro_reduction': 0.016324379877066808, 'mae_macro_reduction': 0.010306239975098386, 'positive_datasets': 4, 'positive_horizons': 8, 'pass': True}`
- exact balance specificity: `{'mse_macro_reduction': 0.0027424321173513677, 'mae_macro_reduction': 0.0011802414744356282, 'positive_datasets': 3, 'positive_horizons': 7, 'pass': False}`
- invariants: `{'pass': True, 'metadata_count': 15, 'fit_count': 315}`
