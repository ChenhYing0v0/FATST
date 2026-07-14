# SC1-D5 Conditioning-Locality Frontier Diagnostic Report

- `decision`: `local_family_headroom_not_supported_basis_component_only`
- `method_training_authorized`: `false`
- selected family counts: `{'block_pca_fit_b96': 15}`

## Gates

- invariants: `{'pass': True, 'fit_count': 585, 'metadata_count': 15, 'selection_count': 15}`
- local headroom: `{'selected_vs_balanced_mse_reduction': 0.00032219822526668995, 'selected_vs_balanced_mae_reduction': -0.0010310942204787565, 'positive_datasets': 2, 'pass': False}`
- global conditioning: `{'selected_vs_dct_mse_reduction': -0.008284433137931924, 'selected_vs_pca_mse_reduction': -0.014722878665364592, 'dct_gap_closure': 0.03759090670432011, 'noninferior_horizons_vs_dct': 5, 'pass': False}`

## Fit-Only Geometry Selection

| Dataset | Checkpoint | Family | Offdiag | Top16 | H48 active |
| --- | ---: | --- | ---: | ---: | ---: |
| ETTh1 | 2021 | block_pca_fit_b96 | 0.863112 | 0.631305 | 96.0 |
| ETTh1 | 2022 | block_pca_fit_b96 | 0.863112 | 0.631305 | 96.0 |
| ETTh1 | 2023 | block_pca_fit_b96 | 0.863112 | 0.631305 | 96.0 |
| ETTh2 | 2021 | block_pca_fit_b96 | 0.902854 | 0.993728 | 96.0 |
| ETTh2 | 2022 | block_pca_fit_b96 | 0.902854 | 0.993728 | 96.0 |
| ETTh2 | 2023 | block_pca_fit_b96 | 0.902854 | 0.993728 | 96.0 |
| ETTm1 | 2021 | block_pca_fit_b96 | 0.832548 | 0.708974 | 96.0 |
| ETTm1 | 2022 | block_pca_fit_b96 | 0.832548 | 0.708974 | 96.0 |
| ETTm1 | 2023 | block_pca_fit_b96 | 0.832548 | 0.708974 | 96.0 |
| ETTm2 | 2021 | block_pca_fit_b96 | 0.892714 | 0.990461 | 96.0 |
| ETTm2 | 2022 | block_pca_fit_b96 | 0.892714 | 0.990461 | 96.0 |
| ETTm2 | 2023 | block_pca_fit_b96 | 0.892714 | 0.990461 | 96.0 |
| Weather | 2021 | block_pca_fit_b96 | 0.325196 | 0.586565 | 96.0 |
| Weather | 2022 | block_pca_fit_b96 | 0.325196 | 0.586565 | 96.0 |
| Weather | 2023 | block_pca_fit_b96 | 0.325196 | 0.586565 | 96.0 |
