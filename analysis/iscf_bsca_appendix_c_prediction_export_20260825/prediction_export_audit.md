# Appendix C prediction export audit

**Date:** 2026-08-25  
**Role:** validation-only qualitative export; no retraining and no formal test  
**Candidate:** `ISCF-BSCA-MAIN-v1` frozen selected profiles

## Export contract

The export used the seven paper-core datasets (`ETTh1`, `ETTh2`, `ETTm1`,
`ETTm2`, `Weather`, `ECL` and `Solar`) and the selected profile manifest
`analysis/iscf_bsca_main_v1_hpo_20260731/final_hpo_freeze_20260806/selected_profile_manifest_final.csv`.
For each dataset, the script loaded the `checkpoint.pt` in the manifest's
`training_artifact_dir`, verified its SHA-256 against
`checkpoint_sha256_before_test`, reconstructed the saved `effective_config.json`,
and evaluated the validation split with `pred_len=720` and horizons
`{96, 192, 336, 720}`. The validation loader was explicitly run with
`shuffle=False`.

The remote execution used `/home/yingch/projects/FATST` at commit
`f47563ff`, `/home/yingch/.conda/envs/r2026-fsa/bin/python`, GPU 0 of an RTX
3090, and data root `/home/yingch/dataset`. No model was trained or modified.
The script rejects a profile whose readout is not
`siff-independent-scope-control`, whose protocol is marked as ablation, or
whose checkpoint hash does not match the frozen manifest.

## Deterministic sample rule

For each validation window, the selection score is the mean of the four
prefix MSEs on channel 0 in the training-standardized scale. The two lowest
scoring windows are selected subject to a minimum separation of 720 raw time
steps; the selected window index, raw forecast origin and per-prefix scores are
recorded in each dataset's `selection.csv` and `metadata.json`. The exported
arrays contain the selected channel's inverse-transformed raw-scale prediction
and ground truth, each with shape `(2, 720)`.

## Selected profiles and outputs

| Dataset | Profile | Trial | Selected validation indices | Raw origins | Checkpoint SHA-256 prefix |
| --- | --- | --- | --- | --- | --- |
| ETTh1 | `h4j_lr3e4` | `ETTh1__h4j_lr3e4` | 8, 936 | 8647, 9575 | `1e402107…` |
| ETTh2 | `h2_lr5e4` | `ETTh2__h2_lr5e4` | 1922, 519 | 10561, 9158 | `bcfbc995…` |
| ETTm1 | `h2_table5_capacity` | `ETTm1__h2_table5_capacity` | 10409, 6519 | 44968, 41078 | `2e8c7e7c…` |
| ETTm2 | `h4m_p6_lr5e5` | `ETTm2__h4m_p6_lr5e5` | 8224, 9248 | 42783, 43807 | `aeb581e8…` |
| Weather | `h4n_seq608_p19_lr2e5` | `Weather__h4n_seq608_p19_lr2e5` | 3914, 0 | 40800, 36886 | `6c6619f0…` |
| ECL | `h2_intermediate_capacity` | `ECL__h2_intermediate_capacity` | 222, 1145 | 18633, 19556 | `97cf661d…` |
| Solar | `h4j_patch4_lr3e4` | `Solar__h4j_patch4_lr3e4` | 1884, 2943 | 38675, 39734 | `42172e2e…` |

The corresponding `appendix_c_predictions.npz` files are the source arrays
for Figure C1. `prediction_sha256sums.txt` provides a local integrity check.
This artifact is illustrative only: it uses validation labels for deterministic
sample selection, does not estimate population prevalence, and does not alter
the Section 5 benchmark tables.
