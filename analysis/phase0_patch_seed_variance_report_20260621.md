# Phase0 PatchEncoderFixedHead Seed Variance Report

## Run Metadata

- [Fact] Remote host: `529_Lab-3090`
- [Fact] Code commit: `03602a605586f05feb79b5d7875707c944bad428`
- [Fact] GPU: physical GPU 2
- [Fact] Output root:
  `/home/yingch/exp_outputs/r-2026-fatst/phase0_patch_seed_variance`
- [Fact] Log:
  `/home/yingch/exp_outputs/r-2026-fatst/phase0_patch_seed_variance/_logs/phase0_patch_seed_variance_20260620_232504.nohup.log`
- [Fact] Matrix:
  `PatchEncoderFixedHead × {ETTh2, ETTm1, Weather} × {96,720} × {2021,2022,2023}`
- [Fact] Status: 18/18 complete

Raw metrics:
`analysis/phase0_patch_seed_variance_metrics_20260621.csv`

Summary:
`analysis/phase0_patch_seed_variance_summary_20260621.csv`

## Summary

| Dataset | Horizon | MSE mean | MSE std | MSE CV | MAE mean | MAE std | MAE CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | 0.305870 | 0.001616 | 0.53% | 0.361330 | 0.004329 | 1.20% |
| ETTh2 | 720 | 0.406913 | 0.003917 | 0.96% | 0.444059 | 0.003504 | 0.79% |
| ETTm1 | 96 | 0.292593 | 0.003234 | 1.11% | 0.345234 | 0.002257 | 0.65% |
| ETTm1 | 720 | 0.415329 | 0.002619 | 0.63% | 0.421685 | 0.001591 | 0.38% |
| Weather | 96 | 0.149441 | 0.003282 | 2.20% | 0.197653 | 0.003857 | 1.95% |
| Weather | 720 | 0.329556 | 0.008127 | 2.47% | 0.339655 | 0.004476 | 1.32% |

## Interpretation

[Strong Evidence] `PatchEncoderFixedHead` has no obvious seed instability on the Phase0 lite
variance matrix. The largest MSE coefficient of variation is `2.47%` on `Weather / 720`; all other
settings are below `2.20%`.

[Strong Evidence] The seed means remain consistent with the Phase0 gate decision: the selected base
is stable enough to serve as the Phase1 internal backbone.

[Inference] For Phase1 mechanism experiments, a single-seed gate is acceptable for quick screening,
but final claims should use at least the same three-seed protocol on sensitive long-horizon settings,
especially `Weather / 720`.

## Phase0 Decision

[Fact] Phase0 internal base is finalized as `PatchEncoderFixedHead`.

[Fact] This base is a clean PatchTST-style internal backbone, not an exact PatchTST paper
reproduction. Paper-facing PatchTST/DLinear comparisons should be reproduced through native upstream
code before final manuscript claims.

[Strong Evidence] Phase1 can now start with Variable-Horizon Decoder experiments on top of
`PatchEncoderFixedHead`.
