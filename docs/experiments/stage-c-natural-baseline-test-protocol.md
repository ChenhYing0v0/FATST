# StageC A6-LBF Natural Baseline Test Protocol

| Field | Value |
| --- | --- |
| `candidate` | `A6-LBF-natural-baseline` |
| `role` | frozen reference; control-only |
| `current_step` | StageC baseline establishment before SC1/SC2 method experiments |
| `selection` | already completed on validation; test cannot change profile |
| `checkpoint` | restored best-validation `checkpoint.pt` |
| `datasets` | Weather, ETTm1, ETTh2 |
| `seeds` | 2021, 2022, 2023 |
| `horizons` | 48, 96, 144, 192, 288, 336, 512, 720 |
| `metrics` | per-seed and three-seed mean/std/CV MSE and MAE |

## Gate

必须满足9/9 checkpoints匹配frozen contract、72/72 test rows完整、所有值finite、contract hash唯一。该实验
没有performance pass/fail阈值；其decision只能是`frozen_test_reference_ready`或`analysis_incomplete`。

## Claim Boundary

这是post-freeze evaluation，不是test-driven calibration。后续method和matched control必须复用同一natural
profile和seed protocol。任何baseline弱点都不能触发profile重选，只能作为method effect-size的参考。
