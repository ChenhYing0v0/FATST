# SC1-D3 Crossed Basis-Group Diagnostic Report

## Decision

- `decision`: `basis_main_effect_supported_return_step4`
- `gate_pass`: `true`
- `method_training_authorized`: `false`
- preregistration: `configs/stage_c_sc1_d3_crossed_basis_group.json`

## What Was Tested

补齐D2缺失的`random basis × random group` cell，形成paired 2×2：
`TT=true basis/true group`、`TR=true basis/random group`、
`RT=random basis/true group`、`RR=random basis/random group`。
每个dataset/checkpoint先平均3个structure-seed blocks，因此primary units为15而不是45。

## Dataset Effects

| Dataset | Basis main MSE reduction | True-group conditional | Random-group conditional | Interaction log effect | Interaction not dominant |
| --- | ---: | ---: | ---: | ---: | --- |
| ETTh1 | 3.7081% | 3.5728% | 3.8433% | -0.002809 | pass |
| ETTh2 | 5.3559% | 5.0783% | 5.6326% | -0.005856 | pass |
| ETTm1 | 0.5834% | 0.5758% | 0.5910% | -0.000153 | pass |
| ETTm2 | 3.8430% | 4.9449% | 2.7284% | +0.023050 | pass |
| Weather | 1.0119% | 1.3221% | 0.7008% | +0.006276 | pass |

## Gates

- basis main: `{'macro_log_effect': 0.029608537478282713, 'macro_relative_reduction': 0.029174499029089196, 'datasets_with_at_least_2_positive_checkpoints': 5, 'pass': True}`
- true-group conditional: `{'macro_log_effect': 0.03165941316496726, 'macro_relative_reduction': 0.031163501149838457, 'datasets_with_at_least_2_positive_checkpoints': 5, 'pass': True}`
- random-group conditional: `{'macro_log_effect': 0.027557661791598152, 'macro_relative_reduction': 0.02718141352642678, 'datasets_with_at_least_2_positive_checkpoints': 5, 'pass': True}`
- MAE guard: `{'macro_relative_reduction': 0.023097990045110373, 'pass': True}`
- interaction guard: `{'datasets_where_abs_interaction_le_abs_main': 5, 'pass': True}`
- invariant gate: `{'pass': True, 'd2_metadata_count': 15, 'd3_metadata_count': 15, 'contract_hashes_match': True, 'd3_config_hash_count': 1, 'd2_invariants_pass': True, 'd3_invariants_pass': True}`

## Failure Attribution Boundary

若random-group conditional失败，只能说明D2的basis优势依赖当前group context；
若interaction guard失败，只能说明main-effect解释不足。non-finite、orthogonality failure、
artifact不完整或split污染均使diagnostic失效，不能否定更广义future-aware architecture。
即便全部通过，也只允许返回Step 4设计新的paper-core idea。
