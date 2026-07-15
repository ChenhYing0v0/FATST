# SC1-D10 Raw Scale Identifiability Result

## Decision

- `decision`: `raw_aligned_scale_not_supported_rollback_step2`；
- `binary_pass`: `false`；
- `detail_monotone_pass`: `false`；
- `invariant_pass`: `true`；
- method/test/SC2 authorization: `false`。

## Canonical Validation Aggregates

| Dataset | binary interaction | global selectivity | detail selectivity | detail monotone gain | best count | permutation p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | -0.001459 | -0.007613 | 0.004695 | 0.044428 | 2/6 | 0.077670 |
| ETTh2 | -0.000691 | 0.000953 | -0.002334 | 0.035374 | 2/6 | 0.147018 |
| ETTm1 | 0.073690 | 0.156866 | -0.009487 | 0.076495 | 2/6 | 0.011096 |
| ETTm2 | 0.053731 | 0.140091 | -0.032628 | 0.042693 | 1/6 | 0.029126 |
| Weather | 0.000006 | 0.000024 | -0.000013 | -0.000016 | 1/6 | 0.843273 |

## Boundary

D10是raw-data、capacity-matched、validation-only diagnostic。任何positive decision只返回Step4 candidate design，不证明architecture effectiveness；negative decision只关闭aligned-scale routing problem，不否定future-side RGNB/projectivity。
