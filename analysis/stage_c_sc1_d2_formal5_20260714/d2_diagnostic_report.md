# SC1-D2 Frozen-Memory Diagnostic Report

## Decision Summary

| Field | Value |
| --- | --- |
| `role` | `diagnostic_only` |
| `suite` | `formal5` |
| `complete` | `true` |
| `invariant_gate` | `true` |
| `decision` | `scale_alignment_not_supported_reformulate_step2` |
| `method_implementation_authorized` | `false` |

## Per-Dataset Attribution

所有数值均为相对control的validation evaluation-space MSE improvement；正值表示后者更好。

| Dataset | Full vs rank256 | Dense nonlinear vs full | True vs best dense | True vs random-group | True vs random-basis | Group/Basis beaten |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 3.6384% | -11.4316% | 6.8719% | -0.5350% | 3.6791% | 0.33/3 / 3.00/3 |
| ETTh2 | 0.1288% | -20.6786% | 12.5011% | -1.0148% | 5.1077% | 0.00/3 / 3.00/3 |
| ETTm1 | -0.8357% | 1.1714% | -0.8040% | -0.1263% | 0.6057% | 1.33/3 / 3.00/3 |
| ETTm2 | 1.4500% | -1.5785% | 3.6213% | 1.6723% | 4.6970% | 3.00/3 / 3.00/3 |
| Weather | -0.9915% | 0.1597% | 0.4104% | 0.4775% | 1.2278% | 3.00/3 / 3.00/3 |

## Gate Reading

- rank expansion macro：`0.6780%`；
- generic nonlinearity macro：`-6.4715%`；
- true scale vs strongest dense macro：`4.5202%`；
- true scale vs random-group median macro：`0.0947%`；
- true scale vs random-basis median macro：`3.0635%`；
- true scale vs strongest dense MAE macro：`2.9166%`。

[Decision] 当前scale-alignment problem未获支持；回到Step 2重定义Contribution 1问题。

## Failure Attribution Boundary

non-finite loss、basis/Parseval invariant失败、缺arm/seed或official validation参与early stopping时，
结果必须标记`diagnostic_invalid_for_direction_rejection`。若full-affine或dense controls的fit/holdout
明显未收敛，只能怀疑optimization protocol，不能据此否定rank/nonlinearity/scale方向。
