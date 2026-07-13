# SC1-D2 Frozen-Memory Diagnostic Report

## Decision Summary

| Field | Value |
| --- | --- |
| `role` | `diagnostic_only` |
| `suite` | `core3` |
| `complete` | `true` |
| `invariant_gate` | `true` |
| `decision` | `core3_precheck_only_formal5_pending` |
| `method_implementation_authorized` | `false` |

## Per-Dataset Attribution

所有数值均为相对control的validation evaluation-space MSE improvement；正值表示后者更好。

| Dataset | Full vs rank256 | Dense nonlinear vs full | True vs best dense | True vs random-group | True vs random-basis | Group/Basis beaten |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | 0.1288% | -20.6786% | 12.5011% | -1.0148% | 5.1077% | 0.00/3 / 3.00/3 |
| ETTm1 | -0.8357% | 1.1714% | -0.8040% | -0.1263% | 0.6057% | 1.33/3 / 3.00/3 |
| Weather | -0.9915% | 0.1597% | 0.4104% | 0.4775% | 1.2278% | 3.00/3 / 3.00/3 |

## Gate Reading

- rank expansion macro：`-0.5661%`；
- generic nonlinearity macro：`-6.4492%`；
- true scale vs strongest dense macro：`4.0358%`；
- true scale vs random-group median macro：`-0.2212%`；
- true scale vs random-basis median macro：`2.3137%`；
- true scale vs strongest dense MAE macro：`2.5806%`。

[Boundary] 本轮是三套已冻结dataset的`core3_precheck`。它可暴露明显机制或数值问题，
但不能形成formal pass，也不能以失败否定方向；ETTh1/ETTm2 profile冻结后必须运行formal5。

## Failure Attribution Boundary

non-finite loss、basis/Parseval invariant失败、缺arm/seed或official validation参与early stopping时，
结果必须标记`diagnostic_invalid_for_direction_rejection`。若full-affine或dense controls的fit/holdout
明显未收敛，只能怀疑optimization protocol，不能据此否定rank/nonlinearity/scale方向。
