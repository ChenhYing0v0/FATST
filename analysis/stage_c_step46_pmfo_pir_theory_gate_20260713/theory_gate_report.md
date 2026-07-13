# StageC Step 4-6 PMFO/PIR Theory Gate

## Decision

- PMFO mixed-radix algebra invariant gate: `pass`；
- PIR is an exact block-diagonal quadratic construction for L2, but not the exact deployment risk when cross-scale blocks are non-zero；
- 本报告只验证代数与measure geometry，不构成method effectiveness evidence。

## PMFO Invariants

| Invariant | Max absolute error |
| --- | ---: |
| `orthogonality_max_abs` | 1.332e-15 |
| `projector_sum_max_abs` | 0.000e+00 |
| `projector_idempotence_max_abs` | 2.220e-16 |
| `projector_cross_max_abs` | 1.293e-17 |
| `basis_projector_match_max_abs` | 2.220e-16 |
| `prefix_restriction_max_abs` | 4.441e-16 |
| `refinement_recovery_max_abs` | 8.882e-16 |

## Domain-Local Tree Counts

`active_total_coefficients` 只统计与 requested prefix 相交的 mixed-radix scaling/detail coefficients；不等价于完整模型 FLOPs。

| H | Active coefficients | Boundary overhead | Avoided outside prefix | Active fraction | A6 basis products |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 9 | 710 | 0.0139 | 256 |
| 48 | 52 | 4 | 668 | 0.0722 | 12288 |
| 96 | 104 | 8 | 616 | 0.1444 | 24576 |
| 192 | 199 | 7 | 521 | 0.2764 | 49152 |
| 336 | 342 | 6 | 378 | 0.4750 | 86016 |
| 720 | 720 | 0 | 0 | 1.0000 | 184320 |

## Measure-Induced Geometry

| Measure | First/last weight | Off-block energy | Mean random-risk gap | Max random-risk gap |
| --- | ---: | ---: | ---: | ---: |
| delta_720 | 1.000 | 0.000000 | 0.000000 | 0.000000 |
| uniform_h | 5153.156 | 0.003456 | 0.003091 | 0.019292 |
| log_uniform_h | 852014.320 | 0.205154 | 0.107832 | 0.462696 |
| benchmark_h | 14.393 | 0.002480 | 0.002686 | 0.014503 |

## Boundary

`delta_720` 令 temporal weights 与 identity 成比例，因此 orthogonal refinement blocks 不产生 cross-scale coupling。其他 horizon measures 通常产生非零 off-block operator；PIR 删除这些 blocks，是由 decoder partition 决定的 structured surrogate，不是 raw deployment risk 的等价改写，也没有一般性的 upper/lower-bound 保证。
