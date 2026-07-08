# Phase5 StageB B12-STBO Small Gate Deep Analysis

## Scope

本报告分析 B12-STBO remote small gate returned artifacts。

| Field | Value |
| --- | --- |
| `candidate_id` | `B12-STBO` |
| `current_step` | StageB Step 9-10 |
| `remote_output_root` | `/tmp/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b12_stbo_small_gate` |
| `local_analysis_root` | `analysis/phase5_stage_b_b12_stbo_small_gate_20260708` |
| `datasets` | `ETTh2`, `ETTm1`, `Weather` |
| `horizons` | `96`, `192`, `336`, `720` |
| `arms` | `a6_clean`, `stbo_shared`, `stbo_bank4`, `stbo_dct`, `stbo_independent` |
| `seed` | `2021` |

The gate is complete: all 15 runs produced `metrics_by_target_horizon.csv`.

## Executive Decision

[Decision] `blocked_by_required_controls`.

B12-STBO as currently implemented does not pass the effectiveness gate or the mechanism gate:

1. all STBO variants are worse than the A6 clean anchor on mean MSE;
2. learned STBO does not beat fixed local DCT;
3. `stbo_bank4` does not learn meaningful tile-bank specialization;
4. `stbo_independent` does not rescue the operator, so the failure is not simply that shared/bank constraints are too strong.

This rejects the tested B12-STBO implementation, not the broader native multi-horizon operator direction.

## Anchor Stability

The `a6_clean` metrics in this B12 run exactly match the validated clean A6 rerun from
`analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706/`.

| Dataset | Horizon | B12 A6 MSE | Clean A6 MSE | Relative Diff |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | `0.244162` | `0.244162` | `+0.000%` |
| ETTh2 | 192 | `0.286193` | `0.286193` | `+0.000%` |
| ETTh2 | 336 | `0.314366` | `0.314366` | `+0.000%` |
| ETTh2 | 720 | `0.395028` | `0.395028` | `+0.000%` |
| ETTm1 | 96 | `0.272766` | `0.272766` | `+0.000%` |
| ETTm1 | 192 | `0.309146` | `0.309146` | `+0.000%` |
| ETTm1 | 336 | `0.346930` | `0.346930` | `+0.000%` |
| ETTm1 | 720 | `0.408370` | `0.408370` | `+0.000%` |
| Weather | 96 | `0.141368` | `0.141368` | `+0.000%` |
| Weather | 192 | `0.182400` | `0.182400` | `+0.000%` |
| Weather | 336 | `0.231650` | `0.231650` | `+0.000%` |
| Weather | 720 | `0.303046` | `0.303046` | `+0.000%` |

Therefore, the result is not caused by baseline drift.

## Main Metric Result

Mean MSE relative to A6:

| Arm | ETTh2 | ETTm1 | Weather | Overall | Wins vs A6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stbo_shared` | `+3.36%` | `+1.27%` | `+0.15%` | `+1.59%` | `0/12` |
| `stbo_bank4` | `+4.59%` | `+1.36%` | `-0.005%` | `+1.98%` | `3/12` |
| `stbo_dct` | `+3.77%` | `+0.91%` | `+0.03%` | `+1.57%` | `2/12` |
| `stbo_independent` | `+3.30%` | `+1.84%` | `+0.19%` | `+1.78%` | `0/12` |

Best arm count over 12 dataset-horizon settings:

| Arm | Best Count |
| --- | ---: |
| `a6_clean` | `9/12` |
| `stbo_bank4` | `3/12` |
| `stbo_shared` | `0/12` |
| `stbo_dct` | `0/12` |
| `stbo_independent` | `0/12` |

The only STBO wins are `stbo_bank4` on Weather H96/H192/H336, with tiny margins:

| Dataset | Horizon | A6 MSE | `stbo_bank4` MSE | Relative |
| --- | ---: | ---: | ---: | ---: |
| Weather | 96 | `0.141368` | `0.141216` | `-0.107%` |
| Weather | 192 | `0.182400` | `0.182188` | `-0.116%` |
| Weather | 336 | `0.231650` | `0.231573` | `-0.033%` |

This is too small and too dataset-specific for a paper-core method.

## DCT Control Reading

The key mechanism question is whether learned local bases outperform a fixed local DCT basis. They do not.

| Comparison | ETTh2 | ETTm1 | Weather | Overall | Wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| `stbo_shared` vs `stbo_dct` | `-0.40%` | `+0.36%` | `+0.12%` | `+0.03%` | `2/12` |
| `stbo_bank4` vs `stbo_dct` | `+0.78%` | `+0.44%` | `-0.04%` | `+0.40%` | `4/12` |

Interpretation:

- `stbo_shared` is essentially tied with DCT on mean MSE and loses on MAE;
- `stbo_bank4` is worse than DCT overall;
- the small Weather gain of `stbo_bank4` is also mirrored by DCT being close to A6.

Therefore, the learned local basis mechanism is not distinct from a generic smooth local basis control.

## Independent-Tile Control Reading

`stbo_independent` is not better than A6 and is not a clear upper bound in training:

| Arm | Overall vs A6 | Wins vs A6 |
| --- | ---: | ---: |
| `stbo_independent` | `+1.78%` | `0/12` |

This means the observed failure is not simply:

```text
shared/bank constraint too strong, independent tile would solve it
```

The independent control has more local basis parameters than shared/bank but still underperforms. The problem is deeper:
the current STBO factorization does not recover A6's useful full-trajectory basis behavior under the current rank and
training setup.

## Bank Specialization Diagnostic

`stbo_bank4` was supposed to learn a soft future-tile mixture over four local basis banks. The trained normalized bank
entropy is nearly maximum:

| Dataset | `stbo_tile_bank_entropy_mean` |
| --- | ---: |
| ETTh2 | `0.9990` |
| ETTm1 | `0.9992` |
| Weather | `0.9995` |

This means tile-bank mixture remains almost uniform. The bank route did not learn stage/tile specialization.

Failure attribution for `stbo_bank4`:

- `hypothesis_false`: not proven;
- `intervention_point_wrong`: possible but not the main evidence here;
- `readout_or_head_design_wrong`: likely, because the bank mixture has symmetry/non-identifiability and no effective
  specialization pressure;
- `optimization_or_numeric_pathology`: no divergence, but bank logits are functionally inactive;
- `capacity_control_explains`: not in the simple sense, because independent tile also fails.

## Capacity Confound

STBO arms have fewer total parameters than A6:

| Dataset | `stbo_shared` | `stbo_bank4` | `stbo_dct` | `stbo_independent` |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | `-12.32%` | `-12.18%` | `-12.36%` | `-11.68%` |
| ETTm1 | `-21.21%` | `-20.95%` | `-21.30%` | `-20.00%` |
| Weather | `-4.51%` | `-4.47%` | `-4.52%` | `-4.34%` |

This matters. The gate does not prove that all tiled-basis operators are inferior. It proves that the current
`tile_len=48`, `rank=16` STBO is not a fair replacement for A6-LBF-r256 as a paper-core method.

However, the DCT result limits the value of a simple capacity-equalized rerun: even if higher rank recovers performance,
the method still needs to beat fixed local DCT and show a non-generic learned-basis mechanism.

## Dataset Pattern

The failure is strongest on ETTh2 and ETTm1:

- ETTh2: all STBO variants lose by `+3.30%` to `+4.59%`;
- ETTm1: all STBO variants lose by `+0.91%` to `+1.84%`;
- Weather: STBO is near-neutral, and `stbo_bank4` has tiny wins on H96/H192/H336.

This pattern suggests Weather is more compatible with local smooth basis factorization, while ETT datasets benefit more
from A6's full-trajectory step basis and rank-256 coefficient interface.

## Research Interpretation

The original B12 story was:

```text
A6 full-720 basis is too trajectory-like;
replace it with native tile-local basis so short horizons activate only needed tiles.
```

The returned evidence says:

1. The story is still coherent as a problem statement, because A6 is indeed full-trajectory/prefix-slicing.
2. The tested solution is not strong enough: local rank16 basis loses A6 capacity and cannot reproduce A6's full-basis
   advantage on ETT.
3. The learned-basis novelty is not supported: DCT matches or beats learned variants.
4. Bank specialization did not happen, so the `stage/subspace bank` mechanism is not active in the trained model.

## Decision and Next Step

[Decision] Do not promote current B12-STBO to paper-core. Do not launch a full matrix.

[Rollback] Return StageB to Step 4 redesign or Step 2/3 architecture search.

If B12 is revisited, it should first be a diagnostic-only capacity/method-separation check:

1. capacity-equalized STBO: raise `stbo_rank` or use a richer coeff path until total parameters are close to A6;
2. keep `stbo_dct` at the same rank to test whether gains are generic local smoothness;
3. add a bank-specialization diagnostic or regularization before claiming stage/tile banks;
4. do not continue if DCT remains tied after capacity equalization.

This is a design-level failure of the current STBO implementation, not a direction-level rejection of native
multi-horizon architecture research.
