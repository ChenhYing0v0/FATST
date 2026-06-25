# Phase4 OP-A Stabilized Routing Gate Report

## 11-Step Record

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10: evaluate stabilized-base adapter-only routing |
| `problem` | RG-B router works against full-time but fails R.3; Phase4 logs show systemic early best epoch |
| `idea` | Pretrain full-time base, freeze base, then train only dynamic residual-stability adapter |
| `gate` | audit freeze, delay early-best collapse, improve Weather vs R.3 and RG-B late segment |
| `artifacts` | `analysis/phase4_stabilized_routing_gate_20260625` |
| `decision` | Fail; protocol stabilizes training timing but damages metrics |

## Overall Result

| Baseline | Settings | MSE wins | MAE wins | Mean relative MSE | Mean relative MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `D1_r3_prefix_risk` | 8 | 0 | 0 | +8.92% | +5.18% |
| `OP-A_pretrain` | 8 | 0 | 0 | +3.57% | +1.93% |
| `RG-B_from_scratch` | 8 | 1 | 1 | +6.64% | +3.43% |

[Fact] Weather vs R.3 remains `0/4`, mean relative MSE `+7.30%`.
[Fact] ETTh2 vs R.3 is `0/4`, mean relative MSE `+10.53%`.

## Per-Horizon MSE Delta

| Dataset | Horizon | vs pretrain | vs R.3 | vs RG-B |
| --- | ---: | ---: | ---: | ---: |
| `ETTh2` | 96 | +5.60% | +17.57% | +15.22% |
| `ETTh2` | 192 | +6.39% | +12.06% | +13.61% |
| `ETTh2` | 336 | +3.37% | +6.51% | +5.69% |
| `ETTh2` | 720 | +3.79% | +5.99% | +7.31% |
| `Weather` | 96 | +1.08% | +5.64% | -0.03% |
| `Weather` | 192 | +2.01% | +6.41% | +2.39% |
| `Weather` | 336 | +2.86% | +7.94% | +3.76% |
| `Weather` | 720 | +3.51% | +9.20% | +5.14% |

## Audit

| Dataset | Freeze | Trainable params | Missing keys | Unexpected keys |
| --- | --- | ---: | ---: | ---: |
| `ETTh2` | `True` | 12848 / 2025312 (0.63%) | 4 | 0 |
| `Weather` | `True` | 12848 / 2025312 (0.63%) | 4 | 0 |

## Trace

| Dataset | Learnable blocks | Noisy blocks | Noisy suppression | Adapter active steps | Mean abs adapter residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ETTh2` | 1.16 | 1.45 | 0.36 | 55.7 | 0.091409 |
| `Weather` | 1.44 | 1.87 | 0.47 | 69.1 | 0.065752 |

## Training Dynamics

| Dataset | Strategy | Epochs | Best epoch | Best val MSE | Last val drift | Train loss change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ETTh2` | `OP-A_pretrain` | 13 | 3 | 0.380270 | +14.45% | -30.94% |
| `ETTh2` | `OP-A_adapter_only` | 12 | 7 | 0.402832 | +2.47% | -0.05% |
| `Weather` | `OP-A_pretrain` | 14 | 4 | 0.534117 | +10.41% | -38.40% |
| `Weather` | `OP-A_adapter_only` | 11 | 6 | 0.542288 | +0.40% | -0.07% |

## Segment Gate

[Fact] Weather h720 `337-720` vs R.3: `+10.24%`.
[Fact] Weather h720 `337-720` vs RG-B: `+5.94%`.

## Interpretation

[Strong Evidence] OP-A validates the audit side: checkpoint loading, base freezing, and adapter-only optimization are functioning. Trainable parameters are only 0.63% of total parameters, and missing keys are the expected adapter head.

[Counter-Evidence] OP-A fails the performance gate. Adapter-only finetune is worse than the full-time pretrain at every horizon, worse than R.3 at every horizon, and mostly worse than from-scratch RG-B.

[Inference] Stabilizing the base delays early-best collapse, but the current adapter-only residual path cannot improve the frozen base. This suggests the problem is not just simultaneous base/routing optimization; the adapter capacity/objective or base objective is insufficient.

[Decision] Do not pursue adapter-only stabilized routing with full-time base. Roll back to Step 5/6 and test whether the base objective itself must be R.3/prefix-risk stabilized, or whether routing needs to update a richer target/readout subset rather than only the small adapter head.
