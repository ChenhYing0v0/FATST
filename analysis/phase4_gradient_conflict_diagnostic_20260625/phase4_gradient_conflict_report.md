# Phase4 Gradient Conflict Diagnostic

## 11-Step Record

| Field | Content |
| --- | --- |
| `current_step` | Step 5/6: theoretical feasibility and concrete diagnostic design |
| `problem` | Loss-only HSS may fail because difficult future units update the same shared representation path |
| `existence_evidence` | SRP-style step-specific representation argument plus Phase4 S1/S2 Weather collapse |
| `idea` | Test whether future-unit supervision groups produce conflicting gradients on shared parameters |
| `theory_check` | If noisy-hard gradients conflict with early/predictable gradients, gradient-routing HSS is justified |
| `design` | Short warmup/full-time checkpoint, then per-group backward passes and gradient cosine by parameter group |
| `gate` | Weather noisy-hard conflicts should be stronger than ETTh2, especially on encoder/target/readout shared paths |
| `artifacts` | `/home/yingch/exp_outputs/r-2026-fatst/phase4_gradient_conflict_diagnostic_20260625` |
| `decision` | Pending: use the diagnostics below to decide whether to implement adapter-isolated HSS |

## Run Config

- datasets: `ETTh2,Weather`
- pred_len: `720`
- warmup_steps: `40`
- diagnostic_batches: `16`
- block_size: `48`
- top_ratio: `0.25`
- checkpoint: `none`

## Focus Pair Summary

| Dataset | Pair | Parameter group | Mean cosine | Negative share | Low share < 0.1 |
| --- | --- | --- | ---: | ---: | ---: |
| ETTh2 | `noisy_hard` vs `early_1_96` | `encoder` | 0.2281 | 0.33 | 0.42 |
| ETTh2 | `noisy_hard` vs `early_1_96` | `target_path` | 0.3083 | 0.25 | 0.42 |
| ETTh2 | `noisy_hard` vs `early_1_96` | `readout_head` | 0.3248 | 0.17 | 0.25 |
| ETTh2 | `noisy_hard` vs `early_1_96` | `all_shared` | 0.3040 | 0.25 | 0.25 |
| ETTh2 | `noisy_hard` vs `predictable_easy` | `encoder` | 0.6804 | 0.08 | 0.08 |
| ETTh2 | `noisy_hard` vs `predictable_easy` | `target_path` | 0.5383 | 0.08 | 0.08 |
| ETTh2 | `noisy_hard` vs `predictable_easy` | `readout_head` | 0.6455 | 0.00 | 0.00 |
| ETTh2 | `noisy_hard` vs `predictable_easy` | `all_shared` | 0.6482 | 0.00 | 0.08 |
| ETTh2 | `learnable_hard` vs `early_1_96` | `encoder` | -0.1222 | 0.62 | 0.69 |
| ETTh2 | `learnable_hard` vs `early_1_96` | `target_path` | 0.0473 | 0.56 | 0.56 |
| ETTh2 | `learnable_hard` vs `early_1_96` | `readout_head` | 0.0917 | 0.44 | 0.56 |
| ETTh2 | `learnable_hard` vs `early_1_96` | `all_shared` | 0.0457 | 0.50 | 0.62 |
| ETTh2 | `late_337_720` vs `early_1_96` | `encoder` | 0.0213 | 0.50 | 0.62 |
| ETTh2 | `late_337_720` vs `early_1_96` | `target_path` | 0.1912 | 0.31 | 0.38 |
| ETTh2 | `late_337_720` vs `early_1_96` | `readout_head` | 0.2379 | 0.31 | 0.31 |
| ETTh2 | `late_337_720` vs `early_1_96` | `all_shared` | 0.1912 | 0.31 | 0.38 |
| Weather | `noisy_hard` vs `early_1_96` | `encoder` | 0.3117 | 0.27 | 0.27 |
| Weather | `noisy_hard` vs `early_1_96` | `target_path` | 0.3415 | 0.00 | 0.13 |
| Weather | `noisy_hard` vs `early_1_96` | `readout_head` | 0.3408 | 0.20 | 0.27 |
| Weather | `noisy_hard` vs `early_1_96` | `all_shared` | 0.3392 | 0.20 | 0.27 |
| Weather | `noisy_hard` vs `predictable_easy` | `encoder` | 0.3234 | 0.27 | 0.33 |
| Weather | `noisy_hard` vs `predictable_easy` | `target_path` | 0.4239 | 0.13 | 0.20 |
| Weather | `noisy_hard` vs `predictable_easy` | `readout_head` | 0.1190 | 0.33 | 0.40 |
| Weather | `noisy_hard` vs `predictable_easy` | `all_shared` | 0.1403 | 0.27 | 0.33 |
| Weather | `learnable_hard` vs `early_1_96` | `encoder` | 0.3362 | 0.19 | 0.25 |
| Weather | `learnable_hard` vs `early_1_96` | `target_path` | 0.2974 | 0.12 | 0.19 |
| Weather | `learnable_hard` vs `early_1_96` | `readout_head` | 0.3715 | 0.06 | 0.12 |
| Weather | `learnable_hard` vs `early_1_96` | `all_shared` | 0.3676 | 0.06 | 0.12 |
| Weather | `late_337_720` vs `early_1_96` | `encoder` | 0.0128 | 0.50 | 0.56 |
| Weather | `late_337_720` vs `early_1_96` | `target_path` | 0.1277 | 0.38 | 0.44 |
| Weather | `late_337_720` vs `early_1_96` | `readout_head` | -0.0219 | 0.56 | 0.62 |
| Weather | `late_337_720` vs `early_1_96` | `all_shared` | -0.0149 | 0.56 | 0.62 |

## Block Bucket Trace

| Dataset | Mean learnable blocks | Mean noisy blocks | Mean predictable blocks |
| --- | ---: | ---: | ---: |
| ETTh2 | 2.38 | 1.62 | 11.00 |
| Weather | 1.88 | 2.12 | 11.00 |

## How To Read

[Fact] `gradient_cosine < 0` means two supervision groups push the same parameter group in opposing directions on that batch.

[Inference] If Weather has a high negative/low-cosine share for `noisy_hard` vs `early_1_96` or `predictable_easy`, then S2's scalar downweight was too weak; the next candidate should isolate noisy-hard gradients into adapters or detached auxiliary branches.

[Counter-check] If cosines are high and positive, the failure is less likely to be representation interference; rollback should prioritize a better predictability proxy rather than architecture-level gradient routing.
