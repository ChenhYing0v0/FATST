# Phase4-FSA-F1 Future-State Anchor Gate Report

## 11-step Record

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10：评估 Future-State Anchored HSS substrate diagnostic |
| `problem` | Phase4 scheduling signal 有效，但当前 `target_states` 可能缺少承接 HSS pressure 的 future-structured representation |
| `existence_evidence` | S1/S2、HSSG、SCC 均有 active signal 但不能稳定超过 R.3；OP-A 否定 full-time base pretrain；R3D 指向 prefix-risk stabilized base |
| `idea` | 用 training-only future teacher 先锚定 `target_states`，再判断 prefix-risk/HSS pressure 是否更可泛化 |
| `theory_check` | 若 future-state geometry 是瓶颈，future anchor 应改善或至少不破坏 R.3/single-prefix base，并给出非 collapse alignment diagnostics |
| `design` | F1-C0/F1-C1 controls；F1-A0/F1-A1 future-anchor candidates；F1-W0 weak full-time control；ETTh2 + Weather，seed2021 |
| `gate` | A1 vs R.3 不劣于 +0.3% 且改善 Weather long/late；或 A0 vs single-prefix 至少 5/8 wins；future leakage/confidence 非 collapse；oracle gap 不是唯一解释 |
| `artifacts` | `analysis/phase4_future_state_anchor_gate_20260626` |
| `decision` | partial_pass_anchor_signal_but_not_core_substrate |

## Main Comparison Summary

| comparison | candidate_arm | baseline_arm | settings | mse_wins | mean_relative_mse_pct | mae_wins | mean_relative_mae_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| anchor_vs_r3_base | F1-A1 | F1-C1 | 8 | 2 | +1.62% | 1 | +1.06% |
| anchor_vs_single_prefix_base | F1-A0 | F1-C0 | 8 | 4 | -0.34% | 6 | -0.51% |
| full_time_anchor_vs_r3_reference | F1-W0 | F1-C1 | 8 | 0 | +2.53% | 0 | +2.56% |
| full_time_anchor_vs_single_prefix_base | F1-W0 | F1-C0 | 8 | 3 | +1.09% | 0 | +1.47% |
| single_anchor_vs_r3_reference | F1-A0 | F1-C1 | 8 | 1 | +1.08% | 1 | +0.55% |

## Dataset Split

| comparison | candidate_arm | baseline_arm | dataset | settings | mse_wins | mean_relative_mse_pct |
| --- | --- | --- | --- | --- | --- | --- |
| anchor_vs_r3_base | F1-A1 | F1-C1 | ETTh2 | 4 | 0 | +3.31% |
| anchor_vs_r3_base | F1-A1 | F1-C1 | Weather | 4 | 2 | -0.06% |
| anchor_vs_single_prefix_base | F1-A0 | F1-C0 | ETTh2 | 4 | 3 | -1.45% |
| anchor_vs_single_prefix_base | F1-A0 | F1-C0 | Weather | 4 | 1 | +0.78% |
| full_time_anchor_vs_r3_reference | F1-W0 | F1-C1 | ETTh2 | 4 | 0 | +1.67% |
| full_time_anchor_vs_r3_reference | F1-W0 | F1-C1 | Weather | 4 | 0 | +3.40% |
| full_time_anchor_vs_single_prefix_base | F1-W0 | F1-C0 | ETTh2 | 4 | 3 | -0.38% |
| full_time_anchor_vs_single_prefix_base | F1-W0 | F1-C0 | Weather | 4 | 0 | +2.56% |
| single_anchor_vs_r3_reference | F1-A0 | F1-C1 | ETTh2 | 4 | 1 | +0.57% |
| single_anchor_vs_r3_reference | F1-A0 | F1-C1 | Weather | 4 | 0 | +1.59% |

## Future Alignment Diagnostics

| arm | dataset | horizons | mean_teacher_student_cosine | mean_future_local_alignment_loss | mean_future_reconstruction_loss | mean_future_alignment_confidence | min_future_alignment_confidence | max_prediction_leakage_abs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1-A0 | ETTh2 | 4 | 0.123149 | 0.883753 | 0.839213 | 0.390767 | 0.050000 | 0.000000 |
| F1-A0 | Weather | 4 | 0.324495 | 0.719214 | 0.537031 | 0.623370 | 0.050000 | 0.000000 |
| F1-A1 | ETTh2 | 4 | 0.265768 | 0.760538 | 0.614898 | 0.465013 | 0.050000 | 0.000000 |
| F1-A1 | Weather | 4 | 0.371749 | 0.667194 | 0.492398 | 0.704149 | 0.050000 | 0.000000 |
| F1-W0 | ETTh2 | 4 | 0.120264 | 0.886066 | 0.839213 | 0.390767 | 0.050000 | 0.000000 |
| F1-W0 | Weather | 4 | 0.322366 | 0.722374 | 0.545123 | 0.609152 | 0.050000 | 0.000000 |

## Segment Region Summary

| comparison | candidate_arm | baseline_arm | dataset | future_region | settings | mse_wins | mean_relative_mse_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| anchor_vs_r3_base | F1-A1 | F1-C1 | ETTh2 | early_1_96 | 4 | 1 | +1.77% |
| anchor_vs_r3_base | F1-A1 | F1-C1 | ETTh2 | late_337_720 | 1 | 0 | +5.70% |
| anchor_vs_r3_base | F1-A1 | F1-C1 | ETTh2 | middle_97_336 | 5 | 0 | +2.26% |
| anchor_vs_r3_base | F1-A1 | F1-C1 | Weather | early_1_96 | 4 | 0 | +0.62% |
| anchor_vs_r3_base | F1-A1 | F1-C1 | Weather | late_337_720 | 1 | 1 | -1.24% |
| anchor_vs_r3_base | F1-A1 | F1-C1 | Weather | middle_97_336 | 5 | 5 | -0.27% |
| anchor_vs_single_prefix_base | F1-A0 | F1-C0 | ETTh2 | early_1_96 | 4 | 4 | -2.25% |
| anchor_vs_single_prefix_base | F1-A0 | F1-C0 | ETTh2 | late_337_720 | 1 | 1 | -2.54% |
| anchor_vs_single_prefix_base | F1-A0 | F1-C0 | ETTh2 | middle_97_336 | 5 | 1 | +0.75% |
| anchor_vs_single_prefix_base | F1-A0 | F1-C0 | Weather | early_1_96 | 4 | 4 | -0.88% |
| anchor_vs_single_prefix_base | F1-A0 | F1-C0 | Weather | late_337_720 | 1 | 0 | +2.91% |
| anchor_vs_single_prefix_base | F1-A0 | F1-C0 | Weather | middle_97_336 | 5 | 0 | +1.43% |

## Training Dynamics

| arm | dataset | future_alignment | epochs_ran | best_epoch | post_best_val_drift_pct | train_loss_drop_pct |
| --- | --- | --- | --- | --- | --- | --- |
| F1-C0 | ETTh2 | off | 14 | 4 | +9.43% | -27.03% |
| F1-C0 | Weather | off | 14 | 4 | +8.01% | -29.27% |
| F1-C1 | ETTh2 | off | 11 | 1 | +19.19% | -46.79% |
| F1-C1 | Weather | off | 13 | 3 | +13.39% | -33.85% |
| F1-A0 | ETTh2 | on | 12 | 2 | +6.69% | -25.02% |
| F1-A0 | Weather | on | 17 | 7 | +7.06% | -33.00% |
| F1-A1 | ETTh2 | on | 13 | 3 | +16.15% | -49.63% |
| F1-A1 | Weather | on | 11 | 1 | +12.26% | -30.97% |
| F1-W0 | ETTh2 | on | 12 | 2 | +10.65% | -21.30% |
| F1-W0 | Weather | on | 16 | 6 | +7.42% | -36.65% |

## Checkpoint Selection Diagnostics

| arm | dataset | selector | best_epoch | official_best_epoch | official_gap_to_selector_best_pct |
| --- | --- | --- | --- | --- | --- |
| F1-C0 | ETTh2 | long_mean | 4 | 4 | +0.00% |
| F1-C0 | ETTh2 | h720 | 4 | 4 | +0.00% |
| F1-C0 | Weather | long_mean | 4 | 4 | +0.00% |
| F1-C0 | Weather | h720 | 4 | 4 | +0.00% |
| F1-C1 | ETTh2 | long_mean | 3 | 1 | +0.21% |
| F1-C1 | ETTh2 | h720 | 3 | 1 | +0.49% |
| F1-C1 | Weather | long_mean | 3 | 3 | +0.00% |
| F1-C1 | Weather | h720 | 3 | 3 | +0.00% |
| F1-A0 | ETTh2 | long_mean | 2 | 2 | +0.00% |
| F1-A0 | ETTh2 | h720 | 2 | 2 | +0.00% |
| F1-A0 | Weather | long_mean | 7 | 7 | +0.00% |
| F1-A0 | Weather | h720 | 7 | 7 | +0.00% |
| F1-A1 | ETTh2 | long_mean | 3 | 3 | +0.00% |
| F1-A1 | ETTh2 | h720 | 3 | 3 | +0.00% |
| F1-A1 | Weather | long_mean | 4 | 1 | +0.70% |
| F1-A1 | Weather | h720 | 4 | 1 | +1.44% |
| F1-W0 | ETTh2 | long_mean | 2 | 2 | +0.00% |
| F1-W0 | ETTh2 | h720 | 2 | 2 | +0.00% |
| F1-W0 | Weather | long_mean | 2 | 6 | +0.44% |
| F1-W0 | Weather | h720 | 2 | 6 | +0.51% |

## Gate Reading

- [Fact] `F1-A0` vs `F1-C0`: `4/8` MSE wins, mean relative MSE `-0.34%`.
- [Fact] `F1-A0` Weather mean relative MSE vs single-prefix `+0.78%`; Weather h720 late `337-720` segment vs single-prefix `+2.91%`.
- [Fact] `F1-A1` vs `F1-C1/R.3`: `2/8` MSE wins, mean relative MSE `+1.62%`.
- [Fact] `F1-A1` ETTh2 mean relative MSE vs R.3 `+3.31%`.
- [Fact] `F1-A1` Weather mean relative MSE vs R.3 `-0.06%`; Weather h720 late `337-720` segment vs R.3 `-1.24%`.
- [Fact] Future leakage max `0.000e+00`; min raw confidence `0.050000`; min mean confidence `0.390767`; max official-to-oracle gap among A0/A1 long/h720 selectors `+1.44%`.

## Decision Rules

- `pass_to_fsa_f2` 需要 A0 或 A1 同时满足 mean/win、dataset split 和 Weather late no-harm gate。
- `partial_pass_anchor_signal_but_not_core_substrate` 表示 future anchor 有局部正信号，但不能直接叠加 HSS/gradient routing。
- 若二者均失败且 diagnostics 非 collapse：future teacher 与 forecasting objective 语义不一致，回 Step 2/3 重新定义 representation 问题。
- 若只有 oracle checkpoint 有收益：先修 validation metric，不继续改 model。
