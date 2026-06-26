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

## Intermediate Diagnostics

[Fact] 结论不只依赖 aggregate MSE/MAE。本报告同步检查了 segment metrics、future alignment/leakage、checkpoint selection、training dynamics、target conditioning、target-state similarity、prefix consistency 和 objective pressure。

### Target Conditioning Summary

| arm | dataset | horizons | mean_abs_gamma | mean_abs_beta | mean_target_state_norm | mean_target_state_norm_std | mean_history_readout_norm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F1-A0 | ETTh2 | 4 | 0.437158 | 0.437516 | 7.153762 | 0.396529 | 10.220398 |
| F1-A0 | Weather | 4 | 0.766411 | 0.417302 | 8.416493 | 1.954961 | 10.715979 |
| F1-A1 | ETTh2 | 4 | 0.513034 | 0.429625 | 7.922244 | 1.117792 | 10.163903 |
| F1-A1 | Weather | 4 | 0.697083 | 0.416682 | 8.396148 | 1.521291 | 11.689461 |
| F1-C0 | ETTh2 | 4 | 0.463760 | 0.429097 | 6.443887 | 0.753688 | 10.597639 |
| F1-C0 | Weather | 4 | 0.723331 | 0.425456 | 8.862094 | 1.715274 | 10.726435 |
| F1-C1 | ETTh2 | 4 | 0.444583 | 0.434325 | 6.470165 | 0.469123 | 10.466527 |
| F1-C1 | Weather | 4 | 0.780469 | 0.409766 | 9.127822 | 2.225557 | 10.463046 |
| F1-W0 | ETTh2 | 4 | 0.432823 | 0.431449 | 7.418954 | 0.422416 | 9.641081 |
| F1-W0 | Weather | 4 | 0.741088 | 0.428028 | 8.157727 | 1.255222 | 11.209194 |

### Target-State Similarity Summary

| arm | dataset | horizons | mean_target_state_cosine | mean_abs_target_state_cosine | mean_adjacent_target_state_cosine | min_target_state_cosine | max_target_state_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F1-A0 | ETTh2 | 4 | 0.981923 | 0.981923 | 0.995412 | 0.903722 | 0.997814 |
| F1-A0 | Weather | 4 | 0.279710 | 0.286924 | 0.192015 | -0.193859 | 0.992190 |
| F1-A1 | ETTh2 | 4 | 0.797603 | 0.797603 | 0.884270 | 0.377783 | 0.997335 |
| F1-A1 | Weather | 4 | 0.130775 | 0.229675 | 0.013286 | -0.286034 | 0.992126 |
| F1-C0 | ETTh2 | 4 | 0.873780 | 0.873780 | 0.938474 | 0.548574 | 0.997353 |
| F1-C0 | Weather | 4 | 0.146128 | 0.222085 | 0.019770 | -0.311883 | 0.993930 |
| F1-C1 | ETTh2 | 4 | 0.912303 | 0.912303 | 0.961652 | 0.649992 | 0.996613 |
| F1-C1 | Weather | 4 | 0.227846 | 0.246360 | 0.133534 | -0.306833 | 0.993345 |
| F1-W0 | ETTh2 | 4 | 0.989593 | 0.989593 | 0.997126 | 0.951085 | 0.998288 |
| F1-W0 | Weather | 4 | 0.261200 | 0.266033 | 0.153050 | -0.147227 | 0.993251 |

### Prefix Consistency Summary

| arm | dataset | pairs | max_prefix_mismatch_mse | max_prefix_mismatch_mae | max_truth_alignment_mse |
| --- | --- | --- | --- | --- | --- |
| F1-C0 | ETTh2 | 3 | 0.000000 | 0.000000 | 0.000000 |
| F1-C0 | Weather | 3 | 0.000000 | 0.000000 | 0.000000 |
| F1-C1 | ETTh2 | 3 | 0.000000 | 0.000000 | 0.000000 |
| F1-C1 | Weather | 3 | 0.000000 | 0.000000 | 0.000000 |
| F1-A0 | ETTh2 | 3 | 0.000000 | 0.000000 | 0.000000 |
| F1-A0 | Weather | 3 | 0.000000 | 0.000000 | 0.000000 |
| F1-A1 | ETTh2 | 3 | 0.000000 | 0.000000 | 0.000000 |
| F1-A1 | Weather | 3 | 0.000000 | 0.000000 | 0.000000 |
| F1-W0 | ETTh2 | 3 | 0.000000 | 0.000000 | 0.000000 |
| F1-W0 | Weather | 3 | 0.000000 | 0.000000 | 0.000000 |

### Objective Pressure Check

| arm | dataset | scope | mode | mean_step_weight | weighted_pressure_share | pressure_share_delta_pct |
| --- | --- | --- | --- | --- | --- | --- |
| F1-C0 | Weather | 1-96 | prefix_risk | 2.611814 | 0.721703 | +50.43% |
| F1-C0 | Weather | 97-192 | prefix_risk | 1.163544 | 0.153976 | -32.98% |
| F1-C0 | Weather | 193-336 | prefix_risk | 0.855834 | 0.077459 | -50.71% |
| F1-C0 | Weather | 337-720 | prefix_risk | 0.610223 | 0.046862 | -64.85% |
| F1-C0 | Weather | horizon_96 | prefix_risk | 2.611814 | 2.611814 | +161.18% |
| F1-C0 | Weather | horizon_192 | prefix_risk | 1.887679 | 1.887679 | +88.77% |
| F1-C0 | Weather | horizon_336 | prefix_risk | 1.445460 | 1.445460 | +44.55% |
| F1-C0 | Weather | horizon_720 | prefix_risk | 1.000000 | 1.000000 | -0.00% |

## Gate Reading

- [Fact] `F1-A0` vs `F1-C0`: `4/8` MSE wins, mean relative MSE `-0.34%`.
- [Fact] `F1-A0` Weather mean relative MSE vs single-prefix `+0.78%`; Weather h720 late `337-720` segment vs single-prefix `+2.91%`.
- [Fact] `F1-A1` vs `F1-C1/R.3`: `2/8` MSE wins, mean relative MSE `+1.62%`.
- [Fact] `F1-A1` ETTh2 mean relative MSE vs R.3 `+3.31%`.
- [Fact] `F1-A1` Weather mean relative MSE vs R.3 `-0.06%`; Weather h720 late `337-720` segment vs R.3 `-1.24%`.
- [Fact] Future leakage max `0.000e+00`; min raw confidence `0.050000`; min mean confidence `0.390767`; max official-to-oracle gap among A0/A1 long/h720 selectors `+1.44%`.
- [Fact] Prefix consistency max MSE `1.523e-14`，说明 unified output prefix 没有因为 anchor 产生数值不一致。
- [Inference] `objective_weight_stats.csv` 显示 prefix-risk pressure 在同类 arms 中一致，因此 F1-A0/A1 的差异不是由 step-weight 配置漂移造成。
- [Inference] `target_conditioning_stats.csv` 与 `target_state_similarity.csv` 证明 anchor 确实改变了 state/conditioning geometry，但这种变化没有稳定转化为跨 dataset 的 main-metric 改善。

## Decision Rules

- `pass_to_fsa_f2` 需要 A0 或 A1 同时满足 mean/win、dataset split 和 Weather late no-harm gate。
- `partial_pass_anchor_signal_but_not_core_substrate` 表示 future anchor 有局部正信号，但不能直接叠加 HSS/gradient routing。
- 若二者均失败且 diagnostics 非 collapse：future teacher 与 forecasting objective 语义不一致，回 Step 2/3 重新定义 representation 问题。
- 若只有 oracle checkpoint 有收益：先修 validation metric，不继续改 model。
