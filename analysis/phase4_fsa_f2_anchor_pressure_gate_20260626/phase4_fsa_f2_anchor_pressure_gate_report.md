# Phase4-FSA-F2 Anchor Pressure Gate Report

## Decision

[Decision] `fail_stop_future_anchor_stacking`.

## Main Summary

| comparison | candidate_arm | baseline_arm | settings | mse_wins | mean_relative_mse_pct | mae_wins | mean_relative_mae_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| selective_anchor_vs_floor005_r3 | F2-A1 | F1-A1 | 8 | 2 | +0.00% | 4 | +0.00% |
| selective_anchor_vs_floor005_single | F2-A0 | F1-A0 | 8 | 3 | -0.00% | 6 | -0.00% |
| selective_anchor_vs_r3_base | F2-A1 | F1-C1 | 8 | 2 | +1.62% | 1 | +1.06% |
| selective_anchor_vs_single_prefix_base | F2-A0 | F1-C0 | 8 | 4 | -0.34% | 6 | -0.51% |

## Dataset Summary

| comparison | candidate_arm | baseline_arm | dataset | settings | mse_wins | mean_relative_mse_pct |
| --- | --- | --- | --- | --- | --- | --- |
| selective_anchor_vs_floor005_r3 | F2-A1 | F1-A1 | ETTh2 | 4 | 2 | +0.00% |
| selective_anchor_vs_floor005_r3 | F2-A1 | F1-A1 | Weather | 4 | 0 | +0.00% |
| selective_anchor_vs_floor005_single | F2-A0 | F1-A0 | ETTh2 | 4 | 0 | +0.00% |
| selective_anchor_vs_floor005_single | F2-A0 | F1-A0 | Weather | 4 | 3 | -0.00% |
| selective_anchor_vs_r3_base | F2-A1 | F1-C1 | ETTh2 | 4 | 0 | +3.31% |
| selective_anchor_vs_r3_base | F2-A1 | F1-C1 | Weather | 4 | 2 | -0.06% |
| selective_anchor_vs_single_prefix_base | F2-A0 | F1-C0 | ETTh2 | 4 | 3 | -1.45% |
| selective_anchor_vs_single_prefix_base | F2-A0 | F1-C0 | Weather | 4 | 1 | +0.77% |

## H720 Segment Deltas

| comparison | candidate_arm | baseline_arm | dataset | segment | relative_mse_pct | mse_win |
| --- | --- | --- | --- | --- | --- | --- |
| selective_anchor_vs_single_prefix_base | F2-A0 | F1-C0 | ETTh2 | 1-96 | -0.32% | True |
| selective_anchor_vs_single_prefix_base | F2-A0 | F1-C0 | ETTh2 | 193-336 | +1.78% | False |
| selective_anchor_vs_single_prefix_base | F2-A0 | F1-C0 | ETTh2 | 337-720 | -2.54% | True |
| selective_anchor_vs_single_prefix_base | F2-A0 | F1-C0 | ETTh2 | 97-192 | +1.12% | False |
| selective_anchor_vs_single_prefix_base | F2-A0 | F1-C0 | Weather | 1-96 | -0.91% | True |
| selective_anchor_vs_single_prefix_base | F2-A0 | F1-C0 | Weather | 193-336 | +1.91% | False |
| selective_anchor_vs_single_prefix_base | F2-A0 | F1-C0 | Weather | 337-720 | +2.90% | False |
| selective_anchor_vs_single_prefix_base | F2-A0 | F1-C0 | Weather | 97-192 | +0.77% | False |
| selective_anchor_vs_floor005_single | F2-A0 | F1-A0 | ETTh2 | 1-96 | -0.00% | True |
| selective_anchor_vs_floor005_single | F2-A0 | F1-A0 | ETTh2 | 193-336 | +0.00% | False |
| selective_anchor_vs_floor005_single | F2-A0 | F1-A0 | ETTh2 | 337-720 | +0.00% | False |
| selective_anchor_vs_floor005_single | F2-A0 | F1-A0 | ETTh2 | 97-192 | -0.00% | True |
| selective_anchor_vs_floor005_single | F2-A0 | F1-A0 | Weather | 1-96 | +0.00% | False |
| selective_anchor_vs_floor005_single | F2-A0 | F1-A0 | Weather | 193-336 | -0.00% | True |
| selective_anchor_vs_floor005_single | F2-A0 | F1-A0 | Weather | 337-720 | -0.00% | True |
| selective_anchor_vs_floor005_single | F2-A0 | F1-A0 | Weather | 97-192 | -0.00% | True |
| selective_anchor_vs_r3_base | F2-A1 | F1-C1 | ETTh2 | 1-96 | -1.18% | True |
| selective_anchor_vs_r3_base | F2-A1 | F1-C1 | ETTh2 | 193-336 | +4.46% | False |
| selective_anchor_vs_r3_base | F2-A1 | F1-C1 | ETTh2 | 337-720 | +5.70% | False |
| selective_anchor_vs_r3_base | F2-A1 | F1-C1 | ETTh2 | 97-192 | +0.80% | False |
| selective_anchor_vs_r3_base | F2-A1 | F1-C1 | Weather | 1-96 | +0.60% | False |
| selective_anchor_vs_r3_base | F2-A1 | F1-C1 | Weather | 193-336 | -0.54% | True |
| selective_anchor_vs_r3_base | F2-A1 | F1-C1 | Weather | 337-720 | -1.24% | True |
| selective_anchor_vs_r3_base | F2-A1 | F1-C1 | Weather | 97-192 | -0.17% | True |
| selective_anchor_vs_floor005_r3 | F2-A1 | F1-A1 | ETTh2 | 1-96 | +0.00% | False |
| selective_anchor_vs_floor005_r3 | F2-A1 | F1-A1 | ETTh2 | 193-336 | -0.00% | True |
| selective_anchor_vs_floor005_r3 | F2-A1 | F1-A1 | ETTh2 | 337-720 | -0.00% | True |
| selective_anchor_vs_floor005_r3 | F2-A1 | F1-A1 | ETTh2 | 97-192 | -0.00% | True |
| selective_anchor_vs_floor005_r3 | F2-A1 | F1-A1 | Weather | 1-96 | +0.00% | False |
| selective_anchor_vs_floor005_r3 | F2-A1 | F1-A1 | Weather | 193-336 | +0.00% | False |
| selective_anchor_vs_floor005_r3 | F2-A1 | F1-A1 | Weather | 337-720 | +0.00% | False |
| selective_anchor_vs_floor005_r3 | F2-A1 | F1-A1 | Weather | 97-192 | +0.00% | False |

## Focused Diagnostics

### Future Alignment

| arm | dataset | horizons | mean_teacher_student_cosine | mean_future_reconstruction_loss | mean_future_alignment_confidence | min_future_alignment_confidence | max_prediction_leakage_abs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F2-A0 | ETTh2 | 4 | 0.122962 | 0.839213 | 0.389083 | 0.000000 | 0.000000 |
| F2-A0 | Weather | 4 | 0.323669 | 0.537031 | 0.619621 | 0.000000 | 0.000000 |
| F2-A1 | ETTh2 | 4 | 0.264843 | 0.614898 | 0.463137 | 0.000000 | 0.000000 |
| F2-A1 | Weather | 4 | 0.370836 | 0.492398 | 0.701015 | 0.000000 | 0.000000 |

### Training Dynamics

| arm | dataset | epochs_ran | best_epoch | post_best_val_drift_pct | train_loss_drop_pct | first_train_future_alignment_confidence_mean | last_train_future_alignment_confidence_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F2-A0 | ETTh2 | 12 | 2 | +6.69% | -25.02% | 0.323431 | 0.350057 |
| F2-A0 | Weather | 17 | 7 | +7.01% | -33.02% | 0.462100 | 0.576543 |
| F2-A1 | ETTh2 | 13 | 3 | +16.13% | -49.63% | 0.383675 | 0.543594 |
| F2-A1 | Weather | 11 | 1 | +12.13% | -31.13% | 0.580295 | 0.459347 |

### Checkpoint Selection

| arm | dataset | selector | best_epoch | official_best_epoch | official_gap_to_selector_best_pct |
| --- | --- | --- | --- | --- | --- |
| F2-A0 | ETTh2 | long_mean | 2 | 2 | +0.00% |
| F2-A0 | ETTh2 | h720 | 2 | 2 | +0.00% |
| F2-A0 | Weather | long_mean | 7 | 7 | +0.00% |
| F2-A0 | Weather | h720 | 7 | 7 | +0.00% |
| F2-A1 | ETTh2 | long_mean | 3 | 3 | +0.00% |
| F2-A1 | ETTh2 | h720 | 3 | 3 | +0.00% |
| F2-A1 | Weather | long_mean | 4 | 1 | +0.73% |
| F2-A1 | Weather | h720 | 4 | 1 | +1.47% |

## Gate Reading

- [Fact] `F2-A0` vs `F1-C0`: mean MSE `-0.34%`; Weather h720 late `+2.90%`.
- [Fact] `F2-A1` vs `F1-C1`: mean MSE `+1.62%`; ETTh2 mean `+3.31%`; Weather h720 late `-1.24%`.
- [Fact] Future leakage max `0.000e+00`.
