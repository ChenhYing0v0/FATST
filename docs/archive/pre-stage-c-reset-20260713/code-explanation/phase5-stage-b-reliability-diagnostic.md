# Phase5 StageB Reliability Diagnostic Code Explanation

本文档解释 StageB B1 reliability diagnostic 的分析脚本。它不是 model code，不改变
`A6-LBF-r256` 的训练、forward path 或 remote runner。

## Functional Module

入口：

- `scripts/analyze_phase5_stage_b_reliability_diagnostic.py`

默认输入：

- `analysis/phase5_timealign_hss_a6_capacity_native_gate_20260703/raw/official-last/`
- run name: `TimeAlignOfficialUnified720_A6_a6_lbf_r256_official-last`

默认输出：

- `analysis/phase5_stage_b_reliability_diagnostic_20260706/stage_b_a6_lbf_unit_reliability.csv`
- `analysis/phase5_stage_b_reliability_diagnostic_20260706/stage_b_a6_lbf_reliability_summary.csv`
- `analysis/phase5_stage_b_reliability_diagnostic_20260706/stage_b_a6_lbf_proxy_alignment.csv`
- `analysis/phase5_stage_b_reliability_diagnostic_20260706/stage_b_a6_lbf_unit_proxy_detrended.csv`
- `analysis/phase5_stage_b_reliability_diagnostic_20260706/stage_b_a6_lbf_trajectory_drift.csv`
- `analysis/phase5_stage_b_reliability_diagnostic_20260706/stage_b_a6_lbf_reliability_report.md`

## Data Flow

The script reads, for each dataset:

1. `predictions_test.npz`;
2. `training_log.csv`;
3. local train split labels under `/Users/river/PaperResearch/Project/datasets`.

It reads `predictions_test.npz`, then treats each 48-step block as a future unit:

```text
pred, true -> squared error by sample / time / channel
48-step block -> unit-level MSE, MAE, sample_error_std, sample_error_cv
```

It computes train-only proxies from train split labels:

- `label_novelty`: future block distance from the last history value;
- `local_variation`: local step-to-step variation inside the future block;
- `seasonal_residual`: distance from a seasonal naive reference.

Held-out prediction error is used only as diagnostic label, not as a training signal.

## Statistics

`stage_b_a6_lbf_unit_reliability.csv`:

| Column | Source | Meaning |
| --- | --- | --- |
| `dataset` | directory name | dataset identity |
| `arm` | constant | `a6_lbf_r256` |
| `target_horizon` | `metrics_by_segment.csv` | selected evaluation horizon; this audit uses `720` only |
| `unit_start`, `unit_end` | `segment_start`, `segment_end` | 96-step future unit boundary |
| `unit_len` | `unit_end - unit_start` | unit width |
| `mse`, `mae` | `predictions_test.npz` | unit error |
| `sample_error_std`, `sample_error_cv` | per-sample unit MSE | sample-level volatility |
| `relative_to_dataset_min_mse_pct` | computed from unit MSE | unit MSE relative to easiest unit in the same dataset |
| `unit_source` | constant | provenance file |

`stage_b_a6_lbf_reliability_summary.csv`:

| Column | Meaning |
| --- | --- |
| `min_unit_mse`, `max_unit_mse`, `mean_unit_mse`, `std_unit_mse` | segment MSE distribution |
| `hard_easy_ratio` | mean of two hardest unit MSEs divided by mean of two easiest unit MSEs |
| `max_vs_min_pct` | max unit MSE relative to min unit MSE |
| `spearman_step_mse` | rank correlation between `segment_start` and segment MSE |
| `pearson_step_mse` | linear correlation between `segment_start` and segment MSE |
| `detrended_mse_std` | unit MSE residual variation after linear step-index regression |

`stage_b_a6_lbf_proxy_alignment.csv`:

| Column | Meaning |
| --- | --- |
| `proxy` | train-only proxy name |
| `spearman_proxy_mse` | proxy alignment with raw held-out unit MSE |
| `spearman_proxy_detrended_mse` | proxy alignment after removing linear step-index trend from held-out MSE |
| `top_quartile_overlap` | overlap between proxy-selected hard units and held-out hard units |
| `top_quartile_detrended_overlap` | overlap between proxy-selected hard units and distance-detrended hard units |

`stage_b_a6_lbf_unit_proxy_detrended.csv`:

| Column | Meaning |
| --- | --- |
| `detrended_mse` | unit MSE residual after linear step-index regression within the dataset |
| `proxy_value` | train-only proxy value for this unit |
| `is_raw_top_quartile` | whether the unit is among raw held-out hard units |
| `is_detrended_top_quartile` | whether the unit is among distance-detrended hard units |
| `is_proxy_top_quartile` | whether the train-only proxy selects the unit as hard |

`stage_b_a6_lbf_trajectory_drift.csv`:

| Column | Meaning |
| --- | --- |
| `first_val_mean_mse`, `best_val_mean_mse`, `last_val_mean_mse` | validation trajectory from `training_log.csv` |
| `last_minus_best_pct` | official-last drift relative to best validation epoch |
| `train_h720_vs_h96_l1_ratio_last` | last-epoch long-prefix train L1 divided by short-prefix train L1 |

## Code-Theory Consistency

[Intended theory] B1 should test whether A6-LBF has future-unit reliability
heterogeneity that can motivate reliability-aware supervision.

[Code realization] The script computes prediction-level future-unit heterogeneity,
sample-level volatility, train-only proxy alignment, and official-last trajectory drift.

[Proxy] `spearman_step_mse` is a confound check. A value near `1.0` means unit
difficulty may be explained by forecast distance. Therefore the stricter evidence is
`spearman_proxy_detrended_mse`, not raw proxy-MSE correlation.

[Falsification] Current B1 is `partial_pass_distance_confounded`: heterogeneity exists,
but immediate B2 is not justified. StageB must return to Step 2/3 unless a stronger
non-distance reliability problem can be defined.
