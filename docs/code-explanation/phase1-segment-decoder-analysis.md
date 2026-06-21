# Phase1 Segment Decoder Analysis

## 目的

`scripts/analyze_phase1_segment_decoder_gate.py` 汇总 Phase1-A remote gate 的 artifacts，
用于判断 `PatchEncoderSegmentQueryHead` 是否通过。

输入目录：

```text
analysis/phase1_segment_decoder_gate_20260621/raw/
```

该目录由远程输出 `/home/yingch/exp_outputs/r-2026-fatst/phase1_future_segment` 同步而来，
包含每个 run 的 `metrics.json`、`metrics_by_segment.csv`、`metrics_by_horizon.csv`、
`training_log.csv`、`environment.json`、`effective_config.json`，以及 SegmentQueryHead 的
`segment_query_similarity.csv`。

## 输出文件

`phase1_segment_decoder_metrics.csv`

- Source: every run's `metrics.json`, `training_log.csv`, and local model parameter count.
- Columns:
  - `model`: model name.
  - `dataset`: dataset name.
  - `horizon`: prediction horizon.
  - `seed`: random seed.
  - `mse`, `mae`: test metrics from `metrics.json`.
  - `epochs`: number of rows in `training_log.csv`.
  - `parameter_count`: parameter count from instantiating the local model with the matching horizon.

`phase1_segment_decoder_comparison.csv`

- Source: paired rows from `phase1_segment_decoder_metrics.csv`.
- Each row compares `PatchEncoderSegmentQueryHead` against `PatchEncoderFixedHead` on the same
  dataset and horizon.
- Key columns:
  - `delta_mse = segment_mse - fixed_mse`
  - `relative_mse_change = delta_mse / fixed_mse`
  - `delta_mae = segment_mae - fixed_mae`
  - `relative_mae_change = delta_mae / fixed_mae`
  - `parameter_ratio = segment_parameter_count / fixed_parameter_count`
  - `segment_passes_mse`: true only if SegmentQueryHead has lower MSE.

`phase1_segment_decoder_segment_comparison.csv`

- Source: paired `metrics_by_segment.csv` files.
- Meaning: checks whether SegmentQueryHead has any local segment-level compensation even when
  whole-horizon MSE is worse.
- `segment_passes_mse` is true only if SegmentQueryHead has lower MSE on that segment.

`phase1_segment_query_similarity_summary.csv`

- Source: `segment_query_similarity.csv`.
- Meaning: summarizes cosine similarity among learned segment query parameters.
- This is a parameter-level diagnostic, not a full hidden-state diagnostic.

`phase1_segment_decoder_summary.json`

- Machine-readable decision summary.
- `status="failed"` means the first SegmentQueryHead candidate did not pass Phase1-A.
- `rollback_step` records the long research loop step to return to.

## Figures

`phase1_segment_decoder_relative_mse_heatmap.png`

- Rows: datasets.
- Columns: horizons.
- Cell value: `relative_mse_change * 100`.
- Positive values mean SegmentQueryHead is worse than FixedHead.

`phase1_segment_decoder_segment_delta_hist.png`

- Distribution of segment-level `relative_mse_change * 100`.
- The vertical zero line marks equal MSE.
- If bars appear mostly or entirely to the right of zero, SegmentQueryHead lacks local segment
  compensation.

## Decision Logic

The script marks the candidate failed when the evidence does not satisfy Phase1-A:

```text
main comparison wins > 0 or segment-level wins > 0
```

In the current run:

```text
main wins = 0 / 12
segment wins = 0 / 30
```

Therefore the script/report recommends rollback to step 5-6 of the long research loop:

```text
theoretical feasibility -> method design
```

The main interpretation is not that future-side modeling is invalid, but that replacing the strong
fixed flatten head with a low-capacity segment query decoder removes too much readout capacity.
