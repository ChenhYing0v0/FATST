# Phase3-C Regime/Segment-Conditioned Target Operator

更新时间：2026-06-24

## 1. 目的

Phase3-C 实现 `PatchEncoderRegimeSegmentTargetOperator`。它不是 output residual correction：

- 不读取 future target；
- 不在 prediction 后加 residual；
- 只在 `segment_output` 前调制 readout hidden state；
- residual/error 仍只作为诊断和 gate label。

## 2. Data Path

训练入口仍是：

- `baselines/patch_encoder_target_set_decoder/train.py`

新增 model variant：

- `--model-variant regime_segment_operator`

数据集新增可选返回：

- 默认：`ForecastDataset.__getitem__ -> (x, y)`；
- 当 `return_index=True`：`ForecastDataset.__getitem__ -> (x, y, window_index_norm)`。

`window_index_norm` 是 split 内窗口位置，范围约为 `[0, 1]`。它只在显式传入
`--use-window-position` 时启用；不传该参数时，新 variant 会把 position feature 置为 0，用于
history-only ablation。

## 3. Forward Computation

输入：

- `x`: `[B, L, C]`;
- optional `window_index_norm`: `[B]`;
- `pred_len`: target horizon。

基础路径沿用 R.3：

1. RevIN normalization 得到 normalized `x`；
2. `_encode(x)` 生成 patch memory `z`: `[B*C, P, d_model]`；
3. `_target_states(z, pred_len, segment_count)` 生成 `target_states`: `[B*C, S, d_model]`；
4. `history_projector(z)` 生成 `history_readout`: `[B*C, readout_dim]`；
5. `condition_head(target_states)` 生成 target-conditioned `gamma/beta`；
6. `conditioned = history_readout * (1 + gamma) + beta`，形状 `[B*C, S, readout_dim]`。

Phase3-C 在第 6 步之后、第 7 步 `segment_output` 之前插入 operator：

1. `_history_regime_features(x, window_index_norm)` 从 history 构造 `[B*C, 10]` features；
2. `regime_encoder(features)` 得到 regime token；
3. `_target_features(...)` 经 `target_feature_embedding` 得到 segment token；
4. 拼接 regime token 和 segment token；
5. `regime_segment_operator` 输出 bounded scale/shift；
6. `conditioned = conditioned * (1 + scale) + shift`；
7. `segment_output(conditioned)` 生成 normalized prediction segments。

## 4. Regime Features

每个 batch-channel 使用 10 个 prediction-before features：

- `history_mean`;
- `history_std`;
- `history_abs_mean`;
- `history_last_abs`;
- `history_recent_mean`;
- `history_recent_std`;
- `history_recent_minus_previous`;
- `history_second_minus_first`;
- `history_slope_abs`;
- `window_index_norm`。

这些量都来自 history input 和窗口位置，不来自 future target。

## 5. Initialization And Safety

`regime_segment_operator` 的最后一层 zero initialization。因此初始 scale/shift 为 0，模型初始行为接近
R.3 的 target-set decoder。

Scale/shift 经过 `0.1 * tanh(...)` bounded transform，避免 operator 在训练早期大幅破坏 base readout。

## 6. Artifacts

当 `--model-variant regime_segment_operator` 时，每个 horizon eval 目录会额外写出：

- `regime_segment_operator_stats.csv`: scale/shift 的 mean/max absolute values；
- `regime_feature_stats.csv`: history/window-position features 的均值、标准差和幅度。

这些 artifact 用于判断 learned operator 是否真的利用 prediction-before regime signal。

## 7. Window-Position Boundary

`window_index_norm` 是 prediction-before signal，但它不是稳定 causal variable。它按 split 内 index
归一化，可能编码 train/val/test split 位置而不是可部署 regime。Phase3-C 的机制结论必须依赖
history-only control 或显式 calendar feature control，而不能只依赖当前 window-position run。

## 8. Verification

本地 smoke 命令：

```bash
conda run -n r2026-fsa python baselines/patch_encoder_target_set_decoder/train.py \
  --dataset-root /Users/river/PaperResearch/Project/datasets \
  --dataset ETTh2 \
  --target-horizons 96,720 \
  --epochs 1 \
  --steps-per-epoch 1 \
  --max-eval-batches 1 \
  --batch-size 4 \
  --model-variant regime_segment_operator \
  --use-window-position \
  --run-name smoke_PatchEncoderRegimeSegmentTargetOperator \
  --output-root artifacts/runs/smoke_phase3_regime_segment_operator \
  --device cpu
```

Smoke result:

- completed one train step and 1-batch eval for H96/H720；
- `prefix_mismatch_mse = 1.015119944076633e-14`；
- `regime_segment_operator_stats.csv` and `regime_feature_stats.csv` were written。
