# Phase3-B Regime/Segment Mechanism Diagnostic

更新时间：2026-06-24

## 1. 目的

Phase3-A 说明 R.3 的剩余问题分成两类：

1. short gaps 来自 short-only extra windows；
2. H720 gaps 来自 late segments。

用户明确指出不希望把下一步做成 output residual correction。Phase3-B 因此只检查一个问题：

> 这些困难 regime/segment 是否能用预测前可用的信息识别？

如果不能识别，就不应实现任何 residual calibration；如果能识别，下一步也应作用在
`target state` / `segment operator` 层，而不是最终输出后加 residual。

## 2. Script

入口：

- `scripts/analyze_phase3_regime_segment_mechanism.py`

输出：

- `analysis/phase3_regime_segment_mechanism_20260624/phase3_regime_segment_mechanism_report.md`;
- `analysis/phase3_regime_segment_mechanism_20260624/phase3_short_regime_preinput_features.csv`;
- `analysis/phase3_regime_segment_mechanism_20260624/phase3_h720_late_segment_preinput_features.csv`;
- `analysis/phase3_regime_segment_mechanism_20260624/phase3_regime_segment_mechanism_summary.json`。

## 3. Inputs

历史输入窗口来自 `ForecastDataset(..., flag="test")`：

- `x`: `[N, 336, C]`;
- no future target is used as feature。

R.3 prediction artifacts 只用于构造 diagnostic labels：

- short extra-window label；
- H720 segment top-quartile error label。

这些 labels 不进入未来模型，只用于判断 history/window-position features 是否能解释已观察到的
失败区域。

## 4. Feature Construction

每个 test window 计算以下预测前特征：

- `history_mean`;
- `history_std`;
- `history_abs_mean`;
- `history_last_abs_mean`;
- `history_recent_mean`;
- `history_recent_std`;
- `history_recent_minus_previous_mean`;
- `history_second_minus_first_mean`;
- `history_slope_abs_mean`;
- `window_index_norm`。

其中 `window_index_norm` 是 split 内位置。它是预测前已知的 time-position signal，但如果后续模型只
依赖它，论文机制会偏弱；更稳妥的模型应把它和 history statistics 一起作为 regime token 的输入。

## 5. Short-Regime Diagnostic

对每个 `(dataset, horizon)`：

1. 构造 `pred_len=horizon` 的完整 test windows；
2. 取 `pred_len=720` 可覆盖的前 `N_720` 个 windows 作为 negative group；
3. 取 short horizon 独有的末端 windows 作为 positive group；
4. 比较 positive vs negative 的 history features；
5. 输出 standardized mean difference 和 rank AUC。

该诊断回答：short-only extra windows 是否是可识别的 input regime。

## 6. H720 Late-Segment Diagnostic

对 H720 每个 segment：

1. 用 R.3 prediction 和 true 计算每个 window 的 segment MSE；
2. top quartile segment error 作为 positive group；
3. lower three quartiles 作为 negative group；
4. 比较同一批 history features。

该诊断回答：late segment hard cases 是否能由预测前 history/window-position features 识别。

## 7. Decision Boundary

[Code Realization] 当前代码只做诊断，不改变模型，不加 residual head。

[Mechanism Boundary] 下一步若实现模型，应是：

- history/window-position -> regime token/router；
- target segment features -> segment-aware operator；
- conditioned target state/readout -> prediction。

不应是：

- prediction -> arbitrary residual correction -> adjusted prediction。

[Gate] 只有当 prediction-before features 对已观察 failure groups 有稳定 separation 时，才允许进入
conditioned target-operator design。
