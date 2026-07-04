# Phase5-A6S2 Official-Last Stability Calibration Gate Report

本文档分析 A6S2 ETTh2-only stability gate。所有 run 使用 `official-last` / without early stop。

## Conclusion

[Fact] 最佳 variant 为 `lbf_r256_ema0999`：相对 ETTh2 best stage control 平均 `+0.67%`，wins `1/4`，last-vs-best validation drift `+9.85%`。

[Strong Evidence] 最佳 variant 相对 A6-LBF-r256 的平均 MSE 变化为 `-1.46%`。

[Fact] 本轮 smoothness regularizer 的最大实际强度为 `weighted_smoothness / train_loss = 2.99e-03`。该值用于判断 regularizer 是否真的进入优化，而不是只看 flag 是否开启。

## Variant Summary

| Variant | mean MSE | vs best control | wins | vs A6-LBF-r256 | last-vs-best val | EMA | smooth weight | smooth/train |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lbf_r256_ema0999` | 0.3038 | +0.67% | 1/4 | -1.46% | +9.85% | 0.999 | 0 | 0.00e+00 |
| `lbf_r256_ema0995` | 0.3074 | +1.98% | 0/4 | -0.17% | +9.85% | 0.995 | 0 | 0.00e+00 |
| `lbf_r256_ema0995_smooth10` | 0.3077 | +2.11% | 0/4 | -0.05% | +10.17% | 0.995 | 10 | 1.78e-03 |
| `lbf_r256_smooth10` | 0.3083 | +2.28% | 0/4 | +0.12% | +10.17% | 0 | 10 | 1.78e-03 |
| `lbf_r256_smooth100` | 0.3101 | +2.85% | 0/4 | +0.68% | +10.81% | 0 | 100 | 2.99e-03 |

## Gate Decision

[Decision] 该 gate 的 effectiveness 必须结合 best-control gap、wins、A6-LBF 相对改善和 regularizer 实际强度判断；不能只看单个 variant 的平均 MSE。

[Decision] 若改善主要来自 EMA，则它首先是 generic trajectory-averaging control evidence，不能直接升级为 paper-core。

[Decision] 若 stronger smoothness 独立改善，才支持继续设计 operator-level stability mechanism；若 stronger smoothness 变差，则应暂停该 route。

## Reader Path

先读取 `phase5_timealign_hss_a6s_summary.csv` 判断 variant-level gate，再读取 `phase5_timealign_hss_a6s_comparison.csv` 判断 prefix-wise wins/gaps，最后回到 stage ledger 写入 11-step decision。

## Artifacts

- `phase5_timealign_hss_a6s_comparison.csv`
- `phase5_timealign_hss_a6s_summary.csv`
- `phase5_timealign_hss_a6s_analysis_config.json`
- ignored raw metrics/logs under `raw/`
