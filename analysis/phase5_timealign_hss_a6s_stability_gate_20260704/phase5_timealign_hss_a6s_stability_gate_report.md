# Phase5-A6S Official-Last Stability Gate Report

本文档分析 A6S ETTh2-only stability gate。所有 run 使用 `official-last` / without early stop。

## Conclusion

[Fact] 最佳 variant 为 `lbf_r256_ema099_smooth1e3`：相对 ETTh2 best stage control 平均 `+2.00%`，wins `0/4`，last-vs-best validation drift `+9.86%`。

[Strong Evidence] `ema_decay=0.99` 只带来约 `-0.15%` 的 A6-LBF 相对改善，不足以修复 best-control gap。

[Fact] 本轮 smoothness regularizer 实际强度很弱：最大 `weighted_smoothness / train_loss = 4.86e-07`。因此该结果只能否定未校准的 `smooth1e-3`，不能严格否定 operator-level stability 方向。

## Variant Summary

| Variant | mean MSE | vs best control | wins | vs A6-LBF-r256 | last-vs-best val | EMA | smooth weight | smooth/train |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lbf_r256_ema099_smooth1e3` | 0.3075 | +2.00% | 0/4 | -0.15% | +9.86% | 0.99 | 0.001 | 4.86e-07 |
| `lbf_r256_ema099` | 0.3075 | +2.01% | 0/4 | -0.14% | +9.85% | 0.99 | 0 | 0.00e+00 |
| `der_ema099` | 0.3079 | +2.15% | 0/4 | -0.01% | +13.44% | 0.99 | 0 | 0.00e+00 |
| `lbf_r256_smooth1e3` | 0.3080 | +2.15% | 0/4 | -0.01% | +9.86% | 0 | 0.001 | 4.86e-07 |
| `lbf_r256_base` | 0.3080 | +2.16% | 0/4 | +0.00% | +9.85% | 0 | 0 | 0.00e+00 |

## Gate Decision

[Decision] A6S minimal gate 未通过 effectiveness gate：最佳 variant 仍对 ETTh2 best stage control `0/4` win，平均 MSE 仍差约 `+2.00%`。

[Decision] 不把 `A6S-EMA` 推为 paper-core；`EMA-0.99` 只作为弱正向 control evidence。

[Decision] `A6S-HeadStability` 需要先做 diagnostic-only strength calibration。当前 `smooth1e-3` 的优化权重过低，不能作为机制失败的充分证据。

## Next Step

回 Step 4/5 设计 `A6S2_stability_calibration_gate`：仍保持 `official-last` / without early stop，使用 ETTh2-only diagnostic 检查 `ema_decay=0.995/0.999` 与更强 operator smoothness weight。该 gate 不宣称 paper-core，只判断 stability route 是否还有继续设计价值。

## Artifacts

- `phase5_timealign_hss_a6s_comparison.csv`
- `phase5_timealign_hss_a6s_summary.csv`
- `phase5_timealign_hss_a6s_analysis_config.json`
- ignored raw metrics/logs under `raw/`
