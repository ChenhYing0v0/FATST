# Phase5-A6ST Official-Last Self-Teacher Gate Report

本文档分析 A6ST stability gate。数据集范围：`ETTm1, Weather`。所有 run 使用 `official-last` / without early stop。

## Conclusion

[Fact] 最佳 variant 为 `a6st_w02_d0999_wu1`：相对 best stage control 平均 `+1.20%`，wins `0/8`，last-vs-best validation drift `+0.14%`。

[Strong Evidence] 最佳 variant 相对 A6-LBF-r256 的平均 MSE 变化为 `+0.95%`。

[Fact] self-teacher 为 train-time detached EMA teacher consistency；最终评估仍使用 raw `official-last` student weights，不使用 `ema_eval`。

## Variant Summary

| Variant | mean MSE | vs best control | wins | vs A6-LBF-r256 | last-vs-best val | EMA | self-teacher | gate | smooth/train |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `a6st_w02_d0999_wu1` | 0.2771 | +1.20% | 0/8 | +0.95% | +0.14% | 0 | w=0.2, d=0.999, wu=1 | none:1.00 | 0.00e+00 |

## Dataset Summary

| Dataset | Variant | mean MSE | vs best control | wins | vs A6-LBF-r256 | last-vs-best val | gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETTm1` | `a6st_w02_d0999_wu1` | 0.3375 | +1.49% | 0/4 | +1.05% | +0.00% | 1.00 |
| `Weather` | `a6st_w02_d0999_wu1` | 0.2166 | +0.91% | 0/4 | +0.86% | +0.28% | 1.00 |

## Gate Decision

[Decision] 该 gate 的 effectiveness 必须同时检查 raw final checkpoint 是否改善、是否跨 dataset 安全、以及是否只是 ETTh2-specific repair。

[Decision] 若 ETTm1/Weather 出现系统性负向，即使 ETTh2 改善，也不能把当前 self-teacher setting 升级为 paper-core universal method。

[Decision] 下一步应回 Step 4/5 重审为什么 stability target 对 ETTh2 有益但对 ETTm1/Weather 负向；不得直接做 full-matrix 扩大实验。

## Reader Path

先读取 `phase5_timealign_hss_a6s_summary.csv` 判断 variant-level gate，再读取 `phase5_timealign_hss_a6s_dataset_summary.csv` 判断 dataset-level 安全性，最后读取 `phase5_timealign_hss_a6s_comparison.csv` 判断 prefix-wise wins/gaps，并回到 stage ledger 写入 11-step decision。

## Artifacts

- `phase5_timealign_hss_a6s_comparison.csv`
- `phase5_timealign_hss_a6s_summary.csv`
- `phase5_timealign_hss_a6s_dataset_summary.csv`
- `phase5_timealign_hss_a6s_analysis_config.json`
- ignored raw metrics/logs under `raw/`
