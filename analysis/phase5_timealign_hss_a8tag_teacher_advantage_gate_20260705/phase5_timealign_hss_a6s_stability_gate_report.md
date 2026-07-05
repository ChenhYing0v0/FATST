# Phase5-A8TAG Official-Last Teacher-Advantage Gate Report

本文档分析 A8TAG stability gate。数据集范围：`ETTh2, ETTm1, Weather`。所有 run 使用 `official-last` / without early stop。

## Conclusion

[Fact] 最佳 variant 为 `a8tag_advratio_w10_d0999_wu1`：相对 best stage control 平均 `+0.91%`，wins `0/12`，last-vs-best validation drift `+3.42%`。

[Strong Evidence] 最佳 variant 相对 A6-LBF-r256 的平均 MSE 变化为 `+0.03%`。

[Fact] 本轮 smoothness regularizer 的最大实际强度为 `weighted_smoothness / train_loss = 0.00e+00`。该值用于判断 regularizer 是否真的进入优化，而不是只看 flag 是否开启。

## Variant Summary

| Variant | mean MSE | vs best control | wins | vs A6-LBF-r256 | last-vs-best val | EMA | self-teacher | gate | smooth/train |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `a8tag_advratio_w10_d0999_wu1` | 0.2858 | +0.91% | 0/12 | +0.03% | +3.42% | 0 | w=1, d=0.999, wu=1 | teacher-advantage-ratio:0.01 | 0.00e+00 |
| `a8tag_advbin_w02_d0999_wu1` | 0.2873 | +1.45% | 0/12 | +0.57% | +3.41% | 0 | w=0.2, d=0.999, wu=1 | teacher-advantage-binary:0.65 | 0.00e+00 |
| `a8tag_advbin_w05_d0999_wu1` | 0.2884 | +1.93% | 0/12 | +1.05% | +3.38% | 0 | w=0.5, d=0.999, wu=1 | teacher-advantage-binary:0.64 | 0.00e+00 |

## Dataset Summary

| Dataset | Variant | mean MSE | vs best control | wins | vs A6-LBF-r256 | last-vs-best val | gate | teacher advantage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETTh2` | `a8tag_advbin_w05_d0999_wu1` | 0.3079 | +2.14% | 0/4 | -0.02% | +9.87% | 0.00 | -0.0156 |
| `ETTh2` | `a8tag_advbin_w02_d0999_wu1` | 0.3080 | +2.15% | 0/4 | -0.01% | +9.86% | 0.00 | -0.0156 |
| `ETTh2` | `a8tag_advratio_w10_d0999_wu1` | 0.3080 | +2.16% | 0/4 | -0.00% | +9.86% | 0.00 | -0.0156 |
| `ETTm1` | `a8tag_advratio_w10_d0999_wu1` | 0.3345 | +0.48% | 0/4 | +0.05% | +0.08% | 0.01 | +0.0039 |
| `ETTm1` | `a8tag_advbin_w02_d0999_wu1` | 0.3376 | +1.46% | 0/4 | +1.02% | +0.04% | 0.95 | +0.0011 |
| `ETTm1` | `a8tag_advbin_w05_d0999_wu1` | 0.3390 | +1.95% | 0/4 | +1.50% | +0.00% | 0.92 | +0.0006 |
| `Weather` | `a8tag_advratio_w10_d0999_wu1` | 0.2148 | +0.09% | 0/4 | +0.04% | +0.34% | 0.01 | +0.0043 |
| `Weather` | `a8tag_advbin_w02_d0999_wu1` | 0.2163 | +0.76% | 0/4 | +0.71% | +0.32% | 1.00 | +0.0028 |
| `Weather` | `a8tag_advbin_w05_d0999_wu1` | 0.2182 | +1.70% | 0/4 | +1.65% | +0.27% | 1.00 | +0.0016 |

## Gate Decision

[Decision] A8TAG effectiveness 必须同时看三点：teacher 是否在 supervised prefix 上确实有正 advantage，teacher-advantage gate 是否避免低质量 teacher imitation，以及 metrics 是否超过 A7DG/A6-LBF。

[Decision] 若 teacher advantage 多数为负或接近零，说明 EMA teacher 不是可靠 target，应停止 self-teacher route。

[Decision] 若 teacher advantage gate 改善 ETTm1/Weather 但损失 ETTh2 gain，则需要回 Step 4/5 重新建模 stability 与 capacity 的冲突，而不是加回 threshold。

## Reader Path

先读取 `phase5_timealign_hss_a6s_summary.csv` 判断 variant-level gate，再读取 `phase5_timealign_hss_a6s_dataset_summary.csv` 判断 dataset-level 安全性，最后读取 `phase5_timealign_hss_a6s_comparison.csv` 判断 prefix-wise wins/gaps，并回到 stage ledger 写入 11-step decision。

## Artifacts

- `phase5_timealign_hss_a6s_comparison.csv`
- `phase5_timealign_hss_a6s_summary.csv`
- `phase5_timealign_hss_a6s_dataset_summary.csv`
- `phase5_timealign_hss_a6s_analysis_config.json`
- ignored raw metrics/logs under `raw/`
