# Phase5-A7DG Official-Last Selective Self-Teacher Gate Report

本文档分析 A7DG stability gate。数据集范围：`ETTh2, ETTm1, Weather`。所有 run 使用 `official-last` / without early stop。

## Conclusion

[Fact] 最佳 variant 为 `a7dg_abs004_t001_w02_d0999_wu1`：相对 best stage control 平均 `+0.46%`，wins `2/12`，last-vs-best validation drift `+1.43%`。

[Strong Evidence] 最佳 variant 相对 A6-LBF-r256 的平均 MSE 变化为 `-0.40%`。

[Fact] self-teacher 为 train-time detached EMA teacher consistency；最终评估仍使用 raw `official-last` student weights，不使用 `ema_eval`。

## Variant Summary

| Variant | mean MSE | vs best control | wins | vs A6-LBF-r256 | last-vs-best val | EMA | self-teacher | gate | smooth/train |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `a7dg_abs004_t001_w02_d0999_wu1` | 0.2844 | +0.46% | 2/12 | -0.40% | +1.43% | 0 | w=0.2, d=0.999, wu=1 | absolute:0.47 | 0.00e+00 |
| `a7dg_ratio010_t002_w02_d0999_wu1` | 0.2845 | +0.50% | 2/12 | -0.37% | +1.43% | 0 | w=0.2, d=0.999, wu=1 | ratio:0.50 | 0.00e+00 |
| `a7dg_ratio008_t002_w02_d0999_wu1` | 0.2848 | +0.60% | 2/12 | -0.27% | +1.42% | 0 | w=0.2, d=0.999, wu=1 | ratio:0.61 | 0.00e+00 |

## Dataset Summary

| Dataset | Variant | mean MSE | vs best control | wins | vs A6-LBF-r256 | last-vs-best val | gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETTh2` | `a7dg_abs004_t001_w02_d0999_wu1` | 0.3021 | +0.16% | 2/4 | -1.95% | +3.81% | 0.88 |
| `ETTh2` | `a7dg_ratio010_t002_w02_d0999_wu1` | 0.3021 | +0.18% | 2/4 | -1.93% | +3.82% | 0.92 |
| `ETTh2` | `a7dg_ratio008_t002_w02_d0999_wu1` | 0.3022 | +0.20% | 2/4 | -1.92% | +3.83% | 0.96 |
| `ETTm1` | `a7dg_abs004_t001_w02_d0999_wu1` | 0.3360 | +0.95% | 0/4 | +0.51% | +0.07% | 0.31 |
| `ETTm1` | `a7dg_ratio010_t002_w02_d0999_wu1` | 0.3360 | +0.96% | 0/4 | +0.53% | +0.07% | 0.31 |
| `ETTm1` | `a7dg_ratio008_t002_w02_d0999_wu1` | 0.3364 | +1.10% | 0/4 | +0.66% | +0.07% | 0.43 |
| `Weather` | `a7dg_abs004_t001_w02_d0999_wu1` | 0.2153 | +0.28% | 0/4 | +0.23% | +0.39% | 0.22 |
| `Weather` | `a7dg_ratio010_t002_w02_d0999_wu1` | 0.2154 | +0.35% | 0/4 | +0.30% | +0.39% | 0.27 |
| `Weather` | `a7dg_ratio008_t002_w02_d0999_wu1` | 0.2157 | +0.50% | 0/4 | +0.45% | +0.37% | 0.45 |

## Gate Decision

[Strong Evidence] A7DG 相对 uniform A6ST 有稳定改善：最佳 `a7dg_abs004_t001_w02_d0999_wu1`
相对 uniform A6ST 平均 MSE `-0.40%`，`11/12` horizons 更好；其中 ETTh2 `-0.05%`
（`4/4`）、ETTm1 `-0.53%`（`3/4`）、Weather `-0.62%`（`4/4`）。

[Strong Evidence] Gate 机制按 dataset 产生了预期分离：最佳 variant 的 `train_self_teacher_gate`
在 ETTh2 为 `0.88`，ETTm1 为 `0.31`，Weather 为 `0.22`。这说明 selective objective
不是空开关，确实在低 drift 数据集上削弱了 consistency force。

[Decision] 但 A7DG 仍未通过 paper-core effectiveness gate：最佳 variant 相对 best controls 仍为
`+0.46%`、wins `2/12`；ETTm1 相对 A6-LBF-r256 仍 `+0.51%`，Weather 仍 `+0.23%`。
因此当前结论是 `partial_positive_selective_stability_signal`，不是 method pass。

[Decision] A7DG effectiveness 必须同时看三点：是否保留 ETTh2 positive signal，是否降低 ETTm1/Weather 的 uniform A6ST 负向，以及 `train_self_teacher_gate` 是否按 dataset 产生选择性降权。

[Decision] 若 A7DG 只优于 uniform A6ST 但仍系统性弱于 A6-LBF 或 best controls，则它只能作为 selective-stability partial evidence，不能直接升级为 paper-core。

[Decision] 若 gate 强度在 ETTh2 显著高于 ETTm1/Weather，且 metrics 接近 A6-LBF，则下一步应围绕 adaptive/selective stability objective 做更严格 narrative gate，而不是继续人工调 threshold。

## Reader Path

先读取 `phase5_timealign_hss_a6s_summary.csv` 判断 variant-level gate，再读取 `phase5_timealign_hss_a6s_dataset_summary.csv` 判断 dataset-level 安全性，最后读取 `phase5_timealign_hss_a6s_comparison.csv` 判断 prefix-wise wins/gaps，并回到 stage ledger 写入 11-step decision。

## Artifacts

- `phase5_timealign_hss_a6s_comparison.csv`
- `phase5_timealign_hss_a6s_summary.csv`
- `phase5_timealign_hss_a6s_dataset_summary.csv`
- `phase5_timealign_hss_a7dg_vs_uniform_a6st.csv`
- `phase5_timealign_hss_a6s_analysis_config.json`
- ignored raw metrics/logs under `raw/`
