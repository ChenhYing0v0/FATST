# Phase5-A6ST Official-Last Self-Teacher Gate Report

本文档分析 A6ST ETTh2-only stability gate。所有 run 使用 `official-last` / without early stop。

## Conclusion

[Fact] 最佳 variant 为 `a6st_w02_d0999_wu1`：相对 ETTh2 best stage control 平均 `+0.21%`，wins `2/4`，last-vs-best validation drift `+3.86%`。

[Strong Evidence] 最佳 variant 相对 A6-LBF-r256 的平均 MSE 变化为 `-1.91%`。

[Fact] 本轮 smoothness regularizer 的最大实际强度为 `weighted_smoothness / train_loss = 0.00e+00`。该值用于判断 regularizer 是否真的进入优化，而不是只看 flag 是否开启。

## Variant Summary

| Variant | mean MSE | vs best control | wins | vs A6-LBF-r256 | last-vs-best val | EMA | self-teacher | smooth/train |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `a6st_w02_d0999_wu1` | 0.3022 | +0.21% | 2/4 | -1.91% | +3.86% | 0 | w=0.2, d=0.999, wu=1 | 0.00e+00 |
| `a6st_w01_d0999_wu1` | 0.3036 | +0.70% | 1/4 | -1.42% | +6.14% | 0 | w=0.1, d=0.999, wu=1 | 0.00e+00 |
| `a6st_w01_d0999_wu3` | 0.3039 | +0.78% | 1/4 | -1.35% | +6.76% | 0 | w=0.1, d=0.999, wu=3 | 0.00e+00 |
| `a6st_w01_d0995_wu1` | 0.3046 | +1.05% | 1/4 | -1.09% | +7.90% | 0 | w=0.1, d=0.995, wu=1 | 0.00e+00 |
| `a6st_w005_d0999_wu1` | 0.3054 | +1.31% | 1/4 | -0.83% | +7.73% | 0 | w=0.05, d=0.999, wu=1 | 0.00e+00 |

## Gate Decision

[Decision] A6ST ETTh2 gate 通过为 `partial_pass_etth2_raw_final_stabilized`。最佳 variant
`a6st_w02_d0999_wu1` 相对 A6-LBF-r256 平均 MSE `-1.91%`，相对 ETTh2 best stage control
仅差 `+0.21%`，并达到 `2/4` wins。

[Strong Evidence] A6ST 已超过 A6S2 `lbf_r256_ema0999` control：A6S2 EMA-0.999 为
`+0.67%` vs best control、`1/4` wins、last-vs-best drift `+9.85%`；A6ST best 为
`+0.21%`、`2/4` wins、drift `+3.86%`。因此它不是简单 test-time EMA 替换，而是让 raw
official-last checkpoint 更稳定。

[Decision] `self_teacher_loss_weight=0.20, decay=0.999, warmup=1` 是当前 active setting。
`weight=0.05/0.10` 与 `decay=0.995` 均较弱，说明足够强的 EMA-teacher consistency 是关键。

[Decision] 该结果仍只是 ETTh2-only partial pass。下一步必须做 cross-dataset sanity gate，
检查 A6ST 是否损害 ETTm1/Weather；若不损害，再进入 full matrix 或 paper-core method refinement。

## Reader Path

先读取 `phase5_timealign_hss_a6s_summary.csv` 判断 variant-level gate，再读取
`phase5_timealign_hss_a6s_comparison.csv` 判断 prefix-wise wins/gaps。A6ST 的关键读法是：
raw official-last checkpoint 是否接近或超过 A6S2 EMA-0.999 control，而不是是否复现 EMA eval。

## Artifacts

- `phase5_timealign_hss_a6s_comparison.csv`
- `phase5_timealign_hss_a6s_summary.csv`
- `phase5_timealign_hss_a6s_analysis_config.json`
- ignored raw metrics/logs under `raw/`
