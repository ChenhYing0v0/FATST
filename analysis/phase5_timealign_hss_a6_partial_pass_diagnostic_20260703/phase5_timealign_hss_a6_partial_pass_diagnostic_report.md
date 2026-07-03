# Phase5-A6 Partial-Pass Diagnostic Report

本文档记录 A6 partial-pass 后的 diagnostic-only 分析。主协议保持 `official-last` / without early stop；
`best-val` 只允许作为后续 optional upper-bound audit，不能替代 paper metric 或作为 main protocol。

## Diagnostic Question

[Fact] A6-LBF 已恢复 dense-capacity path，但对 `best_stage_control` 仍为 `0/12` wins。

[Question] 剩余差距更像是 official-last trajectory drift、learned-basis operator 结构限制，还是 objective conflict？

## Official-Last Trajectory

[Strong Evidence] ETTh2 是主要 trajectory drift 来源：ETTh2 三个 A6 arms 的 last-vs-best validation MSE 平均漂移 `11.81%`，而 ETTm1/Weather 平均仅 `0.12%`。

| Dataset | Arms | mean last-vs-best val | max last-vs-best val | mean gap vs best control |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | 3 | +11.81% | +13.44% | +2.12% |
| ETTm1 | 3 | +0.03% | +0.08% | +0.56% |
| Weather | 3 | +0.21% | +0.32% | +0.07% |

### ETTh2 Arm Detail

| Arm | best epoch | last-vs-best val | mean gap vs best control |
| --- | ---: | ---: | ---: |
| `a6_der` | 1 | +13.44% | +2.15% |
| `a6_lbf_r256` | 3 | +9.85% | +2.16% |
| `a6_lbf_r512` | 3 | +12.13% | +2.04% |

## Learned-Basis Structure

[Strong Evidence] `r512` 的 operator rank99 扩张没有转化为有效性能收益：`r256` induced operator 的 mean rank99 为 `209.33`，`r512` 为 `316.33`，但 mean effective rank 仅从 `82.91` 到 `92.80`，且 A6 gate 中 r512 没有稳定优于 r256。

[Fact] learned temporal basis 的 adjacent-row cosine 较低：`r256` mean `0.10`，`r512` mean `0.09`。这说明它不是简单平滑 Fourier-like basis，而是在学习更接近 dense row bank 的时间位置字典。

| Dataset | Arm | basis rank | basis eff-rank | operator eff-rank | operator rank99 | adjacent cosine |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | `a6_lbf_r256` | 256 | 186.84 | 16.63 | 214 | 0.16 |
| ETTh2 | `a6_lbf_r512` | 512 | 294.32 | 16.14 | 340 | 0.15 |
| ETTm1 | `a6_lbf_r256` | 256 | 199.91 | 86.21 | 172 | 0.12 |
| ETTm1 | `a6_lbf_r512` | 512 | 323.64 | 77.45 | 197 | 0.08 |
| Weather | `a6_lbf_r256` | 256 | 205.55 | 145.89 | 242 | 0.04 |
| Weather | `a6_lbf_r512` | 512 | 319.68 | 184.82 | 412 | 0.03 |

## Statistic Definitions

`phase5_timealign_hss_a6_official_last_trajectory_diagnostic.csv` 的来源是每个 A6 run 的 `training_log.csv` 与 A6 comparison table。`first_val_mean_mse`、`best_val_mean_mse`、`last_val_mean_mse` 分别取 validation MSE 的第 1 epoch、最小值 epoch、最后 epoch；`last_vs_best_val_mse_pct = last_val_mean_mse / best_val_mean_mse - 1`，用于衡量 official-last checkpoint 相对训练轨迹内 best validation point 的漂移；`mean_relative_mse_vs_best_stage_control_pct` 来自 A6 comparison table，表示该 run 在 4 个 horizons 上相对 best stage control 的平均 test MSE gap。

`phase5_timealign_hss_a6_basis_structure_diagnostic.csv` 的来源是 A6-LBF `checkpoint.pt` 中的 `learned_temporal_basis`、`learned_basis_coeff.weight` 和 `learned_temporal_bias`。`basis_effective_rank`、`operator_effective_rank` 分别对 basis matrix 与 induced operator `learned_temporal_basis @ learned_basis_coeff.weight` 的 singular-value energy 分布计算 entropy effective rank；`operator_rank99` 是累计 singular-value energy 达到 99% 所需 rank；`basis_adjacent_cosine_mean` 是相邻 future rows 的平均 cosine similarity，用于判断 learned basis 更接近平滑时间函数还是 dense row dictionary。

## Decision

[Decision] 不启动 `best-val/early-stopping` 主实验。下一步实验应继续保持 official-last，优先解决 ETTh2 上 early-best 后 drift 的 optimization/objective conflict，以及 learned-basis operator 没有转化为 best-control wins 的机制缺口。

[Hypothesis] A6-LBF 的问题不是 rank 不够，而是 learned-basis operator 已近 dense-equivalent ceiling，但 official-last multi-prefix objective 在 ETTh2 上把模型推离 early-best basin；Weather/ETTm1 的剩余差距更像 best controls 本身含有 teacher/nested regularization advantage。

## Artifacts

- `phase5_timealign_hss_a6_official_last_trajectory_diagnostic.csv`
- `phase5_timealign_hss_a6_official_last_trajectory_summary.csv`
- `phase5_timealign_hss_a6_basis_structure_diagnostic.csv`
- `phase5_timealign_hss_a6_partial_pass_diagnostic_config.json`
