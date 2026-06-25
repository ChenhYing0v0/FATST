# Phase4-R3D Decomposition Gate 分析报告

## 11-Step 记录

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10：评估 decomposition artifacts，并决定 rollback 与下一步方向 |
| `problem` | R.3 是 compound baseline，需要拆开 mixed-horizon exposure 与 prefix-risk weighting |
| `existence_evidence` | 远程 gate 包含 full-time、mixed-horizon uniform、single-720 prefix-risk、R.3 四组 controls |
| `idea` | 将 R.3 分解为 exposure-only、weighting-only、compound 三类证据 |
| `theory_check` | 在识别主导因素前，不能直接用 compound baseline 否定或肯定 HSS |
| `design` | ETTh2/Weather，horizons 96/192/336/720，seed 2021，相同 target-set carrier |
| `gate` | 判断 R.3 优势主要来自 exposure、weighting，还是二者 interaction |
| `artifacts` | `analysis/phase4_r3_decomposition_gate_20260625` |
| `decision` | Prefix-risk pressure 是更可用的主导因素；mixed-horizon exposure 有收益但单独不足 |

## 主结果

[Fact] `mixed_horizon_uniform` vs full-time: mean MSE `-4.12%`, beats full-time `8/8`, best among candidates `1/8`.
[Fact] `single_720_prefix_risk` vs full-time: mean MSE `-4.39%`, beats full-time `8/8`, best among candidates `2/8`.
[Fact] compound R.3 vs full-time: mean MSE `-4.83%`, beats full-time `8/8`, best among candidates `5/8`.
[Fact] R.3 vs `single_720_prefix_risk`: mean MSE `-0.43%`; lower is better for R.3.
[Fact] R.3 vs `mixed_horizon_uniform`: mean MSE `-0.74%`.

## 按 Horizon 分解

| Dataset | Horizon | Full | Mixed uniform | Single prefix | R.3 | Winner | Mixed vs full | Single vs full | R.3 vs full |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `ETTh2` | 96 | 0.339358 | 0.308910 | 0.301289 | 0.304796 | `single_720_prefix_risk` | -8.97% | -11.22% | -10.18% |
| `ETTh2` | 192 | 0.388740 | 0.369166 | 0.365306 | 0.369043 | `single_720_prefix_risk` | -5.04% | -6.03% | -5.07% |
| `ETTh2` | 336 | 0.394549 | 0.380801 | 0.381545 | 0.382910 | `mixed_horizon_uniform` | -3.48% | -3.30% | -2.95% |
| `ETTh2` | 720 | 0.419201 | 0.413407 | 0.416938 | 0.410473 | `r3_prefix_risk` | -1.38% | -0.54% | -2.08% |
| `Weather` | 96 | 0.154701 | 0.150202 | 0.149224 | 0.148026 | `r3_prefix_risk` | -2.91% | -3.54% | -4.31% |
| `Weather` | 192 | 0.200707 | 0.194525 | 0.194019 | 0.192409 | `r3_prefix_risk` | -3.08% | -3.33% | -4.13% |
| `Weather` | 336 | 0.256879 | 0.247248 | 0.247812 | 0.244793 | `r3_prefix_risk` | -3.75% | -3.53% | -4.70% |
| `Weather` | 720 | 0.338482 | 0.323659 | 0.326087 | 0.320847 | `r3_prefix_risk` | -4.38% | -3.66% | -5.21% |

## 按 Dataset 汇总

| Dataset | Mixed vs full | Single vs full | R.3 vs full | R.3 vs single | Winner counts M/S/R3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ETTh2` | -4.72% | -5.27% | -5.07% | +0.25% | 1/2/1 |
| `Weather` | -3.53% | -3.52% | -4.59% | -1.11% | 0/0/4 |

## 按 Horizon 汇总

| Horizon | Mixed vs full | Single vs full | R.3 vs full | R.3 vs single | Winner counts M/S/R3 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | -5.94% | -7.38% | -7.25% | +0.18% | 0/1/1 |
| 192 | -4.06% | -4.68% | -4.60% | +0.10% | 0/1/1 |
| 336 | -3.62% | -3.41% | -3.83% | -0.43% | 1/0/1 |
| 720 | -2.88% | -2.10% | -3.65% | -1.58% | 0/0/2 |

## Segment 证据

[Fact] Weather h720 late segment `337-720`: mixed `-4.89%`, single `-3.92%`, R.3 `-5.75%` vs full-time.
[Fact] ETTh2 h720 late segment `337-720`: mixed `-0.68%`, single `+0.54%`, R.3 `-1.87%` vs full-time.

| Future region | Mixed vs full | Single vs full | R.3 vs full | Winner counts M/S/R3 |
| --- | ---: | ---: | ---: | ---: |
| `early_1_96` | -4.90% | -6.06% | -6.39% | 0/2/6 |
| `late_337_720` | -2.78% | -1.69% | -3.81% | 0/0/2 |
| `middle_97_336` | -3.02% | -2.69% | -2.95% | 4/1/5 |

## Training Dynamics

| Dataset | Strategy | Train horizons | Weighting | Best epoch | Best val | Last drift | Train loss change |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `ETTh2` | `full_time_mse720` | `720` | `uniform` | 3 | 0.380270 | +14.45% | -30.94% |
| `Weather` | `full_time_mse720` | `720` | `uniform` | 4 | 0.534117 | +10.41% | -38.40% |
| `ETTh2` | `mixed_horizon_uniform` | `96,192,336,720` | `uniform` | 1 | 0.367096 | +19.84% | -47.29% |
| `Weather` | `mixed_horizon_uniform` | `96,192,336,720` | `uniform` | 1 | 0.528928 | +19.58% | -38.12% |
| `ETTh2` | `single_720_prefix_risk` | `720` | `prefix_risk` | 4 | 0.356905 | +27.53% | -33.51% |
| `Weather` | `single_720_prefix_risk` | `720` | `prefix_risk` | 3 | 0.523890 | +10.37% | -32.31% |
| `ETTh2` | `r3_prefix_risk` | `96,192,336,720` | `prefix_risk` | 1 | 0.374532 | +18.43% | -53.20% |
| `Weather` | `r3_prefix_risk` | `96,192,336,720` | `prefix_risk` | 1 | 0.527042 | +17.01% | -34.75% |

## 解释

[Strong Evidence] `mixed_horizon_uniform` 有用但不足。它在所有 main horizons 上都优于 h720 full-time，但只在 1 个 setting 中成为最优，并且在 Weather 上被 compound R.3 稳定压过。

[Strong Evidence] `single_720_prefix_risk` 在不把 training 绑定到 evaluation horizons 的前提下承载了大量有效信号。它在 ETTh2 h96/h192 最优，并且在 Weather 上接近 R.3，同时保持 h720-only training。

[Fact] Weather h720: `single_720_prefix_risk` vs full-time is `-3.66%`, while R.3 vs full-time is `-5.21%`.
[Fact] ETTh2 h96 is the main exception: R.3 beats single-prefix by `+1.16%`, indicating mixed exposure can help short horizon under ETTh2.

[Inference] 需要重新解释 R.3 的优势：mixed-horizon exposure 确实有帮助，但 prefix-weighted supervision pressure 是更适合 horizon-agnostic 叙事的可操作因素。compound protocol 的额外价值主要体现在 Weather 和 long horizons。

[Decision] 不把 R.3 作为 core story，也不把 mixed-horizon exposure 作为默认主线。下一步应基于 horizon-agnostic、h720-only 的 prefix/stability pressure，进一步从 scalar loss reweighting 推进到控制 gradient 更新位置。

[Next Direction] 在 h720-only prefix-risk base 下测试 architecture-level HSS：允许 prefix-weighted supervision 更新受限的 future-state/readout subspace，同时保护 late/noisy regions，而不是继续采样 target horizons 或 repair R.3。

[Rollback Point] 回到 Step 4/6：将 HSS 重新定义为 horizon-agnostic supervision pressure + gradient routing，而不是 mixed-horizon training schedule。
