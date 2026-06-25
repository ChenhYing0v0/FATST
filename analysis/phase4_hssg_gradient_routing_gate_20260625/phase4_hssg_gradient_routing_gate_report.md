# Phase4-HSSG-A Gradient Routing Gate Report

## 11-step Record

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10：评估远程结果并决定是否通过 gate |
| `problem` | loss-only HSS 无法决定 future-unit gradient 应更新哪些参数子空间 |
| `existence_evidence` | Phase4-R3D、SRP group/tuner 思想、前序 gradient-routing partial evidence |
| `idea` | 在 h720-only prefix-risk objective 下，用 early/middle/late region-routed readout path 控制 gradient/update subspace |
| `theory_check` | 若不同 future regions 的噪声/可预测性不同，region-specific update 应缓解 shared readout conflict；若只带来 tiny residual 或牺牲 short horizon，则理论实现不足 |
| `design` | ETTh2 + Weather，seed2021；比较 full-time、single-prefix、R3、HSSG-A |
| `gate` | vs single overall 改善且 >=5/8 main wins；Weather h720 late vs R.3 gap <= +1%；ETTh2 h96/h192 vs single 不超过 +1%；region path 不 collapse |
| `artifacts` | `analysis/phase4_hssg_gradient_routing_gate_20260625` |
| `decision` | fail_as_core_candidate；见下方证据与 rollback |

## Main Metrics

| strategy | settings | mean_mse | mean_mae |
| --- | --- | --- | --- |
| D0_full_time_mse | 8 | 0.311577 | 0.343633 |
| D1_single_720_prefix_risk | 8 | 0.297778 | 0.334534 |
| D2_r3_prefix_risk | 8 | 0.296662 | 0.333155 |
| HSSG-A_region_routed_readout | 8 | 0.297191 | 0.336479 |

## HSSG-A vs Baselines

| baseline_strategy | settings | mse_wins | mean_relative_mse_pct | mean_relative_mae_pct |
| --- | --- | --- | --- | --- |
| D0_full_time_mse | 8 | 8 | -4.191882 | -1.749032 |
| D1_single_720_prefix_risk | 8 | 4 | 0.230729 | 0.905750 |
| D2_r3_prefix_risk | 8 | 3 | 0.673534 | 1.366862 |

## Dataset Split

| baseline_strategy | dataset | settings | mse_wins | mean_relative_mse_pct |
| --- | --- | --- | --- | --- |
| D0_full_time_mse | ETTh2 | 4 | 4 | -5.749671 |
| D0_full_time_mse | Weather | 4 | 4 | -2.634093 |
| D1_single_720_prefix_risk | ETTh2 | 4 | 3 | -0.451550 |
| D1_single_720_prefix_risk | Weather | 4 | 1 | 0.913009 |
| D2_r3_prefix_risk | ETTh2 | 4 | 3 | -0.699166 |
| D2_r3_prefix_risk | Weather | 4 | 0 | 2.046235 |

## Gate Checks

- [Fact] Absolute mean MSE vs `single_720_prefix_risk`: `-0.20%`；vs R.3: `+0.18%`。
- [Fact] vs `single_720_prefix_risk`: HSSG-A main MSE wins `4/8`, mean relative MSE `+0.23%`.
- [Fact] vs `r3_prefix_risk`: HSSG-A main MSE wins `3/8`, mean relative MSE `+0.67%`.
- [Fact] ETTh2 h96/h192 vs single: `+1.61%` / `-0.74%`；h96 超过 +1% gate，h192 通过。
- [Fact] Weather h720 vs single: `-1.14%`；vs R.3: `+0.47%`。
- [Fact] Weather h720 late segment `337-720` vs single: `-1.74%`；vs R.3: `+0.16%`。
- [Fact] HSSG prefix mismatch max MSE `1.184e-14`，prefix consistency 没有问题。
- [Fact] HSSG Weather h720 region residual all MAE `0.036413`，late MAE `0.038856`；path 非零但 residual magnitude 很小。
- [Fact] HSSG best epoch: ETTh2 `2/12`，Weather `1/11`；Weather best 在 epoch 1。

## Segment/Region Evidence

| baseline_strategy | dataset | future_region | segments | mse_wins | mean_relative_mse_pct |
| --- | --- | --- | --- | --- | --- |
| D2_r3_prefix_risk | ETTh2 | early_1_96 | 4 | 0 | 1.879932 |
| D2_r3_prefix_risk | ETTh2 | late_337_720 | 1 | 1 | -1.181907 |
| D2_r3_prefix_risk | ETTh2 | middle_97_336 | 5 | 5 | -1.729165 |
| D2_r3_prefix_risk | Weather | early_1_96 | 4 | 0 | 3.913519 |
| D2_r3_prefix_risk | Weather | late_337_720 | 1 | 0 | 0.162904 |
| D2_r3_prefix_risk | Weather | middle_97_336 | 5 | 0 | 0.902082 |
| D1_single_720_prefix_risk | ETTh2 | early_1_96 | 4 | 0 | 1.927390 |
| D1_single_720_prefix_risk | ETTh2 | late_337_720 | 1 | 1 | -3.549446 |
| D1_single_720_prefix_risk | ETTh2 | middle_97_336 | 5 | 5 | -1.200653 |
| D1_single_720_prefix_risk | Weather | early_1_96 | 4 | 0 | 3.157773 |
| D1_single_720_prefix_risk | Weather | late_337_720 | 1 | 1 | -1.744018 |
| D1_single_720_prefix_risk | Weather | middle_97_336 | 5 | 2 | -0.188749 |

## Interpretation

[Strong Evidence] HSSG-A 不是无效机制。它的 absolute mean MSE 略优于 `single_720_prefix_risk`，在 ETTh2/Weather 的 h720 主指标上都改善 single-prefix，并且 Weather h720 late segment 已接近 R.3。这说明把一部分更新能力放到 region-routed readout path，确实能吸收 prefix-risk objective 中 shared path 难以处理的 residual。

[Counter-Evidence] HSSG-A 不能作为当前 core candidate 直接推进。它只赢 `single_720_prefix_risk` 的 `4/8` main settings，没有达到 `5/8` gate；per-setting relative MSE 也略差于 single-prefix；ETTh2 h96 相对 single-prefix 变差超过 +1%；Weather early/middle regions 相对 single-prefix 和 R.3 都不稳。也就是说，HSSG-A 修到了一部分 late/long 问题，但把 short/early 优势让出去了。

[Inference] 当前 region-routed readout 更像一个 small residual corrector，而不是充分的 gradient scheduler。证据是 residual path 非零但幅度很小，Weather best epoch 仍在 epoch 1，说明 optimization 仍是早期就被 validation 选择，后续训练主要 drift；这不支持继续简单扩大训练轮数或继续叠 loss weight。

[Decision] HSSG-A 作为 Step 7 最小实现是 `partial evidence`，但 Step 10 gate 不通过。它支持“gradient/update subspace 是有价值轴”，但否定了“固定 early/middle/late low-rank readout residual 足够成为主线方法”。

## Rollback And Next Direction

[Rollback Point] 回到 Step 6，而不是 Step 4：核心 HSSG 主线不撤销，但当前 HSSG-A 设计需要重构。下一步不应 sweep rank/dropout/scale，因为失败点不是单纯 capacity；失败点是 fixed region path 不知道哪些 late residual 可学习、哪些应该被阻断。

[Next] 进入 HSSG-B/C 的混合方向：`learnability-conditioned gradient routing`。具体应让 prefix-stable shared path 保留 single-prefix 的 short-horizon 优势，同时让 late/noisy units 根据 residual stability/predictability 决定进入 region path、shared path、或 no-update path。
