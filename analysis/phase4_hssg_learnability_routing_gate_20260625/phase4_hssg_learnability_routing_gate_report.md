# Phase4-HSSG-B/C Learnability Region Routing Gate Report

## 11-step Record

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10：评估 learnability-conditioned region routing gate |
| `problem` | HSSG-A fixed region path 修复部分 late/long 但牺牲 early；本轮验证 learnability mask 是否能保住 early 并修复 late |
| `existence_evidence` | HSSG-A gate、RG-B residual-stability trace、remote artifacts |
| `idea` | shared base 用 h720-only prefix-risk；learnable residual blocks 只更新 detached region readout path |
| `theory_check` | 如果 proxy 有效，learnability mask 应减少 noisy/ambiguous auxiliary pressure，并避免 fixed region path 的 early 损伤 |
| `design` | ETTh2 + Weather；比较 full-time、single-prefix、R.3、HSSG-A、HSSG-B/C；seed 2021 |
| `gate` | vs single >=5/8 main wins；ETTh2 h96/h192 不超过 +1%；Weather h720 late vs R.3 <= +1%；trace/grad 非 collapse |
| `artifacts` | `analysis/phase4_hssg_learnability_routing_gate_20260625` |
| `decision` | fail_as_core_candidate；learnability mask 有审计信号，但性能和 story gate 均失败 |

## Main Metrics

| strategy | settings | mean_mse | mean_mae |
| --- | --- | --- | --- |
| D0_full_time_mse | 8 | 0.311577 | 0.343633 |
| D1_single_720_prefix_risk | 8 | 0.297778 | 0.334534 |
| D2_r3_prefix_risk | 8 | 0.296662 | 0.333155 |
| HSSG-A_fixed_region_readout | 8 | 0.297191 | 0.336479 |
| HSSG-B_learnability_region_routing | 8 | 0.301085 | 0.336857 |

## HSSG-B/C vs Baselines

| baseline_strategy | settings | mse_wins | mean_relative_mse_pct | mean_relative_mae_pct |
| --- | --- | --- | --- | --- |
| D0_full_time_mse | 8 | 7 | -2.902477 | -1.847563 |
| D1_single_720_prefix_risk | 8 | 3 | 1.570315 | 0.804689 |
| D2_r3_prefix_risk | 8 | 3 | 2.030778 | 1.263458 |
| HSSG-A_fixed_region_readout | 8 | 2 | 1.346370 | -0.084656 |

## Dataset Split

| baseline_strategy | dataset | settings | mse_wins | mean_relative_mse_pct |
| --- | --- | --- | --- | --- |
| D0_full_time_mse | ETTh2 | 4 | 4 | -5.536129 |
| D0_full_time_mse | Weather | 4 | 3 | -0.268825 |
| D1_single_720_prefix_risk | ETTh2 | 4 | 3 | -0.225012 |
| D1_single_720_prefix_risk | Weather | 4 | 0 | 3.365643 |
| D2_r3_prefix_risk | ETTh2 | 4 | 3 | -0.471478 |
| D2_r3_prefix_risk | Weather | 4 | 0 | 4.533033 |
| HSSG-A_fixed_region_readout | ETTh2 | 4 | 1 | 0.228135 |
| HSSG-A_fixed_region_readout | Weather | 4 | 1 | 2.464604 |

## Gate Checks

- [Fact] Absolute mean MSE vs single-prefix `+1.11%`；vs R.3 `+1.49%`；vs HSSG-A `+1.31%`。
- [Fact] Main MSE wins: vs single `3/8`，vs R.3 `3/8`，vs HSSG-A `2/8`。
- [Fact] ETTh2 h96/h192 vs single: `+2.08%` / `-0.69%`，h96 仍超过 +1% gate。
- [Fact] Weather h720 late segment vs single `+4.10%`；vs R.3 `+6.12%`；vs HSSG-A `+5.95%`。
- [Fact] Prefix mismatch max MSE `1.311e-14`，prefix consistency 没有问题。
- [Fact] Weather trace: mean learnable `1.42` blocks，noisy `1.86` blocks，late active steps `62.0`。
- [Fact] Region path 非 collapse：Weather h720 late residual MAE `0.009909`；但它没有转化为 metric gain。
- [Fact] Best epoch: ETTh2 `2/12`，Weather `5/15`。

## Segment/Region Evidence

| baseline_strategy | dataset | future_region | segments | mse_wins | mean_relative_mse_pct |
| --- | --- | --- | --- | --- | --- |
| HSSG-A_fixed_region_readout | ETTh2 | early_1_96 | 4 | 2 | 0.026532 |
| HSSG-A_fixed_region_readout | ETTh2 | late_337_720 | 1 | 0 | 1.000951 |
| HSSG-A_fixed_region_readout | ETTh2 | middle_97_336 | 5 | 3 | -0.006327 |
| HSSG-A_fixed_region_readout | Weather | early_1_96 | 4 | 4 | -0.655057 |
| HSSG-A_fixed_region_readout | Weather | late_337_720 | 1 | 0 | 5.951721 |
| HSSG-A_fixed_region_readout | Weather | middle_97_336 | 5 | 0 | 3.857219 |
| D2_r3_prefix_risk | ETTh2 | early_1_96 | 4 | 0 | 1.903486 |
| D2_r3_prefix_risk | ETTh2 | late_337_720 | 1 | 1 | -0.192787 |
| D2_r3_prefix_risk | ETTh2 | middle_97_336 | 5 | 5 | -1.734477 |
| D2_r3_prefix_risk | Weather | early_1_96 | 4 | 0 | 3.233063 |
| D2_r3_prefix_risk | Weather | late_337_720 | 1 | 0 | 6.124320 |
| D2_r3_prefix_risk | Weather | middle_97_336 | 5 | 0 | 4.788490 |
| D1_single_720_prefix_risk | ETTh2 | early_1_96 | 4 | 0 | 1.953675 |
| D1_single_720_prefix_risk | ETTh2 | late_337_720 | 1 | 1 | -2.584024 |
| D1_single_720_prefix_risk | ETTh2 | middle_97_336 | 5 | 5 | -1.206519 |
| D1_single_720_prefix_risk | Weather | early_1_96 | 4 | 0 | 2.482092 |
| D1_single_720_prefix_risk | Weather | late_337_720 | 1 | 0 | 4.103904 |
| D1_single_720_prefix_risk | Weather | middle_97_336 | 5 | 0 | 3.652144 |

## Trace Summary

| dataset | mean_learnable_blocks | mean_noisy_blocks | mean_noisy_suppression_ratio | mean_region_early_steps | mean_region_middle_steps | mean_region_late_steps | mean_region_abs_residual |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 1.166667 | 1.425141 | 0.356285 | 0.067797 | 1.694915 | 54.237288 | 0.017378 |
| Weather | 1.416500 | 1.859500 | 0.464875 | 0.600000 | 5.424000 | 61.968000 | 0.012458 |

## Interpretation

[Strong Evidence] learnability router 本身没有 collapse。trace 显示 ETTh2/Weather 都能区分 learnable/noisy/ambiguous blocks，region grad 和 residual 也非零，并且 active steps 主要集中在 late region。

[Counter-Evidence] 这条路没有修复 HSSG-A 的核心问题。HSSG-B/C absolute mean MSE 比 single-prefix 差 `+1.11%`，比 HSSG-A 差 `+1.31%`；Weather h720 late segment 比 R.3 差 `+6.12%`，比 HSSG-A 差 `+5.95%`。这直接否定了“residual-stability learnability mask + detached region readout”作为 core method。

[Inference] failure 不是实现未激活，而是 routing target 错了：learnability proxy 将大量 pressure 放到 late region，但 detached low-rank region path 不能承载 Weather 的 late structure，反而削弱了 fixed HSSG-A 已经获得的 late gain。换言之，当前 HSSG-B/C 解决了 noisy pressure 的形式问题，却没有解决 region path 的表达/优化能力问题。

[Decision] 不继续 sweep `aux_weight`、`top_ratio` 或 rank。HSSG-B/C 作为 Step 7 候选失败；HSSG 主线需要回到 Step 5/6 重新评估 carrier，而不是继续在 target-set readout 旁增加 residual paths。

## Next Direction

[Rollback Point] 回到 Step 5/6。保留的事实是：h720-only prefix-risk base 仍强，HSSG-A 的 fixed region path 对 late/long 有 partial signal；但 learnability-conditioned detached region path 会破坏 Weather late。

[Next] 下一步应停止增加 readout residual path，转向 training protocol / representation carrier：优先测试 prefix-risk stabilized base + non-detached richer target/readout subset，或学习率/early-best calibration；若仍坚持 routing，必须让 routing 更新 `condition_head/target_states` 的受控子空间，而不是只更新小 residual head。
