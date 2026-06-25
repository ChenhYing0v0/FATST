# Phase4-R3D R.3 Mechanism Diagnostic

## 11-Step Record

| Field | Content |
| --- | --- |
| `current_step` | Step 5/6: evaluate why R.3 is strong before designing the next method |
| `problem` | R.3 dominates current Phase4 candidates, but its advantage source is confounded |
| `existence_evidence` | Phase4 RG-B and OP-A returned artifacts; Phase2 objective-pressure diagnostic |
| `idea` | Treat R.3 as a compound protocol: mixed-horizon exposure plus prefix-risk pressure |
| `theory_check` | A method cannot claim horizon-agnostic supervision improvement until this compound baseline is decomposed |
| `design` | Compare R.3 vs h720 full-time; audit actual train horizon exposure and objective pressure; use Phase2 as historical reference |
| `gate` | If R.3 strength is compound, run decomposition controls before adding routing/architecture complexity |
| `artifacts` | `analysis/phase4_r3_mechanism_diagnostic_20260625` |
| `decision` | R.3 is not a single simple step-weight baseline; next work should decompose protocol factors |

## Main Finding

[Fact] Current Phase4 R.3 vs `full_time_mse720`: MSE wins `8/8`, mean relative MSE `-4.83%`.
[Fact] ETTh2 R.3 vs full-time: `4/4`, mean `-5.07%`.
[Fact] Weather R.3 vs full-time: `4/4`, mean `-4.59%`.
[Fact] Weather h720 late segment `337-720`: R.3 vs full-time `-5.75%`; ETTh2 h720 late segment `-1.87%`.

## Per-Horizon Delta

| Dataset | Horizon | R.3 MSE | Full-time MSE | R.3 vs full |
| --- | ---: | ---: | ---: | ---: |
| `ETTh2` | 96 | 0.304796 | 0.339358 | -10.18% |
| `ETTh2` | 192 | 0.369043 | 0.388740 | -5.07% |
| `ETTh2` | 336 | 0.382910 | 0.394549 | -2.95% |
| `ETTh2` | 720 | 0.410473 | 0.419201 | -2.08% |
| `Weather` | 96 | 0.148026 | 0.154701 | -4.31% |
| `Weather` | 192 | 0.192409 | 0.200707 | -4.13% |
| `Weather` | 336 | 0.244793 | 0.256879 | -4.70% |
| `Weather` | 720 | 0.320847 | 0.338482 | -5.21% |

## Dataset And Region Summary

| Dataset | Settings | Wins | Mean R.3 vs full MSE |
| --- | ---: | ---: | ---: |
| `ETTh2` | 4 | 4 | -5.07% |
| `Weather` | 4 | 4 | -4.59% |

| Horizon | Settings | Wins | Mean R.3 vs full MSE |
| ---: | ---: | ---: | ---: |
| 96 | 2 | 2 | -7.25% |
| 192 | 2 | 2 | -4.60% |
| 336 | 2 | 2 | -3.83% |
| 720 | 2 | 2 | -3.65% |

| Future region | Segments | Wins | Mean R.3 vs full MSE |
| --- | ---: | ---: | ---: |
| `early_1_96` | 8 | 8 | -6.39% |
| `late_337_720` | 2 | 2 | -3.81% |
| `middle_97_336` | 10 | 10 | -2.95% |

## Actual Training Exposure

[Fact] The directory label contains all evaluation horizons, but the training log determines actual supervision exposure.

| Dataset | Strategy | Step weighting | Mixed train | Exposure source | h96 | h192 | h336 | h720 | Best epoch | Train loss change |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETTh2` | `full_time_mse` | `uniform` | False | `train_supervision_steps_proxy` | 0.00 | 0.00 | 0.00 | 1.00 | 3 | -30.94% |
| `Weather` | `full_time_mse` | `uniform` | False | `train_supervision_steps_proxy` | 0.00 | 0.00 | 0.00 | 1.00 | 4 | -38.40% |
| `ETTh2` | `r3_prefix_risk` | `prefix_risk` | True | `train_steps_h*` | 0.25 | 0.23 | 0.25 | 0.27 | 1 | -53.20% |
| `Weather` | `r3_prefix_risk` | `prefix_risk` | True | `train_steps_h*` | 0.25 | 0.26 | 0.25 | 0.25 | 1 | -34.75% |

## Objective Pressure

| Region | Mean step weight | Uniform pressure | Weighted pressure | Pressure delta |
| --- | ---: | ---: | ---: | ---: |
| `1-96` | 2.612 | 0.4798 | 0.7217 | +50.43% |
| `97-192` | 1.164 | 0.2298 | 0.1540 | -32.98% |
| `193-336` | 0.856 | 0.1571 | 0.0775 | -50.71% |
| `337-720` | 0.610 | 0.1333 | 0.0469 | -64.85% |

## Historical Reference

[Fact] Phase2 compared R.3 against uniform target-set training and found mean R.3 vs uniform MSE `-1.03%` over `12` settings.

| Scope | Phase2 pressure delta | Raw pressure ratio |
| --- | ---: | ---: |
| `1-96` | +50.43% | 2.612 |
| `97-192` | -32.98% | 1.164 |
| `193-336` | -50.71% | 0.856 |
| `337-720` | -64.85% | 0.610 |
| `horizon_96_loss_multiplier` | +161.18% | 2.612 |
| `horizon_192_loss_multiplier` | +88.77% | 1.888 |
| `horizon_336_loss_multiplier` | +44.55% | 1.445 |
| `horizon_720_loss_multiplier` | +0.00% | 1.000 |

## Interpretation

[Strong Evidence] 当前 R.3 的优势不能解释为“简单 step 权重”单因素。它同时改变了训练 horizon 暴露分布和 step pressure：R.3 每个 epoch 近似均匀采样 h96/h192/h336/h720，而 full-time 实际只用 h720 supervision。

[Strong Evidence] `prefix_risk` 又把 expected pressure 从 late region 推向 early prefix：`1-96` pressure share 增加约 `+50%`，`337-720` 降低约 `-65%`。这解释了为什么 R.3 对 Weather 全 horizon 都强，但也暴露了它对 late future 的叙事风险。

[Counter-Evidence] Phase2 里 R.3 相对 uniform target-set 的均值收益只有约 `-1%`。因此 Phase4 里 R.3 相对 h720 full-time 的更大优势，很可能来自 compound protocol，而不是 prefix-risk 单独强大。

[Decision] 下一步不应继续 repair R.3，也不应把 R.3 包装成 core story。必须先做 decomposition controls：`mixed_horizon_uniform` 用来隔离 mixed-horizon exposure，`single_720_prefix_risk` 用来隔离 prefix-risk weighting，当前 R.3 保留为 compound reference。

[Rollback Point] 回退到 Step 6 设计实验。只有当 decomposition 证明 architecture/gradient-routing 仍能在同等 exposure/objective 条件下带来稳定收益，才继续推进 HSS 主线升级。
