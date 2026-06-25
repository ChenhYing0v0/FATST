# Phase4 Protocol Calibration Gate Report

## 结论

[Strong Evidence] Lower LR 对 training trajectory 有帮助，但不能把 HSSG-A 修成主线候选。
最佳 HSSG-A setting 是 `lr_0p00005`，8 个 horizon 平均 MSE 为 `0.295466`；
最佳 single-prefix 是 `lr_0p0001`，平均 MSE `0.297778`；
最佳 R.3 是 `lr_0p00005`，平均 MSE `0.294135`。

[Decision] Protocol calibration gate 不通过。下一步不应继续调 HSSG readout residual path，
应回到 Step 6，设计 richer carrier：让 supervision scheduling 作用到更高语义的 state / condition / adapter 子空间，
而不是只改变 horizon-independent loss 权重或小 residual readout。

## HSSG-A vs Single

| lr_dir | settings | mse_wins | mean_mse | mae_wins | mean_mae |
| --- | --- | --- | --- | --- | --- |
| lr_0p00003 | 8 | 6 | -0.73% | 3 | -0.18% |
| lr_0p00005 | 8 | 4 | -0.81% | 4 | -0.20% |
| lr_0p0001 | 8 | 4 | +0.23% | 2 | +0.91% |

## HSSG-A vs R.3

| lr_dir | settings | mse_wins | mean_mse | mae_wins | mean_mae |
| --- | --- | --- | --- | --- | --- |
| lr_0p00003 | 8 | 4 | +0.31% | 4 | +0.27% |
| lr_0p00005 | 8 | 2 | +0.60% | 2 | +0.87% |
| lr_0p0001 | 8 | 3 | +0.67% | 2 | +1.37% |

## Dataset-Level Delta

| lr_dir | dataset | baseline | mse_wins | mean_relative_mse_pct |
| --- | --- | --- | --- | --- |
| lr_0p00003 | ETTh2 | single_720_prefix_risk | 3/4 | -1.01% |
| lr_0p00003 | ETTh2 | r3_prefix_risk | 4/4 | -1.76% |
| lr_0p00003 | Weather | single_720_prefix_risk | 3/4 | -0.46% |
| lr_0p00003 | Weather | r3_prefix_risk | 0/4 | +2.39% |
| lr_0p00005 | ETTh2 | single_720_prefix_risk | 4/4 | -2.20% |
| lr_0p00005 | ETTh2 | r3_prefix_risk | 2/4 | -0.20% |
| lr_0p00005 | Weather | single_720_prefix_risk | 0/4 | +0.58% |
| lr_0p00005 | Weather | r3_prefix_risk | 0/4 | +1.39% |
| lr_0p0001 | ETTh2 | single_720_prefix_risk | 3/4 | -0.45% |
| lr_0p0001 | ETTh2 | r3_prefix_risk | 3/4 | -0.70% |
| lr_0p0001 | Weather | single_720_prefix_risk | 1/4 | +0.91% |
| lr_0p0001 | Weather | r3_prefix_risk | 0/4 | +2.05% |

## Training Trajectory

| lr_dir | strategy | runs | mean_best_epoch | mean_epochs_ran | mean_post_best_val_drift_pct | max_post_best_val_drift_pct |
| --- | --- | --- | --- | --- | --- | --- |
| lr_0p00003 | hssg_region_routed_readout | 2 | 6.500000 | 16.500000 | +6.56% | +7.96% |
| lr_0p00003 | r3_prefix_risk | 2 | 3 | 13 | +13.62% | +18.17% |
| lr_0p00003 | single_720_prefix_risk | 2 | 8 | 18 | +5.44% | +6.43% |
| lr_0p00005 | hssg_region_routed_readout | 2 | 3.500000 | 13.500000 | +7.57% | +8.85% |
| lr_0p00005 | r3_prefix_risk | 2 | 2 | 12 | +16.29% | +19.19% |
| lr_0p00005 | single_720_prefix_risk | 2 | 4 | 14 | +8.72% | +9.43% |
| lr_0p0001 | hssg_region_routed_readout | 2 | 1.500000 | 11.500000 | +14.86% | +20.22% |
| lr_0p0001 | r3_prefix_risk | 2 | 1 | 11 | +17.72% | +18.43% |
| lr_0p0001 | single_720_prefix_risk | 2 | 3.500000 | 13.500000 | +18.95% | +27.53% |

## Gate Assessment

- best epoch 后移：部分成立，`3e-5` 的 HSSG-A mean best epoch 后移到 6.5，但 R.3 仍为 3.0。
- validation drift 下降：部分成立，HSSG-A 从高 LR 的 mean drift 14.87% 降到 6.56%，但 R.3 仍有 13.62%。
- HSSG-A vs single 至少 5/8 wins：只在 `3e-5` 成立，达到 6/8。
- ETTh2 h96/h192 不超过 +1%：`3e-5`、`5e-5` 成立。
- Weather h720 late 接近 R.3：不成立，`3e-5` Weather h720 比 R.3 差 +3.37%。

## 11-Step Decision

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10：评估 Phase4-PTC 结果并做 go/no-go 决策 |
| `problem` | early-best 与 validation drift 会污染 HSSG-A carrier 判断 |
| `existence_evidence` | 18 个 run 均完成；log 可解析 best epoch 与 drift；metrics 覆盖 ETTh2/Weather × 4 horizons |
| `idea` | 用 lower LR 判断 protocol 是否是 HSSG-A 失败主因 |
| `theory_check` | 若过快 optimization 是主因，lower LR 应同时推迟 best epoch、降低 drift，并让 HSSG-A 保持 Weather/R.3 竞争力 |
| `design` | LR `1e-4/5e-5/3e-5`；single/R.3/HSSG-A；ETTh2 + Weather；seed 2021 |
| `gate` | HSSG-A ≥5/8 wins vs single；ETTh2 short 不坏；Weather h720 late 接近 R.3；drift 降低 |
| `artifacts` | 本目录 CSV 与 report；raw artifacts under `artifacts/runs/phase4_protocol_calibration_gate` |
| `decision` | Fail as core-route repair；rollback to Step 6，设计 state/condition carrier，而不是继续 LR sweep |
