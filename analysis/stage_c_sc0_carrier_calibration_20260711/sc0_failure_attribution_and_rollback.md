# StageC SC0 Standardized Carrier Calibration：失败归因与回滚

> 2026-07-12 correction：本文中的31.63%-44.95%是full-720 **validation** degradation，不是test
> degradation。后续用同一fixed20 checkpoints评估test，ETTh2三臂last相对best的H720 test MSE分别
> 恶化13.70%/4.69%/9.68%，均值9.35%。因此validation恶化在test上收缩但没有消失。完整修订见
> `analysis/stage_c_sc0_checkpoint_test_gap_20260712/`。

## 1. Decision Summary

- `candidate`: `SC0-MCP`
- `current_step`: StageC Step 9-10；decision 后回滚到 Step 2/3
- `role`: validation-only protocol control，不是 paper-core method
- `profile_hash`: `79a037f751c0c24eea98ff0b516cb0dfeaef950871b3bbc515904754f54fd900`
- `runs`: `3 datasets × 3 arms × seed2021 = 9/9`
- `test_metrics_used_for_selection`: `false`
- `effectiveness_gate`: `fail`
- `decision`: `exact_sc0_fixed20_protocol_not_frozen`
- `rollback_point`: StageC Step 2/3，先重审统一 optimization/checkpoint policy

SC0 的 best-validation global winner 是 `sc0_p24_d64`，其 macro regret 为 `0.4051%`，且三个
dataset 的最大 regret 为 `1.2153%`，通过预注册的 `<=3%` gate。但 last-checkpoint global winner
变为 `sc0_p48_d32`，因此 selector stability gate 失败。按预注册规则，不能启动 seed 2022/2023，也
不能冻结 `p24/d64`。

## 2. What Was Tested

三臂都满足 `P*D=1536`，active-forward parameters 为
`718,672/719,168/718,576`，spread 约 `0.08%`。训练统一使用 full-720 L1、AdamW、LR `1e-4`、
cosine schedule、effective batch 32、20 epochs、dropout 0.1 和 LayerNorm。配置、数值、dual
checkpoint 与 dense validation artifacts 均完整；remote log 无 traceback、OOM 或 non-finite error。

因此该结果不是容量不匹配、`patch_num=1`、缺失运行或 test leakage 所致。

## 3. Global Selection Evidence

| Selector | Winner | Macro regret | Max dataset regret | Per-dataset gate |
| --- | --- | ---: | ---: | --- |
| best validation | `sc0_p24_d64` | 0.4051% | 1.2153% | pass |
| last epoch | `sc0_p48_d32` | 0.8108% | 2.4324% | pass |

best-validation 下，`p24/d64` 在 ETTm1 与 ETTh2 最优，在 Weather 相对 `p12/d128` 仅损失
1.2153%。这说明“一个跨 dataset 的 common granularity profile”在 primary selector 下有初步可行性；
但它尚未达到可冻结的稳定程度。

## 4. Optimization And Checkpoint Pathology

| Dataset | Arm | Best epoch | Best val MSE | Last val MSE | Last vs best |
| --- | --- | ---: | ---: | ---: | ---: |
| Weather | `p12/d128` | 11 | 0.586420 | 0.592000 | +0.95% |
| Weather | `p24/d64` | 2 | 0.593547 | 0.601231 | +1.29% |
| Weather | `p48/d32` | 4 | 0.595930 | 0.606400 | +1.76% |
| ETTm1 | `p12/d128` | 6 | 0.975055 | 1.008003 | +3.38% |
| ETTm1 | `p24/d64` | 1 | 0.960369 | 0.992866 | +3.38% |
| ETTm1 | `p48/d32` | 2 | 0.962710 | 0.985982 | +2.42% |
| ETTh2 | `p12/d128` | 1 | 0.653424 | 0.932300 | +42.68% |
| ETTh2 | `p24/d64` | 1 | 0.637995 | 0.924785 | +44.95% |
| ETTh2 | `p48/d32` | 2 | 0.673673 | 0.886735 | +31.63% |

ETTh2 三臂都在第 1-2 epoch 达到最优，继续训练到第 20 epoch 后 validation MSE 恶化
31.63%-44.95%。这是一致作用于全部 arms 的 training-policy pathology，而不是某一 patch topology 的
独有失败。ETTm1 也呈现较弱但一致的过训练。

## 5. Failure Attribution

- `hypothesis_false`: **未成立**。best-validation 下存在满足跨 dataset regret gate 的 common arm。
- `intervention_point_wrong`: **不适用**。SC0 是 carrier control，没有新增 paper mechanism。
- `readout_or_head_design_wrong`: **未被本实验隔离**。三臂共用同一 A6 readout，不能归因于 patch arm。
- `optimization_or_numeric_pathology`: **主要归因**。固定 20-epoch trajectory 在 ETTh2 发生严重 validation
  degradation，并直接改变 global winner。
- `capacity_control_explains`: **已控制**。active capacity spread 约 0.08%，不能解释 selector reversal。

[Decision] 该结果否定的是 exact `fixed20 + cosine + last-sensitivity` SC0 research instrument，不能否定
token-MLP common carrier topology，更不能否定 unified varied-horizon forecasting、projective decoder 或
horizon-measure learning。状态应记为 `design_fault_suspected`，而不是方向级 rejection。

## 6. Protocol Audit Finding

原 protocol 还存在一个确认逻辑缺口：它一方面只授权 selected arm 追加 seeds 2022/2023，另一方面要求
至少 `2/3` seeds 保持“global winner方向”。如果其他 arms 不在相同 seed 下运行，global winner 无法被
重新计算。因此不能用 selected-arm-only confirmation 声称 winner stability。

后续修订必须在启动前二选一并写清 claim：

1. 若要确认 **global winner**，每个 confirmation seed 必须运行全部三臂；
2. 若只运行 selected arm，只能确认该 arm 的 absolute stability，不能确认其相对 winner 身份。

## 7. Next Research Action

打开 control-only `SC0-R1`，但暂不启动训练。先在 StageC Step 2/3 预注册统一的
validation-controlled training policy，至少回答：

- 固定 20 epochs 是否应改为统一 `max_epochs + patience + restore-best` rule；
- checkpoint robustness 应如何定义，避免用明显过训练的 raw last epoch替代 deployed checkpoint；
- confirmation 是全臂多 seed winner confirmation，还是 selected-arm absolute stability confirmation；
- 如何保留 test-blind、跨 dataset 同一 optimizer/LR/stopping rule 与无 dataset-specific preset 的边界。

只有修订后的 narrative/protocol gate 通过，才允许重新运行 SC0-R1。SC1/SC2 的 method implementation
继续被 standardized carrier freeze 阻塞，但其 prior-art 与 problem diagnostic 可以并行推进。

## 8. Artifacts

- 机器生成 gate report：`sc0_carrier_calibration_report.md`
- global selection：`sc0_global_selection.csv`
- run/config diagnostics：`sc0_run_diagnostics.csv`
- dense validation horizons：`sc0_validation_horizon_metrics.csv`
- raw synced artifacts：本目录 `raw/`（由 gitignore 排除 checkpoints/predictions）
- remote output：`/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc0_carrier_calibration`
