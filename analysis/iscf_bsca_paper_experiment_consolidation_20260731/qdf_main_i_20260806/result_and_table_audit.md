# QDF Solar 复现结果与 Main I 表审计（2026-08-06）

## 1. Outcome

`current_step=Step 9 artifact audit complete -> Step 10 Main I consolidation complete for QDF scope`。

QDF Solar 的 `4 horizons × seed2023` 已完整训练并执行正式 test；4/4 checkpoint、4/4 learned QDF loss、4/4 effective configs 与 4/4 MSE/MAE rows 均存在。QDF 已按用户要求放在 Main I 中 TimeAlign 的右侧。

## 2. Solar results

| Horizon | MSE | MAE |
|---:|---:|---:|
| 96 | 0.190262 | 0.224629 |
| 192 | 0.218261 | 0.249375 |
| 336 | 0.334881 | 0.333303 |
| 720 | 0.380994 | 0.358703 |
| Avg. | 0.281099 | 0.291503 |

这组结果完整保留，未按有利 horizon 选择。QDF Solar 相对本地 TimeAlign Solar 的 four-H mean MSE/MAE `0.195970/0.216647` 明显更弱；相对 ISCF-BSCA Solar `0.190/0.211` 也更弱。主要差距来自 H336/H720。由于 Solar 采用 ECL-derived released preset，而不是 QDF 论文提供的 Solar preset，这一结果只能标记为 `official_native_source_informed_solar_single_seed`，不作 QDF 在 Solar 上最优调参的声明。

## 3. Infrastructure retry audit

初次完整训练使用 release default `num_workers=10`。四个 checkpoint 均已由 validation early stopping 选定，但正式 test DataLoader 因 remote file-descriptor limit 报 `Too many open files`。修复仅把 host-side loader 固定为 `num_workers=0`：

- evaluation-only retry 均记录 `checkpoint_retrained=false`；
- 原训练与首次 test failure 日志保留为 `training_stdout_before_test_retry.log`；
- checkpoint mtime 均早于 retry metrics mtime；
- retry 执行 commit=`2c147bf1e77af0ec98285fc8036df5123b6571cc`；
- data SHA256=`230327ef72d2abb387939d4a35d6fd34f1066071bc7c40ce7ecf5531a0122ac2`。

因此这是 infrastructure-only test retry，不是 checkpoint retraining、test-based selection 或 profile change。

## 4. Main I source roles

QDF 完整 7-dataset surface 共 28 rows：

- ETTm1、ETTm2、ETTh1、ETTh2、Weather、ECL：QDF Table 6 的 published three-run means，共 24 rows；
- Solar：本轮 official-code source-informed seed2023 reproduction，共 4 rows。

完整 Main I dense table 为 14 models × 7 datasets × 4 horizons = 392 standard rows，加入逐 dataset arithmetic Avg. 后为 490 long rows。模型顺序固定为 `ISCF-BSCA | TimeAlign | QDF | ...`。QDF 加入后，ISCF-BSCA 仍为 27/56 best、19/56 second（按三位小数 displayed value，允许并列）；原因是 ISCF-BSCA 对 QDF 的 MSE 为 28/28 cells 更优、MAE 为 27/28 cells 更优。原先 `33/56` 仍只属于冻结的 five-comparator scope，不得与 14-model full table 混写。

## 5. Artifact and claim boundary

- table PDF：`output/pdf/iscf_bsca_main_i_qdf_20260806.pdf`
- table long data：`main_i_final_qdf_20260806/table_data_long.csv`
- QDF published source rows：`qdf_table6_published.csv`
- Solar local metrics：`qdf_solar_local_metrics.csv`
- artifact manifest：`qdf_solar_artifact_manifest.csv`
- checkpoint/loss hashes：`checkpoint_and_loss_hashes.txt`

Decision=`QDF_Solar_4_of_4_complete_Main_I_14_model_table_finalized_with_mixed_source_disclosure`。这项结果闭合 QDF scope，但不闭合 AMD、SimpleTM、TimePerceiver、SRSNet 或 Exchange 的既有缺口，也不构成 matched mechanism attribution。
