# TimeAlign ETTm2/Weather official reproduction result

日期：2026-08-05

## 结论

本轮TimeAlign复现为8/8 artifact-complete：ETTm2、Weather × H96/H192/H336/H720 × seed2021。8个official-last checkpoints SHA256唯一，8个training logs均为10 epochs且无early stopping，8个test metric rows均标记`native_external`、`official_test_mode=1`和`evaluation_split=test`。执行角色为`official-source model/config + FATST test-hygiene/artifact adapter`，不是byte-identical raw runner，也不构成matched mechanism attribution。

## 四horizon结果

| Dataset | H | MSE | MAE | Paper MSE | Paper MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTm2 | 96 | 0.154997 | 0.241157 | 0.155 | 0.241 |
| ETTm2 | 192 | 0.210228 | 0.280474 | 0.210 | 0.280 |
| ETTm2 | 336 | 0.263393 | 0.317077 | 0.263 | 0.315 |
| ETTm2 | 720 | 0.342938 | 0.371385 | 0.343 | 0.372 |
| Weather | 96 | 0.140421 | 0.178258 | 0.140 | 0.179 |
| Weather | 192 | 0.182363 | 0.220420 | 0.182 | 0.220 |
| Weather | 336 | 0.233109 | 0.262647 | 0.232 | 0.262 |
| Weather | 720 | 0.307309 | 0.317576 | 0.307 | 0.317 |

四H平均：

- ETTm2=`0.242889/0.302523`，相对论文Table 6的`0.242750/0.302000`为MSE `+0.057%`、MAE `+0.173%`；
- Weather=`0.215800/0.244725`，相对论文`0.215250/0.244500`为MSE `+0.256%`、MAE `+0.092%`。

相对2026-06-26 historical local rerun，本轮任一cell最大绝对偏差仅MSE `0.195%`、MAE `0.298%`。这说明官方source/config路径在当前adapter/runtime下高度稳定复现论文量级。论文值是three-run mean，本轮是单seed2021，因此不能要求逐cell精确相等，也不能把单seed差异解释为method优势。

## Evidence role

当前可以把本轮结果作为ETTm2/Weather的artifact-complete official-native reproduced baseline，同时保留论文Table 6作为published three-run context。`license_status=license_unresolved`不影响本地科研运行，但禁止声称upstream redistribution已获许可。

Canonical artifacts：

- `reproduced_metrics_and_comparison.csv`；
- `artifact_manifest.csv`；
- `reproduction_summary.json`；
- `remote_lite/`中的effective configs、environment、training logs、metrics与run logs。

Decision=`TimeAlign_ETTm2_Weather_single_seed_official_native_reproduction_pass`。
