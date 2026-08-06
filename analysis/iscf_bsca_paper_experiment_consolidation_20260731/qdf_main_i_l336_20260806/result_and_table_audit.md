# QDF Main I `L=336` 全矩阵结果与 table replacement audit（2026-08-06）

## 1. Decision

`QDF_L336_32_of_32_COMPLETE_MAIN_I_REPLACEMENT_PASS`。

QDF `seq_len=336` 的八数据集完整复现已通过 artifact、config、numeric 与 provenance gates。Main I 中的 QDF block 现已从“六个 published L96 datasets + 本地 Solar L96”原子替换为八数据集全部本地复跑的 single-seed L336 结果；未混入 partial、failed 或 L96 values。七数据集 dense table 使用 28/28 本地 QDF cells，Exchange companion 增加 4/4 本地 QDF cells。

## 2. Artifact audit

- Remote queue：2026-08-06 19:31:57--20:42:52 +08:00，32 starts、32 completions、正常终止。
- Required artifacts：32 checkpoints、32 learned QDF losses (`A.pth`)、32 metrics、32 effective configs、32 stdout logs，共160 rows。
- Hashes：32 checkpoint hashes、32 learned-loss hashes均唯一；manifest paths和sizes均非空。
- Sync integrity：本地保留的32 metrics、32 effective configs和32 stdout共96 files已逐文件重算SHA-256并与remote manifest一致；checkpoint/A binaries留在remote。
- Config audit：32/32逐字段匹配frozen contract，包括`seq_len=336`、`label_len=48`、dataset loader/channels/cycle/dropout、逐H learning rates/meta settings、30 epochs、patience 5、seed2023、`num_workers=0`、uncapped train/eval以及`final_evaluation_split=test`。
- Numeric/log audit：32/32 MSE/MAE finite；无Traceback、OOM、NaN或file-descriptor failure。
- Test role：每个fixed-H system由upstream validation early stopping选择checkpoint，训练结束后一次test；本轮不以test重新选择QDF hyperparameters。

## 3. Dataset-level results

| Dataset | QDF L336 MSE | QDF L336 MAE | ISCF-BSCA MSE | ISCF-BSCA MAE | TimeAlign MSE | TimeAlign MAE |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | 0.438867 | 0.442165 | 0.393520 | 0.421034 | 0.417990 | 0.429396 |
| ETTh2 | 0.352528 | 0.393573 | 0.307332 | 0.365116 | 0.346665 | 0.386517 |
| ETTm1 | 0.352795 | 0.379712 | 0.330699 | 0.363879 | 0.339704 | 0.366959 |
| ETTm2 | 0.256994 | 0.314927 | 0.248733 | 0.305693 | 0.242889 | 0.302523 |
| Weather | 0.236614 | 0.271482 | 0.214887 | 0.245821 | 0.215800 | 0.244725 |
| ECL | 0.166655 | 0.259948 | 0.151625 | 0.245635 | 0.154704 | 0.243853 |
| Solar | 0.208124 | 0.258174 | 0.190485 | 0.210792 | 0.195970 | 0.216647 |
| Exchange | 0.399989 | 0.427612 | 0.398836 | 0.425081 | 0.512558 | 0.459692 |

七个dense datasets的QDF L336 macro MSE/MAE为`0.287511/0.331426`。相对当前ISCF-BSCA为`+9.541%/+7.508%`，相对本地TimeAlign为`+5.166%/+5.905%`；lower is better，因此QDF整体明显弱于两者。

## 4. Cell-level interpretation

在七个dense datasets × four H × MSE/MAE的56 cells中：

- QDF仅在`ETTm2-H192 MSE`优于ISCF-BSCA，即QDF领先`1/56`，ISCF-BSCA严格领先其余`55/56`；
- QDF在`ETTh1-H96 MSE/MAE`和`ETTh2-H336 MSE/MAE`优于TimeAlign，共`4/56`；
- 将QDF L336加入完整14-model表后，ISCF-BSCA的best/second仍为`27/56`和`19/56`，与上一版相同。

Exchange是单独的三系统companion。QDF four-H mean相对ISCF-BSCA略差`+0.289% MSE / +0.595% MAE`，但在H96 MSE、H192 MSE/MAE和H336 MSE四个cells领先ISCF；相对TimeAlign mean则低`21.962% MSE / 6.979% MAE`，主要来自H336/H720。由于TimeAlign和QDF都没有official Exchange script，这些结果必须保留ETTh1-derived source-informed披露，不能升级为matched attribution。

## 5. L336 versus previous L96 block

在七个共同datasets上，L336 QDF macro相对上一版mixed-source L96 block为`-5.462% MSE / -1.277% MAE`。其中ETTm1、ETTm2、ETTh2和Weather的MSE改善，Solar改善最大（MSE `-25.961%`，MAE `-11.433%`）；ETTh1和ECL则退化。

该对比仅为descriptive：六个旧值是QDF paper的L96 three-run means，Solar旧值是本地single-seed reproduction，而本轮全部是本地single-seed L336。因此它不能被写成严格matched lookback ablation；论文主表只采用本轮统一来源的L336值。

## 6. Table artifacts and claim boundary

- Main dense table：14 models × 7 shared datasets × four H，392 raw rows、490 rows with averages。
- Exchange companion：ISCF-BSCA、TimeAlign、QDF × four H + averages，共15 rows。
- QDF provenance：28 dense cells + 4 Exchange cells全部为official-code local seed2023 L336；Solar=`ECL-derived`，Exchange=`ETTh1-derived`。
- TimeAlign：32 cells全部为local seed2021 reproduction；其他11个baselines仍是TimeAlign Table 6 published context。
- QDF是`modern_native_fixed_H` effectiveness baseline，不是matched unified control，不能用于ISCF-BSCA decoder mechanism attribution。

## 7. Failure attribution and next cursor

QDF L336没有artifact或numeric failure。其弱于ISCF-BSCA属于`external_baseline_accuracy_result`，不是QDF实现失败，也不改变ISCF-BSCA architecture claim。当前cursor回到Main I baseline consolidation：QDF scope关闭；AMD、SimpleTM、TimePerceiver和SRSNet等尚未闭合的baseline evidence仍是独立后续任务，不应由本结果自动启动。

## 8. Machine-readable field definitions

`qdf_l336_cell_comparison.csv`每行对应一个dataset-horizon system。`*_mse/*_mae`来自各方法相应official-test scorecard；`qdf_l96_*`来自上一版published-plus-local历史block。`qdf_vs_<reference>_<metric>_pct=100*(QDF_L336/reference-1)`，负值表示QDF更好；`qdf_l336_vs_l96_*_pct`同理。`qdf_beats_<reference>_<metric>`仅在QDF原始未round数值严格更低时为true。

`qdf_l336_dataset_summary.csv`把上述cell按dataset对四个horizons取算术均值；所有`*_mean_*`均为four-H mean，percentage字段使用这些mean计算，`qdf_beats_*_cells`是在该dataset的四H × 两metrics共8 cells中的strict-win数量。Exchange没有L96值，因此相应字段为空。

`qdf_l336_result_summary.json`中的`macro_7`是七个dense datasets的28个逐H values等权平均，`macro_8`加入Exchange成为32 values；`*_cells_7_dense_datasets`的分母为56。`artifact_audit.role_unique_hash_counts`按artifact role统计不同SHA-256个数，不是文件总数。

`remote_lite/audit/qdf_main_i_l336_artifact_manifest.csv`中的`artifact_role`区分checkpoint、learned QDF loss、metrics、effective config和stdout；`path`是remote原始绝对路径，`sha256`对完整原始文件计算，`bytes`为文件大小。同步的`remote_lite`保留metrics/config/log用于本地复核，checkpoint binaries留在remote，由manifest hashes提供不可变身份。
