# SC-D22-HFA D22-C raw-history target-access diagnostic：代码说明

## 1. 模块角色

本次新增的是`diagnostic_only` runner与aggregator，不是production forecasting model。它不修改
`baselines/timealign_official/models/TimeAlign.py`，不恢复A6 representation，也不实现requested-H
embedding/router。

相关文件：

- `configs/stage_c_d22c_target_access_diagnostic.json`：冻结arms、shape、optimization、selector、gates与授权；
- `scripts/run_stage_c_d22c_target_access.py`：按dataset训练完全同参数的六臂neutral raw-history models；
- `scripts/analyze_stage_c_d22c_target_access.py`：汇总20-cell scorecard、bins、internal health与failure attribution；
- `scripts/remote/run_stage_c_d22c_target_access.sh`：3090 dataset-major调度；
- `scripts/sync_stage_c_d22c_target_access_results.sh`：同步raw artifacts并本地重算decision。

## 2. Forward tensor flow

输入`batch_x`为`[B,720,C]`。`prepare_batch`先按window/channel进行RevIN，再转为：

- `history_rows`: `[N,720]`，$N=B\times C$；
- `future_rows`: `[N,720]`；
- `mean_rows/std_rows`: `[N,1]`；
- `row_ids`: `[N]`，只由source window index和channel index组成。

`NeutralTargetAccess.forward`执行：

1. `history_rows.reshape(N,24,30)`得到raw patches；
2. shared `patch_encoder`把`[N,24,30]`映射到`[N,24,32]`；
3. fixed sinusoidal patch position形成ordered memory；
4. fixed target positions经shared `query_encoder`形成`[N,720,32]`；
5. shared four-head `MultiheadAttention`输出context `[N,720,32]`；
6. `cat([coordinate_query, context], -1)`得到`[N,720,64]`；
7. shared fusion与scalar projection输出normalized forecast `[N,720]`；
8. evaluation用`prediction * std_rows + mean_rows`恢复到dataset-standardized scale。

六臂拥有完全相同的module/parameters。`GLOBAL_COMPRESSED`与`POOLED_MEMORY`只改变memory construction；
`ORDER_SHUFFLED`只改变patch content与position的对应；`TARGET_SHUFFLED_QUERY`只改变row-specific coordinate
identity；`GENERIC_MATCHED`只让memory retrieval不再受coordinate控制，canonical coordinate仍进入共享readout。

## 3. Training、selector与artifacts

每个arm从相同`base_state`开始，重新创建相同AdamW optimizer与相同seed DataLoader order。model先输出
normalized forecast，再用window/channel的history mean/std重建到dataset-standardized scale；training loss与
validation selector都在该scale计算pointwise MSE。v1曾在normalized scale直接计算loss，Weather/ETTm2被
near-zero history variance rows放大到$10^3$量级；该run在任何完整dataset/test artifact前终止，由v1.1取代。
validation selector是H96/H192/H336/H720 prefix MSE平均。
best checkpoint选定后才评估official test。

每个`dataset/arm`写出：

- `training_history.csv`：epoch、train normalized MSE、validation selector、best flag、耗时；
- `metrics.csv`：validation/test的四个prefix与五个coordinate bins的MSE/MAE；
- `checkpoint.pt`与SHA256；
- `summary.json`：best epoch、selector、params与internal health。

dataset级`metadata.json`记录commit、config hash、split window counts、test role、授权、runtime与matrix completeness。
aggregator生成cell、bin、aggregate、parameter、health CSV，`decision.json`和中文result report。

## 4. Statistic definitions

- `mse_gain`：`(control_mse - ordered_mse) / control_mse`；
- `mae_gain`：同式替换为MAE；
- `attention_entropy`：每个target的attention entropy除以$\log S$，再对rows/targets平均；
- `attention_target_dispersion`：attention weights沿target维的standard deviation均值；
- `prediction_coordinate_dispersion`：每行720个normalized predictions的standard deviation均值；
- `relative_gap`：同dataset六臂最大与最小trainable parameter count之差除以最大值。

internal health只判断路径是否active/collapsed，不能替代20-cell performance或matched attribution gate。

## 5. Code-theory consistency

[Intended Theory] 不同lead times可能需要从同一fixed past选择不同evidence；该需求属于finite computation
organization，不是requested-H改变Bayes target。

[Code Realization] only `ORDERED_TARGET_ACCESS`同时保留canonical coordinate identity与ordered
query-to-memory retrieval；五个controls逐项移除global compression、token availability、order semantics、
coordinate semantics或coordinate-specific retrieval。

[Proxy Boundary] raw patch attention只是测试information-access necessity的载体；它不等同于最终
`lead-time-conditioned evidence operator`，也没有证明novelty。

[Falsification] ordered若不能同时超过五个controls，或validation/test不transfer，exact D22-C hypothesis失败；
non-finite、matrix缺失、parameter mismatch或严重退化只使protocol无效，不能方向级拒绝。
