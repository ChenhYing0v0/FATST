# Stage C D18 Step 9 Deep Audit Code Explanation

## 1. 功能边界

`scripts/analyze_stage_c_d18_step9_deep_audit.py`只读取已经完成的D18 remote artifacts与冻结
`A6_MEASURE` validation artifacts，不训练模型、不选择checkpoint、不访问新的test labels。

输入：

1. `--raw-root`：D18 25-unit remote artifact root；
2. `--control-root`：冻结SIFF attribution raw root中的A6 validation metrics；
3. `--output-dir`：curated Step9 output directory。

输出：

- `validation_test_cells.csv`；
- `checkpoint_summary.csv`；
- `protocol_invariants.csv`；
- `effect_decomposition.csv`；
- `probe_interval_gains.csv`；
- `deep_summary.json`。

## 2. 指标来源与计算

### `validation_gain_over_a6_measure_percent`

source为specialist与`A6_MEASURE`各自`metrics_by_target_horizon.csv`的own-H MSE：

$$
100\left(1-\frac{\operatorname{MSE}_{specialist}}
{\operatorname{MSE}_{A6\_MEASURE}}\right).
$$

### `test_gain_over_a6_measure_percent`

source为remote frozen analyzer的`analysis/own_horizon_cells.csv`，使用official-test own-H MSE；脚本不重新
读取raw dataset labels。

### `best_epoch_at_budget_boundary`

从specialist `training_log.csv`找到minimum `val_mean_mse`对应epoch，并判断其是否等于实际训练epoch数。
该量只检查明显训练预算截断，不表示更长训练必然无收益。

### `prefix_gap`

直接读取每个run的`test_audit_invariants.json/full_prefix_max_abs`。它比较full forecast的prefix与相同forward
contract下的cropped output，验证numeric crop一致性。

### `probe_interval_gains.csv`

每个specialist与D18同步保存的`A6_MEASURE` NPZ都包含：

- `probe_fused[256,720]`；
- `probe_targets[256,720]`。

脚本先要求两arm的`probe_targets`逐元素完全相等，再对固定区间计算sample-time mean MSE及relative gain。
区间为：

- H96：`1–48`、`49–96`；
- H192：`1–96`、`97–192`；
- H336：`1–96`、`97–192`、`193–336`。

`prediction_nrmse`定义为：

$$
\frac{
\sqrt{\operatorname{mean}(\hat Y_{spec}-\hat Y_{measure})^2}
}{
\sqrt{\operatorname{mean}(\hat Y_{measure}^2)}+10^{-12}
}.
$$

probe只有256 rows，因此只作内部定位，不能替代full official-test effectiveness。

### `effect_decomposition.csv`

对15个own-H cells的cell-wise relative gains取算术平均，分别报告：

1. `A6_MEASURE` over `A6_FULL`；
2. specialist over `A6_FULL`；
3. specialist over `A6_MEASURE`；
4. validation specialist over `A6_MEASURE`。

## 3. Decision逻辑

脚本不重新发明gate，`remote_gate_result`直接保留预注册D18 analyzer的七项结果。新增validation、checkpoint与
probe统计只用于failure attribution：

- 若protocol/hash/prefix/numeric失败，则不能方向级拒绝；
- 若predictions不变，则intervention可能未执行；
- 若大量best epochs卡预算边界，则optimization解释需保留；
- 若上述均正常而specialist仍不稳定超过`A6_MEASURE`，则主归因为精确projectivity-cost hypothesis false。

## 4. 验证边界

代码验证应至少包括：

1. `python -m py_compile scripts/analyze_stage_c_d18_step9_deep_audit.py`；
2. 重新生成全部六个outputs；
3. `deep_summary.json`确认25/25、25 unique hashes与D18 remote summary一致；
4. probe target mismatch必须立即报错，不允许比较错位rows。
