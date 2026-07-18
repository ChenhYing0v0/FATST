# CCSF temperature pilot result analyzer代码说明

`scripts/analyze_stage_c_siff_ccsf_temperature_pilot_result.py`读取同步后的15个run目录与remote selection artifacts。

每个run必须包含`effective_config.json`、`training_log.csv`、四horizon metrics、environment、initialization contract和
model diagnostics。analyzer检查：

- 15 runs与60 cells完整；
- MSE/MAE及training log无NaN/Inf；
- `final_evaluation_split=val`、`official_test_mode=false`、`checkpoint_policy=best-val`；
- remote cell artifact包含15个unique checkpoint SHA256；
- 本地重算macro MSE与remote选择一致；
- pilot checkpoint不复用且formal test仍未授权。

`temperature_comparison.csv`的macro MSE/MAE是每个temperature的20个dataset-horizon cells算术平均。
`cell_winners.csv`逐dataset-horizon选择最低MSE；`aggregation_stability.csv`分别按dataset和horizon平均后选择最低MSE。
这些稳定性统计只解释shared selection，不是formal effectiveness。

输出`pilot_result_gate.json`固定candidate name、tau、macro scores、cell/dataset/horizon wins、config/selection hashes与
下一决策。若任一完整性或no-test category失败，analyzer拒绝冻结formal candidate。
