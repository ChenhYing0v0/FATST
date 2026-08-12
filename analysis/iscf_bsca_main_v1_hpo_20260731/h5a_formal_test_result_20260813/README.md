# H5A Formal-Test Audit Artifacts

`test_audit_completeness.json`、`test_audit_ledger.jsonl`、
`all_trial_scorecard.csv`与`profile_aggregates.csv`是48-checkpoint formal-test的
canonical完整性与metric artifacts。

通用formal-test analyzer还按其legacy mean-MSE fallback生成了根目录下的
`profile_ranking.csv`、`selected_profiles.json`和`selected_profile_scorecard.csv`。
这三个文件的selection role为`excluded_generic_mse_only_diagnostic`，不符合H5A冻结的
Main II best-cell selector，因此不得用于profile或table decision。

H5A权威selection artifacts位于`frozen_main_ii_selector/`：

- `all_profile_main_ii_ranking.csv`：historical+H5A完整profile pool；
- `selected_profile_scorecard.csv`：三个selected dataset profiles的12个four-H cells；
- `h5a_selection_result.json`：guard、best-cell与projected global gates。
