# StageC SC0 Best-vs-Last Test Diagnostic Code Explanation

## Purpose

SC0原报告中的31.63%-44.95%只来自validation trajectory。TimeAlign作者明确说明论文主表采用固定轮数
的last checkpoint，并指出validation/test distribution shift可能使validation-best训练不足。因此本诊断在
SC0 calibration已经结束后，打开test比较同一fixed-20 trajectory的best与last checkpoint。

该test结果只用于判断checkpoint机制，不允许反向选择patch profile或修改已观察checkpoint。

## Artifact Flow

`evaluate_stage_c_sc0_checkpoint_test_gap.py`对每个SC0 run读取：

- `effective_config.json`：恢复dataset、模型结构、dense horizons与official test flag；
- `checkpoint_best_val.pt`：full-720 validation MSE最低epoch；
- `checkpoint_last.pt`：同一trajectory的epoch20 state；
- 两个validation metric CSV：建立validation/test同horizon对照。

模型重建后strict load两个state，并在相同test loader上分别计算H48/96/144/192/288/336/512/720的
MSE/MAE。三个dataset分配到GPU 0/1/2并行；不重新训练。

## Output Definitions

每行对应`dataset × arm × horizon`：

- `validation_best_mse`、`validation_last_mse`：原SC0 validation artifacts；
- `validation_last_vs_best_mse`：`(last-best)/best`；
- `test_best_mse`、`test_last_mse`：同一checkpoint在test上的MSE；
- `test_last_vs_best_mse`：负值表示last test更好；
- `test_best_mae`、`test_last_mae`及相对差：MAE对应量；
- `selection_role=diagnostic_only_after_profile_freeze`：禁止用于超参数选择。

`analyze_stage_c_sc0_checkpoint_test_gap.py`汇总72个比较，并分别报告全部dense horizons、H720整体和
各dataset H720的last win count、mean delta与最坏/最好差值。

## Interpretation Boundary

- validation显著恶化但test不恶化：支持TimeAlign作者关于distribution shift和last策略的解释；
- validation与test同时恶化：说明exact fixed-20 policy对当前A6 carrier也存在真实泛化损失；
- dataset间方向不同：checkpoint policy应成为dataset-aware protocol field或报告双口径，不能用一个
  全局规则覆盖；
- seed2021单次诊断不能证明multi-seed稳定，必要时再设计确认实验。
