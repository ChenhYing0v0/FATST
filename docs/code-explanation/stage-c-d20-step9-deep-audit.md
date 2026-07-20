# Stage C D20 Step 9 Deep Audit 代码说明

## 1. 目的与输入

`scripts/analyze_stage_c_d20_deep_audit.py`不修改模型或冻结gate，只对已完成的`SC-D20-CST-v1` artifacts做
post-hoc failure attribution。输入为同步后的轻量run metadata、validation/test horizon metrics、training log，以及
remote frozen analyzer输出。

## 2. 统计定义

任一candidate相对reference的gain定义为：

$$
G=100\left(1-\frac{M_{candidate}}{M_{reference}}\right).
$$

正值表示candidate更好。`macro_gain_percent`是5 datasets × 4 standard horizons的20个cell gain等权平均；
`cell_wins`统计$G>0$的cell数。`dataset_wins`先在每个dataset内平均四个horizon，再统计正值；
`horizon_wins`先在每个horizon内平均五个dataset，再统计正值。

脚本分别对validation和official test计算三组comparison：SPEC-vs-A6 transfer、SPEC-vs-RANDOM specificity、
RANDOM-vs-A6 capacity control。validation只用于解释checkpoint/generalization，不重新决定正式effectiveness。

## 3. Dense horizon与checkpoint审计

`dense_comparison_by_horizon.csv`读取test audit的H1到H720累积prefix指标，对每个horizon先算逐dataset gain再宏平均；
`dense_bin_summary.csv`按Step7B冻结的8个future bins平均这些dense gains。它们只解释收益随预测距离如何变化，
不替代standard-horizon gate。

`checkpoint_audit.csv`从`training_log.csv`重算minimum validation mean-MSE epoch，并与日志记录的best epoch对齐；
`last_vs_best_degradation_percent`为最后epoch相对best validation score的恶化比例。它用于识别early stopping和
generalization pathology，不读取test选择checkpoint。

## 4. Internal intensity

`internal_intensity_vs_gain.csv`将每个dataset的summary prediction contribution standard deviation、SPEC/RANDOM
contribution ratio与test transfer/specificity gain并列。该表只检查路径是否活跃、是否可能过强或冗余；由于只有五个
datasets，不把相关性解释为因果证据。

## 5. Decision boundary

若validation transfer为正而test为负，脚本标记`transfer_validation_test_reversal`。按项目Diagnostic Failure
Attribution Rule，这属于`optimization_or_numeric_pathology(validation_test_mismatch)`，因此只能关闭exact tested
design，不能据此拒绝整个history-spectrum方向。若SPEC相对RANDOM为正但未过冻结0.3% gate，则标记为
`specificity_directional_only`：保留弱正向设计证据，但不允许confirmation或paper claim。
