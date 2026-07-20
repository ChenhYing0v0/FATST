# SC-D23-FCMI Step9/10 Formal Audit

Decision=`fails_a6_internal_valid`。

## Four evidence layers

- paper-facing effectiveness: `False`；
- matched attribution: {"capacity_fcmi_vs_dense_dual": false, "decomposition_fcmi_vs_standard_dual": true, "effectiveness_fcmi_vs_a6": false, "interaction_fcmi_vs_generic_dual": true, "order_fcmi_vs_order_shuffled": false, "target_coordinate_fcmi_vs_target_shuffle": true}；
- internal health: {"ETTh1": true, "ETTh2": true, "ETTm1": true, "ETTm2": true, "Weather": true}；
- failure attribution: 见machine decision与冻结decision map。

该报告只允许由完整五dataset矩阵生成；不得选择性删除负dataset、horizon、control或seed。
