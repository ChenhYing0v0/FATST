# Stage C ISCF-RSCC Step9 Analyzer

## 作用

`scripts/analyze_stage_c_iscf_rscc_step9.py`只读取冻结的RSCC validation artifacts与历史EQUAL parent artifacts，
生成protocol audit、standard-horizon comparison、internal mechanism health和machine decision。它不训练模型、不读取
official test，也不修改checkpoint。

## 输入与统计量

- effective runs：`iscf_rscc`、`iscf_equal_armerr`、`iscf_rscc_shuffled`与历史`iscf_equal`，共20 runs；
- performance cells：每run读取H96/H192/H336/H720 validation MSE/MAE，共80 cells；
- `gain_percent=(reference-candidate)/reference*100`，正值表示candidate更优；
- `dataset_wins`和`horizon_wins`分别对相应4个horizons或5个datasets的MSE gain取macro mean后计正值数；
- coalition oracle headroom：从`probe_arms [R,5,T]`、`probe_fused [R,T]`、
  `probe_direct_policy [R,T,5]`和`probe_targets [R,T]`精确构造leave-one-scope-out risk，再计算normalized-positive
  credit forecast相对actual fused L1的gain；
- policy-credit alignment：对每个row/target的five-scope policy与exact coalition credit计算Spearman，报告median；
- gradient health：读取每epoch五个`train_pcc_scope_s*_mode_grad_norm`，统计非零scope数与全局minimum。

## Gate与failure attribution

Analyzer直接读取`configs/stage_c_iscf_rscc_step7b.json`中的预注册阈值。primary要求RSCC相对EQUAL达到
MSE至少`+0.3%`、MAE正、3/5 datasets与3/4 horizons为正；两个matched controls各要求MSE至少`+0.1%`。
internal gate要求finite/nonzero gradients、至少3个policy scopes使用、RSCC coalition headroom为正，且policy-credit
Spearman高于EQUAL。

- primary fail：`hypothesis_false`，关闭exact RSCC；
- primary pass但control fail：`capacity_control_explains`；
- performance/control pass但internal fail：`intervention_point_wrong`；
- 全部通过：只请求独立formal-test design/authorization，不自动访问test。

## Code-theory consistency

实现严格对应“reliability-preserving coalition calibration”的验证边界：EQUAL比较决定额外coalition loss是否产生净收益，
ARMERR与SHUFFLED分别隔离standalone error和scope binding，internal health检查credit是否真的进入既有policy。该analyzer
不能证明泛化到official test，也不能把positive validation单独升级为paper-core pass。
