# D20 CST Step 7A Production Implementation Gate

## What we test

本地门槛验证`SC-D20-CST-v1`是否忠实实现Step6冻结的三臂contract，而不判断预测性能。目标是排除CLI未接线、
projection错误、三臂初始函数不一致、prefix不一致或summary path不可训练等implementation fault。

## Artifact construction

checker按五个dataset profile分别构造`A6_MEASURE_RETRAIN/A6_CST_SPEC/A6_CST_RANDOM`，复用相同初始化源，
执行synthetic forward/backward。projection、base operator、initial output、prefix与summary-head gradient均直接从
production model读取；没有加载dataset、checkpoint、validation或test。

## Metrics

- `max_initial_output_gap`：三臂zero-init时full prediction最大绝对差；
- `max_prefix_gap`：full-$T$ crop与相同trajectory prefix的最大绝对差；
- `max_projection_orthogonality_gap`：$\|Q^TQ-I\|_\infty$；
- `max_spectrum_dc_leakage`：SPEC columns的DC泄漏；
- `minimum_spec_random_summary_gradient_difference`：两种projection产生的summary-head gradient最小差异，确认
  control不是同一signal的别名。

## Result

- gate：`9/9 pass`；
- 15 CLI cases、15 model constructors、60 shape/prefix cases、10 summary-gradient cases；
- initial output gap：`0`；prefix gap：`0`；
- maximum production projection orthogonality gap：`3.136e-08`；
- maximum spectrum DC leakage：`1.602e-07`；
- minimum SPEC/RANDOM summary-gradient difference：`0.0769151`；
- SPEC与RANDOM均只比A6多`16,384`个trainable parameters，且两者相同。

Machine-readable evidence：`gate_summary.json`。

## Decision

`step7a_pass_step7b_prelaunch`。production实现与Step6 theory contract一致，允许冻结remote matrix；该结论不是
accuracy evidence，不授权paper-method promotion。

## Failure-attribution boundary

当前没有implementation pathology。后续若SPEC在训练中失败，必须再区分transfer hypothesis、frequency
specificity、optimization、path activation与generic capacity control；不能因Step7A正确就预设机制有效。
