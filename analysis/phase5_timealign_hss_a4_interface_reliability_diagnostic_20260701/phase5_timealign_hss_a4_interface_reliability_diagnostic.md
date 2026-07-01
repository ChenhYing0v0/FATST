# Phase5 A4 Interface Reliability Diagnostic

## 诊断目标

[Step 2/3] A3E 失败后，本诊断不提出新 head，也不把 dataset/horizon 手工选择写成最终方法。
它只回答：现有 capacity-preserving / prefix-aware paths 是否存在稳定可靠性差异，以及这种差异是否大到值得设计 learned reliability routing。

Dataset universe: `ETTh2 + ETTm1 + Weather`；每个 dataset 使用 `96/192/336/720` 四个 target horizons。

## Path-Level Reliability

| rank | path_id | family | wins_as_best | within_0.2%_best | mean_gap_to_best_% | vs_fixed_% | vs_h1_% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `a3d_teacher_preserved` | `teacher_preserved_nested` | 2 | 5 | 0.439 | -4.712 | -0.056 |
| 2 | `h1_target_set` | `dense_target_set` | 1 | 4 | 0.495 | -4.658 | 0.000 |
| 3 | `a3e_target_conditioned_scratch` | `target_conditioned_nested` | 1 | 5 | 0.677 | -4.509 | 0.181 |
| 4 | `a3e_target_conditioned_warm` | `target_conditioned_nested` | 0 | 4 | 0.692 | -4.517 | 0.199 |
| 5 | `h1c_row_gated` | `dense_target_set` | 0 | 4 | 0.949 | -4.276 | 0.453 |
| 6 | `a3c_warm_nested` | `primary_nested` | 3 | 4 | 0.951 | -4.295 | 0.458 |
| 7 | `a2_nested` | `primary_nested` | 2 | 4 | 1.063 | -4.188 | 0.566 |
| 8 | `fixed_specialist` | `fixed` | 3 | 3 | 5.969 | 0.000 | 5.458 |

## Family-Level Reliability

| rank | family | path_count | wins_as_best | within_0.2%_best | mean_gap_to_best_% | vs_fixed_% | vs_h1_% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `teacher_preserved_nested` | 1 | 2 | 5 | 0.439 | -4.712 | -0.056 |
| 2 | `target_conditioned_nested` | 2 | 1 | 9 | 0.684 | -4.513 | 0.190 |
| 3 | `dense_target_set` | 2 | 1 | 8 | 0.722 | -4.467 | 0.226 |
| 4 | `primary_nested` | 2 | 5 | 8 | 1.007 | -4.241 | 0.512 |
| 5 | `fixed` | 1 | 3 | 3 | 5.969 | 0.000 | 5.458 |

## Best Path Map

| dataset | horizon | best_path | second_path | margin_vs_second_% | best_vs_fixed_% | best_vs_h1_% |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | `a3e_target_conditioned_scratch` | `h1_target_set` | -0.215 | -10.845 | -0.215 |
| ETTh2 | 192 | `h1_target_set` | `a3d_teacher_preserved` | -0.222 | -17.145 | 0.000 |
| ETTh2 | 336 | `a3d_teacher_preserved` | `h1_target_set` | -0.306 | -19.283 | -0.306 |
| ETTh2 | 720 | `a3d_teacher_preserved` | `a3e_target_conditioned_warm` | -0.536 | -4.700 | -0.968 |
| ETTm1 | 96 | `a3c_warm_nested` | `a3e_target_conditioned_warm` | -0.270 | -2.872 | -1.622 |
| ETTm1 | 192 | `a3c_warm_nested` | `a3e_target_conditioned_warm` | -0.199 | -4.325 | -0.847 |
| ETTm1 | 336 | `a3c_warm_nested` | `a3e_target_conditioned_warm` | -0.147 | -0.902 | -0.569 |
| ETTm1 | 720 | `fixed_specialist` | `h1c_row_gated` | -0.081 | 0.000 | -0.282 |
| Weather | 96 | `fixed_specialist` | `a2_nested` | -0.614 | 0.000 | -0.825 |
| Weather | 192 | `fixed_specialist` | `a3d_teacher_preserved` | -0.023 | 0.000 | -0.103 |
| Weather | 336 | `a2_nested` | `a3e_target_conditioned_scratch` | -0.003 | -0.425 | -0.073 |
| Weather | 720 | `a2_nested` | `a3e_target_conditioned_scratch` | -0.060 | -1.121 | -0.078 |

## Oracle Routing Upper Bound

| dataset | settings | best_static_path | oracle_vs_best_static_% | oracle_vs_h1_% | oracle_vs_fixed_% |
| --- | --- | --- | --- | --- | --- |
| ETTh2 | 4 | `a3d_teacher_preserved` | -0.166 | -0.430 | -12.865 |
| ETTm1 | 4 | `a3c_warm_nested` | -0.043 | -0.762 | -1.848 |
| Weather | 4 | `a2_nested` | -0.130 | -0.205 | -0.514 |
| ALL | 12 | `a3d_teacher_preserved` | -0.431 | -0.504 | -5.758 |

## 机制判断

- [Fact] 当前 best static path 是 `a3d_teacher_preserved`；per-setting oracle 相对它仍有 `-0.431%` MSE 改善上限。
- [Strong Evidence] 没有单一路径在 12 个 setting 上稳定最优；best path map 同时出现 dense target-set、teacher-preserved nested、target-conditioned nested 和 warm-started nested。
- [Strong Evidence] 这说明 Stage A 的核心问题不应再写成寻找一个 universal prefix-aware head；更合理的问题是 capacity-preserving path 的可靠性在不同 future context 下变化。
- [Self-Critique] 但该诊断仍基于 test horizon MSE 的离线 oracle。它只能证明 reliability 差异存在，不能证明可部署的 routing signal 已经存在。

## 下一步

进入 A4R：`Reliability-Aware Capacity-Preserving Interface Diagnostic`。
下一轮不应该按 dataset/horizon id 手工选择路径，而应导出可训练前或 validation-time 获得的 reliability signals，例如 teacher-student disagreement、prefix residual、validation gap、segment volatility。
只有当这些 signals 能解释 best-path map 或 gap-to-best，才进入 learned/estimated interface routing method。
