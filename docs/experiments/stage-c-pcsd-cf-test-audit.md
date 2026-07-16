# StageC PCSD-CF-v1 Milestone Test Audit Protocol

## Status

| Field | Value |
| --- | --- |
| `audit_id` | `SC-D15-T1` |
| `candidate_version` | `SC1-PCSD-CF-v1`（frozen after validation screen） |
| `current_step` | 60/60 completed；PCSD-CF-v1 Step10 fail；PCC Step6 test-informed design authorized |
| `role` | primary milestone test audit；not retraining |
| `user_authorization` | 2026-07-16 explicit |
| `test_access_count` | one formal complete-matrix audit for v1 |
| `test_informed_after_completion` | true |
| `test_retraining/checkpoint_selection` | false / historical best-validation-H720 |
| `matrix` | 12 arms × 5 datasets × seed2021 = 60 frozen checkpoints |
| `test_horizons` | dense H1..720；full-H720 output only prefix crop |
| `PCC Step6` | authorized for design only；implementation/remote仍false |

Local prelaunch artifact：
`analysis/stage_c_pcsd_cf_test_audit_prelaunch_20260716/prelaunch_gate_report.md`。

## What We Plan To Test

validation screen已证明PCSD-CF-v1的plain DIRECT相对A6为0/5，并出现25/25 joint-arm under-training。但最终
paper effectiveness必须由test performance决定。本audit只回答：

> 对完全冻结、由validation选择的同一批checkpoints，validation上的method failure、arm credit starvation与
> control attribution是否在official test split上成立？

它不允许重新训练，也不使用test选择checkpoint、epoch、dataset-specific setting或horizon-specific setting。

## Frozen Matrix

五datasets固定为`ETTh1/ETTh2/ETTm1/ETTm2/Weather`，profiles、seed2021、architecture、training objective、
optimizer与best-validation checkpoint均来自已完成Step7B。十二arms为：

- `A6`与exact-paired `M0`；
- `FIXED_{1,48,144,360,720}`；
- `EQUAL`、`STATIC_TARGET`、`DIRECT`、`RANDOM_PARTITION`；
- parameter-matched `DENSE_NONLINEAR`。

每个checkpoint在audit前后计算SHA-256；任何hash变化使对应run及整个audit失效。test evaluator只新增
metrics/diagnostics/invariant artifacts，不写model parameters。

## Metrics

对每个frozen checkpoint一次性生成full-H720 prediction，并以prefix cumulative error精确计算：

- `test_dense_h1_h720_mse_auc`：720个prefix MSE的算术平均，primary；
- `test_h720_mse`与`test_dense_h1_h720_mae_auc`，secondary；
- DIRECT内部`arm_row_bin_mse`、same-run oracle、policy usage与arm separation；
- DIRECT内部每个scope arm相对对应独立fixed-scope checkpoint的test退化；
- validation与test的DIRECT-relative-control gain差及sign reversal。

`num_rows_channels`是test samples × channels。Horizon-$H$ MSE由前$H$个position的总SSE除以
`num_rows_channels × H`，不是对batch means再平均。

## Gates And Decision Map

primary method threshold复用test访问前已冻结的Step7B threshold：DIRECT至少3/5超过A6且macro gain
`>=0.3%`，同时审计equal/static/dense/random controls。test是effectiveness primary gate，validation只作为
对照与failure attribution。

- `test_method_pass`：先审计validation-test reversal，再授权unchanged seed confirmation design；不直接宣布论文pass；
- `test_fail_with_arm_headroom`：维持`training_blocked`，允许PCC Step6，但标记PCC为`test_informed`；
- `test_fail_without_arm_headroom`：在PCC implementation前回SC1 Step4，重新评估shared-field/readout ceiling；
- `validation_test_reversal`：暂停method claim，先审计split representativeness、checkpoint rule和seed stability；
- `artifact_or_hash_failure`：test audit无效，只允许artifact repair，不增加method版本。

完整60/60 matrix是hard requirement；不得以部分完成结果作方向决策。

## Artifacts

每个historical run目录新增：

- `test_audit_metrics_by_target_horizon.csv`；
- `pcsd_test_audit_diagnostics.npz`；
- `test_audit_invariants.json`。

aggregate写入远程`_test_audit_seed2021/`，返回后同步至新的`analysis/`目录并形成Step9-10报告。所有report必须
记录test exposure，后续PCC设计不得再声称official test完全untouched。

## Completed Result And Decision

2026-07-16一次性official test audit已完成，60/60 runs、60/60 invariant files与checkpoint hash gate均通过；
所有checkpoint均为historical best-validation checkpoint，`checkpoint_retrained=false`。

| Reference | Validation macro gain | Test macro gain | Test wins |
| --- | ---: | ---: | ---: |
| A6 | -1.5833% | -1.3994% | 1/5 |
| EQUAL | -0.0294% | -0.4984% | 1/5 |
| STATIC | -0.6266% | -0.5304% | 2/5 |
| DENSE matched | +2.3492% | -0.8942% | 1/5 |
| RANDOM partition | +0.4499% | -0.1164% | 2/5 |

DIRECT相对A6仅ETTh1为正（+1.5338%），ETTh2/ETTm1/ETTm2/Weather分别为
-0.7352%/-2.6867%/-3.8896%/-1.2196%。因此primary method gate明确失败，PCSD-CF-v1不得成为paper claim。
validation上的`DIRECT > DENSE`发生明显test reversal，但`DIRECT > A6`在两split均为整体失败，不能把结论归因于
validation-only误判。

另一方面，same-run oracle test headroom macro为+2.0197%，3/5 datasets为正；25/25 DIRECT arms仍低于对应
independent fixed training，median degradation 90.6647%。故预注册decision为`test_fail_with_arm_headroom`：

- exact PCSD-CF-v1 method被拒绝；
- coupling-scope问题与same-run credit starvation线索保留；
- PCC只获准进入`test_informed` Step6 design，不获准实现或远程训练；
- 新设计必须升级candidate version，且不得根据本次test逐dataset/horizon调参。

正式报告：`analysis/stage_c_pcsd_cf_test_audit_seed2021_20260716/test_audit_report.md`。
