# StageC Five-Dataset Mechanism Validation Policy

## Status

| Field | Content |
| --- | --- |
| `scope` | future StageC paper-core mechanism screening and confirmation |
| `dataset_suite` | ETTh1, ETTh2, ETTm1, ETTm2, Weather |
| `screen_seed` | 2021 |
| `confirmation_seeds` | 2021, 2022, 2023 |
| `current_ready_profiles` | ETTh2, ETTm1, Weather |
| `profile_calibration_pending` | ETTh1, ETTm2 |
| `effective_after` | ETTh1/ETTm2 validation-only natural profiles frozen |

## Why Five Datasets

五数据集覆盖hourly ETT、15-minute ETT与21-channel Weather，可降低candidate只适配某一sampling frequency、
dataset family或恰好被选中三数据集的风险。这减少的是cross-dataset selection uncertainty，不等价于减少
training stochasticity；后者必须用paired multi-seed实验评估。

## Profile Rule

- 已冻结的ETTh2、ETTm1、Weather natural profiles保持不变；
- ETTh1与ETTm2必须使用与当前contract相同的validation-only两阶段natural coarse grid：seed2021选择，
  seeds2022/2023只确认selected profile的absolute stability；
- dataset之间允许不同`patch_num/d_model/d_ff`，但test不得参与profile选择；
- profile一经冻结，所有candidate和matched controls必须共用，不做per-method/per-mechanism tuning；
- ETTh1/ETTm2不得从旧`R_2026_FSA`或archive静默继承profile或结果。

ETTh1/ETTm2冻结后生成新的five-dataset contract hash；现有三dataset hash仍只描述历史Step 7B，不得被
追溯改写。

## Staged Experiment Rule

### Broad screen

```text
5 datasets × all preregistered candidate/control arms × seed2021
```

它用于快速关闭明显失败的exact design。单seed结果只能形成rollback或`partial_pass`，不能形成paper claim。

### Confirmation

只有broad screen未触发rollback，才运行：

```text
5 datasets × all decisive candidate/control arms × seeds {2021,2022,2023}
```

必须在相同dataset与seed内作paired comparison；不能只给candidate补seed而让control停留在seed2021。报告
per-dataset mean/std、至少`2/3` seeds方向一致性和五dataset macro，不用增加dataset数量替代seed分析。

## Historical Boundary

本政策只约束未来候选。PMFO-RCT v1的三dataset Step 7B已经按预注册gate完成并rollback；新增ETTh1/ETTm2
不会追溯性重开v1，也不会改变其`readout_or_head_design_wrong`结论。
