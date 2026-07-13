# StageC Five-Dataset Mechanism Validation Policy

## Status

| Field | Content |
| --- | --- |
| `scope` | future StageC paper-core mechanism screening and confirmation |
| `dataset_suite` | ETTh1, ETTh2, ETTm1, ETTm2, Weather |
| `screen_seed` | 2021 |
| `confirmation_seeds` | 2021, 2022, 2023 |
| `current_ready_profiles` | ETTh1, ETTh2, ETTm1, ETTm2, Weather |
| `profile_calibration_pending` | none |
| `effective_after` | active since 2026-07-14 |
| `extension_contract` | `configs/stage_c_five_dataset_profile_extension.json` |
| `frozen_contract` | `configs/stage_c_five_dataset_natural_profiles.json` |

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

## ETTh1/ETTm2 Extension Protocol

该extension严格复用历史R2的coarse-grid原则，但使用新的active config、runner与analyzer，不直接启动archive：

1. Phase A：`patch_num={12,24,48}`，固定`d_model=64,d_ff=128`，每dataset只用seed2021；
2. Phase B：固定Phase A选中的patch，比较`32/64`、`64/128`、`128/256`三档width；medium直接复用
   Phase A结果，因此每dataset只新增narrow/wide两次训练；
3. selection按8个dense horizons的macro normalized regret最小；依次用max regret、H720 regret和profile
   name打破tie；parameter count不参与选择；
4. Phase C：只为selected profile补seeds2022/2023。8 horizons的mean MSE CV不高于3%、max CV不高于
   5%才允许freeze；
5. 共`6 + 4 + 4 = 14`个validation-only runs，所有metrics的`evaluation_split`必须为`val`。

若stability gate失败，不得切换到test或围绕D2单独调参；应回到control protocol审计失败源，再决定是增加
confirmation seeds还是重开coarse profile calibration。

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
