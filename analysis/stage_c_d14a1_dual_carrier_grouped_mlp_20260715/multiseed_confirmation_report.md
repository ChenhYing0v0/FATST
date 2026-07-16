# StageC D14-A1 Three-Seed Confirmation Report

## 1. What We Tested

D14-A1检验一个problem-level问题：在fixed past与fixed future domain下，point、block与global
future-output coupling scopes是否学成不同函数，并且其相对优劣是否随sample和future region稳定交叉。

本次补齐seeds 2022/2023，并与seed2021按预注册规则聚合：

- datasets：ETTh1、ETTh2、ETTm1、ETTm2、Weather；
- carriers：`neutral_raw` primary direction gate与`a6_natural` sensitivity carrier；
- 每个seed包含40个neutral runs和45个A6 runs，共`3 × 85 = 255`个训练结果；其中本轮新增
  `2 × 85 = 170`个结果；
- 全部使用train/validation，`test_used=false`；
- stable evidence要求同一dataset至少2/3 seeds复现；
- GroupedMLP仍是`diagnostic_only`，不等于PCSD或CCRL method。

远端结果同步后，使用本地`analyze_stage_c_d14a1_multiseed.py`从原始per-seed artifacts独立重算。
远端与本地decision一致：

`dual_carrier_confirmation_pass_authorize_d14b_design`

## 2. What Each Metric Means

- `stable crossing`：同一对canonical scales在short/mid/long regions中出现双向胜负，并在至少2/3 seeds复现；
  它回答“一个fixed coupling scope是否统一支配”。
- `strict oracle`：逐sample × region选择最佳scale，相对validation上事后最佳fixed scale的MSE改善；它扣除了
  train-only scale selection不稳造成的虚假headroom。
- `sample over bin policy`：逐sample × region oracle相对“每个region固定一个scale”的改善；它隔离
  instance-specific增量，而不是只允许short/mid/long各用一个静态规则。
- `contiguity`：canonical contiguous partitions相对matched random partitions的oracle改善；它检查temporal
  邻近分组是否比任意等容量分组更合理。
- `A6-LBF comparison`：只判断当前GroupedMLP是否已达到paper-carrier performance；不参与scale problem gate。

## 3. Aggregate Result

| Carrier | Function separation | Carrier skill | Stable crossing | Strict oracle | Sample over bin | Stable contiguity | Contiguity macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| neutral raw-history | 5/5 | 5/5 | 5/5 | +7.1107% | +6.7948% | 4/5 | +0.4230% |
| A6-natural E2E | 5/5 | 5/5 | 5/5 | +9.1259% | +8.5990% | 4/5 | +0.4667% |

所有trained invariants通过；没有non-finite、severe degradation或split/test leakage。两个carrier均独立通过
预注册confirmation gate。

## 4. Per-Dataset Evidence

| Dataset | Neutral strict | Neutral instance | Neutral contiguous seeds | A6 strict | A6 instance | A6 contiguous seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | +11.6604% | +10.4744% | 3/3 | +8.2706% | +8.2147% | 1/3 |
| ETTh2 | +5.5070% | +5.3734% | 1/3 | +8.4745% | +7.3120% | 3/3 |
| ETTm1 | +4.7456% | +4.6449% | 3/3 | +8.6257% | +7.8754% | 3/3 |
| ETTm2 | +8.5071% | +8.4241% | 3/3 | +13.8792% | +13.4720% | 2/3 |
| Weather | +5.1335% | +5.0569% | 2/3 | +6.3795% | +6.1209% | 3/3 |

[Strong Evidence] scale crossing与strict/instance oracle不是seed2021偶然现象：两种carrier均为5/5 datasets
stable crossing，且每个dataset的three-seed mean strict oracle均为正。

[Strong Evidence] static target-region policy不足以兑现headroom。相对validation-bin policy仍有neutral
+6.7948%、A6 +8.5990%，说明需要研究history/sample-dependent coupling choice，而不只是为short/mid/long
预设不同scale。

[Qualified Evidence] temporal contiguity总体有用，但不是universal law。neutral在ETTh2仅1/3 seeds为正，A6在
ETTh1仅1/3；失败dataset还随carrier改变。因此未来可把ordered contiguous scopes作为有根据的default/control，
不能声称所有series都天然按邻近future blocks组织，也不应删除non-contiguous control。

## 5. A6-LBF Performance Boundary

A6 carrier上的train-selected GroupedMLP相对A6-LBF H720 three-seed macro为`-2.6886%`；即GroupedMLP MSE
更差2.6886%。即使事后使用validation-best GroupedMLP，macro仍为`-1.4879%`。逐dataset的train-selected差距为：

| Dataset | GroupedMLP vs A6-LBF H720 |
| --- | ---: |
| ETTh1 | -1.0390% |
| ETTh2 | -5.5236% |
| ETTm1 | -2.2873% |
| ETTm2 | -4.2841% |
| Weather | -0.3089% |

因此本次结果不能写成“adaptive decoder已经提升forecast accuracy”。它证明的是：强carrier内部也存在稳定的
coupling-choice headroom，但当前fixed grouped heads没有保留A6-LBF的强global forecast function。

## 6. Failure Attribution And What Remains Untested

本次diagnostic没有方向级失败；其边界是：

1. `hypothesis_false`：不支持。stable crossing 5/5且strict/instance headroom显著超过冻结阈值。
2. `intervention_point_wrong`：A0问题已由A1的nonlinear parameter-sharing topology修复；function separation 5/5。
3. `readout_or_head_design_wrong`：仍可能解释A6-LBF performance gap。GroupedMLP是诊断头，不是A6-containing
   adaptive decoder。
4. `optimization_or_numeric_pathology`：未观察到；全部invariants通过。
5. `capacity_control_explains`：full-affine containment、matched params与random partition controls未复制全部结果。

仍未测试的核心问题是D14-B：仅使用inference-visible history与target coordinate，是否能预测哪个coupling scope
对当前sample/region更好。oracle使用了真实future error，只是上界，不能在inference直接获得。

## 7. Decision And Rollback Point

[Decision] D14-A从`single-seed problem evidence`升级为`three-seed dual-carrier problem confirmed`。当前只授权
D14-B返回11-step loop的Step 4-6，完成source-informed design、theory feasibility、controls与narrative gate。

- `d14_b_step4_6_design_authorized=true`；
- `d14_b_implementation_authorized=false`；
- `paper_method_implementation_authorized=false`；
- `test_used=false`。

若D14-B在matched、cross-fitted、leakage-free条件下失败：关闭CCRL，PCSD单独返回Step 4并重新寻找第二项
training contribution；不得把D14-A oracle直接包装成可部署router。若D14-B只由target-only policy解释，则关闭
instance-adaptive claim。

## 8. Artifacts

- local gate：`local_reanalysis_multiseed/gate.json`；
- per-dataset table：`local_reanalysis_multiseed/multiseed_dataset_metrics.csv`；
- compact machine interpretation：`local_reanalysis_multiseed/research_interpretation.md`；
- frozen design：`configs/stage_c_d14a1_dual_carrier_grouped_mlp.json`。
