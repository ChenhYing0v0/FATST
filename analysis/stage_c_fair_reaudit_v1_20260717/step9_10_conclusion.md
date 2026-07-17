# SC-RETRO-FAIR-v1 Step 9–10 结论与归因

## 1. 审计有效性

[Fact] 70/70 runs完成，280/280 test standard-horizon cells有效。全部runs：

- 从头joint训练Encoder–Decoder；
- 使用相同dataset profile、seed、optimizer class与训练预算；
- 使用validation H96/H192/H336/H720平均MSE选择checkpoint；
- test阶段checkpoint SHA256未变化；
- prefix projectivity、finite values、test authorization与paired Encoder initialization全部通过。

因此本次没有frozen replacement、capacity mismatch或artifact缺失导致的方向级阻塞。

## 2. 最重要的结论

[Decision] 结果不是“PCSD、PCC、SIFF全部失败”，也不是“原两项contribution已经成立”，而是：

> `SIFF_EQUAL`形成了目前最强、跨四horizon稳定的paper-facing performance carrier；旧SIFF失败包含明显的
> best-H720 checkpoint假失败。但PCSD架构本身仍失败，PCC specificity失败，SIFF也尚未通过完整的
> objective-robust与control-specific attribution。因此当前状态是
> `performance_partial_pass / two-contribution_attribution_fail`。

## 3. 预注册comparison结果

Primary MSE gate要求macro $\ge0.3\%$、至少11/20 cells、3/5 datasets与3/4 horizons。

| Comparison | Macro gain | Cells | Datasets | Horizons | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| PCSD_DIRECT vs A6 | -0.8562% | 10/20 | 2/5 | 0/4 | fail |
| PCSD_MEASURE vs dense matched | +0.0960% | 10/20 | 2/5 | 2/4 | fail |
| PCC vs EQUAL on PCSD | +0.1689% | 10/20 | 2/5 | 4/4 | fail |
| PCC vs PRIOR on PCSD | +0.0806% | 13/20 | 2/5 | 4/4 | fail |
| SIFF vs PCSD under EQUAL | **+0.5906%** | **14/20** | **3/5** | **4/4** | **pass** |
| SIFF vs PCSD under PRIOR | +0.0613% | 8/20 | 2/5 | 3/4 | fail |
| SIFF vs PCSD under PCC | +0.1548% | 9/20 | 2/5 | 3/4 | fail |
| PCC vs EQUAL on SIFF | **-0.2663%** | 7/20 | 1/5 | 1/4 | fail |
| PCC vs PRIOR on SIFF | +0.1725% | 14/20 | 3/5 | 4/4 | fail: macro |
| SIFF+PCC vs A6 | **+1.3812%** | **16/20** | **4/5** | **4/4** | **pass** |
| ordered vs constant | **+0.6986%** | **18/20** | **5/5** | **4/4** | **pass** |
| ordered vs permuted | +0.2949% | 15/20 | 3/5 | 4/4 | fail: macro by 0.0051 pp |
| ordered vs Q1-wide | **+0.8658%** | **12/20** | **3/5** | **4/4** | **pass** |
| ordered vs independent | +0.1600% | 12/20 | 3/5 | 3/4 | fail: macro |

## 4. 当前最佳模型不是SIFF+PCC，而是SIFF+EQUAL

`SIFF_EQUAL`相对A6：

- test MSE macro `+1.6436%`，17/20 cells、4/5 datasets、4/4 horizons；
- test MAE macro `+0.9084%`，16/20 cells；
- horizon gains：H96 `+1.642%`、H192 `+1.462%`、H336 `+1.694%`、H720 `+1.776%`；
- dataset gains：Weather `+1.228%`、ETTm1 `+2.538%`、ETTm2 `-0.419%`、
  ETTh1 `+1.515%`、ETTh2 `+3.355%`。

相比之下，加入PCC后`SIFF_PCC`相对`SIFF_EQUAL`：

- MSE `-0.2663%`；
- MAE `-0.3577%`；
- 只在1/5 datasets、1/4 horizons形成MSE win；
- H720平均退化`-0.652%`。

因此`SIFF+PCC vs A6`虽然通过performance gate，但不能被解释为两个contributions的joint成功；PCC实际上
降低了更简单`SIFF_EQUAL`的性能。

## 5. PCSD结论

[Strong Evidence] `PCSD_DIRECT vs A6=-0.8562%`，且validation/test分别为`-1.5853%/-0.8562%`，
无numeric或protocol pathology。因此PCSD-CF v1的native coupling-field architecture仍是exact failure。

PCSD在EQUAL/PRIOR/PCC下相对A6分别约`+1.0578%/+1.1472%/+1.2266%`，但`PCSD_MEASURE`本身已达到
`+1.0219%`，而EQUAL/PCC相对MEASURE只增加`+0.0177%/+0.1842%` MSE且MAE反而更差。更合理的解释是
measure-aligned training或generic auxiliary supervision提供了主要收益，不是PCSD架构成立。

## 6. PCC结论

[Strong Evidence] PCC未超过EQUAL或PRIOR controls的预注册gate，并在SIFF carrier上明确劣于EQUAL。
这不是“PCC需要再调一点系数”，而是其transport-specific credit claim没有带来可归因优势。

`PCC-v1-TI`应关闭为exact contribution candidate，rollback Step2/4。其nested-risk/transport algebra可以保留
为negative evidence或future ingredient，但不应进入seed confirmation。

## 7. SIFF结论

[Strong Evidence] SIFF在EQUAL objective下超过matched PCSD并同时改善MSE/MAE，说明scale-indexed
forecast field不是全面失败。ordered相对constant为5/5正，相对Q1-wide也通过，证明：

1. 非constant scale variation有价值；
2. $Q=2$ scale field不是单纯依靠更宽Q1 capacity；
3. 四-horizon收益不是只来自H720。

[Unresolved] 但SIFF在PRIOR/PCC下的architecture effect没有通过，ordered也没有超过independent matched
control的macro gate。现有constant/permuted/Q1/independent controls均在PCC objective下，不能直接解释
`SIFF_EQUAL`的正向结果。

因此SIFF状态应从`failed_exact_design`修正为
`partial_pass_under_equal_skill / attribution_blocked`，不能直接进入论文claim或two-seed confirmation。

## 8. Checkpoint规则确实修复了旧假失败

旧best-H720 artifacts中，`SIFF_EQUAL vs PCSD_EQUAL`的四-horizon validation macro为`-2.3897%`，
其中ETTm2为`-14.2617%`。本次four-horizon checkpoint下：

- validation macro变为`+0.1469%`；
- test macro为`+0.5906%`；
- ETTm2 test差距缩小到`-0.9784%`；
- Weather、ETTm1、ETTh2 test分别为`+1.111%/+1.619%/+1.310%`。

这证明旧SIFF negative不能全部归因于架构理论失败；best-H720 checkpoint确实丢弃了更适合unified
multi-horizon scorecard的状态。CTD无需恢复才能得到这一结论，因为当前matched retraining已经完成更直接的
effectiveness audit。

## 9. Validation/test分工的新规则得到支持

本次再次出现显著split reversal：

- `dense_measure vs A6`：validation `-3.1488%`，test `+0.8696%`；
- `PCC on SIFF`：validation `+0.1091%`，test `-0.2663%`；
- `SIFF_EQUAL vs PCSD_EQUAL`：validation `+0.1469%`，test `+0.5906%`。

因此validation继续用于checkpoint是必要的，但用它判机制pass/fail会产生不同方向的误判；test-primary规则
在当前benchmark上具有实际必要性。

## 10. 仍缺少的关键controls

当前结果暴露两个归因缺口：

1. 缺少`A6_MEASURE_ONLY`，无法隔离`dense_measure/PCSD_measure`的收益究竟来自harmonic target measure还是
   新readout；
2. 缺少EQUAL objective下的`SIFF_CONSTANT/PERMUTED/Q1_WIDE/INDEPENDENT`，无法证明
   `SIFF_EQUAL`收益来自ordered scale coordinate，而不是更一般的multi-arm/nonlinear capacity。

这些不是为失败cell调参，而是现有positive result必需的mechanism attribution controls；由于已观察test，
后续必须建立新版本并标记`test_informed`。

## 11. Step 10决策

- `PCSD-CF-v1`: exact architecture failure，关闭；
- `PCC-v1-TI`: exact contribution failure，关闭并rollback Step2/4；
- `SIFF-v1`: 从失败修正为partial pass，rollback Step6补齐EQUAL-context attribution；
- `SIFF+PCC`: performance pass，但joint contribution attribution fail；
- `CTD`: 继续暂停；
- `paper_status`: 已获得一个有希望的`SIFF_EQUAL` performance carrier，但尚不足以支撑“两项创新”的论文主体。

下一候选不应直接补seed2022/2023，而应先冻结`SC1-SIFF-v2-EQ-attribution`：

`A6_FULL / A6_MEASURE / PCSD_EQUAL / SIFF_EQUAL / SIFF_CONSTANT_EQUAL /
SIFF_PERMUTED_EQUAL / SIFF_Q1_WIDE_EQUAL / SIFF_INDEPENDENT_EQUAL`。

只有SIFF_EQUAL在同一EQUAL objective下超过全部controls后，才补seed2022/2023。Contribution 2则需独立回到
Step2-4重新定义，不再以PCC为默认方向。
