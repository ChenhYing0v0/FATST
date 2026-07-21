# ISCF-v1-CPSI Step9/10 Result and Rollback

## 1. 审计问题与结论

本次正式审计回答：在ISCF-v0 fixed carrier上，native pre-synthesis common-private multiplicative interaction是否能在
五dataset、四standard horizons的official test上改善MSE/MAE，并超过SELF、LINEAR、COMMON和POST-SYNTH
matched controls？

Decision=`cpsi_v1_exact_performance_fail_return_step4_5`。

- [Fact] 25/25 new trainings、25/25 formal tests与10个historical references均protocol valid；
- [Strong Evidence] CPSI相对ISCF-v0 MSE/MAE为`-2.2128%/-1.6987%`，material negative；
- [Strong Evidence] LINEAR control显著优于CPSI，但相对ISCF-v0仅`+0.0217%/+0.0472%`，属于tie；
- [Fact] 五arms internal paths均finite/nonzero，无OOM、NaN、dead path或checkpoint mutation；
- [Decision] 关闭exact CPSI-v1与confirmation/rescue；保留ISCF-v0 strong carrier，回Step4/5重审operator hypothesis。

该结论拒绝的是“当前common-private product在当前intervention point提供有效inductive bias”，不是所有cross-scope
interaction的方向级否定。不得用rank、seed、loss、router或dataset-specific tuning救回exact v1。

## 2. Data、split与统计量

new runs来自`stage_c_iscf_v1_cpsi_v1`，historical ISCF-v0/A6_FULL来自
`stage_c_siff_equal_attribution_v2`。所有模型用validation H96/H192/H336/H720 mean MSE选择checkpoint；test不选择
epoch或配置。formal test在25/25 training完成后一次性执行，checkpoint SHA256前后不变。

对任一candidate/reference cell，gain定义为

$$
g=100\left(1-\frac{metric_{candidate}}{metric_{reference}}\right).
$$

`macro_gain_percent`是5 datasets × 4 horizons共20个$g$的算术平均；`dataset_wins`先对每个dataset的四个horizons
求平均再计正值；`cell_wins`直接计20 cells中的正值。`max_dataset_degradation_percent`是最差dataset平均gain的负部。
validation只解释transfer，不改变test decision。

## 3. Paper-facing effectiveness

### 3.1 CPSI vs ISCF-v0

| Metric | Macro gain | Cell wins | Dataset wins | Worst dataset degradation |
| --- | ---: | ---: | ---: | ---: |
| Test MSE | -2.2128% | 4/20 | 1/5 | ETTh2 -5.0538% |
| Test MAE | -1.6987% | 5/20 | 1/5 | ETTh2 -4.0698% |
| Validation MSE | -3.1209% | 3/20 | 1/5 | ETTh1 -6.8475% |
| Validation MAE | -1.9852% | 2/20 | 1/5 | ETTh1 -4.5019% |

MSE按horizon分别为H96 `-1.7390%`、H192 `-2.0362%`、H336 `-1.9874%`、H720 `-3.0887%`；H720为
0/5 dataset wins。negative在validation/test同向，不是split reversal。

Step6 material-negative gate要求macro MSE不高于`-0.5%`且至少4/5 datasets为负，或任一dataset退化达到5%。本结果
同时满足两条，因此不是`[-0.5%,+0.3%)` mild/inconclusive band。

### 3.2 CPSI vs A6_FULL

CPSI相对A6_FULL test MSE/MAE为`-0.7775%/-1.0606%`，仅1/5 datasets正向。即使不使用更强ISCF parent作为
reference，CPSI仍未形成paper-facing improvement。

## 4. Matched mechanism attribution

| Comparison (CPSI relative to control) | Test MSE gain | Cell wins | Dataset wins |
| --- | ---: | ---: | ---: |
| SELF | +0.9560% | 17/20 | 4/5 |
| LINEAR | -2.2586% | 3/20 | 0/5 |
| COMMON | +0.7648% | 13/20 | 3/5 |
| POST-SYNTH | -1.7093% | 4/20 | 1/5 |

CPSI优于SELF和COMMON，说明结果不能简单归因于任意same-parameter nonlinear depth；但LINEAR与POST均显著优于
CPSI，否定“common-private multiplicative nonlinearity”和“pre-synthesis placement”对当前收益的必要性。

LINEAR是全矩阵最强新增arm：相对ISCF-v0 test MSE/MAE仅`+0.0217%/+0.0472%`，11/20 cells、3/5 datasets；
validation则为`+0.4491%/+0.2700%`。test macro落在`±0.3%` attribution tie band内，且ETTh1 MSE平均退化
`3.2163%`，不能视为stable extension。理论上fixed linear common/private mixing可吸收到ISCF各scope affine maps，
所以该arm最多是optimization reparameterization/control，不构成新的function-class或paper contribution。

POST相对ISCF-v0 test MSE/MAE为`-0.4882%/-0.6362%`，MSE恰在inconclusive band内但MAE与dataset coverage均弱；
它相对A6_FULL的MSE `+0.9082%`只说明ISCF family仍是strong carrier，不支持POST mechanism promotion。

## 5. Internal mechanism health

25/25 arm-dataset internal rows全部finite。CPSI五dataset的message RMS为`0.0888–0.3074`，latent RMS为
`2.7585–8.2330`，trained output projection norm为`0.3777–1.6897`。LINEAR、SELF、COMMON、POST的对应量也均
非零；POST message RMS最高但没有paper-facing gain。

因此：

- 不是`optimization_or_numeric_pathology`；
- 不是zero-init path永久关闭；
- internal activity不能覆盖negative effectiveness；
- message magnitude与performance不形成“越活跃越有效”的证据。

## 6. Failure attribution

Primary attribution=`hypothesis_false_for_exact_CPSI_v1`：D1.1发现的scope response relation是真实problem clue，但
“mean × private-deviation product在pre-synthesis modes上提供有益inductive bias”的可证伪推论没有成立。

Secondary boundary=`readout_or_intervention_design_wrong_remains_possible`：更广义cross-scope operator仍未被方向级否定，
因为本实验只覆盖one low-rank elementwise product family；但LINEAR tie、POST negative与四horizon同向failure共同表明，
不能直接做rank/seed/placement rescue。下一候选必须重新通过Step4/5，证明其function-class增量既不被LINEAR吸收，也不
退化为generic extra depth。

`capacity_control_explains`不用于解释CPSI gain，因为CPSI没有gain；它只说明更简单的LINEAR optimization path足以超过
CPSI，但不能超过ISCF parent。

## 7. Rollback与授权状态

- exact CPSI-v1：closed；
- SELF/COMMON：diagnostic complete，closed；
- LINEAR：retained as optimization/control evidence，不是method candidate；
- POST-SYNTH：inconclusive placement diagnostic，不promote；
- ISCF-v0：继续作为frozen strong carrier；
- active paper-core method：none；
- confirmation seeds、rank rescue、new loss/router：false。

下一步回Step4/5做source-informed operator redesign。允许使用本次证据形成新的hypothesis，但必须在实现前重新冻结
problem、function-class boundary、matched controls与narrative gate；不得把LINEAR的tie结果改名为创新。

## 8. Artifacts

- formal machine decision：`primary/decision.json`；
- protocol audit：`primary/run_audit.csv`；
- paper-facing cells：`primary/metrics.csv`；
- preregistered comparisons：`primary/comparison_cells.csv`与`primary/comparison_summary.csv`；
- all-arm validation/test context：`primary/arm_reference_context.csv`；
- internal health：`primary/internal_health.csv`；
- lite remote artifacts：`raw_lite/`（checkpoint与large NPZ excluded）。
