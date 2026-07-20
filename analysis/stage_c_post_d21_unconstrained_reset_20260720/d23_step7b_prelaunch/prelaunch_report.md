# SC-D23-FCMI Step7B Formal Design / Prelaunch Audit

## 1. Decision

Step7B machine gate为`21/21 pass`：

```text
step7b_prelaunch_pass_waiting_remote_test_authorization
```

这表示candidate、controls、matrix、evaluation、four-layer gates与rollback均已冻结且local tooling可执行。
它不授权remote training或official test；当前runner在normal launch时必须exit 3。

## 2. Dense capacity control

Step7A发现FCMI相对A6少83%–95% active parameters。Step7B新增
`DENSE_DUAL_MATCHED`，其primary path为`STANDARD_DUAL_MATCHED`，并增加：

$$
c(x)=W_c\operatorname{vec}(M(x))+b_c,\qquad
r(x)=B_dc(x)+b_d,
$$

最终forecast为standard-dual forecast加$r(x)$。$W_c,b_c,b_d$均zero-init，$B_d$随机初始化，因此
initial residual严格为0，新增capacity不是inactive parameter padding：第一步$W_c$获得gradient，更新后第二步
$B_d$获得gradient。

dense rank只由冻结profile和A6–FCMI active-parameter gap反解，不读取validation/test：

| Dataset | Rank | A6 active | Dense active | Relative gap |
| --- | ---: | ---: | ---: | ---: |
| Weather | 234 | 419216 | 419675 | 0.1095% |
| ETTm1 | 250 | 391408 | 390891 | 0.1321% |
| ETTh1 | 241 | 613904 | 613266 | 0.1039% |
| ETTh2 | 234 | 419216 | 419675 | 0.1095% |
| ETTm2 | 247 | 1006160 | 1005240 | 0.0914% |

五个profiles的initial standard-output gap为$6.71\times10^{-8}$到$8.94\times10^{-8}$，dense residual
均为0。coefficient first-step gradient norm为1.56–3.01；basis second-step gradient norm为
0.0173–0.0640。

## 3. Frozen matrix

所有runs均from scratch、seed2021、same natural profile、harmonic `measure_only` objective与best-validation
four-horizon selector。

| Role | Arms | Runs | Cells |
| --- | --- | ---: | ---: |
| formal official test | A6、STANDARD_QUERY、STANDARD_DUAL、GENERIC_DUAL、FCMI、ORDER_SHUFFLED、DENSE_DUAL、TARGET_SHUFFLED | 40 | 160 |
| all-arm validation | 8 arms | 40 | 160 |

`TARGET_SHUFFLED_QUERY`最初拟作validation-only，但它参与方向级matched attribution；按项目治理规则，
validation不得pass/reject mechanism，因此在任何training/test access前将其纳入完整official-test control
matrix。完整manifest为`manifest.csv`。

## 4. Four-layer gates

### 4.1 Paper-facing effectiveness

`FCMI vs A6_MEASURE`必须同时满足：

- test MSE macro gain至少0.3%；
- 至少11/20 cells、3/5 datasets、3/4 horizons正向；
- test MAE macro gain非负。

### 4.2 Matched mechanism attribution

必须逐项通过：

1. FCMI vs `STANDARD_DUAL_MATCHED`：main–interaction decomposition；
2. FCMI vs `GENERIC_DUAL_MATCHED`：coordinate interaction；
3. FCMI vs `FCMI_ORDER_SHUFFLED`：ordered binding；
4. FCMI vs `DENSE_DUAL_MATCHED`：capacity explanation；
5. test FCMI vs `TARGET_SHUFFLED_QUERY`：coordinate semantics。

前3项与target shuffle使用0.3%/11 cells/3 datasets/3 horizons/MAE非负gate。capacity control使用MSE
macro非负、10 cells、3 datasets、2 horizons以及MAE不低于-0.1%的noninferiority-attribution gate。
`STANDARD_QUERY`只作source-family context，不单独决定promotion。

### 4.3 Internal mechanism health

evaluator保存context coordinate std、main/interaction RMS、attention target dispersion、within-model
interaction prediction contribution与dense residual。所有量必须finite/nonzero；paired encoder/common/dual
initialization hashes必须逐dataset一致；dense与A6 parameter gap必须不超过0.2%。

### 4.4 Failure attribution

- non-finite、hash、checkpoint mutation或matrix缺失：
  `numeric_or_protocol_invalid`，回Step7修复，不方向拒绝；
- FCMI未超过A6且protocol valid：关闭FCMI-v1 paper-core，回Step4；
- 超过A6但任一matched control失败：`performance_partial_pass`，按control映射
  `capacity_control_explains`或mechanism-specific rollback；
- attribution通过但internal path inactive：`readout_or_head_design_wrong`，回Step5/7；
- seed2021全部通过：仅`performance_partial_pass_confirmation_required`，冻结candidate后再申请confirmation。

## 5. Local prelaunch evidence

- Step7A current-code regression：11/11；
- 40个production CLI cases；
- dense five-profile capacity/morphism/two-step gradient gate：5/5；
- runner syntax与40-job dry-run：pass；
- unauthorized normal launch：exit 3；
- checkpoint evaluator synthetic smoke：pass；
- four-layer analyzer synthetic decision smoke：pass；
- frozen hashes、matrix、gates、rollback与promotion boundary：pass。

## 6. Current authorization

```text
remote_training_authorized=false
formal_test_access_authorized=false
confirmation_seeds_authorized=false
paper_method=false
```

下一步若用户单独授权，必须先更新authorization并commit/push，然后remote pull、`nvidia-smi`、
Weather-FCMI与ETTm2-DENSE两项resource smoke；resource smoke通过后才允许启动40-run matrix。
