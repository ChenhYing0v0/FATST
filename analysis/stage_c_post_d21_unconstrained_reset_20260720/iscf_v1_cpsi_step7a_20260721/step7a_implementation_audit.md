# ISCF-v1-CPSI Step7A Production-Local Implementation Audit

## 1. Decision

```text
decision = step7a_local_pass_step7b_prelaunch_next
cases = 81/81
remote_training = false
formal_test = false
```

Step7A只验证production code与Step6 theory/design contract一致。没有dataset training、validation comparison或official
test access，不能据此判断CPSI performance。

## 2. Implemented path

新增`CPSIReadout`继承independent `SIFFCouplingFieldReadout`。parent constructor先完成encoder-facing independent
mode maps、scope synthesis与direct policy初始化，随后才创建interaction matrices，因此同一seed下五个arms的base ISCF
parameters保持exact paired。

五个production readout modes：

| CLI mode | Interaction | Placement |
| --- | --- | --- |
| `iscf-v1-cpsi` | common × private product | modes `[B,C,S,D,K]` |
| `iscf-v1-cpsi-self` | self × self product | modes `[B,C,S,D,K]` |
| `iscf-v1-cpsi-linear` | common + private linear | modes `[B,C,S,D,K]` |
| `iscf-v1-cpsi-common` | common × common product | modes `[B,C,S,D,K]` |
| `iscf-v1-cpsi-post` | forecast-common × forecast-private | arms `[B,C,S,T]` |

pre-synthesis modes先flatten为`[B,C,S,L]`；interaction后unflatten回`[B,C,S,D,K]`并进入原有
`_scope_forecast`。POST在五个`[B,C,T]` scope forecasts stack后、direct policy fusion前执行。

## 3. Gate coverage

| Category | Passed / Cases | Meaning |
| --- | ---: | --- |
| readout | 50/50 | shapes、parent pairing、zero morph、parameter count、finite diagnostics |
| equivariance | 5/5 | joint scope permutation使output按同一permutation变化 |
| semantics | 1/1 | candidate在private deviation为零时interaction为零 |
| gradient | 10/10 | 五arms均通过first/second backward opening |
| model | 5/5 | five CLI modes完成真实`TimeAlign.Model` full/prefix forward |
| CLI | 5/5 | rank、mode rank、direct policy、equal-skill与validation split可解析 |
| profile parameters | 5/5 | five natural profiles的`3DKr`与Step6 freeze一致 |

完整machine artifacts：`checks.csv`与`summary.json`。

## 4. Exact containment and pairing

small production witness中，五个arms相对independent parent均得到：

- `parent_gap=0.0`；
- `morph_gap=0.0`；
- `arm_gap=0.0`；
- `policy_gap=0.0`；
- full/prefix crop gap=`0.0`。

五个`TimeAlign.Model`的`cpsi_parent_initialization_hash`均为
`e9f3c1cb2c7442ff290ff403c96fb42fc2f3df3132a288f4c630bc8801529167`。这证明interaction module创建没有改变
parent RNG path；formal run仍需在真实profile上把hash与historical ISCF checkpoint initialization contract对齐。

## 5. Two-stage gradient opening

$W_o=0$时，五个arms的first backward均满足：

- output projection gradient finite且`>0`；
- common/private input projection gradients恰为0。

对$W_o$执行一次deterministic synthetic update后，second backward中五个arms的两组input gradients与message RMS全部
finite且`>0`。所以zero-init是one-step delayed opening，不是permanent credit starvation。

该检查只证明synthetic differentiability。真实训练中仍需按epoch记录projection norms、gradient与message/base ratio；若
长期不打开，应归因`optimization_or_numeric_pathology`，不能用本地witness掩盖。

## 6. Parameter and placement checks

five natural profiles的candidate added parameters保持：ETTh1 `41,856`，ETTh2/ETTm1/Weather `44,544`，
ETTm2 `40,704`。POST production code按`round(D*K*32/720)`自动派生rank；small witness因$L=16$得到rank1，真实
profiles将在Step7B construction manifest中再次核验为`19/21/21/19/21`。

## 7. Failure attribution

- `optimization_or_numeric_pathology`：未触发；81 cases finite；
- `intervention_point_wrong`：未判断；需test CPSI vs POST；
- `readout_or_head_design_wrong`：未触发local hard invalidity；effectiveness未知；
- `capacity_control_explains`：未判断；需test CPSI vs SELF；
- `hypothesis_false`：未判断；无MSE/MAE。

默认Python缺少PyTorch的首次调用报`ModuleNotFoundError: torch`；按仓库规定切换到conda `r2026-fsa`后同一checker
81/81通过。这是environment selection，不是model failure。

## 8. Authorization

```text
active_method = ISCF-v1-CPSI implementation_ready_effectiveness_pending
next_step = Step7B prelaunch
remote_training = false
formal_test = false
confirmation = false
router_or_second_loss = false
```
