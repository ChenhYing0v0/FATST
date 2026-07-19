# SC-D18-SPC Step 9 四层诊断与 Step 10 决策

## 1. 当前节点

| Field | Content |
| --- | --- |
| `current_step` | D18 Step 9 complete；Step 10 close；Contribution 1 rollback Step 2 |
| `candidate_version` | `SC-D18-SPC-v1` |
| `problem` | strong `A6_MEASURE` exact-projective unified model是否相对own-H specialists付出稳定accuracy cost |
| `existence_evidence` | 15个specialists、10个reused controls、15个own-H official-test cells |
| `idea` | 保持full-domain A6 architecture不变，仅把training/selection改为H96、H192或H336 own-H |
| `theory_check` | 只有specialists稳定超过`A6_MEASURE`，才存在研究controlled soft projectivity的必要性 |
| `design` | 5 datasets × 3 specialists；seed2021；same profile、rank、parameter count与initialization class |
| `narrative_gate` | `diagnostic_only`；不是method gate |
| `effectiveness_gate` | `2/7` categories；fail |
| `artifacts` | 本目录CSV/JSON、remote 25-run raw root |
| `decision` | `measure_training_explains_close_soft_architecture_route_return_step2` |

## 2. Test audit

| Audit | Result |
| --- | ---: |
| `test_access_date` | 2026-07-19 |
| `user_authorization` | 用户明确通知远程实验完成并要求继续推进 |
| expected / valid artifact units | 25 / 25 |
| new / reused checkpoints | 15 / 10 |
| unique checkpoint hashes | 25 |
| finite / protocol / readout pass | 25 / 25 |
| maximum prefix crop gap | $3.58\times10^{-7}$ |
| matched initialization and parameter count | pass |
| `matrix_complete` | true |
| `test_role` | test-informed problem-existence diagnostic |

[Fact] 15个specialists均实际训练7–20 epochs，`0/15`的best checkpoint落在允许预算的最后边界。
因此不存在整批未收敛、checkpoint缺失、cross-arm checkpoint复用或明显训练预算截断。

## 3. Layer 1：paper-facing problem effectiveness

specialist相对`A6_MEASURE`的own-H MSE gain如下：

| Dataset | H96 | H192 | H336 | Dataset macro |
| --- | ---: | ---: | ---: | ---: |
| Weather | +0.378% | +0.090% | -0.946% | -0.159% |
| ETTm1 | +1.806% | -0.349% | -1.185% | +0.091% |
| ETTh1 | +0.795% | -0.229% | -0.913% | -0.116% |
| ETTh2 | +3.160% | -0.193% | +0.115% | +1.027% |
| ETTm2 | +0.235% | -0.012% | -0.264% | -0.014% |
| Horizon macro | +1.275% | -0.139% | -0.638% | **+0.166%** |

MAE macro为`+0.252%`，`8/15` cells为正。MSE只得到`7/15` positive cells、`2/5`
positive datasets、`1/3` positive horizons；H336 macro为`-0.638%`，也违反“不显著损害任一horizon”
的预注册约束。

七项gates中仅以下两项通过：

1. specialists确实改变了forecast，minimum prediction NRMSE为`0.0174`；
2. protocol、numeric、initialization与parameter contract通过。

macro、dataset、horizon、cell support及no-regression五项均失败。因此Layer 1=`fail`。

## 4. Layer 2：matched attribution

| Comparison | MSE macro | MSE cells | MAE macro | MAE cells |
| --- | ---: | ---: | ---: | ---: |
| `A6_MEASURE` vs `A6_FULL` | **+1.798%** | **15/15** | **+1.266%** | **15/15** |
| specialists vs `A6_FULL` | +1.957% | 15/15 | +1.512% | 15/15 |
| specialists vs `A6_MEASURE` | **+0.166%** | **7/15** | **+0.252%** | **8/15** |

[Strong Evidence] 相对弱`A6_FULL`时，specialists看起来全部成功；但同architecture的
`A6_MEASURE`已在15/15 cells稳定赢过`A6_FULL`，并解释了大部分增益。specialists在这个强control之上只剩
`+0.166%`且分布不稳定。

这正是预注册的`measure_training_explains`情形。不能把“只为own-H训练比flat H720 loss更好”解释成
exact projectivity的accuracy cost。

## 5. Layer 3：internal mechanism health

[Fact] intervention没有退化为同一个模型：

- 所有specialist与`A6_MEASURE`的prediction NRMSE均大于零；
- local gradient precheck中prefix gradient非零、tail gradient为零；
- 25个checkpoint hash全部不同；
- 无non-finite、>100% degradation或projectivity failure。

但是，H96正信号并不构成稳定的局部机制：

- 正式full-test H96 macro为`+1.275%`；
- 固定256-row prediction probe进一步切为`1–48`和`49–96`后，仅`4/10`区间为正；
- probe interval macro为`-1.628%`。

[Boundary] probe只是内部定位样本，不能覆盖正式full-test metric；它的作用是说明H96收益并非在五dataset的
两个半区间上普遍存在，不能据此启动H96-specific architecture。

## 6. Validation、test与checkpoint解释

specialist相对`A6_MEASURE`：

| Split | macro gain | positive cells | datasets | horizons | minimum horizon |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | +0.420% | 7/15 | 4/5 | 2/3 | -0.730% |
| official test | +0.166% | 7/15 | 2/5 | 1/3 | -0.638% |

validation/test符号一致为`9/15`。存在split差异，但validation本身也未通过macro、cell和no-regression gates。
因此结果不能归因为“validation成功、test分布偏移导致假失败”。checkpoint audit也没有显示系统性训练不足。

## 7. Layer 4：failure attribution

### `hypothesis_false`

[Strong Evidence] primary。被否定的精确命题是：

> 在当前A6 natural carrier与strong measure-trained control上，exact projectivity会造成稳定、跨dataset且跨
> horizon的own-H accuracy cost。

该命题没有获得支持。

### `intervention_point_wrong`

[Fact] 不支持为主因。training target、gradient support与prediction均实际改变；本轮不是一个没有生效的
intervention。

### `readout_or_head_design_wrong`

[Fact] 不支持为本轮主因。specialists与control共享同一A6 readout，变化只在loss domain与checkpoint selection，
所以不存在新head过弱这一额外混杂。

### `optimization_or_numeric_pathology`

[Fact] 不支持。25/25协议完整，0个best epoch卡预算边界，numeric与prefix invariants全部通过。

### `capacity_control_explains`

[Strong Evidence] `A6_MEASURE`是关键解释性control。相对`A6_FULL`的表面specialization增益主要来自统一
prefix measure training，而不是放弃projectivity。

## 8. Step 10 决策

1. `SC-D18-SPC-v1`关闭，不补seeds2022/2023；
2. 不实现soft-projective decoder、不做consistency-$\lambda$ sweep、不引入requested-horizon feature；
3. H96 specialist signal仅保留为局部training线索，不升级为paper problem；
4. `A6_MEASURE`升级为所有后续decoder research的mandatory effectiveness control；
5. raw measure weighting仍是prior-covered protocol/control，不自动升级为Contribution 2；
6. Contribution 1回滚Step 2，重新审查fixed-past trajectory generation还剩什么可证伪的decoder问题；
7. Contribution 2保持Step 2，不允许用新的loss/router挽救一个尚未成立的architecture problem。

最终decision：
`measure_training_explains_close_soft_architecture_route_return_step2`。

## 9. Artifact定义

- `validation_test_cells.csv`：每个dataset-own-H cell的validation/test MSE与相对gain；
- `checkpoint_summary.csv`：specialist训练epoch、best epoch与early stopping；
- `protocol_invariants.csv`：checkpoint hash、retrained flag、finite/protocol/prefix invariants；
- `effect_decomposition.csv`：measure、specialist相对两个controls的macro分解；
- `probe_interval_gains.csv`：256-row probe中，specialist own-H内部固定区间相对`A6_MEASURE`的MSE gain与
  prediction NRMSE；
- `deep_summary.json`：本报告machine-readable摘要。

Remote canonical raw root：
`/home/yingch/exp_outputs/r-2026-fatst/stage_c_d18_soft_projectivity_cost_v1`。
