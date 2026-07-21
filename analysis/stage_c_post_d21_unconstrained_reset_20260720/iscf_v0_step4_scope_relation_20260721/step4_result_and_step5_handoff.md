# ISCF-v0 Step4 Result and Step5 Handoff

## 1. Long-stage record

| Field | Record |
| --- | --- |
| `current_step` | ISCF-v0 Step4 existence/narrative audit complete；advance Step5 theory/design only |
| `problem` | ISCF-v0的independent future-output coupling scopes是否学得超越shared target difficulty与architecture prior的stable local response relation？ |
| `existence_evidence` | D1.1在disjoint validation rows上15/15超过direction-null与random-init；common/private response median=`0.2803/0.7197`；topology 4/5 datasets跨seed稳定 |
| `idea` | 将问题收紧为“learned pre-synthesis response dependence与late-only forecast fusion之间的错位”；研究non-ordered scope-set interaction，不做router或第二loss |
| `theory_check` | label-free $J_s(h)u$ proxy不依赖target或requested H；response relation不等于method gain；output low-rank与ordered scale仍被既有证据否定 |
| `design` | D1 primary 16 directions；post-hoc D64 validity check；pre-registered D1.1使用offset64 disjoint rows、64 directions、128 null、8 random-init controls |
| `narrative_gate` | `conditional_pass_to_step5_as_single_pre_synthesis_architecture_problem` |
| `effectiveness_gate` | not applicable；no method exists |
| `artifacts` | D1/D64/D1.1 CSV+JSON、two frozen configs、diagnostic code与source audit |
| `decision` | `scope_response_relation_confirmed_for_step5_theory`；active_method=none；implementation/training/test false |

## 2. Evidence chain and split roles

| Phase | Role | Rows/directions | Result |
| --- | --- | --- | --- |
| D1 primary | pre-registered validation diagnostic | first 32 rows；16 directions；16 random controls | relation/null/noncollapse pass；topology 2/5 fail |
| D1-D64 | declared post-hoc estimator validity | same rows；64 directions；4 random controls | topology恢复5/5；primary gate不改写 |
| D1.1 | newly frozen confirmation | disjoint rows offset64；64 directions；8 random controls | all four gates pass；topology 4/5 |

全部计算只读取validation histories。loader返回的`batch_y`未被读取；没有MSE/MAE、test input/label、training、checkpoint
mutation或parameter update。D1.1 remote固定为commit `afb6a59`、environment `moe`、GPU0；preflight为18 MiB /
24576 MiB、0% utilization。remote output：
`/home/yingch/exp_outputs/r-2026-fatst/iscf_v0_scope_response_d1_1_20260721/full`。

## 3. Confirmatory result

### 3.1 Global gates

| Gate | D1.1 result | Decision |
| --- | ---: | --- |
| synchronized above direction null | `15/15` | pass |
| learned above matched random-init | `15/15` | pass |
| scope-specific noncollapse | private median `0.7197`；distance median `1.3440` | pass |
| cross-seed topology | `4/5` datasets | pass |

Machine decision：

```text
scope_response_relation_confirmed_for_step5_theory
```

对五个unit-RMS scope responses，independent baseline的expected common energy约为$1/5=0.2$。D1.1 observed median为
`0.2803`，且15/15均超过各run的direction-null和architecture-identical random-init p95。这说明同步relation不是共同
target term造成，也不能仅由固定scope pooling/synthesis结构解释。

### 3.2 Dataset heterogeneity

| Dataset | Common | Direction-null p95 | Random-init p95 | Private | Topology rho |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | `0.2405` | `0.2040` | `0.2032` | `0.7595` | `0.5879` |
| ETTh2 | `0.2611` | `0.2051` | `0.2018` | `0.7389` | `0.4182` |
| ETTm1 | `0.3791` | `0.2071` | `0.2008` | `0.6209` | `0.8545` |
| ETTm2 | `0.2755` | `0.2059` | `0.2019` | `0.7245` | `0.5515` |
| Weather | `0.4573` | `0.2142` | `0.2029` | `0.5427` | `0.7576` |

Common response强度具有明显dataset heterogeneity，但每个dataset均超过两类controls；private始终占多数。故证据支持
“shared response + substantial scope-specific response”，不支持固定common/private比例。ETTh2 topology未过0.5，禁止
claim universal fixed pairwise relation graph。

## 4. Diagnostic validity and failure attribution

D1 primary的16-direction topology只有2/5稳定；同一rows扩至64 directions后恢复5/5，随后D1.1在disjoint rows上确认4/5。
因此primary topology failure归因为`diagnostic_estimator_variance/design_fault_suspected`，不能作为方向级拒绝。其余三个
primary gates本来即通过，D1.1再次通过。

- `hypothesis_false`：未触发；label-free synchronized relation得到confirmation；
- `capacity_control_explains`：未触发；15/15超过matched random-init readout；
- `optimization_or_numeric_pathology`：未触发；全部finite，central-response RMS非退化；
- `intervention_point_wrong`：当前hidden-to-pre-synthesis intervention可测到relation，但尚未证明这里是最佳method注入点；
- `readout_or_head_design_wrong`：不适用，尚无新method。

保留的不确定性：frozen hidden/readout仍是co-adapted conditional evidence；D1.1不能证明新增interaction会改善end-to-end
forecast。ETTh2 fine topology不稳定也意味着不可把一个universal pairwise matrix写进claim。

## 5. What is and is not supported

### Supported

1. ISCF-v0不只是五个随机冗余heads：training使其对相同history perturbation形成超越结构先验的同步response；
2. private response median约72%，Q1/common-only不足以描述carrier；
3. relation是non-ordered且在4/5 datasets具有跨seed topology stability；
4. ISCF当前只在完整arm forecasts后做scalar fusion，因此存在一个可研究的“pre-synthesis dependence / late-only
   composition”错位。

### Not supported

1. output functions是two-factor/low-rank：previous audit为0/15；
2. canonical scale order或ordered SIFF恢复：order evidence仍弱；
3. universal fixed graph：ETTh2 confirmation不稳定；
4. relation-aware method一定提升MSE/MAE；
5. oracle headroom可由router实现；
6. 第二loss、requested-H conditioning或第二contribution的必要性。

## 6. Narrative gate

Source audit确认：[MoLE](https://proceedings.mlr.press/v238/ni24a.html)已覆盖forecast experts与learned output
mixture；[TimeMixer](https://openreview.net/pdf?id=7oLshfEIC2)和
[DMSC](https://arxiv.org/abs/2508.02753)覆盖multi-scale predictors/coordination；
[Cross-Stitch](https://openaccess.thecvf.com/content_cvpr_2016/html/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.html)
覆盖generic shared/private mixing；[Deep Sets](https://arxiv.org/abs/1703.06114)与
[Set Transformer](https://proceedings.mlr.press/v97/lee19d.html)覆盖permutation-equivariant/invariant set interaction。
因此“多个experts互相通信”或“set attention”不能成为component novelty。

仍有条件成立的完整贡献边界是：

```text
single-model varied-horizon forecasting
-> future-output coupling scopes as native internal operators
-> confirmed label-free pre-synthesis response dependence
-> non-ordered scope-set interaction before scope-specific synthesis
-> matched ISCF/ordered/Q1/set-mixing/capacity attribution
```

Narrative decision：

```text
conditional_pass_to_step5_as_single_pre_synthesis_architecture_problem
```

这是problem+narrative gate，不是method gate。论文当前最多预留一个architecture contribution；不预设第二loss/router。

## 7. Step5 theory/design constraints

下一步工作名仅为`scope-set response coupling operator`，不是最终method name。Step5必须先完成：

1. **Tensor contract**：作用于`component/history modes [B,C,S,D,K]`，在`_scope_forecast`前交互；说明每个tensor如何
   进入后续scope-specific synthesis；
2. **Non-ordered contract**：scope slots及其关联metadata同时置换时，relation operator必须equivariant；不得输入
   log-scale order或requested H；
3. **Containment**：zero interaction精确退化为ISCF-v0；common-only/Q1、independent ISCF、ordered SIFF-v2均为controls；
4. **Minimal primitive comparison**：mean-based DeepSets-style interaction先于attention；若使用Set Transformer/attention，
   必须证明pairwise expressivity的必要性，不能因D1.1通过就默认选择attention；
5. **Capacity/fairness**：增加参数必须有matched no-relation control；最终paper gate仍需same-init-class end-to-end joint training；
6. **Single objective**：保持现有forecast/equal-skill contract，不新增第二loss、router或oracle teacher；
7. **Falsification**：若理论只能退化为generic set mixing，或无法给出超越late fusion的任务特异必要性，Step5 narrative fail，
   rollback到ISCF-v0 carrier而不实现。

## 8. Current authorization

```text
active_method = none
current_step = ISCF Step5 theory/design
method_implementation = false
remote_training = false
formal_test = false
```

Step5可阅读Deep Sets、Set Transformer、Cross-Stitch及forecast-expert source implementations，并形成shape-level operator与
matched-control design；任何production model code必须等待Step5 theory+narrative/design gate明确通过。
