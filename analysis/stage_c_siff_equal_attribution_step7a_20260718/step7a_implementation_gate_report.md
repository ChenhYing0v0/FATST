# SC1-SIFF-v2-EQ-ATTR Step 7A Implementation Gate

## 1. 当前节点

| Field | Content |
| --- | --- |
| `current_step` | Step 7A local implementation complete |
| `candidate_version` | `SC1-SIFF-v2-EQ-ATTR-v1` |
| `problem` | 将10-arm归因协议转成不会丢失control、不会越权读取test、并能导出scale-component证据的production path |
| `existence_evidence` | Step6 16/16；现有TimeAlign/PCSD/SIFF已有全部readout与objective primitives |
| `idea` | 不修改SIFF forecast公式；增加leave-one-component-out diagnostic、10-arm manifest、四层analyzer与authorization guard |
| `theory_check` | component intervention固定policy，只移除一个scale-field component；它定位该component是否实际改变输出，但不假设nonlinear synthesis下可加 |
| `design` | 50 CLI jobs、35 construction cases、10 gradient cases、5 component cases、2 matched-control families |
| `narrative_gate` | unchanged conditional pass；implementation不能替代effectiveness/attribution |
| `effectiveness_gate` | pending；本阶段未训练、未读取validation/test结果 |
| `artifacts` | `local_gate.json`、`jobs_seed2021.csv`、`model_construction.csv`、`gradient_paths.csv`、`parameter_matching.csv`、`scale_component_cases.csv` |
| `decision` | 13/13 categories pass；进入Step7B prelaunch audit；remote/test仍为false |

## 2. 实现内容

### 2.1 10-arm production wiring

`check_stage_c_siff_equal_attribution_step7a.py`从冻结config与five-dataset profile contract生成：

$$
5\ \text{datasets}\times10\ \text{arms}\times1\ \text{seed}=50\ \text{jobs}.
$$

每个job显式携带dataset profile、readout、objective、matched rank、four-horizon validation selector与
`full-crop` evaluation contract。50/50 CLI均可由`train_repo.py`解析，未出现PCC/MCCA objective混入。

### 2.2 Scale-component intervention artifact

对SIFF readout：

1. `hidden [B,C,R]`生成`components [B,C,Q,D,K]`；
2. `scale_basis [S,Q]`组合为`history_modes [B,C,S,D,K]`；
3. scope synthesis得到`arms [B,C,S,T]`；
4. 固定原policy `weights [B,C,T,S]`；
5. 逐个令component $q$ 为零，重新生成`ablated [B,C,Q,T]`；
6. 保存raw-scale
   `scale_component_contribution = full - ablated`，artifact shape为
   `[row_channel,Q,T]`。

该量是intervention effect，不是additive attribution。由于SIFF synthesis含GELU，各component effects之和不要求
等于full forecast。

### 2.3 四层analyzer

新增analyzer会在完整Phase A返回后同时输出：

1. standard-horizon MSE/MAE effectiveness；
2. 七项hard matched comparisons；
3. oracle、arm NRMSE、policy entropy、component RMS、ordered-vs-constant contrast与projectivity；
4. failure class、rollback与confirmation authorization。

synthetic pass matrix已验证四层decision逻辑；它不包含真实结果。

### 2.4 Remote authorization guard

remote runner已经具备50-job manifest、dataset-major scheduling、checkpoint hash guard、test evaluator与final
analyzer接口。但当前config中`remote_training_authorized=false`且`formal_test_access_authorized=false`，非
dry-run调用固定以exit code 3停止。Step7A pass不会自动解锁远程或test。

## 3. Local gate结果

| Category | Result |
| --- | --- |
| profile hash / matrix / arm count | pass |
| 50 CLI contracts | pass |
| 35 unique dataset-readout-rank constructors | pass |
| paired Encoder initialization | pass |
| 10 objective forward/backward paths | pass |
| Q1/independent parameter matching | pass |
| 5 scale-component cases | pass |
| checkpoint evaluator smoke | pass |
| four-layer analyzer smoke | pass |
| remote runner authorization guard | pass |
| authorization boundary | pass |

matched parameter relative gap最大为`0.0038386`，低于`0.005`。所有10个arms的Encoder与readout gradient均
non-zero且finite。component intervention满足：

- full-path consistency gap：`0.0`；
- nonconstant component RMS：`0.48267`；
- ordered-vs-constant RMS contrast：`1.48785`；
- nonconstant component gradient norm：`0.12519`。

这些component数值来自random-initialized local witness，只证明接口非退化、可观测、可反传；不能当作训练后
internal mechanism health evidence。

## 4. Code-theory consistency

### Intended theory

SIFF的paper claim依赖ordered scale coordinate实际进入forecast，而不只是额外参数或多个arms。

### Code realization

原production forward保持不变。新增diagnostic用相同components、scope synthesis与policy做leave-one-component-out
counterfactual，并在denormalization后计算prediction delta。

### Remaining proxy

Step7A没有证明trained nonconstant component足够大、policy使用合理或ordered mapping优于controls。这些问题只能由
Phase A的official test effectiveness、matched controls和trained internal artifacts回答。

### Falsification evidence

- Step7B发现任一arm不能按冻结contract复现；
- resource smoke无法生成完整component artifact；
- trained component RMS低于阈值或arms collapse；
- 七项hard comparison任一失败。

## 5. Decision

Step7A local gate=`13/13 categories pass`。下一步只进入Step7B prelaunch audit：检查remote environment、
50-job commands、resource smoke计划、config authorization与完整test audit metadata。

当前不得启动remote training、official test evaluation或confirmation seeds。
