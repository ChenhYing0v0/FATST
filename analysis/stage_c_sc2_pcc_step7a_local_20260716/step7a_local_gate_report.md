# SC2-PCC-v1-TI Step 7A Local Implementation Gate

## Decision Summary

| Field | Value |
| --- | --- |
| `candidate` | `SC2-PCC-v1-TI` |
| `current_step` | Step 7A complete；Step 7B prelaunch audit next |
| `problem` | Step6公式能否在真实PCSD training path中以one-forward、one-stage方式无歧义实现 |
| `existence_evidence` | frozen PCSD-CF-v1 official test中25/25 joint arms under-trained，same-run oracle仍有+2.0197% macro headroom |
| `idea` | dense nested-prefix risk经harmonic incidence输运为natural target-coordinate arm/router credit |
| `theory_check` | vectorized与direct loop最大差`2.22e-16`；transport identity gap `0` |
| `design` | raw-scale L1、九个frozen modes、continuous schedule、stop-gradient capability、inference unchanged |
| `narrative_gate` | 保持Step6 conditional pass；实现未扩大claim |
| `effectiveness_gate` | 未执行；无validation/test performance evidence |
| `artifacts` | `step7a_cases.csv`、`local_gate.json`、`adapter_smoke/initialization_contract.json` |
| `decision` | `step7a_pass_prelaunch_audit_next`；remote/test仍false |

## What We Tested And Why

Step6只证明数学设计可行，尚不能证明训练代码没有把tensor layout、RevIN scale、control term或gradient path接错。
Step7A因此只回答以下implementation questions：

1. $R_s(H)$与$c_s(t)$的$O(BCTS)$ vectorized实现是否等于direct nested loops；
2. 九个arms是否只激活各自预注册的skill/router terms；
3. 同一次PCSD forward取得的raw-scale arms能否重构默认fused forecast；
4. capability是否严格stop-gradient，而conditional no-stopgrad仅改变gradient path；
5. continuous schedule是否从equal skill连续到final coefficients；
6. 一个真实PCSD model batch能否产生finite parameter gradients，且五个scope均获得非零auxiliary output gradient；
7. 默认forward signature、parameter count、full-prefix crop与test/remote boundary是否保持不变；
8. training adapter是否实际完成一次optimizer update并登记diagnostics，而不打开test loader。

本次不评估MSE/MAE，不读取validation/test dataset，不训练checkpoint，不启动remote runner。

## Implemented Tensor Path

PCSD在同一次forward中产生：

```text
history x [B,720,C]
  -> normalized memory [B,C,P,D]
  -> hidden [B,C,R=P*D]
  -> normalized arms [B,C,S=5,T=720]
  -> policy [B,C,T=720,S=5]
  -> normalized fused [B,T,C]
```

训练details path用history RevIN保存的`mean/stdev [B,1,C]`把每个scope反归一化：

```text
raw arms [B,C,S,T]
  -> permute [B,C,T,S]
target [B,T,C] -> [B,C,T]
  -> point L1 errors e [B,C,T,S]
  -> prefix risk R [B,C,T,S]
  -> prefix capability q [B,C,T,S], stop-gradient
  -> transported credit c [B,C,T,S]
  -> fused + skill + route objective
```

默认`Model.forward(...)`仍返回三元组；只有显式设置
`return_pcsd_training_details=True`时才附加raw-scale arms与policy。无新增trainable parameter。

## Frozen Objective Modes

所有modes共享dense-prefix measure fused L1。差异仅在auxiliary terms：

| Mode | Skill credit | Router credit |
| --- | --- | --- |
| `measure_only` | none | none |
| `equal_skill` | uniform | none |
| `pointwise_route_only` | none | pointwise capability |
| `pointwise_capability_skill_only` | floored pointwise | none |
| `pointwise_prior_composed` | uniform | pointwise capability |
| `pointwise_pcc_v0` | floored pointwise | pointwise capability |
| `transport_skill_only` | floored transported | none |
| `transport_route_only` | none | transported capability |
| `pcc_transport_full` | floored transported | transported capability |

固定参数为$	au=1$、$delta=10^{-6}$、$lambda_{skill}=1$、
$lambda_{route}^{final}=0.1$、$epsilon_{final}=0.2$、ramp fraction $0.25$。
training progress按optimizer-update group计算，不按dataset、horizon或test result调节。

## Metric And Column Definitions

`step7a_cases.csv`：

- `case`：local invariant唯一名称；
- `value`：对应直接测量值、bool或shape/count；
- `threshold`：Step6冻结的通过条件；
- `pass`：`value`是否满足条件。

新增`training_log.csv` fields均由当前epoch各train batch均值得到：

- `train_pcc_total_loss`：fused、weighted skill与weighted route之和；
- `train_pcc_fused_measure_l1`：$omega_t$加权的raw-scale fused L1；
- `train_pcc_skill_loss`：active arm credit对raw-scale arm L1的$omega_t$加权值，inactive为0；
- `train_pcc_route_kl`：active credit到policy的$mathrm{KL}/\log S$，inactive为0；
- `train_pcc_weighted_skill_loss`、`train_pcc_weighted_route_loss`：乘以当前coefficient后的两项；
- `train_pcc_skill_floor`：active skill credit的实际floor（equal=1、inactive=0、capability mode按schedule）；
- `train_pcc_route_weight`：active router term的当前schedule coefficient，inactive为0；
- `train_pcc_credit_normalized_entropy`：active router credit的$omega_t$加权entropy除以$\log S$；
- `train_pcc_policy_normalized_entropy`：policy的对应normalized entropy；
- `train_pcc_policy_usage_max`：先按$omega_t$聚合target，再对batch/channel平均后的最大scope usage；
- `train_pcc_credit_policy_kl`：active credit与policy的measure-weighted normalized KL；
- `train_pcc_credit_argmax_accuracy`：policy与active credit argmax的一致率，按$omega_t$加权；
- `train_pcc_credit_min/max`：active router credit的batch最小/最大值；
- `train_pcc_arm_s{k}_measure_l1`：第$k$个scope raw-scale error的dense-prefix measure L1。

这些训练统计用于optimization与failure attribution，不替代validation/test MSE、MAE或arm degradation audit。

## Results

[Fact] `35/35` local cases通过：

- vectorized prefix risk与direct loops最大差`2.220446e-16`；
- vectorized harmonic transport与direct loops最大差`2.220446e-16`；
- nested-risk transport identity gap=`0`；
- pointwise与transported credit crossed-case最大差=`0.618931`，不是相同control；
- identical arms capability与uniform的最大差=`0`；
- final transported skill minimum=`0.059821`，高于理论下界`0.04`；
- default/details output gap=`0`，raw-scale arm fusion identity gap=`8.88e-16`；
- arbitrary $H=337$ prefix crop gap=`0`；parameter count前后均为`1,775,109`；
- 五个scope的最小auxiliary output-gradient L1 sum=`0.078423`；
- real PCSD one-batch parameter gradients全部finite；
- adapter one-update loss=`2.341146`，所有20个PCC diagnostics fields已写入；
- loader访问序列严格为`train,val`，未访问test。
- production CLI接受PCSD+PCC contract，并拒绝PCC与非PCSD readout组合。

## Code-Theory Consistency

[Strong Evidence] 代码准确实现了Step6所声明的
`nested prefix risk -> harmonic target transport -> arm/router credit`，且未改变inference function class。
九个controls共享同一model forward与measure，loss decomposition可精确归因。

[Uncertainty] local synthetic/model batch不证明真实capability可由history-target policy预测，也不证明shared field训练后
五个arms会保持差异，更不证明candidate超过A6、plain PCSD或pointwise priors。

[Self-critique] 所有scope获得非零output-level auxiliary gradient，只排除了“loss没有传到某个arm”的实现错误；它不排除
shared parameters上的gradient cancellation、arm homogenization或validation/test reversal。

## Decision And Rollback

[Decision] `step7a_pass_prelaunch_audit_next`。Step7A不需要回Step5/6修复。下一步只能进入Step7B prelaunch audit：
生成45-run manifest/runner/analyzer，检查资源、hash、resume与validation-only contract。该prelaunch gate通过前，remote仍未
授权；本次也不授权test、confirmation seeds或conditional no-floor/no-stopgrad Phase B。
