# SC2-PCC Step 6 Source-Informed Redesign And Experiment Contract

## Status

| Field | Value |
| --- | --- |
| `current_step` | Step 6 complete；Step 7A local implementation next |
| `candidate` | `SC2-PCC-v1-TI` |
| `parent` | `SC2-PCC-v0` pointwise prototype；降为mandatory control |
| `test_informed` | true；来源为`SC-D15-T1`完整test audit |
| `problem` | plain fused training使structured coupling arms与router缺少可恢复的multi-horizon credit |
| `narrative_gate` | conditional pass；必须超过pointwise与prior-composed controls |
| `theory/design_gate` | 19/19 pass；transport identity gap `0` |
| `implementation/remote/test` | Step7A local true / false / false |

## Why PCC-v0 Was Not Advanced Directly

2026-07-16以external primary-source为主重新检索。结果表明PCC-v0的两个primitive不能继续作为novelty：

1. [Expert Loss Integration](https://arxiv.org/html/2605.10330)已在time-series forecasting中把global forecast
   loss与direct expert-specific losses联合训练，明确针对gate-filtered weak expert gradients；
2. [Diverse and Sparse MoE for OOD Graph Learning](https://iclr.cc/virtual/2026/poster/10011548)已用negative
   per-expert losses构造teacher distribution，以KL训练gate，并采用uniform-routing warm-up；
3. [Expert-Router Coupling](https://openreview.net/pdf?id=MpeyjgWbKt)已把router–expert capability alignment作为
   ICLR 2026核心问题；
4. [AME-TS](https://arxiv.org/abs/2605.25166)已在forecasting中用training-only structural prior稳定expert routing；
5. [GradNorm](https://proceedings.mlr.press/v80/chen18a.html)、
   [PCGrad](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)与
   [CAGrad](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html)
   已覆盖generic loss/gradient balancing与conflict manipulation。

[Decision] `expert loss + pointwise loss-teacher gate`只能作为closest-prior control。PCC的contribution boundary必须
来自unified multi-horizon problem本身，而不是给普通MoE换名。因此研究从Step6短暂回滚Step4/5，将v0收紧为
`nested prefix risk -> target-coordinate credit transport`，再完成本地代数检查后返回Step6。

## PCC-v1-TI: Projective Nested-Risk Credit Transport

同一次full-domain forward产生$S=5$个scope arms $F_s(t)$、policy $\pi_s(t)$与fused forecast。以训练criterion
一致的pointwise L1定义

$$
e_s(t)=|F_s(t)-Y_t|,
\qquad
R_s(H)=\frac1H\sum_{t=1}^{H}e_s(t).
$$

$R_s(H)$不是benchmark horizon子集，而是全部$H=1,\ldots,T$的dense nested-prefix risk。对scope维做
stop-gradient standardization后形成prefix capability：

$$
q_s(H)=\operatorname{softmax}_s\left(
-\frac{\operatorname{sg}[R_s(H)-\bar R(H)]}
{\max(\operatorname{std}_s R(H),\delta)\tau}
\right).
$$

router不能接收requested $H$。因此不能直接学习$q(H)$，而把所有包含target $t$的prefix credit输运回natural
target coordinate：

$$
c_s(t)=
\frac{\sum_{H=t}^{T}q_s(H)/H}
{\sum_{H=t}^{T}1/H}.
$$

对arm skill使用带floor的$q_s^\epsilon(H)=(1-\epsilon)q_s(H)+\epsilon/S$，并同样输运为
$c_s^\epsilon(t)$。由

$$
\omega_t=\frac1T\sum_{H=t}^{T}\frac1H
$$

可得exact identity：

$$
\frac1T\sum_{H=1}^{T}\sum_s q_s^\epsilon(H)R_s(H)
=\sum_{t=1}^{T}\omega_t\sum_s c_s^\epsilon(t)e_s(t).
$$

这不是简单地用$\omega_t$重加权pointwise q：当best arm随prefix accumulation改变时，transported credit与
pointwise credit不同。本地crossed case最大差为`0.616407`，而两种nested-risk计算形式误差为`0`。

## Frozen Objective And Continuous Schedule

$$
\mathcal L_{\mathrm{PCC}}
=\underbrace{\sum_t\omega_t e_{\mathrm{fuse}}(t)}_{\mathcal L_{\mathrm{fuse}}^\mu}
+\lambda_{\mathrm{skill}}
\underbrace{\sum_t\omega_t\sum_s c_s^\epsilon(t)e_s(t)}_{\mathcal L_{\mathrm{skill}}^P}
+\lambda_{\mathrm{route}}
\underbrace{\sum_t\omega_t
\frac{\mathrm{KL}(\operatorname{sg}[c(t)]\|\pi(t))}{\log S}}_{\mathcal L_{\mathrm{route}}^P}.
$$

全数据集共享：$\tau=1$、$\lambda_{skill}=1$、$\lambda_{route}^{final}=0.1$、
$\epsilon_{final}=0.2$。不做dataset-specific coefficient tuning。

以optimizer progress $u\in[0,1]$定义$a(u)=\min(1,u/0.25)$：

$$
\epsilon(u)=1-(1-0.2)a(u),
\qquad
\lambda_{route}(u)=0.1a(u).
$$

起点是equal skill、route auxiliary为零；随后连续过渡到capability-specific credit。所有model parameters始终通过
fused loss active，不冻结router、不训练teacher、不切换dataset、不进行第二阶段fine-tuning。因此它是one-run、
one-forward、one-stage E2E schedule，而不是CCRL式teacher/student pipeline。

## Phase A Frozen Controls

既有A6、plain DIRECT与DENSE matched仅作锁定reference；新训练9 arms × 5 datasets = 45 runs：

| Arm | What it isolates |
| --- | --- |
| `MEASURE_ONLY` | dense-prefix measure本身 |
| `EQUAL_SKILL` | forecasting Expert Loss Integration/generic deep supervision |
| `POINTWISE_ROUTE_ONLY` | pointwise loss-teacher router |
| `POINTWISE_CAPABILITY_SKILL_ONLY` | pointwise capability-weighted arms |
| `POINTWISE_PRIOR_COMPOSED` | equal expert loss + pointwise gate teacher closest-prior composition |
| `POINTWISE_PCC_V0` | 原PCC-v0完整pointwise formulation |
| `TRANSPORT_SKILL_ONLY` | nested-risk transport只训练arms |
| `TRANSPORT_ROUTE_ONLY` | nested-risk transport只训练router |
| `PCC_TRANSPORT_FULL` | candidate v1；transported arm+router credit |

所有arms使用five natural profiles、seed2021、full-T forward、best-val-H720 checkpoint、validation-only。test=false。
只有Phase A同时通过method与specificity gates，才允许conditional `NO_FLOOR/NO_STOPGRAD` Phase B。

## Diagnostics And Metric Definitions

除dense-H1..720 MSE AUC外，Step7实现必须记录：

1. `arm_degradation_percent`：joint arm dense MSE AUC相对对应independent fixed-scope run的退化；
2. `arm_pair_improved`：PCC arm退化是否低于plain DIRECT同dataset/scope；
3. `credit_policy_kl`：以$\omega_t$加权的$\mathrm{KL}(c(t)\|\pi(t))/\log S$；
4. `credit_argmax_accuracy`：policy与transported capability的best-scope一致率，仅作predictability diagnostic；
5. `arm_pairwise_nrmse`：scope forecasts两两normalized RMSE，防止equal-skill把arms同质化；
6. `policy_normalized_entropy/usage_max`：router collapse guard；
7. `shared_gradient_cosine`：固定train batch上五个scope losses对shared field parameters的pairwise cosine，只作
   failure attribution，不进入optimizer、不用作method gain。

gradient diagnostic只在预注册epoch snapshots执行，不做PCGrad/CAGrad式gradient surgery。

## Hard Gates And Rollback

Phase A要求full PCC同时满足：

- vs A6：至少3/5 wins且macro gain不低于0.3%；
- vs plain DIRECT：至少3/5 wins且macro gain不低于0.5%；
- vs `POINTWISE_PCC_V0`与`POINTWISE_PRIOR_COMPOSED`：各至少3/5 wins、macro gain不低于0.2%；
- plain的arm degradation median至少相对降低30%，25 pairs中至少15个改善；
- pairwise NRMSE至少保留plain的50%，normalized entropy不低于0.3、usage max不高于0.9。

决策边界：

- pointwise/prior composition解释收益：PCC降为training control，回Step4；
- arms恢复但仍不超过A6：training策略不是主瓶颈，回SC1 Step4改shared-field/readout；
- arms不恢复：回Step4审计shared-parameter gradient cancellation或intervention point；
- moving-target/numeric pathology：只否定当前objective，回Step5；
- 全部门槛通过：才授权Phase B，然后再决定confirmation；不会直接访问test。

## Step 6 Decision

[Fact] 19/19 local design cases通过；nested-prefix transport identity gap为`0`，simplex/floor/stop-gradient/schedule/
control-matrix/protocol contracts均通过。

[Strong Evidence] v1相对v0形成了multi-horizon-specific数学对象，而且closest prior controls已进入同一matrix。

[Uncertainty] transport target是否可由history预测、shared field是否能保留scope差异、以及45-run validation上是否超过
A6仍完全未知。

[Decision] `step6_pass_step7a_local_authorized`。只允许实现objective、diagnostics与local invariants；remote与test继续false。
