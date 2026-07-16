# SC2-PCC Step 5 Theory Feasibility

## Status

| Field | Value |
| --- | --- |
| `candidate` | `SC2-PCC-v0`（Projective Coupling Credit） |
| `current_step` | Step 5 theory feasibility complete |
| `problem` | plain fused loss使PCSD scope arms的训练credit与router权重绑定，25/25 arms出现skill starvation |
| `narrative_gate` | conditional pass；完整claim必须落在projective coupling field的same-forward forecast-risk credit |
| `effectiveness_gate` | not started |
| `artifacts` | `theory_and_synthetic_cases.csv`；`local_gate.json` |
| `decision` | `conditional_pass_step6_design_only` |
| `implementation/remote/test` | false / false / false |

## What We Tested And Why

本次不证明PCC能提高真实预测性能，只检查它是否具备进入Step 6的最低理论条件：

1. plain fused loss是否确实让arm gradient乘上当前policy，形成自我强化式credit starvation；
2. PCC是否在output level给每个arm一个严格正的skill credit，并给router一个可解释的capability gradient；
3. dense-prefix measure是否精确对应所有prefix risk的平均，而非人为挑选benchmark horizons；
4. full-domain generation后prefix crop是否保持projectivity；
5. 当best scope同时依赖history与target position时，history-target policy能否在一个简单合成问题中恢复capability target。

脚本使用`torch.float64`自动微分与解析式逐元素比较。所有检查均为local algebra/synthetic case，未读取真实
dataset、test split或远程结果。

## Gradient Accounting

令

$$
F(t)=\sum_s\pi_s(t)F_s(t),\qquad
\ell(t)=\frac12(F(t)-Y_t)^2.
$$

在包含batch与position measure系数$\omega_t$后，plain fused loss对arm output的梯度为

$$
\frac{\partial\mathcal L_{\mathrm{fuse}}}{\partial F_s(t)}
=\omega_t\pi_s(t)(F(t)-Y_t),
$$

对router logits $z_s$的梯度为

$$
\frac{\partial\mathcal L_{\mathrm{fuse}}}{\partial z_s(t)}
=\omega_t\pi_s(t)(F_s(t)-F(t))(F(t)-Y_t).
$$

因此低$\pi_s$会同时削弱该arm的学习与router纠错信号。自动微分与解析式最大误差分别为
`3.47e-18`与`5.20e-18`。

PCC定义same-forward capability target

$$
q_s(t)=\operatorname{softmax}_s\left(-\operatorname{sg}[\tilde e_s(t)]/\tau\right),
\qquad
q_s^\epsilon(t)=(1-\epsilon)q_s(t)+\epsilon/S,
$$

并加入

$$
\lambda_{\mathrm{skill}}\sum_{t,s}\omega_tq_s^\epsilon(t)e_s(t)
+\lambda_{\mathrm{route}}\sum_t\omega_t
\operatorname{KL}(\operatorname{sg}[q(t)]\|\pi(t)).
$$

此时arm output获得附加梯度

$$
\lambda_{\mathrm{skill}}\omega_tq_s^\epsilon(t)(F_s(t)-Y_t),
$$

router logits获得附加梯度

$$
\lambda_{\mathrm{route}}\omega_t(\pi_s(t)-q_s(t)).
$$

两项自动微分恒等式最大误差均为`3.47e-18`；实测最小$q_s^\epsilon=0.02629`，高于
$\epsilon/S=0.02$。

[Theory Boundary] 该lower bound只存在于每个arm **output gradient coefficient**。多个positions经shared mode
field反向传播后仍可能发生parameter-gradient cancellation，所以不能声称PCC数学上保证每个shared parameter都被
有效训练。

## Prefix Measure And Projectivity

对full future domain $T$，定义

$$
\omega_t=\frac1T\sum_{H=t}^{T}\frac1H.
$$

则

$$
\frac1T\sum_{H=1}^{T}\frac1H\sum_{t=1}^{H}e_t
=\sum_{t=1}^{T}\omega_te_t.
$$

在$T=720$随机误差上，两侧差为`4.44e-16`，$\sum_t\omega_t=1$误差为`5.55e-16`，且
$\omega_t$单调不增。任意$H\in\{1,48,96,144,360,719,720\}$均只对同一个full-domain output做prefix crop，
最大差为`0`。

[Novelty Boundary] 该重排identity不是创新点。它只是规定PCC如何在不向模型输入requested horizon的前提下，对
全部nested prefixes分配training credit。

## Synthetic Recoverability

合成case令teacher capability同时依赖history coordinate、target coordinate及两者interaction，并使用4维
history-target features训练一个softmax policy。1200步Adam后：

- capability KL=`1.50e-11`；
- best-scope argmax accuracy=`1.0`；
- 每个history row内best scope随target改变的比例=`1.0`；
- 每个target column内best scope随history改变的比例=`1.0`；
- 被teacher实际选中的scope数=`4/5`。

这只证明PCC的router target在一个可表达、无噪声问题中可由inference-visible features学习，不证明真实PCSD
capability具有同样predictability，也不证明五个scope都会在真实数据中形成skill。

## Gate Result And Failure Risks

`15/15` local cases通过。Step 5因此是conditional pass，但以下风险未被解除：

1. $q$随arms共同更新，可能在训练初期追随noise或形成self-confirming assignment；
2. skill floor可能只把arms训练成相似predictors，而非保留不同coupling capability；
3. shared-field gradient cancellation仍可能使output-level floor失效；
4. fixed-scope独立训练只在2/5 datasets优于A6，readout ceiling尚未排除；
5. dense-prefix measure本身可能解释部分收益，必须作为独立control；
6. generic equal-arm supervision可能已足够，若如此PCC只能降为training control。

## Step 6 Authorization

下一步只授权Step 6 concrete design，必须预注册至少以下arms：

- `A6`、plain `DIRECT`与`DENSE_MATCHED`；
- `MEASURE_ONLY`：只替换为dense-prefix measure；
- `EQUAL_SKILL`：所有arms等权full supervision；
- `CAPABILITY_SKILL_ONLY`：$q^\epsilon$只训练arms，不监督router；
- `ROUTE_ONLY`：capability KL只训练router；
- full `PCC`；
- `STOPGRAD/NO_FLOOR` ablations，用于识别moving-target与skill-floor作用。

Step 6必须定义warm-up、$q$ normalization、共享gradient冲突诊断、arm skill保留指标及hard rollback gate。
在这些设计完成前，PCC implementation、remote、test和confirmation seeds继续保持false。
