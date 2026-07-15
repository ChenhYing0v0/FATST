# StageC D14-A1 Dual-Carrier Grouped-MLP Source/Theory Audit

## 0. Decision Card

| Field | Value |
| --- | --- |
| `current_step` | Step 4-7A passed；Step7B neutral seed2021 next |
| `problem` | A0 nominal scales未形成不同forecast functions，不能验证output-sharing scope |
| `idea` | E2E grouped nonlinear heads + neutral primary / A6 sensitivity dual carriers |
| `method_status` | diagnostic_only；不是PCSD implementation |
| `neutral_remote` | true；仅neutral raw-history seed2021 |
| `a6_remote` | false；neutral problem gate后才重新授权 |
| `D14-B/test` | false |
| `rollback` | neutral valid fail -> close pair；neutral invalid -> diagnostic redesign；A6-only fail -> no direction rejection |

## 1. External Primary-Source Audit

检索日期为2026-07-15，Zotero只作seed，本轮采用external primary sources：

- [Learning to Branch for Multi-Task Learning](https://proceedings.mlr.press/v119/guo20e.html)说明过度共享会造成
  over-generalization与negative transfer，branch topology需要由任务关系决定；
- [Efficiently Identifying Task Groupings for Multi-Task Learning](https://arxiv.org/abs/2109.04617)直接把哪些tasks
  应共同训练视为需要验证的问题，并用task influence识别grouping；
- [Learning Task Grouping and Overlap in Multi-task Learning](https://arxiv.org/abs/1206.6417)在task parameters的
  low-dimensional subspace中表达group-specific与overlapping sharing；
- [CATS](https://openreview.net/pdf?id=iN43sJoib7)把future horizons写成独立queries，同时强调跨horizons的
  parameter sharing；它构成point/query endpoint与prior-art边界；
- [Implicit Forecaster](https://openreview.net/forum?id=gqoeQPhQcE)指出逐future-point独立预测可能缺少global view，
  但其wave decoding不是本诊断采用的grouping mechanism；
- [PatchTST](https://openreview.net/pdf?id=Jbdc0vTOcol)使用flatten representation与linear forecasting head，支持
  把简单history carrier与forecasting head分开审计。

[Boundary] 上述工作支持“sharing topology可能重要”，不证明temporal-contiguous future grouping必然有效；
task grouping、branch search、future queries、wave decoding都不能成为PCSD单项novelty。

## 2. Why A0 Is Replaced Rather Than Tuned

A0从同一个OLS map做post-hoc rank projection；global rank 60接近carrier rank 64，最终five-scale risk spread最多
0.04036%。它匹配factor storage count，但未匹配rank-manifold DoF，也没有强制trained function separation。

A1不调A0 ranks或PCA width。它直接改变E2E decoder parameter-sharing topology，并在看validation performance前
证明每个scale都包含full-affine map、params近似匹配、group cover与gradient sharing contract成立。

## 3. Grouped Nonlinear Forecast Family

carrier输出$h\in\mathbb R^R$。对scale $s$，future domain被分为$G=T/s$个groups。每group独立计算：

$$
z_g=\operatorname{GELU}(hA_g+a_g),\qquad
\hat y_{B_g}=z_gB_g+b_g.
$$

$A_g\in\mathbb R^{R\times k_s}$、$B_g\in\mathbb R^{k_s\times s}$。不同groups没有decoder parameter sharing；
同一group内的$s$个targets共享$k_s$个nonlinear hidden units。因此：

- $s=1$：future targets拥有独立nonlinear banks；
- $s=720$：全部targets共享同一个hidden bank；
- intermediate $s$：真正的block sharing；
- random partition：保持同一个$s,k_s$与parameter count，只打乱future coordinate membership。

scale不作为learned input；requested $H$只crop full-domain output。

## 4. Capacity And Affine-Containment Proof

point arm固定$k_1=4$，其decoder params为：

$$
P_1=T\,[Rk_1+k_1+k_1+1].
$$

其他$k_s$选择为使

$$
P_s=G\,[Rk_s+k_s+k_ss+s]
$$

最接近$P_1$的整数；每个carrier/dataset最大gap必须小于1%。这不是用params决定dataset profile，而是隔离
sharing attribution。

更重要的是，exact GELU满足：

$$
\operatorname{GELU}(u)-\operatorname{GELU}(-u)=u.
$$

任意rank-$r$ affine block map可用$r$对正负hidden units表示。A1要求：

$$
k_s\ge2\min(R,s).
$$

对$R\in\{720,768,1536,3072\}$与全部scales均成立。因此所有arms都能精确包含任意full-affine
history-to-future map；差异来自nonlinear feature sharing，不是linear forecast能力缺失。

## 5. Dual-Carrier Causal Contract

### N0 neutral raw-history carrier — primary

normalized raw history直接作为$h[B,C,720]$；没有learned encoder，也没有global basis。GroupedMLP本身从头训练。
它负责方向级problem gate，必须先证明carrier skill、trained function disagreement、crossing、oracle headroom与
canonical-vs-random contiguity。

### A6N A6-natural architecture carrier — sensitivity only

复用五数据集冻结natural profiles的`timealign-token-mlp` encoder architecture，但不加载checkpoint；encoder与
GroupedMLP从相同seed、相同协议E2E joint training。另运行exact A6-LBF作为paper-carrier reference。

[User Constraint] natural profiles与encoder hyperparameters是围绕global basis decoder冻结的。即使GroupedMLP在
A6N上失败，也可能是carrier/head interface或profile inductive bias，不得单独否定scale mechanism。

## 6. Result Interpretation Matrix

| Neutral primary | A6 sensitivity | Decision |
| --- | --- | --- |
| pass | pass | strongest problem evidence；进入multi-seed confirmation |
| pass | fail/flat | scale problem retained；`carrier_interface_or_profile_incompatibility`；不得方向拒绝 |
| fail validly | any/not run | hypothesis unsupported；close PCSD/CCRL pair |
| invalid | pass | insufficient；A6-only positive不能通过方向gate |
| invalid | fail | redesign/close according to neutral failure attribution |

因此execution order固定为`N0 seed2021 -> neutral gate -> A6N authorization -> confirmation`，不是把两个carrier
简单平均。

## 7. Function-Separation And Effectiveness Gates

### Step7A local gates

1. all shapes/prefix crop exact；
2. canonical/random partitions均为exact disjoint cover；
3. target-pair decoder-gradient parameter overlap iff同group；
4. five-scale params最大gap$\le1\%$；
5. affine witness max gap$\le10^{-5}$；
6. all carrier/head gradients finite且nonzero；
7. no frozen parameters，A6 grouped arms encoder initialization paired bydataset/seed。

### Remote neutral gates

1. carrier skill至少3/5 datasets，gain$\ge0.5\%$；
2. 至少3/5 datasets的trained canonical pair median prediction disagreement达到0.5%；否则
   `diagnostic_invalid_for_direction_rejection`；
3. stable future-bin crossing至少3/5；
4. sample × bin oracle相对train-selected fixed scale macro gain$\ge0.5\%$；
5. canonical oracle相对random oracle至少3/5为正且macro$\ge0.1\%$；
6. no non-finite、split leakage或$>100\%$ pathology。

只有neutral全部problem gates通过，才授权A6 sensitivity。A6结果不与neutral取平均，不改变neutral方向判定。

## 8. Staging Decision

[Returned Gate] Step7A完成80个parameter/partition/affine cases、20个full forward/gradient cases与五profiles的
encoder pairing。最大parameter gap 0.1646%，最大affine witness gap $2.3842\times10^{-7}$，prefix gap 0，全部
通过。

[Decision] 只授权neutral raw-history、seed2021的Step7B remote diagnostic。A6-natural仍held，只有neutral返回
`neutral_problem_pass_authorize_a6_sensitivity`才可启动。D14-B、PCSD method、test与confirmation seeds仍未授权。
