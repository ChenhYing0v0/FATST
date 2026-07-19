# SC-D20-CST Step 6 Diagnostic Design

## 1. Current record

| Field | Frozen value |
| --- | --- |
| `current_step` | `SC-D20-CST Step6 complete` |
| `problem` | compact history spectrum能否transfer到A6 coefficient operator并超过同维random history subspace |
| `existence_evidence` | D19 IF skip vs no-skip `+1.6191%` MSE、16/20 cells；但IF vs A6 `-3.6117%` |
| `idea` | 在相同A6 full-trajectory operator上比较fixed spectrum与fixed random orthogonal summary |
| `theory_check` | compact orthonormal subspace + matched random control；no explicit H；full-T后只crop |
| `design` | three-arm matched E2E，5 datasets × seed2021，15 runs / 60 official-test cells |
| `narrative_gate` | `diagnostic_only_pass`；generic statistic concatenation不是method contribution |
| `effectiveness_gate` | `not_started` |
| `artifacts` | config、static checker、14/14 gate artifact、本报告 |
| `decision` | `step6_pass_step7a_local_only` |

## 2. What we plan to test

D20不测试“frequency decoder是否优于A6”，而只测试两个更窄的问题：

1. `transfer`：D19中IF的spectrum-skip收益能否迁移到strong A6 coefficient operator；
2. `specificity`：若迁移成功，收益是否超过同维、同参数的fixed random history projection。

两项必须同时通过。只超过A6但不超过RANDOM，意味着generic direct history access或新增active capacity已经足以
解释收益，不能建立frequency claim。

## 3. Why the intervention is placed here

A6的真实forward contract为：

$$
x_{raw}[B,720,C]
\rightarrow x_{norm}[B,720,C]
\rightarrow memory[B,C,P,D]
\rightarrow h[B,C,R]
\rightarrow a[B,C,256]
\rightarrow \hat y[B,720,C].
$$

D19表明raw-history spectrum在IF readout中有增量信息，但IF整体head过大且效果较差。D20不再替换整个decoder，
而是在最小干预点——`full-trajectory coefficient operator`——检测A6压缩状态$h$之外是否还需要低维history
statistic。该位置直接影响整条720-point trajectory，仍服务fixed-past unified multi-horizon generation。

## 4. Exact summary construction

### 4.1 Spectrum summary

固定$T=720$、$k=1,\ldots,32$，构造64列real Fourier projection：

$$
Q^{cos}_{t,k}=\sqrt{\frac{2}{720}}\cos\frac{2\pi kt}{720},\qquad
Q^{sin}_{t,k}=\sqrt{\frac{2}{720}}\sin\frac{2\pi kt}{720}.
$$

按`cos_k, sin_k`交替排列得到$Q_{spec}\in\mathbb{R}^{720\times64}$。DC被排除，因为输入使用A6
`normalization_x(..., "norm")`后的mean-centered history。所有dataset共用相同frequency indices，不做
energy ranking、dataset-specific bins或dimension sweep。

$q=64$对应32个complex modes，覆盖period不短于22.5 samples的固定低频子空间，同时仅占720-dimensional
history的8.89%。它是预注册的compact diagnostic width，不是从test或parameter count选择的超参数；D20不据此
claim该宽度最优。

### 4.2 Random projection control

使用seed `20260719`生成float64 Gaussian matrix$G\in\mathbb{R}^{720\times64}$，经reduced QR与positive-diagonal
sign canonicalization得到$Q_{random}$。它与$Q_{spec}$均为fixed、non-trainable、orthonormal projection，并在所有
dataset中共享。

normalized history进入两者的操作完全相同：

$$
s=x_{norm}^{\top}Q,\qquad s\in\mathbb{R}^{B\times C\times64}.
$$

因此SPEC-vs-RANDOM只改变projection semantics，不改变summary dimension、scaling、head shape或parameter count。

## 5. Exact coefficient operator

三臂为：

| Arm | Coefficient input | Role |
| --- | --- | --- |
| `A6_MEASURE_RETRAIN` | $h[B,C,R]$ | same-run E2E anchor |
| `A6_CST_SPEC` | $[h;s_{spec}][B,C,R+64]$ | transfer candidate |
| `A6_CST_RANDOM` | $[h;s_{random}][B,C,R+64]$ | generic history-access + exact parameter control |

SPEC/RANDOM使用同一个architecture mode：

$$
a=W_hh+W_ss+b,\qquad
\hat y_T=\Phi_Ta+b_T,\qquad
\hat y_H=P_H\hat y_T.
$$

其中$Phi_T\in\mathbb{R}^{720\times256}$继续是A6 learned temporal basis。没有activation、dropout、residual
prediction path或requested-horizon input。任意$H$只从同一full output取prefix。

## 6. Paired initialization

为避免“新增history path一开始破坏A6”成为混杂因素，冻结以下function-preserving initialization：

1. 同一dataset/seed下三臂Encoder、temporal basis与temporal bias完全相同；
2. SPEC/RANDOM的$W_h$与bias复制同一随机初始化的A6 coefficient head；
3. $W_s$固定zero initialization；
4. SPEC与RANDOM完整trainable weights相同，只有fixed projection buffer不同。

所以训练开始前三臂输出完全相同，但$W_s$在第一个training batch即可获得非零梯度。这里的“function-preserving”
只表示公平的initialization parity，不表示复制或保留任何已训练capacity。

## 7. Matrix and evaluation

- datasets：Weather、ETTm1、ETTh1、ETTh2、ETTm2；
- seed：2021；
- arms：3；全部from scratch，不复用历史A6 checkpoint；
- runs：15；
- official-test cells：5 × 3 × 4 = 60；
- horizons：96、192、336、720；
- metrics：MSE、MAE；
- checkpoint：validation四horizon MSE平均；
- optimization：沿用frozen natural profile与A6_MEASURE training contract；
- seeds2022/2023：仅预留为conditional confirmation，当前未授权。

## 8. Primary effectiveness and attribution gates

gain定义为：

$$
gain(A,B)=\frac{metric(B)-metric(A)}{metric(B)}\times100\%,
$$

正值表示$A$优于$B$。

`SPEC vs A6`与`SPEC vs RANDOM`分别必须满足：

- macro MSE gain至少`0.3%`；
- MSE cell wins至少`11/20`；
- 至少`3/5` datasets与`3/4` horizons为正；
- macro MAE gain至少`0%`。

这两组gate对称，不能因为random control较强而事后放宽specificity要求。

## 9. Internal mechanism health

Step7A/9必须输出：

1. projection orthogonality与DC leakage；
2. SPEC/RANDOM exact parameter equality；
3. three-arm paired state hashes与initial prediction equality；
4. summary weight/gradient norm与prediction deformation；
5. full-T/prefix projectivity；
6. per-dataset/per-horizon SPEC-RANDOM signature；
7. train/validation trajectory、best epoch、budget ceiling与NaN/Inf；
8. optional summary contribution norm，但它不能替代test performance。

## 10. Static feasibility result

checker共14项，结果`14/14 pass`：

| Quantity | Result |
| --- | ---: |
| spectrum orthogonality max error | `3.7629e-15` |
| random orthogonality max error | `8.8818e-16` |
| spectrum DC leakage max | `1.7153e-14` |
| SPEC/RANDOM parameter gap | `0` |
| augmentation parameters | `16,384` for every profile |
| initial A6-SPEC/RANDOM output gap | `0` |
| active-summary prefix gap | `0` |
| active intervention deformation NRMSE | `0.273060` |
| zero-init summary-weight gradient norm | `3331.4684` |

参数增加只用于attribution说明，不参与profile或candidate选择。static synthetic数值只证明contract可实现、可训练，
不证明真实数据上的optimization或effectiveness。

## 11. Failure attribution and rollback

| Result | Attribution | Rollback |
| --- | --- | --- |
| SPEC不超过A6，internal valid | current compact spectrum不transfer | 关闭direct compact-spectrum route，Contribution 1回Step2 |
| SPEC超过A6但不超过RANDOM | `capacity_control_explains`或generic history access | frequency claim关闭，回Step2/4 |
| primary positive但gradient/collapse/numeric fail | `optimization_or_numeric_pathology`或`readout_or_head_design_wrong` | exact diagnostic无效，Step6/7只允许修复一次 |
| SPEC同时过两组primary且internal healthy | compact frequency-specific problem supported | 先申请confirmation；之后只回Step4设计native non-residual operator |

SPEC与RANDOM若同时严重失败，也不能方向级否定history information；它只说明当前`coefficient-input` intervention
不适合，必须标记`intervention_point_wrong`而不是`hypothesis_false`。

## 12. Code-theory consistency

- intended theory：A6 compressed state之外可能存在compact、frequency-specific history information；
- current realization：用两个同维orthonormal fixed subspaces直接进入shared coefficient operator；
- remaining proxy：low-32 Fourier subspace只是frequency semantics的一个固定proxy，不代表所有time-frequency
  representation；random projection也不穷尽所有alternative history summaries；
- falsification：SPEC不能同时超过A6与RANDOM，或summary path没有真实梯度/使用，即否定当前exact diagnostic chain。

## 13. Decision

`step6_pass_step7a_local_only`。

Step7A只实现production readout、buffers、paired initialization、CLI wiring与local shape/projectivity/gradient/hash
tests。remote training、official-test access、confirmation与paper-method promotion继续为false。

## 14. Artifacts

- config：`configs/stage_c_d20_cst_step6.json`
- config SHA256：`ae7408801edf26f4cff793bbbc70f18ebac8b9528cfec8425c1749e68e86594d`
- checker：`scripts/check_stage_c_d20_cst_step6.py`
- checker SHA256：`1a6d9bf57b983f33983fb164e2a977e2b6e3befb5275f87644e201a5fe2932c9`
- static artifact：`analysis/stage_c_post_ccsf_step24_reset_20260719/d20_step6/step6_static_gate.json`
