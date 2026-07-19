# SC-D19-IFC Step 4/5 Source、Theory 与 Control Audit

## 1. 当前节点

| Field | Content |
| --- | --- |
| `current_step` | D19 Step 4/5 complete；Step 6 control design next |
| `candidate_version` | `SC-D19-IFC-control-v0` |
| `problem` | A6 learned-basis decoder是否已经充分利用trajectory structure，还是implicit wave generation仍有headroom |
| `existence_evidence` | D18关闭soft projectivity；IF paper显示forecasting-phase structured decoding可跨backbone增益 |
| `idea` | 将official IF机制适配到same A6 Encoder/full-T/measure protocol，仅作为source-informed control |
| `theory_check` | full-spectrum output先生成720 points再crop，exact projectivity成立；function class与A6不同 |
| `design` | Step 6拟冻结A6、IF、IF-no-skip、matched nonlinear direct control |
| `narrative_gate` | not applicable：control-only；不得升级为本项目创新 |
| `effectiveness_gate` | not started |
| `artifacts` | IF paper、official GitHub code、本文 |
| `decision` | `step5_conditional_pass_step6_control_design_only` |

## 2. Official implementation核对

核对日期：2026-07-19。来源均为official repository
`https://github.com/rakuyorain/Implicit-Forecaster`。

### 2.1 Upstream tensor path

official `IFT`使用：

$$
X[B,L,C]\xrightarrow{\operatorname{RevIN}}X_n[B,L,C],
$$

$$
X_n\rightarrow E[B,C,D],
$$

同时对原始input做：

$$
\operatorname{rFFT}(X^\top)\rightarrow
(A_x,\Phi_x)[B,C,L/2+1].
$$

Amplitude head：

$$
[E,A_x]\rightarrow
\operatorname{Linear}\rightarrow\operatorname{GELU}
\rightarrow\operatorname{Dropout}\rightarrow\operatorname{Linear}
\rightarrow\operatorname{ALU}_{0.5}
\rightarrow \hat A[B,C,P/2+1].
$$

其中`ALU`实现为`abs(where(z < 0, 0.5*z, z))`，保证nonnegative amplitude。

Phase head使用两个独立two-layer MLP：

$$
[E,\Phi_x]\rightarrow (\hat s,\hat c)
\xrightarrow{\operatorname{atan2}}\hat\Phi[B,C,P/2+1],
$$

两个head末层均为`tanh`。最终：

$$
\hat Y_{full}
=
\operatorname{irFFT}
\left(\hat A\exp(i\hat\Phi)\right)
\in\mathbb R^{B\times720\times C}.
$$

### 2.2 Critical defaults

| Item | Official default / script |
| --- | --- |
| lookback | 96 |
| `spectrum_size` | 720 |
| Fourier normalization | `ortho` |
| RevIN | on，affine off |
| decoder dropout | 0.1 |
| head hidden width | global `d_ff`，default 2048 |
| head activation | GELU |
| phase representation | tanh sine/cosine + `atan2` |
| amplitude activation | `ALU(w=0.5)` |
| optimizer | Adam，LR $10^{-4}$ |
| batch size | 32 |
| epochs / patience | 16 / 3 |
| upstream loss | MSE |

official ETT/Weather scripts分别训练H96/H192/H336/H720模型，但architecture始终生成`spectrum_size=720`
的full output；`main.py`只用`pred[:, :pred_len]`计算loss和metric。因此upstream本身已经采用
“full 720 generation + requested prefix crop”，只是training objective仍是horizon-specific。

这点非常重要：D19不需要为projectivity修改IF synthesis，只需把upstream own-H loss替换为项目冻结的
measure objective。

## 3. 与A6的function-class关系

A6：

$$
h[B,C,R]\rightarrow c=W_ch+b_c\in\mathbb R^{256},
$$

$$
\hat Y=B_{\theta}c+b_t,
\qquad
B_{\theta}\in\mathbb R^{720\times256}.
$$

对给定参数，A6 output对$h$是linear-affine。temporal basis全局learned，但每个sample只预测一组coefficients。

IF：

$$
(h,\operatorname{rFFT}(X))
\rightarrow(\hat A,\hat\Phi)
\rightarrow\operatorname{irFFT}
\left(\hat A e^{i\hat\Phi}\right).
$$

它对history state是nonlinear，对每个sample同时改变amplitude与phase，且通过input spectrum skip获得A6
hidden之外的固定history summary。

[Strong Evidence] IF不是A6的简单reparameterization；但IF比A6同时多了两件事：

1. nonlinear head；
2. raw input spectrum skip。

所以仅比较`IF vs A6`不能把收益归因于wave synthesis，Step 6必须加入matched nonlinear direct control与
no-skip control。

## 4. Projectivity证明

令IF full synthesis为$G_\theta(x)\in\mathbb R^{720}$，定义任意$H\le720$：

$$
F_H(x)=P_HG_\theta(x).
$$

则对任意$H\le K$：

$$
P_HF_K(x)
=P_HP_KG_\theta(x)
=P_HG_\theta(x)
=F_H(x).
$$

因此，只要：

1. `spectrum_size=720`固定；
2. requested horizon不进入amplitude/phase heads；
3. 所有horizon只crop同一full output；

exact projectivity严格成立。D19不得根据requested horizon改变frequency pool、phase normalization或head
parameters。

## 5. 本地适配contract

建议Step 6冻结以下shape：

1. A6 normalized history
   $X_n[B,96,C]$；
2. A6 memory
   $M[B,C,P,D]$；
3. flattened encoder state
   $h=M.\operatorname{flatten}(-2)[B,C,R]$；
4. raw-history spectrum
   $(A_x,\Phi_x)[B,C,49]$；
5. amplitude/phase heads输出
   $[B,C,361]$；
6. `irfft(n=720, norm="ortho")`
   输出$[B,C,720]$；
7. 转置、A6 `Normalize.denorm`后得到
   $\hat Y[B,720,C]$。

source-faithful部分：

- separate amplitude/sine/cosine MLPs；
- `ALU(0.5)`；
- tanh + `atan2`；
- input amplitude/phase skip；
- orthonormal rFFT/iFFT与P=720。

项目适配部分：

- Encoder替换为A6 natural Encoder；
- hidden size从upstream $D$变为A6 $R=P\times D$；
- objective使用冻结`A6_MEASURE` harmonic-L1；
- checkpoint使用validation four-H mean MSE；
- 所有数据集使用各自冻结natural profile。

这些适配必须写入candidate identity，不能称为upstream exact reproduction。

## 6. Step 6 mandatory controls

### `A6_MEASURE`

固定effectiveness reference。

### `IF_MEASURE`

完整source-informed wave head与input-spectrum skip。

### `IF_NOSKIP_MEASURE`

amplitude/phase heads只读$h$。为了避免参数减少本身成为解释，Step 6需决定zero-spectrum input还是matched
constant features；不得用不同head width补偿后再称为纯skip ablation。

### `DIRECT_NONLINEAR_MATCHED_MEASURE`

读取与`IF_MEASURE`相同的信息，直接生成720 points；参数量匹配IF，但不经过polar spectrum与iFFT。该control
回答收益是否只来自nonlinear/capacity。

[Decision] 不增加Transformer decoder。IF paper已用其作external ablation，而本地核心归因只需最小四arm
matrix；额外heavy decoder会扩大实验而不改变当前问题。

## 7. Numeric与optimization风险

1. `atan2(s,c)`在$(0,0)$附近gradient不稳定；upstream没有显式epsilon。Step7A必须做zero/near-zero输入
   gradient test与真实batch finite smoke。
2. `ALU`在0点不可微但PyTorch可给subgradient；需检查初始amplitude是否全零。
3. A6 natural profiles的$R$从768到3072不等，直接把$R+49$送入三个head会产生明显dataset-specific params。
   params只报告，不作为profile选择因素；但matched direct control必须逐dataset匹配。
4. upstream用MSE、ours用harmonic-L1；这是为了same-objective comparison，不是source exact reproduction。
5. upstream IF full spectrum rank为361 complex polar components，表达维度高于A6 rank256。不能把正收益仅归因为
   “wave structure”；matched direct control与parameter accounting不可省略。

## 8. Step 5 Gate

| Gate | Result |
| --- | --- |
| source mechanism完整可追溯 | pass |
| full-T exact projectivity | pass |
| A6 tensor contract可接入 | pass conditional |
| function class区别清晰 | pass |
| generic nonlinearity/capacity control可定义 | pass |
| prior-art boundary清晰 | pass：IF只能作control |
| numeric risks可在Step7A证伪 | pass conditional |

Decision=`step5_conditional_pass_step6_control_design_only`。

## 9. 下一步

进入Step 6，冻结：

1. four-arm exact parameterization；
2. `IF_NOSKIP`公平ablation方式；
3. matched direct MLP width与parameter tolerance；
4. initialization pairing；
5. validation-only screening还是直接formal test的阶段边界；
6. performance、attribution、internal health与rollback gates。

Step 6完成前：

- model code=false；
- local gradient test=false；
- remote=false；
- official test=false；
- D19仍为`control_only`，不是Contribution 1。
