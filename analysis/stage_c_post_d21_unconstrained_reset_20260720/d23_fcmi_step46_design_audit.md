# SC-D23-FCMI Step4-6：Future-Coordinate Main–Interaction operator

## 1. 11-step record

| Field | Current Record |
| --- | --- |
| `current_step` | Step4-6 source/narrative/theory/design gate |
| `problem` | standard future-query decoder把trajectory-wide evidence与coordinate-specific interaction混在同一context中，不能原生回退到generic retrieval，也难以排除multi-branch capacity解释 |
| `existence_evidence` | D22-C ordered > generic `+2.5228%` MSE，15/20 cells、4/5 datasets；Weather 4/4负 |
| `idea` | 将query-retrieved evidence精确分解为future-main evidence与zero-mean coordinate interaction，并分别变换后合成 |
| `theory_check` | generic与standard query decoder均为exact contained cases；不输入requested H；不改变Bayes target |
| `design` | paired E2E A6 encoder + FCMI decoder；standard/generic/multi-branch/order controls；same objective |
| `narrative_gate` | `conditional_pass`；claim位于problem→decomposition→contained fallback→attribution chain，不claim query/attention primitive |
| `effectiveness_gate` | pending implementation；official-test matrix尚未授权 |
| `artifacts` | D22-C complete artifacts + current Step4-6 audit |
| `decision` | `step46_conditional_pass_step7a_local_only` |

## 2. Source-informed boundary

CATS与TimePerceiver已经覆盖future/target query cross-attention；MQTransformer覆盖context-dependent alignment；
TQNet覆盖temporal query结合raw input K/V。故普通query decoder、learned query、patch memory、positional encoding、
cross-attention和parameter sharing均不能单独claim novelty。

另一方面，D22-C最强generic control与ordered之间存在split-stable aggregate gap，但Weather为负；这说明问题不是
“是否使用attention”，而是“如何让同一native operator区分generic evidence main effect与coordinate-specific
interaction”。2026 attention critique进一步要求用matched multi-branch no-interaction control排除Q/K/V branch
capacity解释。

## 3. Core operator

A6/TimeAlign-style encoder从normalized history得到memory
$M(x)\in\mathbb{R}^{B\times C\times P\times D}$。future coordinate $t$的fixed/continuous positional query为
$q_t$。standard cross-attention context为

$$
S_t(x)=\operatorname{Attn}(q_t,K(M),V(M))\in\mathbb{R}^D.
$$

FCMI不直接把$S_t$作为monolithic decoder state，而作exact main–interaction decomposition：

$$
\bar{S}(x)=\frac{1}{T}\sum_{t=1}^{T}S_t(x),\qquad
\Delta_t(x)=S_t(x)-\bar{S}(x),
$$

其中

$$
\frac{1}{T}\sum_{t=1}^{T}\Delta_t(x)=0.
$$

native forecast state定义为

$$
U_t(x)=W_{\mathrm{main}}\bar{S}(x)
+W_{\mathrm{int}}\Delta_t(x)+E(q_t),
\qquad
\hat y_t=w_o^\top \sigma(U_t).
$$

这里没有requested-H embedding、router或第二loss。`main`与`interaction`是同一decoder operator的两个可识别
输入坐标。

## 4. Exact containment与initial morph

### 4.1 Generic contained case

若$W_{\mathrm{int}}=0$，则history evidence对所有future coordinates相同；coordinate仍通过$E(q_t)$进入shared
readout。这对应D22-C的generic retrieval boundary。

### 4.2 Standard query contained case

若$W_{\mathrm{main}}=W_{\mathrm{int}}=W$，则

$$
W\bar S+W(S_t-\bar S)=WS_t,
$$

精确恢复standard query decoder。

Step7A可从同一个random $W$复制到两个branches，使FCMI与standard query在初始化时function-identical。该操作只
是`exact initial morph`，不是preserved learned capacity，也不使用trained checkpoint。

## 5. 为什么不是一个普通双分支patch

FCMI的claim不在“两层Linear”：

1. D22-C先证明coordinate-specific retrieval超越query-independent retrieval与五类controls；
2. Weather negative要求operator原生包含generic boundary；
3. zero-mean $\Delta_t$禁止interaction branch携带trajectory-wide evidence main effect；
4. standard query与generic query均为可证明contained functions；
5. matched dual-branch control检验收益来自main–interaction coordinates，而不是branch数量。

如果matched control解释收益，FCMI必须按`capacity_control_explains`关闭；不能只靠performance promotion。

## 6. Frozen control design

| Arm | State construction | Role |
| --- | --- | --- |
| `A6_MEASURE` | accepted learned-basis carrier | strong performance/objective control |
| `STANDARD_QUERY` | $W S_t$ | source-family control；参数较少，不单独作attribution |
| `STANDARD_DUAL_MATCHED` | $\frac12(W_1+W_2)S_t$ | exact parameter/multi-branch control |
| `GENERIC_DUAL_MATCHED` | $\frac12(W_1+W_2)\bar S$ | contained generic control |
| `FCMI` | $W_1\bar S+W_2\Delta_t$ | candidate |
| `FCMI_ORDER_SHUFFLED` | memory values与positions固定置换 | order specificity ablation |

`TARGET_SHUFFLED_QUERY`只在local diagnostic/sanity中保留，不进入第一轮formal 5-dataset test matrix，避免把显然
collapsed的negative control当paper baseline；若FCMI进入formal attribution，必须在validation diagnostics报告。

## 7. Objective、split与comparison roles

- all E2E arms from scratch；
- same Encoder、initialization class、data、harmonic measure objective、optimizer、checkpoint selector与test matrix；
- validation只按H96/H192/H336/H720平均MSE选checkpoint；
- official test仍是`test_informed` complete five-dataset × four-horizon gate；
- A6与FCMI若parameter gap超过1%，增加`DENSE_DUAL_MATCHED`而不是删control或宣称capacity无关；
- CATS、TimePerceiver、TQNet需作为source-faithful external baselines/related work，不机械复制进local candidate。

## 8. Narrative gate

### Problem motivation

[Strong Evidence] D22-C控制后表明target-coordinate-specific evidence access具有aggregate必要性，但不是universal；
generic fallback不可缺。

### Mechanism novelty

[Conditional] main–interaction evidence decomposition、zero-mean identifiability与dual containment形成task-specific
operator contract。未发现primary prior覆盖该完整chain，但attention/query primitives均已有直接prior，novelty
必须限定为完整coupling而非组件。

### Explainable tensor/gradient path

$M\to S_t\to(\bar S,\Delta_t)\to(W_{\rm main},W_{\rm int})\to U_t\to\hat y_t$；
main与interaction gradient/norm可以独立测量，interaction不能通过mean carry generic evidence。

### Contribution boundary

Contribution 1暂定为：

> controlled evidence diagnosis + a generic-contained, main–interaction future-coordinate forecasting operator。

不设Contribution 2；只有FCMI E2E暴露新的训练瓶颈且经problem gate后才讨论。

Decision：`narrative_gate=conditional_pass`，只授权Step7A local implementation与synthetic/real-batch
shape、morphism、gradient、parameter audits；remote training与official test暂不授权。

## 9. Step7A falsification

local implementation必须同时通过：

1. memory/query/context/main/interaction/output shapes；
2. $\operatorname{mean}_t\Delta_t=0$ numeric invariant；
3. `W_main=W_int`时FCMI与standard output max gap `<1e-6`；
4. generic control不读取$\Delta_t$；
5. main、interaction、query、output paths均有finite nonzero gradients；
6. order shuffle只改变value-position binding，不改marginal/capacity；
7. paired base hashes、initial output equality与parameter audit；
8. production CLI synthetic smoke。

任一失败只回Step5/6修复，不得启动remote。Step7A通过后还需独立freeze formal matrix与effectiveness/narrative
threshold；不因D22-C positive自动授权method test。
