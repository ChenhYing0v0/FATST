# SC1-SIFF-v2-CCSF Step 7A Implementation Gate

## 1. 结论

`SC1-SIFF-v2-CCSF-v1-preimplementation`完成本地production implementation，Step7A gate为`18/18`通过。
当前decision=`step7a_local_pass_step7b_next`：只允许进入Step7B prelaunch设计；validation temperature pilot、
remote training、formal test与confirmation仍为false。

[Fact] 本轮没有训练数据集、没有选择temperature、没有读取validation/test结果。所有数值均来自synthetic
construction tests，只证明实现与Step6理论/控制合同一致，不能证明CCSF有效。

## 2. 11-step record

| Field | Content |
| --- | --- |
| `current_step` | Step7A local implementation complete |
| `problem` | 把Step6的contrast-conditioned fusion与relative calibration转成不会改变parent initialization、projectivity或control semantics的production path |
| `existence_evidence` | Step5 contrast identifiability 5/5；Step6 narrative/control static 5/5 |
| `idea` | SIFF arms生成后构造target-free scope contrast，以shared scorer修正parent logits；same-forward detached relative regret弱监督policy |
| `theory_check` | parent base hash/initial output exact、full-domain crop、teacher stop-gradient、relative scale invariance、capacity/semantic controls均通过local gate |
| `design` | 10 arms × 5 datasets manifest；4 CCSF readouts；2 calibration objectives；18类construction gates |
| `narrative_gate` | unchanged conditional pass；implementation不扩大Step6 claim |
| `effectiveness_gate` | not started；无dataset training或official-test evidence |
| `artifacts` | `local_gate/*.csv/json`、Step7A config、production module、adapter与guard |
| `decision` | `step7a_local_pass_step7b_next`；pilot/remote/test=false |

## 3. Production architecture

### 3.1 Forward tensor path

Encoder与SIFF arm generator保持不变：

$$
H[B,C,R]\rightarrow A[B,C,S,T],\qquad S=5,\ T=720.
$$

`CCSFCouplingFieldReadout`从同一次forward的arms计算：

1. consensus与centered contrast；
2. cross-scope normalized contrast；
3. 各scope native group内的mean、RMS与endpoint difference；
4. relative cross-scope disagreement。

得到`contrast_descriptor [B,C,T,S,6]`。它只读取model forecasts，不读取target、requested horizon或benchmark
horizon bins。shared scorer输入为：

$$
32\ \text{history}+4\ \text{target coordinate}
+1\ \text{scale coordinate}+6\ \text{contrast}=43,
$$

经过`43 -> 64 -> 1`产生`correction_logits [B,C,T,S]`，新增参数量严格为2,881。最终：

$$
\alpha=\operatorname{softmax}(\ell^{v1}+\Delta\ell),\qquad
\widehat Y=\sum_s\alpha_sA_s.
$$

final correction layer零初始化。相同seed下，CCSF、zero-control与permuted-control的SIFF base parameter hash均与
parent完全相同，初始输出maximum gap均为`0.0`。因此CCSF从v1函数开始训练，但这仍不是trained checkpoint
warm-start。

### 3.2 Control semantics

实现了四个readout modes：

| Readout | Scale field | Contrast input |
| --- | --- | --- |
| `ccsf-coupling-field` | ordered Q2 | true descriptor |
| `ccsf-no-contrast-control` | ordered Q2 | exact zero，参数不删减 |
| `ccsf-permuted-contrast-control` | ordered Q2 | scope轴循环左移`[1,2,3,4,0]` |
| `ccsf-independent-scope-control` | independent Q5 | true descriptor，dataset-wise matched rank |

local intervention确认true descriptor RMS为`0.869589`，zero maximum为`0.0`，permuted tensor与冻结映射exact；
给相同nonzero scorer后，true相对zero/permuted correction RMS gap分别为`0.246947/0.288452`。contrast到arm
forecast的gradient norm为`0.009272`，证明该信息确实进入可训练computation graph，而非只作为日志导出。

### 3.3 Projectivity

四个CCSF readouts均先计算完整T=720 arms、contrast、policy与forecast，最后crop。对prefix
`{1,96,192,337,720}`，四类readout的maximum prefix gap均为`0.0`，通过`1e-7`阈值。

## 4. Training objective

实现两个新objective modes：

- `ccsf_relative_calibration`：relative regret teacher + entropy confidence；
- `ccsf_standardized_calibration`：旧cross-arm standardized teacher geometry control。

两者都保留harmonic dense-prefix fused L1与weight=1的equal-skill arm loss；calibration weight固定为`0.1`。
relative teacher由same-forward arm absolute error构造并stop-gradient。local algebra结果：

- 未加入calibration时，与现有`equal_skill`目标的gap为`0.0`；
- prediction/target共同乘7后relative teacher maximum gap为`7.15e-7`；
- arms完全tie时confidence maximum为`0.0`；
- teacher不需要gradient，policy gradient norm为`0.009125`；
- relative与standardized objectives均finite。

Step7A使用`tau=0.1`只作为local smoke value，不是formal hyperparameter selection。正式temperature仍需从
`{0.05,0.1,0.25}`按五dataset共同validation macro score选择；该15-run pilot尚未授权。

## 5. Optimization-path audit

由于correction final layer零初始化，第一步中hidden-layer gradient按构造为零，但output-layer必须先得到gradient。
本次对四个CCSF modes执行两步SGD witness：

1. 第一步correction output gradient均非零；
2. 一次更新后correction logits RMS均非零；
3. 第二步correction hidden gradient均非零；
4. true/permuted/independent的contrast-column gradient非零；zero-control的contrast-column gradient严格为零，
   但history/coordinate columns仍可训练。

这排除了“零初始化导致correction branch永久死亡”的实现故障。它不说明20-epoch optimization一定稳定。

## 6. Manifest、parameters与diagnostics

Step7A config生成5 datasets × 10 arms × seed2021 = 50 jobs。50/50 CLI均能由`train_repo.py`解析，包含four-H
validation checkpoint rule、full-crop、dataset profile、matched rank与objective mode。

30个unique dataset-readout-rank constructors全部通过，且每个dataset内所有arms的Encoder initialization hash一致。
ordered/independent CCSF parameter relative gap：

- ETTh1：`0.2463%`；
- ETTh2/ETTm1/Weather：`0.3833%`；
- ETTm2：`0.0777%`。

均低于冻结`0.5%`阈值。production training details现同时暴露：arms、policy、base logits/policy、contrast descriptor
与correction logits，shape/finite 7/7通过，为后续internal mechanism health提供直接artifact source。

## 7. Local gate结果

18类categories全部通过：

1. Step6 design hash与profile hash；
2. exact arm contract与50-job matrix；
3. 50 CLI contracts与30 constructors；
4. paired Encoder、parent base initialization与initial function containment；
5. ordered/independent parameter matching；
6. true/zero/permuted control semantics；
7. four-readout prefix projectivity；
8. objective algebra与teacher stop-gradient；
9. 10-arm Encoder/readout gradient paths；
10. four-readout two-step correction optimization；
11. diagnostic tensor contract；
12. remote manifest dry-run、remote refusal与authorization boundary。

remote guard在非dry-run调用时固定exit code 3。runner当前只是50-arm manifest/refusal template；正式training、
evaluation、artifact completeness与scheduling必须到Step7B后才能补齐和授权。

## 8. Code-theory consistency与未测试边界

### Intended theory

relative scope competence不能只由history state推断；同一个projective decoder内部arms对每个target的deterministic
contrast提供额外信息，relative calibration只在这条information path上形成有意义的co-design。

### Code realization

- arms先生成，contrast再进入policy correction；
- target只进入detached training teacher，不进入inference；
- requested H只在完整forecast后crop；
- zero/permuted/independent controls与parent initialization均已显式实现；
- 2×2 architecture/objective arms由同一training adapter执行。

### Remaining proxy

Step7A只证明shape、algebra、gradient与control语义。以下仍完全未知：

- trained contrast correction是否超过zero/permuted controls；
- relative calibration是否优于architecture-only/standardized teacher；
- full model是否超过A6_MEASURE与v1；
- ordered field是否超过matched independent field；
- policy allocation、oracle、diversity与correction RMS是否满足Step6内部健康gate。

## 9. Decision

Decision=`step7a_local_pass_step7b_next`。下一步是Step7B prelaunch：先冻结15-run shared-temperature validation
pilot如何独立执行、候选版本如何在temperature选择后固化，再补齐正式runner/evaluator/internal-artifact contract。

当前不得启动temperature pilot、remote training、official test或confirmation seeds。
