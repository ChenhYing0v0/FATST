# SC1-JAPO Step 6 Method And Control Design

## Decision Summary

| Field | Decision |
| --- | --- |
| `candidate` | `SC1-JAPO`（Joint Atom-History Projective Operator） |
| `current_step` | Step 6 complete；只授权 Step 7A local implementation |
| `candidate_status` | `narrative_ready` |
| `architecture` | $E=2$ independent full-rank RGNB experts；$K=256$；joint router width $G=32$ |
| `history_interface` | A6 memory的bijective flatten；无atom-to-patch attention |
| `horizon_contract` | requested $H$只选择active atoms；不进入learned path |
| `initialization` | independent experts；near-uniform nonzero router；禁止identical copy/warm-start |
| `controls` | A6 + JOINT/UNIFORM/HISTORY/ATOM/PERM/RANDOM，共7 arms |
| `training` | full-H720 pointwise L1保持不变；SC2 auxiliary loss继续held |
| `local_gate` | 5 profiles全部通过；projectivity最大`3.331e-16`；gradient path 5/5 |
| `parameter_policy` | JAPO readout约A6 readout的`2.065–2.102x`；只报告，不参与选择 |
| `narrative_gate` | pass only for complete forecasting contract；generic MoE/geometry gate不作claim |
| `remote_training` | `false`；Step 7A production implementation与local invariants通过后才可申请 |
| `rollback` | local fault -> Step 6 repair；controls解释收益 -> Step 4；problem失效 -> Step 2/3 |

## 1. What We Planned To Design

Step 5只证明JAPO function class可行。本轮必须把它收紧成一个不可事后调参、可由controls证伪的具体方法：

1. expert maps到底是共享、split-rank还是每个expert完整rank？
2. history与atom geometry如何形成joint interaction，同时保持continuity和projectivity？
3. 如何初始化，既不落入identical-expert symmetry，又不让softmax一开始saturate？
4. 如何把capacity、generic history routing、geometry-only routing与descriptor semantics逐一隔离？
5. 单seed screen何时可早停、何时必须补seed，而不是看到边缘结果后临时决定？

本轮实现的是design-only checker与protocol config，不是production model。没有dataset I/O、forecast training、
validation metric或test读取。

## 2. Source-Informed Audit

### 2.1 Search record

- search date: 2026-07-14；
- topics: geometric operator gating、operator mixture、soft expert averaging、expert symmetry、router initialization；
- sources: external arXiv/OpenReview paper、official GitHub code；Zotero只视为seed，本轮key additions均由external
  search发现；
- completeness: GNOT paper PDF与official implementation均核对；Cluster-Aware Upcycling只核对arXiv
  metadata/abstract，未把其vision experiment结果外推为本项目结论；operator-mixture papers只用于claim边界。

### 2.2 What was adopted and rejected

| Source | Verified evidence | JAPO decision |
| --- | --- | --- |
| [GNOT paper](https://openreview.net/forum?id=JomvpMQ6NF) + [official code](https://github.com/thu-ml/GNOT/blob/master/models/mmgpt.py) | query coordinates进入MLP gate，softmax后加权多个expert FFNs；official default为2 experts | 说明geometry-only soft gating已有直接先例；保留dense expert softmax作为稳定primitive，但`ATOM`只能是control |
| [MoNO](https://arxiv.org/abs/2404.09101) | mixture of neural operators具有distributed approximation视角 | operator mixture本身不作为novelty或有效性证据 |
| [Spatially conditioned operator experts](https://arxiv.org/abs/2502.04562) | spatially conditioned experts用于boundary/domain decomposition | geometry/spatial conditioning不作claim；JAPO必须证明history-dependent interaction |
| [Cluster-Aware Upcycling](https://arxiv.org/abs/2604.13508) | identical upcycled experts与symmetry/limited early specialization相关 | 与Step5 gradient identity一致；采用independent init，不采用clustering、teacher或distillation |
| [Soft MoE](https://openreview.net/forum?id=jxpsAj7ltE) | softmax mixing可避免hard dispatch，但其slot/token dispatch服务vision backbone | 不采用slot dispatch或batch-axis routing；JAPO只在每个history-atom pair上跨experts归一化 |

[Decision] upstream工作只提供soft gating、geometry conditioning与symmetry风险证据。JAPO的贡献边界仍是
multi-horizon forecasting专属的完整contract，而不是上述任一组件。

## 3. Frozen Architecture

### 3.1 Encoder interface

A6 Encoder输出：

$$
M\in\mathbb R^{B\times C\times P\times D},\qquad
h=\operatorname{vec}(M)\in\mathbb R^{B\times C\times R},\quad R=PD.
$$

flatten是bijective reshape。首版不加入atom-specific patch retrieval，也不修改Encoder；所有arms从相同
initialization class端到端训练Encoder与Decoder。

### 3.2 Independent expert bank

固定$E=2$、每个expert rank $K=256$：

$$
z_e=A_eh+a_e\in\mathbb R^K,
$$

$$
r_{j,e}=V_{e,j:}z_e+c_{j,e},
$$

其中：

- $A\in\mathbb R^{E\times K\times R}$；
- $V\in\mathbb R^{E\times T\times K}$；
- $r\in\mathbb R^{B\times C\times T\times E}$。

不采用两个rank-128 experts。后者总rank接近A6，但每个expert都不能承载任意rank-256 A6 map，会削弱Step5
containment。两个full-rank experts增加capacity，但所有JAPO controls使用完全相同的expert bank与paired seed，
所以capacity不能单独解释`JOINT > controls`。

### 3.3 Factorized joint router

history path：

$$
\widetilde h=\operatorname{LayerNorm}(h),\qquad
s=\operatorname{RMSNorm}(\tanh(W_h\widetilde h+b_h))
\in\mathbb R^G.
$$

atom path使用既有8维canonical RGNB descriptor $d_j$：

$$
\phi_j=\operatorname{RMSNorm}(\tanh(W_dd_j+b_d))
\in\mathbb R^G.
$$

固定$G=32$，joint feature与logits为：

$$
u_j=\operatorname{RMSNorm}(s\odot\phi_j),
$$

$$
\ell_{j,e}=\frac{w_e^Tu_j}{\sqrt G},\qquad
\pi_{j,:}=\operatorname{softmax}_e(\ell_{j,:}).
$$

最终：

$$
\alpha_j=\sum_e\pi_{j,e}r_{j,e},
$$

$$
\widehat y_H=Q_{[0,H),\mathcal A_H}\alpha_{\mathcal A_H}.
$$

该factorized multiplicative router比full $R_e\in\mathbb R^{G\times G}$更小、更容易归因；它仍保留Step5
scalar non-collapse witness所需的history-geometry product。它不是per-horizon router，也不在atoms之间归一化。

## 4. Initialization Contract

### 4.1 Expert initialization

- 每个$A_e$使用与`nn.Linear`一致的independent Kaiming-uniform initialization；
- 不复制同一expert，不从trained A6 warm-start；
- $V_e$独立初始化为

$$
V_{e,j,k}\sim\mathcal N\left(0,\frac{E}{K}\right).
$$

当初始gate近似uniform时，两个independent expert outputs平均会把variance缩小为$1/E$；$V_e$的$\sqrt E$
scale把该variance恢复为A6式$1/\sqrt K$ basis initialization的理论水平。checker得到
`uniform_variance_ratio_theory=1.0`。

### 4.2 Router initialization

- $W_h,W_d$独立使用standard linear initialization；
- $w_e\sim\mathcal N(0,0.01^2)$，router bias为0；
- 不使用exact zero output layer，因为那会让$W_h,W_d$首步gradient为0；
- 不使用large logits、temperature sweep、load-balancing或specialization loss。

[Fact] 五profiles的minimum normalized gate entropy为`0.999855`，mean expert usage范围为
`0.498003–0.501997`；所有expert、history projection、descriptor projection与gate parameters的gradient均
finite且nonzero。

[Decision] 该初始化近uniform但不是symmetry trap：experts彼此不同，router所有层从第一步就有gradient。

## 5. Mandatory Seven-Arm Matrix

| Arm | Gate | Descriptor | What it excludes |
| --- | --- | --- | --- |
| `A6-LBF-natural` | none | none | accepted carrier effectiveness reference |
| `JAPO-JOINT-GEO` | $s\odot\phi$ | canonical | complete candidate |
| `JAPO-UNIFORM` | fixed $(0.5,0.5)$ | none | extra expert capacity / ensemble effect |
| `JAPO-HISTORY` | $s$ | none | sample conditioning without atom geometry |
| `JAPO-ATOM` | $\phi$ | canonical | geometry-only fixed operator |
| `JAPO-JOINT-PERM` | $s\odot\phi$ | permuted | canonical atom-geometry alignment |
| `JAPO-JOINT-RANDOM` | $s\odot\phi$ | moment-matched random | descriptor semantics versus arbitrary coordinates |

所有六个JAPO arms共享：

- 相同$E=2,K=256$ expert-bank architecture；
- 相同expert initialization seed；
- 相同Encoder seed、dataset profile、optimizer、objective、epoch budget与checkpoint policy；
- PERM/RANDOM只改变router descriptor buffer，不改变RGNB synthesis columns或free expert tables。

params差异只报告，不参与gate。UNIFORM没有active router参数并不构成selection bias；它的作用正是检验相同
expert bank在没有routing mechanism时能否解释收益。

## 6. Design Checker Results

五profiles对应$R\in\{768,1536,3072\}$。design-only checker结果：

| Check | Result |
| --- | ---: |
| profile contract hash | exact match |
| profile cases | 5/5 |
| selected prefix cases | 30/30 |
| projectivity max gap | `3.331e-16` |
| uniform mixture identity max gap | `0` |
| minimum control functional difference | `8.211e-4` |
| joint gradient paths | 5/5 finite and nonzero |
| minimum initial gate entropy | `0.999855` |
| expert usage range | `0.498003–0.501997` |
| independent expert pairs | 5/5 different |

JAPO readout parameter ratio相对A6 readout为：ETTh1 `2.086x`、ETTh2/ETTm1/Weather `2.065x`、ETTm2
`2.102x`。差异主要来自两个完整expert maps；router本身只有`24,960–98,688` parameters。

[Self-Critique] checker只证明formula、gradient与controls可执行。它没有证明2x readout capacity能够被controls
完全消除所有optimization差异，也没有证明near-uniform router会形成有意义specialization。

## 7. Frozen Training Protocol

继承five-dataset natural profile contract：

- datasets：ETTh1、ETTh2、ETTm1、ETTm2、Weather；
- dataset-specific fields只允许既已冻结的`patch_num/patch_len/d_model/d_ff`；
- target/validation horizon均为720；training objective保持full-H720 pointwise L1；
- optimizer=`AdamW`，LR=`1e-4`，cosine，effective batch size=32；
- max epochs=20，patience=5，best-val restore；
- evaluation读取validation的dense H1..720与segments H48/96/192/336/720；
- test禁止；SC2-MIPR继续held；
- 所有arms from-scratch end-to-end joint training，不做frozen replacement。

因此首轮实验回答纯architecture question，不把decoder与新training loss混在一起。

## 8. Staged Experiment Matrix And Gates

### 8.1 Seed-2021 screen

固定5 datasets × 7 arms=`35 runs`。

primary metric为dense H1..720 MSE AUC，improvement定义为
$100(1-\mathrm{candidate}/\mathrm{reference})$。

**Immediate fail**只在严重且方向一致时触发：

1. JOINT vs A6 macro $\le-10\%$且正向datasets $\le1$；或
2. JOINT vs five same-bank controls的per-dataset median macro $\le0$且正向datasets $\le1$；或
3. NaN/divergence、missing gradients、prefix inconsistency或artifact不完整。

**Provisional pass**要求同时满足：

1. JOINT vs A6 macro $>0$，至少4/5 datasets为正；
2. JOINT分别相对UNIFORM/HISTORY/ATOM/PERM/RANDOM的macro均$>0$，每项至少3/5 datasets为正；
3. JOINT vs same-bank median macro至少`+1%`，至少4/5 datasets为正。

若provisional pass，直接以完全相同design补seed2022/2023。若既非严重失败也未pass，状态为
`screen_inconclusive`，只补seed2022；在two-seed mean上重用同一pass threshold，pass才补seed2023，否则停止，
不得调router width、init scale、expert rank或epoch。

### 8.2 Three-seed confirmation

三seed full seven-arm matrix最多为105 runs。最终要求：

1. JOINT vs A6 three-seed macro $>0$，至少4/5 dataset means与2/3 seed macros为正；
2. 相对每个same-bank control的three-seed macro均$>0$，每项至少3/5 datasets与2/3 seed macros为正；
3. 相对same-bank median macro至少`+1%`且4/5 datasets为正；
4. MAE macro不得低于`-2%` guard；
5. short与long horizon segments相对A6及same-bank median均为正，防止仅靠AUC局部获益。

这些是architecture effectiveness gates，不是最终test claim。只有完成design freeze与three-seed validation确认后，
才允许使用此前冻结的test reference组织正式test comparison。

## 9. Narrative Gate

| Criterion | Result |
| --- | --- |
| real problem | pass：D8显示geometry有效但fixed separable PAF严重弱于A6 |
| mechanism targets failure | pass：free full-rank expert maps恢复operator freedom，joint gate引入sample-specific atom operator |
| explainable tensor path | pass：`h -> experts`与`(h,d_j) -> gate -> alpha_j -> RGNB` |
| A6 containment/projectivity | pass from Step5；Step6 design preserves invariants |
| continuity | pass：dense softmax，无H、top-k或active-set normalization |
| capacity attribution | pass at design level：UNIFORM + same-bank five controls frozen |
| initialization feasibility | pass locally：balanced/non-saturated/full gradient |
| novelty boundary | conditional pass only for complete task-specific contract |
| empirical effectiveness | not started |

[Decision] `SC1-JAPO`从`proposed`更新为`narrative_ready`。Step 7A可实现production module、runner、analyzer与
local invariants；remote training仍为false。

## 10. Failure Attribution And Rollback

- Step 7A shape/gradient/prefix失败：`implementation_or_numeric_fault`，回Step 6修复，不否定JAPO；
- JOINT不超过UNIFORM：`capacity_control_explains`，当前method失败，回Step 4；
- JOINT超过UNIFORM但不超过HISTORY：joint atom geometry不必要，回Step 4；
- JOINT超过history controls但不超过ATOM/PERM/RANDOM：geometry semantics未被识别，回Step 4；
- controls全过但显著弱于A6：`readout_or_optimization_design_wrong`，回Step 4，不直接否定joint operator problem；
- 多次稳定E2E redesign仍无法超过controls：才允许回Step 2/3重审problem contract。

禁止在任一失败后直接加入auxiliary loss、更多experts、patch retrieval或explicit H挽救结果。

## 11. Step 7A Required Deliverables

1. 在`layers/PLGO.py`实现独立`JAPOReadout`，不改写旧PAF历史实现；
2. 在`TimeAlign.py`注册七arms需要的readout modes；
3. 五profiles × 七arms × 六prefix的shape/projectivity checks；
4. 35个gradient cases，验证Encoder、两个expert maps与active router paths；
5. expert-bank paired initialization hashes与within-bank independence checks；
6. descriptor/basis hashes、initial entropy/usage与patch-block rewrite diagnostics；
7. 35-job remote runner dry-run和analyzer synthetic fixture；
8. 明确`final_evaluation_split=val`、`test=false`、`SC2=false`；
9. model code explanation与code-theory consistency report；
10. 本地全部通过、commit/push后，才检查3090 GPU并申请Step 7B remote screen。

## 12. Artifact Map

- `configs/stage_c_sc1_japo_step6_design.json`：冻结architecture、protocol与gates；
- `scripts/check_stage_c_japo_step6_design.py`：design-only tensor/gradient/protocol checker；
- `profile_design_checks.csv`：五profiles的shape、params、init与gradient结果；
- `arm_contract.csv`：七arms的唯一变化和归因角色；
- `screen_gate_matrix.csv`：screen/confirmation decision rules；
- `design_gate.json`：machine-readable Step6 decision；
- 本报告：reader-facing theory-to-experiment chain。

## 13. 11-Step Record

| Field | Record |
| --- | --- |
| `current_step` | Step 6 complete；Step 7A next |
| `problem` | fixed separable PAF无法兼具RGNB geometry与A6 operator freedom |
| `existence_evidence` | D6 support interaction；D8 geometry positive/operator negative |
| `idea` | independent full-rank experts + history-atom multiplicative router + RGNB projectivity |
| `theory_check` | Step5 pass；Step6 initialization/gradient/control checker pass |
| `design` | E2/K256/G32、seven arms、staged seeds与hard gates frozen |
| `narrative_gate` | pass for complete contract；status=`narrative_ready` |
| `effectiveness_gate` | validation screen/confirmation defined；not started |
| `artifacts` | config + checker + 3 CSV + JSON + report |
| `decision` | `narrative_ready_step7a_local_implementation_only`；remote false；SC2 held |
