# Post-PCC SIFF/MCCA Step 6 Source-Informed Method And Control Design

## Decision

| Field | Value |
| --- | --- |
| `current_step` | Step 6 complete；Step 7A local implementation next |
| `architecture_candidate` | `SC1-SIFF-v1` |
| `training_candidate` | `SC2-MCCA-v1` |
| `local_design_gate` | 22/22 pass |
| `narrative_gate` | conditional pass |
| `implementation/remote/test` | Step7A local true / false / false |
| `rollback_point` | SIFF generic-capacity explanation → Step4；MCCA same-mass PCC/generic OT explanation → Step4 |

[Decision] Step 6允许实现**一个冻结版本**，不证明两项贡献有效。SIFF与MCCA都使用已有primitive，因此允许的创新
边界必须落在完整`problem -> constraint -> mechanism -> implementation -> claim`链上；Step7B若没有分别可归因的
architecture与training主效应，不得以joint arm的小幅收益把两项同时写入论文。

## Why The Problem Remains Real

PCC-v1-TI相对A6与plain PCSD有正收益，并恢复25/25 arms；但`EQUAL_SKILL`解释了其相对A6收益的88.90%，同时
pairwise output diversity只保留plain的20.57%-41.13%。因此当前矛盾不是“arms完全不会预测”，而是：

1. current PCSD让五个scopes读取同一份history modes，scale identity只在后续pooling中出现；
2. PCC为了防starvation，在每个target持续向所有arms撒同标签credit，恢复skill的同时推动同质化；
3. 论文需要同时改变decoder的可辨识自由度和credit的分配位置，不能继续只调floor、temperature或loss weight。

## Source Audit And Claim Boundary

本次检索日期为2026-07-17，采用external-first primary-source policy。详细记录见`source_matrix.csv`。

- Bontempi/Ben Taieb与Stratify已经覆盖multi-output dependency及固定block-size `DirMO-$\sigma$`；SIFF不能claim
  “首次研究output coupling scale”，只能claim一个不按requested $H$选strategy、在同一full-domain decoder中共享并
  索引多种coupling operators的实现链；
- CViT、conditioned neural fields与HyperDeepONet已经覆盖coordinate conditioning和generated parameters；SIFF的
  `scale coordinate`、basis或weight generation本身都不新；
- BASE、Expert Choice与Selective Sinkhorn Routing已经覆盖balanced assignment、fixed expert capacity和OT routing；
  MCCA不能claim Sinkhorn、双marginal或anti-starvation本身；
- Expert Loss Integration、AME-TS、MoHETS与specialization regularizers覆盖direct expert loss、structure-guided
  routing、heterogeneous experts及orthogonality/variance。它们分别进入controls或claim exclusions。

[Novelty Boundary] Step 6只保留下列完整链作为可检验candidate：

1. `fixed past -> one full future function -> internal coupling-scale coordinate -> continuously shared scale-indexed history
   modes -> point/block/global scope reads -> prefix crop`；
2. `all-prefix risk measure -> natural-target transported capability -> same total scope skill mass as PCC -> competitive
   target placement by I-projection -> one-stage arm/router co-training`。

## SC1-SIFF-v1 Frozen Method

### Tensor Contract

设A6-natural Encoder输出`hidden [B,C,R]`。固定$Q=2,D=4,K=256$，其中$R$继续由各dataset natural profile决定：

$$
W\in\mathbb R^{Q\times D\times R\times K},\qquad
b\in\mathbb R^{Q\times D\times K}.
$$

对内部scope $s\in\{1,48,144,360,720\}$定义

$$
z_s=\frac{\log s}{\log T},\qquad
\phi(s)=[1,\tilde z_s],
$$

其中$\tilde z$在五个scopes上zero-mean、unit-RMS。forward为：

```text
hidden [B,C,R]
 -> component modes [B,C,Q,D,K]
 -> phi [S,Q] contraction
 -> scale-indexed modes [B,C,S,D,K]
 -> existing scope pooling and shared target synthesis
 -> arms [B,C,S,T]
 -> target-wise policy [B,C,T,S]
 -> fused full forecast [B,T,C]
 -> prefix crop only
```

$Q=1$或linear component为零时精确退化为current PCSD history field；requested $H$、dataset ID和future truth均不
进入field或policy。该设计不是five independent decoders，也不增加residual forecast path。

### Why Q=2

$Q=2$是能表达有序scale contrast的最小版本。quadratic/cubic basis会在只有五个训练scopes时迅速接近离散expert
bank，增加“只是容量”与过拟合解释；因此Step 6禁止搜索$Q$。$K=256$沿用PCSD/A6 operator rank，不因dataset或
validation结果改变。

### Capacity Is Attribution, Not Selection

用户规则保持不变：parameter count不参与dataset profile或candidate选择。但SIFF新增参数必须被controls解释。

| Dataset | $R$ | SIFF field params | matched Q1 rank / gap | matched independent rank / gap |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | 1536 | 3,517,136 | 463 / 0.0902% | 109 / 0.2501% |
| ETTh2 | 768 | 1,944,272 | 430 / 0.0860% | 116 / 0.3892% |
| ETTm1 | 768 | 1,944,272 | 430 / 0.0860% | 116 / 0.3892% |
| ETTm2 | 3072 | 6,662,864 | 485 / 0.0319% | 106 / 0.0789% |
| Weather | 768 | 1,944,272 | 430 / 0.0860% | 116 / 0.3892% |

`SIFF_CONST`保留完全相同的$Q=2$ tensor storage和active computation，但令两个basis columns恒定；其function可合并
回Q1。`SIFF_PERMUTED_SCALE`打乱$\tilde z_s$与真实pooling scope的对应。`PCSD_Q1_WIDE`和
`INDEPENDENT_SCOPE_MATCHED`按上表公式自动计算rank，不进行dataset tuning。

## SC2-MCCA-v1 Frozen Method

### Same-Mass Redesign Of PCC

PCC从dense prefix risk得到transported capability $c_{nts}$，其中$n$合并batch和channel，$t$为natural target，
$s$为scope。训练进度$\alpha\in[0,1]$沿用原25% ramp：

$$
\bar c_{nts}=(1-\alpha)\frac1S+\alpha c_{nts}.
$$

令projective row mass为$a_{nt}=\omega_t/N$。current PCC的skill credit等价于

$$
A^{\mathrm{PCC}}_{nts}=a_{nt}\left(0.8\bar c_{nts}+\frac{0.2}{S}\right).
$$

因此它的scope总skill mass是

$$
\rho_s=0.8\sum_{n,t}a_{nt}\bar c_{nts}+\frac{0.2}{S}.
$$

MCCA**不改变这个总量**，只把同一$\rho_s$重新放到targets上：

$$
A^*=\arg\min_{A\ge0}\mathrm{KL}\left(A\middle\|a\bar c\right),
\quad
\sum_sA_{nts}=a_{nt},
\quad
\sum_{n,t}A_{nts}=\rho_s.
$$

这使PCC与MCCA成为严格matched comparison：每个scope获得同样总credit；PCC在每个target均匀撒floor，MCCA则在
global coverage约束下尽量保留capability的竞争结构。$p_{nts}=A^*_{nts}/a_{nt}$同时作为skill weight和router KL
target；assignment、capability与marginals全部stop-gradient，fused measure loss保持不变。

### Numerical Policy

- rows在当前mini-batch内部展开为`(B*C*T) × S`，不跨batch、不跨dataset；
- 直接对positive reference measure做log-domain Sinkhorn/I-projection，不再引入额外entropic temperature；
- `kernel_floor=1e-8`、固定64 iterations、float32 terminal row/column gap必须不超过`2e-5`；
- solver failure必须停止该step并标记`optimization_or_numeric_pathology`，禁止静默fallback到PCC或uniform；
- assignment只在training loss存在，inference graph与PCSD/SIFF完全不变。

synthetic full-$T=720$检查中，float64/float32最大marginal gap分别为`3.86e-10/1.04e-7`；MCCA与PCC的column
mass gap分别为`5.55e-17/2.98e-8`。在完全相同marginals下，MCCA相对PCC mixed credit的reference-KL降低
`0.107352`，minimum scope mass为`0.175609 > 0.04`。

## Frozen Experiment Logic

### Core $2\times3$ Factorial

| Architecture | EQUAL_SKILL | PCC_MIXED | MCCA |
| --- | --- | --- | --- |
| PCSD | `PCSD_EQUAL` | `PCSD_PCC` | `PCSD_MCCA` |
| SIFF | `SIFF_EQUAL` | `SIFF_PCC` | `SIFF_MCCA` |

该矩阵同时回答：SIFF是否在相同training下有architecture主效应；MCCA是否在相同architecture下超过same-mass PCC；
joint arm是否存在非冗余interaction。`EQUAL_SKILL`不是MCCA最近邻，但用于验证两种architecture在统一直接监督下的
基本trainability。

### Mandatory Attribution Controls

- architecture：`SIFF_CONST_MCCA`、`SIFF_PERMUTED_SCALE_MCCA`、`PCSD_Q1_WIDE_MCCA`、
  `INDEPENDENT_SCOPE_MATCHED_MCCA`、`DENSE_SIFF_MATCHED`；
- training：`PCSD_POINTWISE_MCCA`、`PCSD_UNIFORM_BALANCED_OT`；
- locked references：A6-LBF、plain PCSD、prior PCSD-matched dense；只有config、checkpoint和metric hash全部一致
  时才允许复用旧seed2021 reference，否则重跑；
- Phase A固定五dataset、seed2021、validation dense-H1..720 MSE AUC、best-val-H720 checkpoint；预计55个new runs；
  confirmation seeds、remote与test均未授权。

## Narrative And Effectiveness Gates

### Contribution 1 Gate

SIFF必须在至少3/5 datasets显示architecture main effect且macro gain至少0.3%，并且不能被`CONST`、permuted、Q1-wide、
dense或independent-scope controls解释。若仅independent bank同样有效，最多保留“heterogeneous scopes”工程结论，不能
claim continuous scale-indexed field。

### Contribution 2 Gate

MCCA必须在至少3/5 datasets超过**same-mass PCC**且macro gain至少0.2%，同时超过generic uniform balanced OT和
pointwise MCCA。若MCCA只优于equal而不优于PCC，则exact contribution失败，因为“更多/更均匀skill mass”仍可解释。

### Joint Gate

`SIFF_MCCA`还必须相对A6至少3/5 wins、macro gain至少0.3%，pairwise diversity retention不低于50%，policy entropy
不低于0.3且maximum usage不高于0.9。两项单独main effect任一失败时，不得依赖positive interaction同时宣称两项
贡献；应回到对应Step4。

## Failure Attribution

[Fact] 当前只验证了tensor/function containment、matched parameter controls、same-mass identity、Sinkhorn numeric
path和experiment completeness，未实例化production SIFF/MCCA model，也未读取dataset或test。

[Uncertainty] SIFF可能仍被generic width或independent experts解释；MCCA可能因batch-dependent moving target而不稳定，
也可能在真实capability近uniform时几乎等于PCC。任何这类失败首先归到exact design，不自动否定“coupling-scale
identifiability”或“competitive projective credit”方向；但必须按冻结rollback返回Step4/5，不能继续堆机制。

decision=`step6_conditional_narrative_pass_step7a_local_next`。
