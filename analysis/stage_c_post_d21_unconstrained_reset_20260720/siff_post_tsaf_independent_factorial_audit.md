# SIFF Post-TSAF Independent-Field Factorial Audit

## 1. Long-stage record

| Field | Record |
| --- | --- |
| `audit_date` | `2026-07-21` |
| `current_step` | SIFF-first Step 2/4 existing-artifact problem audit |
| `parent` | immutable `SC1-SIFF-v2-EQ-ATTR-v1` |
| `audited_lead` | `siff_independent_target_only` single-seed weak signal |
| `problem` | 该弱信号是否证明“independent scale field与target-only policy存在稳定正interaction”，从而足以提出新的SIFF-first paper problem？ |
| `existence_evidence` | historical/new E2E validation与official-test artifacts构成一个不完全parameter-matched $2\times2$ factorial |
| `idea` | 仅审计`field sharing × policy information` interaction；不设计新method、loss或router |
| `theory_check` | fixed past、pointwise MSE且policy不获得额外信息时，去掉history只能改变finite-capacity/regularization，不能改变Bayes target |
| `design` | ordered-Q2/direct、ordered-Q2/static、independent-Q5/direct、independent-Q5/static；MSE/MAE log-ratio interaction；validation/test transfer；same-rank sensitivity |
| `narrative_gate` | fail：generic independent experts、multi-scale gating与query-specific expert selection已有强prior，且本地interaction不稳定 |
| `effectiveness_gate` | not applicable；本轮不创建method candidate，也不重新访问test |
| `artifacts` | 两次已完成E2E matrices的metrics、configs、initialization与training logs |
| `decision` | `independent_target_only_weak_lead_not_supported_for_step4` |

## 2. Executive decision

[Fact] `siff_independent_target_only`相对SIFF-v2 parent的official-test MSE/MAE为
`+0.2383%/+0.0898%`，但它不是“只移除history-conditioned policy”的单因素arm。它同时：

1. 把SIFF-v2的ordered rank-2 scale basis改为五个identity/independent scale fields；
2. 把`direct` history-conditioned policy改为`static-target` policy；
3. 在ETTh2、ETTm1与Weather把independent field rank从116改为115，以匹配TSAF而非parent的active parameters。

[Strong Evidence] 现有四个E2E arms构成一个可审计但部分rank-confounded的$2\times2$ factorial。全20-cell
official-test interaction看似为MSE `+0.5265%`、MAE `+0.4246%`；然而在rank严格相同的ETTh1+ETTm2八个
cells上，interaction变为MSE `-0.3097%`、MAE `-0.1175%`。全矩阵正值主要来自三个rank发生变化的数据集，
且dataset/split方向不稳定。

[Decision] 该weak lead不足以授权Step 4新候选。它不能证明target-only allocation、independent scale experts或
二者interaction是新的paper problem；也不能把预注册control post-hoc升级为method。SIFF-v2保持immutable
paperization parent；TSAF-v1保持关闭；不补seed、rank、readout、loss或router rescue，不启动remote training。

## 3. Tensor and function-class audit

四个arms的关键合同如下：

| Field family | Policy | Arm | Scale basis | Policy input | Rank |
| --- | --- | --- | --- | --- | --- |
| ordered | direct | `siff_equal` / SIFF-v2 parent | $Q=2$，`[1, normalized log scale]` | history state + target coordinate | 256 |
| ordered | static | `siff_categorical_target_only` | 同一$Q=2$ ordered basis | target coordinate；history state置零 | 256 |
| independent | direct | `siff_independent_equal` | $Q=5$ identity basis；每个scale独立field | history state + target coordinate | 109/116/116/106/116 |
| independent | static | `siff_independent_target_only` | 同一$Q=5$ identity basis | target coordinate；history state置零 | 109/115/115/106/115 |

rank按`ETTh1/ETTh2/ETTm1/ETTm2/Weather`顺序列出。由此可见：

- `ordered direct -> ordered static`是干净的history-policy removal control；
- `ordered direct -> independent direct`审计field sharing/function class，但rank因parameter matching而改变；
- `independent direct -> independent static`只在ETTh1、ETTm2上严格同rank；
- 新weak lead不是任何一个main effect，而是两个变化同时出现后的结果。

## 4. Statistics

对metric $m\in\{\mathrm{MSE},\mathrm{MAE}\}$、dataset $d$与horizon $h$，定义static policy相对direct
policy的log improvement：

$$
E_{f,d,h}^{(m)}=100\log\frac{m_{f,\mathrm{direct},d,h}}
{m_{f,\mathrm{static},d,h}},
$$

其中field family $f\in\{\mathrm{ordered},\mathrm{independent}\}$。正值表示static更好。定义factorial
interaction：

$$
I_{d,h}^{(m)}=E_{\mathrm{independent},d,h}^{(m)}-
E_{\mathrm{ordered},d,h}^{(m)}.
$$

正值表示“移除history policy”在independent field下比ordered field下更有利。报告量包括：

- `macro interaction`：对dataset-horizon cells等权平均$I$；
- `interaction-positive cells`：$I>0$的cell数，只作方向稳定性，不作显著性检验；
- `same-rank sensitivity`：只保留ETTh1与ETTm2；
- `validation/test transfer`：同一冻结checkpoint在validation与official test上的符号和量级一致性。

聚合metrics不能重建sample-level paired uncertainty，因此本报告不作$p$值、confidence interval或
“statistically significant”声明。

## 5. Controls and split roles

### 5.1 Controls

1. `ordered direct`是immutable parent；
2. `ordered static`隔离history-conditioned policy main effect；
3. `independent direct`隔离independent function-class main effect；
4. `independent static`是被审计的组合arm；
5. `same-rank sensitivity`排除116到115的一阶rank变化，但只覆盖2/5 datasets；
6. dataset/horizon cell map防止macro掩盖异质性。

### 5.2 Validation and test

- validation仍只解释checkpoint/transfer behavior，不决定method pass；
- official test metrics来自两次已经发生、已经记录为`test_informed`的完整formal audits；本轮只读复用，未重新
  运行evaluator或选择checkpoint；
- test不能用于按dataset/horizon选择新结构；因此任何局部positive只能形成未来pre-registered问题，不能直接
  晋升method；
- historical direct-independent与new static-independent来自不同但protocol/profile hash相同的matrices；
  encoder initialization按各自matrix匹配，不能宣称四arm完全paired initialization。

## 6. Results

### 6.1 Main effects already rule out simple explanations

本小节沿用既有analyzer的arithmetic percentage gain $100(1-m_{\rm new}/m_{\rm ref})$；6.2起的factorial
interaction使用第4节定义的symmetric log ratio，因此数值会有轻微差异。

| Comparison | Validation MSE | Test MSE | Interpretation |
| --- | ---: | ---: | --- |
| ordered static vs ordered direct | `+0.2951%` | `-0.0246%` | 干净history removal不transfer |
| independent direct vs ordered direct | not used as causal main effect | `-0.2660%` | independent field alone不优于parent |
| independent static vs independent direct | `+0.4123%` | `+0.4969%` | 表面正向，但3/5 datasets有rank变化 |
| independent static vs ordered direct | positive but below final gate | `+0.2383%` | 低于预注册`+0.3%`且不是单因素 |

MAE给出同样的因果边界：ordered static vs parent test为`-0.0397%`，independent direct vs parent为
`-0.2953%`，而组合arm vs parent仅`+0.0898%`。

### 6.2 Factorial interaction

| Split / scope | MSE interaction | positive cells | MAE interaction | positive cells |
| --- | ---: | ---: | ---: | ---: |
| validation，all 20 | `+0.1237%` | 10/20 | `-0.1253%` | 12/20 |
| test，all 20 | `+0.5265%` | 12/20 | `+0.4246%` | 13/20 |
| validation，same-rank 8 | `+0.0090%` | 5/8 | `+0.0209%` | 6/8 |
| test，same-rank 8 | `-0.3097%` | 4/8 | `-0.1175%` | 4/8 |
| test，rank-confounded 12 | `+1.0839%` | 8/12 | `+0.7859%` | 9/12 |

[Strong Evidence] 严格same-rank子集在validation近零、test为负；而rank-confounded子集承担几乎全部positive
interaction。即使一阶rank变化未必必然导致收益，它也足以阻止归因。

### 6.3 Dataset transfer

MSE interaction的dataset宏平均如下：

| Dataset | Validation | Test | Rank matched? | Audit |
| --- | ---: | ---: | --- | --- |
| ETTh1 | `+0.0446%` | `+0.1955%` | yes | weak positive，量级小 |
| ETTh2 | `-0.8542%` | `-0.4287%` | no | stable negative |
| ETTm1 | `+1.6570%` | `+1.4357%` | no | positive但rank-confounded |
| ETTm2 | `-0.0267%` | `-0.8150%` | yes | exact matched negative on test |
| Weather | `-0.2023%` | `+2.2447%` | no | clear split reversal |

MAE更弱：validation all-cell interaction为负；Weather从validation `-1.5065%`反转到test `+1.2220%`。
不存在跨五dataset统一的interaction。

### 6.4 Horizon behavior

test MSE interaction在H96/H192/H336/H720分别为`+0.5428%/+0.4425%/+0.7233%/+0.3972%`，但positive
cell仅`3/5、3/5、4/5、2/5`。validation对应为`+0.0668%/+0.0937%/+0.3045%/+0.0297%`。
量级不随lead time单调变化，也没有支持统一target-scale frontier。

### 6.5 Checkpoint and optimization health

四个arms在五datasets上均以相同four-horizon validation selector early-stop，所有logs finite，且都触发预期
early stopping。组合arm的best epoch为ETTh1/ETTh2/ETTm1/ETTm2/Weather=`3/2/10/1/9`；对照arms亦覆盖
epoch 1到11。没有divergence、NaN、失效selector或单一epoch-cap saturation。

[Decision] 因此不能用`optimization_or_numeric_pathology`解释negative/unstable interaction；但也不能把正常
optimization当作interaction成立的证据。

## 7. Latest external primary-source audit

检索日期：`2026-07-21`。query scope包括`time-series forecasting multi-scale experts`、`target/query-specific
expert selection`、`multi-scale gating`与`shared plus input-dependent residual`。来源使用PMLR、arXiv与
OpenReview primary pages；以下新增项均由external search发现，Zotero presence未在本次delta中审计。

| Primary source | Covered mechanism | Boundary for a possible SIFF successor |
| --- | --- | --- |
| [MoLE, AISTATS 2024](https://proceedings.mlr.press/v238/ni24a.html) | forecast experts + end-to-end input-dependent router | generic expert fusion/router不是贡献 |
| [TimeExpert, 2025 preprint](https://arxiv.org/abs/2509.23145) | query-specific local temporal experts + shared global expert | target/query-specific expert selection已有直接覆盖 |
| [MoHETS, 2026 preprint](https://arxiv.org/abs/2601.21866) | heterogeneous convolution/Fourier experts、patch routing、single-model arbitrary horizons | heterogeneous/multi-scale experts与arbitrary-H不是贡献 |
| [M²FMoE, AAAI 2026](https://arxiv.org/abs/2601.08631) | spectral-band experts、multi-resolution fusion、temporal gating | frequency/multi-resolution expert fusion高度重叠 |
| [TA-SparseMG, 2026 preprint](https://arxiv.org/abs/2606.27908) | scale-adaptive denoising与multiscale gated-attention prediction head | generic decoder-side multi-scale gating已拥挤 |
| [Self-Gating Attention, 2026 preprint](https://arxiv.org/abs/2607.02344) | shared attention matrix + input-dependent residual | “shared field + adaptive residual”本身也不能作为novelty claim |
| [GMM-TS, withdrawn ICLR 2026](https://openreview.net/forum?id=NS22e2Vgnv) | per-future-time-step expert weights用于multi-modal forecasts | per-step weights有直接但较弱的withdrawn prior；只作边界，不作权威benchmark |

[Strong Evidence] 最新文献没有证明SIFF完整operator chain已被覆盖，但它显著压缩了下一候选的可辩护空间：
“independent scale experts + target/time-step gating”“shared + residual gating”或“再加一个router”都不足以形成
SCI-level贡献。primitive overlap不是自动拒绝；本轮拒绝来自prior pressure与本地existence evidence同时不足。

## 8. Failure attribution

1. `hypothesis_false`：对“independent field × target-only policy具有稳定、可归因正interaction”的exact问题，
   当前证据为not supported；
2. `capacity_control_explains`：3/5 datasets的independent rank变化与positive interaction重合，严格same-rank
   subset不支持interaction，因此capacity/rank confound未被排除；
3. `optimization_or_numeric_pathology`：未发现；所有runs finite并正常early-stop；
4. `intervention_point_wrong` / `readout_or_head_design_wrong`：无法由现有factorial区分，但这只保留更广义方向，
   不授权exact candidate；
5. rejection scope：只关闭把该weak control signal升级为new Step4 candidate的路径，不否定SIFF-v2 parent、
   不否定所有future-coordinate operators，也不把frozen replacement用于方向拒绝；
6. rollback：保持SIFF-first Step 4 paper-claim consolidation，先判断immutable SIFF-v2的窄claim是否足以承载论文，
   再决定是否需要新实验；不回Step7做参数修补。

## 9. Paperization handoff

SIFF-v2现有证据允许保留但必须收紧：

- 可陈述：相对A6_FULL、PCSD_EQUAL、constant、permuted与Q1-wide controls为正，内部多arm路径健康；
- 不可陈述：ordered field严格优于independent fields、history-conditioned policy不可替代、target-only allocation
  已被证明、或SIFF-v2已超过最强A6_MEASURE；
- 不再把TSAF或independent target-only写成第二贡献；
- 下一节点是`SIFF-v2 final paper-claim consolidation`，回答“在不夸大上述失败controls的前提下，完整
  `problem -> scale-indexed output operator -> evidence`链是否仍达到SCI narrative gate”；
- 在该claim gate完成前，不进入modern-baseline performance execution，不实现successor，也不启动remote training。

最终decision：

```text
independent_target_only_weak_lead_not_supported_for_step4
```
