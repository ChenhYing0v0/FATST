# ISCF-v0 Post-CPSI Step 4/5 Scope-Independence Narrative Gate

## 1. Long-stage record

| Field | Record |
| --- | --- |
| `audit_date` | `2026-07-21` |
| `current_step` | exact CPSI-v1关闭后回滚Step 4/5；ISCF-v0 paper-core problem/narrative重审 |
| `problem` | varied-horizon decoder是否需要同时表示多种future-output coupling extents，并避免用单一shared mode map强迫不同extents共适应？ |
| `existence_evidence` | ISCF-v0相对A6_FULL three-seed test MSE/MAE=`+1.3584%/+0.9144%`，5/5 datasets、4/4 horizons、3/3 seeds正向；scope functions与local responses均非退化 |
| `idea` | 不再给independent scopes添加interaction；把ISCF本身收紧为future-output coupling-scope factorization，并用near-matched-width与exact random-partition controls检验scope-specificity和temporal structure |
| `theory_check` | fixed past与pointwise MSE的Bayes target不因requested H改变；ISCF只改变finite-capacity output-sharing bias，不增加future information |
| `design` | exact frozen ISCF-v0；下一步只做Scope Attribution Confirmation（SAC），不修改candidate、不增加loss/router |
| `narrative_gate` | `conditional_pass_as_output_coupling_scope_architecture_pending_sac` |
| `effectiveness_gate` | carrier performance pass vs A6_FULL；scope-specificity attribution pending |
| `artifacts` | FCC、function audit、D1.1、CPSI Step9/10、latest primary-source audit、SAC config与`static_contract_audit.json` |
| `decision` | 保留ISCF-v0为唯一paperization candidate；active paper-core method仍为none，SAC前不promote |

## 2. Executive decision

[Decision] exact CPSI-v1的material failure不支持继续堆叠cross-scope interaction，也不能反向证明“independence必然最优”。但它与FCC共同给出一个更简单、可前瞻验证的paper path：**把ISCF-v0本身作为future-output coupling-scope architecture候选，而不是把它只当SIFF control或下一机制的carrier。**

该路径暂时通过Step 4/5 narrative gate，理由是：

1. [Strong Evidence] exact frozen ISCF-v0相对A6_FULL的three-seed official-test MSE/MAE为`+1.3584%/+0.9144%`，5/5 datasets、4/4 horizons、3/3 seeds正向；
2. [Strong Evidence] ordered SIFF-v2相对ISCF三seed MSE/MAE为`-0.1272%/-0.1733%`，说明收益不需要ordered log-scale field；
3. [Strong Evidence] D1.1中15/15 learned scopes超过direction-null和architecture-identical random-init，private response median=`0.7197`，说明五个scope-specific maps没有训练成随机冗余副本；
4. [Strong Evidence] exact CPSI相对ISCF MSE/MAE=`-2.2128%/-1.6987%`，LINEAR只与ISCF tie，说明当前证据更支持保留简单独立factorization，而不是增加generic interaction；
5. [Fact] 以上结果全部`test_informed`。ISCF最初是matched control，不能仅凭post-hoc正结果直接promote为已通过paper core。

因此下一步不是发明ISCF-v2，也不是启动modern baselines，而是用预先冻结的matched controls回答两个仍阻塞论文归因的问题：

```text
ISCF > Q1-WIDE ?    独立scope maps是否超越near-matched预算的单一shared map
ISCF > RANDOM ?     contiguous/nested future-output partitions是否超越同形随机分组
```

任一问题失败，都必须收窄或关闭ISCF paper-core claim；两项通过才进入modern-baseline/generalization阶段。

## 3. Exact architecture and tensor contract

Encoder输出`hidden [B,C,R]`。ISCF-v0固定$S=5$个coupling scopes
$\{1,48,144,360,720\}$、coordinate dimension $D=4$与dataset-matched rank $K$。

### 3.1 Independent history-to-mode maps

`mode_weight [S,D,R,K]`与`mode_bias [S,D,K]`产生

$$
M_{b,c,s,d,k}
=\sum_r h_{b,c,r}W_{s,d,r,k}+b_{s,d,k},
$$

即`scale_modes [B,C,S,D,K]`。这里的independent只指五个scope拥有独立的history-to-mode affine maps；它们仍共享同一个Encoder、target coordinate field、identity/nonlinear synthesis以及final policy。

### 3.2 Future-output coupling scope

第$s$个scope把$T=720$个future coordinates划分为$T/s$个连续groups。每个group先对固定coordinate field求均值，再用对应`M_s [B,C,D,K]`生成group state：

```text
group coordinates [G_s,D]
M_s               [B,C,D,K]
state_s            [B,C,G_s,K]
```

group内所有coordinates共享同一`state_s`，但由`identity_synthesis [T,K]`和
`nonlinear_synthesis [T,K]`产生各自target value，最终得到`arm_s [B,C,T]`。因此scope不是input downsampling rate，也不是requested horizon；它直接规定**哪些future coordinates在nonlinear synthesis前共享latent state**。

### 3.3 Late fusion and request boundary

五个arms堆叠为`arms [B,C,S,T]`。既有direct policy由history state与target coordinate产生
`weights [B,C,T,S]`，逐target凸融合为`full [B,C,T]`；requested H只在最后crop为`[B,H,C]`。

direct policy、equal-skill objective和full-domain crop都是frozen implementation contracts，不作为第二项创新。SAC不检验新router或第二loss。

## 4. Problem and theory boundary

### 4.1 Paper problem

现有multi-scale forecasting多数从history sampling、patch resolution、frequency band、expert task或forecast horizon分解问题。ISCF研究的是不同对象：一个full-domain decoder在生成future trajectory时，应该让多宽的future-coordinate region共享latent state。

单一global map可能把point/local/block/global sharing biases压进同一个mode space；完全独立的scope maps则允许每种coupling extent在相同Encoder上学习自己的finite-capacity projection，再由同一target-wise policy组合。

### 4.2 Bayes boundary

[Fact] 对fixed past $X$、同一future coordinate $Y_t$与pointwise MSE，Bayes predictor仍为

$$
\mathbb E[Y_t\mid X],
$$

它不依赖用户请求的prefix length。ISCF没有向model提供requested H、future label、oracle error或额外context。

[Hypothesis] ISCF可能有效的唯一合法解释是finite-capacity inductive bias：不同output-coupling partitions对同一history representation施加不同parameter-sharing geometry，独立maps避免它们在一个shared mode map中发生不必要的共适应。

该解释不保证population-risk优势，也不支持“independence普遍优于sharing”。

## 5. Evidence balance

| Evidence layer | Result | Meaning |
| --- | --- | --- |
| paper-facing carrier | vs A6_FULL MSE/MAE `+1.3584%/+0.9144%`；5/5 datasets、4/4 horizons、3/3 seeds | stable package headroom |
| ordered control | SIFF ordered vs ISCF MSE/MAE `-0.1272%/-0.1733%` | ordered scale不是必要解释 |
| frozen function relation | common/private、complementarity、4/5 topology pass；low-rank 0/15 fail | scopes有关系但不是简单ordered/low-rank field |
| label-free response | 15/15超过direction-null/random-init；private median `0.7197` | learned scope-specific maps非随机冗余 |
| interaction attempt | CPSI vs ISCF `-2.2128%/-1.6987%`；LINEAR vs ISCF tie | exact nonlinear interaction不提供增益 |
| scope-specificity control | seed2021 derived ISCF vs Q1-WIDE MSE/MAE=`+0.8980%/+0.6406%`；15/20 cells | promising，但只有single-seed、post-hoc，且active-param gap非零 |
| temporal-structure control | independent random partition尚未测试 | blocking gap |

[Self-critique] “复杂扩展失败、简单base更强”只说明当前实现排序，不证明independence的科学必要性。ISCF还可能只是多分支capacity。Q1-WIDE只能把active-param gap压到`0.4646%`以内而不能exact match；RANDOM才是exact parameter/initialization control。两项必须联合解释。

## 6. Latest primary-source audit

检索日期：`2026-07-21`。query scope包括`multi-scale predictors`、`multi-branch complementary prediction`、`forecasting sub-task specialization`、`multi-resolution experts`、`future prediction mixing`。来源仅使用conference proceedings、OpenReview、PMLR、AAAI与official paper pages；本轮未以Zotero coverage判断novelty，FSA subset presence记为`unknown`。

| Primary work | Covered mechanism | ISCF claim boundary |
| --- | --- | --- |
| [TimeMixer, ICLR 2024](https://arxiv.org/pdf/2405.14616) | average-downsampled history scales、PDM与Future-Multipredictor-Mixing | multiple future predictors与complementary fusion不新；ISCF不得claim generic multi-scale prediction |
| [FreqMoE, AISTATS 2025](https://proceedings.mlr.press/v258/liu25i.html) | frequency-band experts、gating、residual refinement | independent experts与output gating不新 |
| [MAFS, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f34f0630c33be15b8c89426bb8056798-Abstract-Conference.html) | forecasting sub-task specialization，包括不同future horizons/resolutions；agent communication与voting | generic horizon/resolution subtask decomposition与multi-head collaboration不新；ISCF必须限定为单decoder内的output-coupling partition |
| [HMformer, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39355) | hierarchical cross-scale mixing与multi-branch complementary prediction | multi-branch complementarity不构成novelty |
| [M²FMoE, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39362) | multi-view frequency experts、inter-expert collaboration、multi-resolution adaptive fusion | multi-resolution expert coordination与temporal gating高度拥挤 |
| [QDF, ICLR 2026](https://openreview.net/forum?id=vpO8n9AqEG) | future-step quadratic objective与heterogeneous task weights | equal-skill/objective不能包装成第二贡献；本轮不改变loss |

[Decision] 最新文献显著压缩但尚未覆盖以下完整链：

```text
future-output coupling extent as a decoder partition
-> independent history-to-mode maps for each extent
-> shared target synthesis over full T
-> one target-wise fused varied-horizon forecast
-> near-matched shared-width and exact random-partition attribution
```

这只构成`conditional contribution-level novelty`。如果SAC不能证明scope-specific maps与temporal partition结构同时必要，完整链断裂，ISCF应降为strong engineering carrier/control。

## 7. Scope Attribution Confirmation design

冻结candidate=`ISCF-v0`，不修改任何tensor path、rank、objective、policy或training protocol。SAC包含两个primary controls：

### 7.1 Q1-WIDE near-matched-width control

`siff-q1-wide-control`保留五个canonical scope arms、相同synthesis、policy与equal-skill objective，但只用一个shared
history-to-mode map；以dataset-specific wide rank近似匹配ISCF/SIFF active parameter budget。

历史`model_diagnostics.json`给出的signed active-param gap
$100(N_{\rm ISCF}-N_{\rm Q1})/N_{\rm ISCF}$在ETTh1/ETTh2/ETTm1/ETTm2/Weather分别为
`-0.1564%/+0.4582%/+0.4646%/+0.1085%/+0.4582%`，最大绝对值`0.4646%`。因此该control是
near-matched而非exact-matched；报告不得省略这一限制，Q1 primary MSE margin也提高到`+0.5%`。

- 已有seed2021；
- 新增seeds2022/2023 × 5 datasets=`10` trainings；
- primary question：independent scope maps是否超过“一个更宽shared map”。

### 7.2 RANDOM-PARTITION matched-structure control

使用与ISCF完全相同的readout、rank、参数量、policy与objective，只把中间scope的连续groups改为各scope独立的固定随机分组。scale1和scale720端点保持不变，中间48/144/360不再连续且不再nested。

- seeds2021/2022/2023 × 5 datasets=`15` trainings；
- primary question：收益是否来自future temporal contiguity/nesting，而不是五个任意independent heads；
- partition seed全局冻结，不按dataset、horizon或结果选择。

本地静态合同已在conda `r2026-fsa`中验证：canonical与random control的参数量、parameter initialization与global
RNG post-state完全一致；中间scope group indices不同；两者均输出finite `[B,720,C]`且同input forecast非恒等。

合计`25`个new trainings。历史ISCF、A6_FULL、SIFF ordered各15 runs以及Q1-WIDE seed2021五runs只作frozen references。全部25 trainings完成并通过protocol audit后，才允许一次完整official-test access；validation仍只用四horizon mean MSE选择checkpoint。

### 7.3 Frozen gates

两项primary comparison共享dataset/horizon/seed/MAE gates，但MSE macro margin不同：Q1-WIDE至少`+0.5%`，
RANDOM-PARTITION至少`+0.3%`。其余必须同时满足：

1. dataset wins至少`3/5`；
2. horizon wins至少`3/4`；
3. seed macro至少`2/3`为正；
4. MAE macro严格为正；
5. 全部dataset/horizon/seed cells完整报告。

两项都通过，decision=`iscf_scope_architecture_supported_pending_modern_baselines`。Q1-WIDE失败则
`capacity_or_shared_width_explains`；RANDOM失败则`temporal_scope_structure_not_supported`；任一失败均不做rank、seed、partition、loss或router rescue。

## 8. Failure attribution and authorization

- exact CPSI-v1：`hypothesis_false_for_exact_CPSI_v1`，closed；
- broader interaction：仍为direction-level unresolved，但本轮不继续；
- ISCF-v0：`conditional paperization candidate`，不是passed core；
- strongest open confounds：matched width与random partition；
- rollback：若SAC失败，回Step2/portfolio decision，ISCF只保留strong carrier/control；
- model implementation：false；candidate code unchanged；
- remote training：false，未获本SAC matrix新授权；
- formal test：false，必须在25/25 training完成后单次执行并记录新authorization；
- router/second loss/requested-H conditioning：false。

最终decision：

```text
conditional_pass_as_output_coupling_scope_architecture_pending_sac
```
