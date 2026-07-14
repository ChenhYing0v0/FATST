# StageC Frozen Replacement Fairness Audit

## Decision Summary

| Field | Decision |
| --- | --- |
| `audit_date` | 2026-07-14 |
| `current_step` | protocol correction；rollback from D7 Step 10 to PLGO Step 6/7A |
| `problem` | frozen A6 Encoder与A6 decoder存在co-adaptation，replacement-head gap不具备end-to-end公平性 |
| `main_correction` | D7 free-M0 gap只能衡量A6-representation compatibility，不能判定PAF method readiness |
| `evidence_retained` | GEO vs PERM/RANDOM matched within-family geometry effect，conditional on frozen A6 memory |
| `candidate_status` | `SC1-PLGO-PAF narrative_ready_reopened_for_end_to_end_gate` |
| `next` | Step 6冻结end-to-end contract；Step 7A实现；随后D8-E2E五dataset screen |
| `rollback_if_fail` | stable joint-training PAF仍失败且matched controls解释geometry -> Step 4 redesign |

## 1. Audit Question

本审计区分两个不同问题：

1. 给定已经与A6 decoder共同训练的`memory [B,C,P,D]`，哪个replacement head更容易使用它？
2. 当Encoder与Decoder从相同初始化类别端到端共同学习时，哪种完整架构更有效？

D2-D7回答第一个问题。它们不能替代第二个问题，因为A6 Encoder并非task-agnostic feature extractor，而是
在A6 decoder与forecast loss的梯度作用下形成。free-M0/A6-compatible head天然处于in-distribution interface，
PAF则被要求在不改变Encoder的情况下适配已有representation。

## 2. Code Evidence

[Fact] `scripts/run_stage_c_sc1_d2_diagnostic.py:329-334`重新实例化A6、加载其checkpoint、设为`eval()`，并对
所有parameters执行`requires_grad_(False)`。D3-D7复用该loader；训练循环的optimizer只接收
`head.parameters()`（同文件450-454行）。

[Fact] Step7B不同。`baselines/timealign_official/train_repo.py:751-752`新建完整model并将
`model.parameters()`全部交给AdamW；runner为每个A6/PMFO/control arm独立训练。因此Step7B是end-to-end
architecture screen，而不是frozen Encoder replacement。旧文档中的“frozen objective/contract”指固定实验
协议，不表示Encoder weights frozen，应避免继续使用该歧义表述。

## 3. External Source Audit

search date：2026-07-14。queries/topics：`split co-adapted neurons frozen transfer`、`fixed feature extractor vs
fine-tuning vs random initialization`。source type：external primary sources；未依赖Zotero覆盖，Zotero presence
本轮未复核。

- Yosinski et al., [How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792),
  NeurIPS 2014。作者报告，迁移性能不仅受feature specialization影响，也受在co-adapted neurons之间拆分网络
  引起的optimization difficulty影响。这直接支持“冻结一半、替换另一半”不能作为无偏architecture gate。
- Kornblith et al., [Do Better ImageNet Models Transfer Better?](https://openaccess.thecvf.com/content_CVPR_2019/html/Kornblith_Do_Better_ImageNet_Models_Transfer_Better_CVPR_2019_paper.html),
  CVPR 2019。该研究明确把fixed feature extraction、fine-tuning与from-random-initialization作为三种不同
  evaluation settings，并显示fixed-feature transfer对pretraining recipe敏感。因此fixed compatibility与joint
  adaptation不可互换。

[Inference] 两篇工作来自vision transfer而非forecasting，不能直接证明PAF端到端一定成功；它们证明的是
protocol逻辑：frozen replacement performance同时混合representation specialization、interface compatibility与
head quality，因而不足以单独否定完整架构。

## 4. Historical Experiment Reclassification

| Evidence | Training path | Revised validity | False-failure risk |
| --- | --- | --- | --- |
| natural A6 baseline | full end-to-end training；之后只冻结作reference | unchanged | none |
| PMFO-RCT Step7B | 每个arm独立end-to-end joint training | exact PMFO-RCT v1 failure仍成立 | low |
| D1-v2 | frozen A6 counterfactual/probe | 只证明信息在当前A6 memory中可访问；Encoder sufficiency不成立 | medium if generalized |
| D2 | frozen A6 + replacement heads | depth grouping negative只在当前representation/head family成立 | high for direction-level rejection |
| D3 | matched$2\times2$ heads on same frozen memory | basis main effect条件性成立；不证明end-to-end gain | low for conditional claim |
| D4 | matched basis families on same frozen memory | basis ranking/locality attribution条件性成立；不否定balanced end-to-end role | medium if generalized |
| D5 | frozen head selector | 已标记`design_fault_suspected`，边界不变 | already contained |
| D6 | frozen disjoint-window confirmation | support × horizon interaction在A6 representation下成立 | low for conditional problem evidence |
| D7 GEO vs PERM/RANDOM | 同一PAF family、同width、同frozen memory | geometry attribution条件性成立 | low |
| D7 GEO vs free-M0 | A6-co-adapted memory上比较native-compatible与replacement head | **不能评估method readiness** | **high；可能是假失败** |
| Step5/6 algebra/prior art | no forecast optimization comparison | unchanged | none |

### D7 correction

raw values`-37.38%/-39.10%`没有算错；错误在于将其解释为“descriptor-only PAF architecture不ready”。它最多
证明：

> 在A6 decoder共同塑造的frozen representation上，PAF不能作为drop-in head达到free-M0的compatibility。

它没有证明PAF Encoder-Decoder joint training后的function class、optimization或generalization失败。因此原
`descriptor_geometry_supported_paf_not_ready_return_step4`更正为
`conditional_geometry_supported_end_to_end_gate_required`。当前不能断言PAF已经成功，也不能断言其失败；
“是否是假失败”必须由end-to-end实验回答。

## 5. Revised Research Direction

不立即设计capacity-preserving geometry patch。先回到PLGO Step 6，冻结一个真正end-to-end的PAF contract，
然后进入Step 7A/D8-E2E：

1. A6、GEO、PERM、RANDOM以及compact/matched PAF arms均从random initialization独立训练；
2. 所有Encoder与Decoder parameters可训练，禁止加载A6 Encoder checkpoint作为primary evidence；
3. 五dataset使用已冻结natural profiles；共享objective、optimizer class、checkpoint selection与validation-only
   evaluation；在同一runner中重跑A6，不直接借用旧test reference；
4. seed2021完成35-run screen；只有通过预注册gate的decisive arms才进入五dataset三seed confirmation；
5. frozen cross-swap可在完整模型训练后作为secondary attribution，但不是next gate。

## 5.1 Patch-Level Interface Audit

[Fact] 当前`memory [B,C,P,D] -> hidden [B,C,R]`由`flatten(start_dim=-2)`完成，$R=P D$。五profiles的
$(P,D,R)$分别为ETTh1 $(24,64,1536)$、ETTh2 $(12,64,768)$、ETTm1 $(24,32,768)$、ETTm2
$(48,64,3072)$与Weather $(12,64,768)$。flatten是固定顺序的bijective reshape，不是pooling，patch identity与
全部elements均保留。

A6与PAF的真正tensor boundary是：

$$
\begin{aligned}
h &= \operatorname{vec}(M)\in\mathbb{R}^{R},\\
c_{A6} &= W h+b,\quad W\in\mathbb{R}^{256\times R},\\
z_{PAF} &= A h+a,\quad A\in\mathbb{R}^{256\times R},\\
\alpha_j &= \psi(d_j)^Tz_{PAF}+b_j.
\end{aligned}
$$

将$A=[A_1,\ldots,A_P]$按patch切块，可得$z_{PAF}=\sum_p A_pm_p+a$。因此flatten Linear与“每个patch
使用独立$A_p$后求和”exact equivalent；每个atom对patch $p$的effective map为$\psi(d_j)^TA_p$。当前设计并非
看不见patch，而是把sample-to-atom interaction限制为shared latent上的separable bilinear form。

[Risk] A6与PAF都把$R$压到256维，所以bottleneck不是PAF独有。PAF额外受descriptor-generated atom map约束；
如果不同future supports确实需要sample-dependent patch routing，shared latent可能不足。但B14对unit-specific
retrieval只有1/6 settings、0/3 datasets通过，Step6也已因source overlap禁止atom-to-memory attention，当前没有
足够证据直接加入patch retrieval。

[Decision] D8不增加未经narrative gate的patch-attention arm，而增加强制interface evidence：

1. local gate验证flatten与patch-block sum数值等价；
2. 确认每个patch block到shared latent的gradient finite/nonzero；
3. trained A6/PAF报告per-patch latent contribution、patch-block norm/entropy与effective atom-patch Jacobian diversity；
4. 若E2E PAF失败且这些统计显示相对A6发生patch collapse，再返回Step4提出patch-aware candidate；否则失败首先
   归于exact shared-latent descriptor generator，而不是“flatten丢信息”。

## 6. Outcome Interpretation

| End-to-end outcome | Decision |
| --- | --- |
| GEO接近/超过A6，且超过PERM/RANDOM | D7 free-gap确认是假失败；PAF进入3-seed effectiveness confirmation |
| GEO大幅关闭D7 gap但仍略低A6，且超过PERM/RANDOM | co-adaptation confound confirmed；geometry有效，exact PAF仍需Step4 redesign |
| GEO仍弱于A6，但超过PERM/RANDOM，patch usage未collapse | D7 geometry不是假阳性；exact shared-latent generator仍不足，Step4 redesign justified |
| GEO仍弱于A6且patch contribution/Jacobian明显collapse | `intervention_point_wrong` plausible；Step4审计patch-aware interface，不拒绝PLGO family |
| GEO约等于PERM/RANDOM | frozen geometry signal可能是A6-representation-specific；收缩geometry claim |
| divergence/epoch-cap/validation instability | `optimization_or_numeric_pathology`；只否定protocol，不能拒绝方向 |

## 7. Self-Critique And Uncertainty

[Uncertainty] co-adaptation风险足以使旧method gate无效，但不足以估计端到端恢复幅度。PAF也可能因受限的
atom-conditioned generator而在joint training下继续失败。

[Counterargument] frozen Encoder能提供严格相同输入，似乎是更“控制变量”的比较；但这里控制掉的正是新
Decoder改变representation learning的主要机制。该设计适合问compatibility，不适合问完整架构。

[Decision] 本次不删除D1-D7 raw artifacts，也不把所有frozen diagnostics判为无效；只更正它们可支持的
claim level。paper-core failure必须等待D8-E2E。
