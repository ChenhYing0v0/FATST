# Stage C ISCF-BSCA Paper-Writing Restart Handoff

## 0. Authority and use

本文件是2026-07-31之后新对话继续ISCF-BSCA论文写作工作的current首读入口。
它取代
`docs/stage-ledgers/stage-c-post-d21-d22-restart-handoff-20260720.md`
作为paper-writing handoff，但不删除或改写后者的历史研究记录。并行实验工作使用
`docs/stage-ledgers/stage-c-iscf-bsca-paper-experiments-restart-handoff-20260731.md`。

如果本文件与旧聊天、旧handoff、archive或主线文档的历史段落冲突，以本文件及
三份主线文档顶部的最新cursor为准。

新对话必须严格按以下顺序读取：

1. `AGENTS.md`；
2. 本handoff；
3. `docs/paper-drafts/iscf-bsca-introduction-initial-draft.md`；
4. `docs/iscf-bsca-paper-architecture.md`；
5. `analysis/iscf_bsca_intro_evidence_full_search_20260730/result_selection_and_figure_report.md`；
6. `analysis/iscf_bsca_intro_evidence_full_search_20260730/design_and_figure_contract.md`；
7. `analysis/iscf_bsca_intro_concept_figure_20260730/figure_contract_and_qa.md`；
8. `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_bsca_v1_confirmation_step9_10_20260722/step9_10_three_seed_result_and_paper_handoff.md`；
9. `docs/code-explanation/stage-c-iscf-bsca-v1.md`；
10. `docs/paper-mainline.md`；
11. `docs/research-roadmap.md`；
12. `docs/stage-ledgers/stage-c-unified-forecasting-redesign.md`。

只有在下一任务确实需要追溯早期实验时，才按上述文件中的artifact links继续读取
历史analysis。不要从旧D22或旧SIFF路线重新开始研究。

## 1. Current authoritative state

| Field | Content |
| --- | --- |
| `project` | `R_2026_FATST` |
| `stage` | `StageC-UVHF paper consolidation` |
| `handoff_date` | `2026-07-31` |
| `source_commit_before_handoff` | `8e3eb99` |
| `paper_candidate` | exact frozen `ISCF-BSCA-v1` |
| `paper_core_status` | `passed_core_candidate_ready_for_paper_consolidation` |
| `active_workstream` | manuscript writing and paper-facing evidence consolidation |
| `active_section` | Section 3: Problem Formulation and Empirical Motivation |
| `introduction_status` | `v0.9-author-refinement`=`temporarily_frozen_usable` |
| `active_method_search` | none |
| `method_implementation_authorized` | false |
| `remote_training_authorized` | false |
| `formal_test_authorized` | false |
| `next_action` | draft and freeze Section 3 definitions, statistics, controls, Figures 2--3 captions and claim boundaries |
| `conditional_next` | after Section 3, design Method Figure 4 and draft Method in computation-flow order |

当前目标是把已冻结模型与已有证据组织成连贯论文，不是继续architecture search。
不要因为旧ledger包含大量closed candidates而恢复D17--D24、TSAF、CPSI、SAC、
SPS、FRSC、SCC或RSCC。

## 2. Frozen Introduction

Canonical clean draft：

`docs/paper-drafts/iscf-bsca-introduction-initial-draft.md`

当前版本：

`v0.9-author-refinement`

Highlighted comparison仅供审阅，不是manuscript source：

`docs/paper-drafts/iscf-bsca-introduction-v0.9-highlighted-review.md`

用户已确认当前Introduction可作为暂时固定的可用版本。后续默认不再改写P1--P6，
除非Section 3、Method或main results出现明确矛盾，并且用户同意解冻。

### 2.1 Frozen paragraph roles

1. P1：multi-horizon实际需求、horizon-specific主流协议、少量varied-horizon先例；
2. P2：horizon-specific systems的fragmentation、overlap disagreement与系统冗余；
3. P3：horizon无关future-step-indexed mapping与CHPC；
4. P4：uniform output mechanism、latent-state sharing extent与
   future-region sharing-demand heterogeneity；
5. P5：single-scope对照、ISCF scope-indexed forecast field、
   target-conditioned allocation与BSCA；
6. P6：task/problem、ISCF、BSCA三项贡献及预期paper-facing results。

### 2.2 Important provisional sentence

P6当前写为：

> Experiments across datasets from multiple application domains show that a
> single unified model outperforms separately trained horizon-specific
> forecasters. Component-wise ablations confirm the effectiveness of each
> component, while backbone transfer studies demonstrate decoder portability.

这是用户批准的预期paper-facing claim，不是当前已由完整tables证明的事实。提交前
必须由以下三类完整结果逐项兑现：

1. unified ISCF-BSCA versus separately trained horizon-specific models；
2. core component-wise ablations；
3. decoder transfer across forecasting backbones。

如果对应结果不支持，必须降低相应动词强度。未完成统计检验前不要写
`statistically significant`。

## 3. Frozen terminology and claim boundaries

| Layer | Canonical term | Boundary |
| --- | --- | --- |
| task | `varied-horizon forecasting` | 一个system服务多个nested requested horizons |
| position | `future time step` | 不使用`future coordinate`或`forecast step`作正文中心术语 |
| interface | `horizon-agnostic future-step-indexed mapping` | 不写成先生成max-$T$再crop |
| property | `cross-horizon prefix consistency (CHPC)` | basic system property，不是独立算法创新 |
| problem | `future-region sharing-demand heterogeneity` | 问题层不预先使用scope定义 |
| method dimension | `future-step latent-state sharing scope` | scope描述多个future steps间的state-sharing关系 |
| architecture | `Independent Scope-Conditioned Forecasting (ISCF)` | 单一`scope-indexed forecast field`，不是多个独立models |
| integration | `target-conditioned scope allocation` | 沿scope轴weighted contraction，不包装成generic router |
| training | `Balanced Scope Co-Adaptation (BSCA)` | ISCF-specific train-only objective，不claim generic KL novelty |

保持以下理论边界：在fixed history、pointwise MSE且requested horizon不携带额外
信息时，同一future step的Bayes conditional mean不依赖requested horizon。本文
讨论finite-capacity output-side sharing，不讨论H-conditioned Bayes target。

不要声称：

- 所有已有unified models都缺少CHPC；
- CHPC本身是算法创新；
- short horizon必然对应fine scope、long horizon必然对应broad scope；
- canonical contiguous scopes已被证明优于random partition；
- ISCF的多个scopes是多个独立forecasting models；
- BSCA证明了semantic expert specialization或universal gain。

## 4. Figure inventory and numbering

### Figure 1: Introduction concept figure

- role：只做概念说明；
- panels：horizon-specific overlap disagreement；region-dependent sharing demand；
- source：`analysis/iscf_bsca_intro_concept_figure_20260730/`；
- manuscript assets：`paper-figures/figure_intro_conceptual_problem.{svg,pdf,png,tiff}`；
- status：`approved_for_manuscript_draft`；
- boundary：constructed illustration，不是empirical evidence；
- note：caption已按用户决定精简，图内footer仍声明`not empirical data`。

### Figure 2: Section 3 prefix-disagreement evidence

- selected case：`ETTh2 / DLinear / origin805 / channel0`；
- role：展示horizon-specific models在共同future steps上的material disagreement；
- source：`analysis/iscf_bsca_intro_evidence_full_search_20260730/selected_figures/figure_intro_prefix_disagreement.*`；
- stable assets：`paper-figures/figure_intro_prefix_disagreement.*`；
- boundary：validation-only maximum illustrative example；不估计prevalence；
- required disclosure：maximum validation selection、shared-96 raw differences、
  all-validation NCHPD与test未访问。

### Figure 3: Section 3 sharing-demand evidence

- selected case：`ETTm2 / origin4177`；
- role：展示matched fixed sharing extents的region-wise risk ordering发生变化；
- source：`analysis/iscf_bsca_intro_evidence_full_search_20260730/selected_figures/figure_intro_sharing_heterogeneity.*`；
- stable assets：`paper-figures/figure_intro_sharing_heterogeneity.*`；
- boundary：validation-only descriptive diagnostic；
- required disclosure：five-scale winners、10/10 qualified crossings、
  `8.112%` descriptive headroom不是learned out-of-sample gain。

### Planned Figure 4: Method overview

Figure 4尚未创建。它应放在Method开篇，而不是作为Introduction第二张图。冻结的
visual contract为：

1. single-scope forecasting；
2. independent history projections and multiple sharing scopes；
3. `scope_field:[B,C,T,S]`与`allocation:[B,C,T,S]`；
4. weighted contraction得到`forecast:[B,T,C]`；
5. BSCA以train-only虚线路径表示，inference graph不增加module。

在Figure 4生成并冻结编号前，Introduction P5不加入空forward reference。

## 5. Method and evidence status

### 5.1 Frozen method

ISCF-BSCA-v1的forward与ISCF-v0一致：

- `arm_forecasts:[B,C,T,5]`；
- `policy:[B,C,T,5]`；
- `fused_forecast:[B,T,C]`。

BSCA只改变training objective：

- uniform arm-skill supervision；
- dense-prefix-measure-weighted normalized `KL(uniform || policy)`；
- route weight在前25% optimizer progress从0 ramp到0.1；
- 不增加parameter、requested-H input或inference operation。

Canonical code-facing explanation：

`docs/code-explanation/stage-c-iscf-bsca-v1.md`

Frozen configs：

- `configs/stage_c_iscf_bsca_v1.json`；
- `configs/stage_c_iscf_bsca_v1_confirmation.json`。

### 5.2 Completed BSCA evidence

Three-seed official-test confirmation完整：

- macro MSE gain versus ISCF-EQUAL：`+0.3541%`；
- macro MAE gain：`+0.3073%`；
- MSE-positive cells：`41/60`；
- positive seed means：`3/3`；
- positive dataset means：`4/5`；
- positive horizon means：`4/4`；
- ETTm2 mean：`-0.6506%`，必须报告；
- cluster-bootstrap interval跨0，不能写universal或statistically conclusive gain。

该证据支持BSCA作为ISCF-native training contribution；不证明P6中的
horizon-specific superiority或decoder portability。后两项仍需正式实验。

## 6. Next work sequence

### Immediate: Section 3

下一对话应先完成
`3. Problem Formulation and Empirical Motivation`，不修改模型。

建议按以下顺序推进：

1. `3.1 Horizon-Specific and Unified Multi-Horizon Forecasting`
   - 定义history、future target、requested horizon与supported horizon set；
   - 分别定义horizon-specific predictors和horizon无关step-indexed predictor；
   - 正式定义CHPC；
   - 区分forecast horizon $H$与future time step $\tau$。
2. `3.2 Cross-Horizon Disagreement of Horizon-Specific Models`
   - 定义CHPD/NCHPD及其source tensors；
   - 说明baseline、validation role、maximum-example selection与controls；
   - 嵌入Figure 2和完整caption；
   - 结论只到“不保证CHPC + system redundancy”，不声称accuracy更差。
3. `3.3 Performance Compromise in Naive Unified Forecasting`
   - 先审计已有证据是否足以保留本节；
   - D18未支持specialists普遍胜过unified carrier，因此不得把unified penalty写成
     已证事实；
   - 若证据不足，保留为待实验设计或缩短，不为叙事强行造结论。
4. `3.4 Future-Region Sharing-Demand Heterogeneity`
   - 定义future regions、sharing extent、matched region risk；
   - 明确neutral/raw-history diagnostic与capacity matching；
   - 嵌入Figure 3和完整caption；
   - 区分descriptive oracle headroom与learned policy gain。
5. `3.5 Design Requirements`
   - 从前三节归纳unified system、CHPC、multiple sharing extents、
     target-conditioned integration与stable joint learning；
   - 本节末尾才能过渡到ISCF-BSCA。

建议新建clean manuscript file：

`docs/paper-drafts/iscf-bsca-problem-formulation-initial-draft.md`

完成后同步：

- `docs/iscf-bsca-paper-architecture.md`；
- `docs/paper-mainline.md`；
- `docs/research-roadmap.md`；
- `docs/stage-ledgers/stage-c-unified-forecasting-redesign.md`。

### Conditional next

Section 3冻结后：

1. 使用`nature-figure`设计并生成Method Figure 4；
2. 按真实forward tensor flow起草Method；
3. 与并行paper-experiments workstream对齐main results、with/without ablations与
   decoder transfer的已冻结protocol和返回结果。

## 7. Authorization and safety boundaries

本handoff只授权论文阅读、写作、定义审计、caption设计、已有artifact分析和
非破坏性验证。

它不自动授权：

- 新method或loss实现；
- 远程训练；
- official-test访问；
- 根据旧test结果做dataset/horizon/cell tuning；
- 恢复closed candidates；
- 为兑现P6而选择性报告favorable results。

如果Section 3写作暴露必须补充的实验，先完成problem/narrative/design gate并在
文档中冻结完整matrix、controls、test role和failure attribution，再请求用户
授权。

## 8. Working-tree preservation

以下untracked目录与当前论文工作无关，必须原样保留、忽略，不得删除或提交：

- `SRP-7C55/`；
- `analysis/phase5_timealign_hss_h1c_capacity_preserving_gate_20260701_partial/`。

## 9. Copy-ready startup prompt

```text
请在 /Users/river/PaperResearch/Project/R_2026_FATST 中继续 ISCF-BSCA 论文工作。

首先严格阅读并遵守仓库 AGENTS.md，然后按顺序完整阅读：
1. docs/stage-ledgers/stage-c-iscf-bsca-paper-writing-restart-handoff-20260731.md
2. docs/paper-drafts/iscf-bsca-introduction-initial-draft.md
3. docs/iscf-bsca-paper-architecture.md
4. analysis/iscf_bsca_intro_evidence_full_search_20260730/result_selection_and_figure_report.md
5. analysis/iscf_bsca_intro_evidence_full_search_20260730/design_and_figure_contract.md
6. analysis/iscf_bsca_intro_concept_figure_20260730/figure_contract_and_qa.md
7. analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_bsca_v1_confirmation_step9_10_20260722/step9_10_three_seed_result_and_paper_handoff.md
8. docs/code-explanation/stage-c-iscf-bsca-v1.md
9. docs/paper-mainline.md
10. docs/research-roadmap.md
11. docs/stage-ledgers/stage-c-unified-forecasting-redesign.md

当前权威状态：
- exact ISCF-BSCA-v1 已冻结为 paper-core candidate；
- Introduction v0.9-author-refinement 已由用户确认，可作为暂时固定的可用版本；
- Figure 1 已批准且只作 constructed conceptual illustration；
- Figures 2--3 已批准用于 Section 3，均为 validation-only illustrative evidence；
- Method Figure 4 仅完成设计规划，尚未生成；
- 当前没有 active method search，新implementation、remote training和formal test均未授权。

请不要重新讨论或改写 Introduction，除非后续章节产生明确矛盾并先向我说明。当前第一任务是推进 Section 3: Problem Formulation and Empirical Motivation：
1. 完成3.1 task formulation与CHPC正式定义；
2. 完成3.2 horizon-specific prefix disagreement，定义CHPD/NCHPD并整合Figure 2；
3. 审计3.3 naive unified forecasting是否有足够证据，证据不足时不得强写成已证事实；
4. 完成3.4 future-region sharing-demand heterogeneity，定义matched statistics/controls并整合Figure 3；
5. 完成3.5 design requirements，之后才引出ISCF-BSCA。

请使用专业时序预测研究员和高水平期刊作者的标准，严格区分problem、evidence、method和claim boundary。P6中“统一模型优于horizon-specific模型、组件有效、decoder可迁移”是待后续main/ablation/transfer tables兑现的provisional paper-facing claim，不得当作当前已完成证据。

建议新建 docs/paper-drafts/iscf-bsca-problem-formulation-initial-draft.md，并同步更新paper architecture、paper-mainline、research-roadmap与Stage C ledger。完成最小诚实验证后，按AGENTS.md提交并推送。

请保留并忽略以下untracked目录，不要删除或提交：
- SRP-7C55/
- analysis/phase5_timealign_hss_h1c_capacity_preserving_gate_20260701_partial/
```
