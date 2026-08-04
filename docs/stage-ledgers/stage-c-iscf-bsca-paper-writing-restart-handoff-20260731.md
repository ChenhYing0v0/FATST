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
4. `docs/paper-drafts/iscf-bsca-problem-formulation-initial-draft.md`；
5. `docs/iscf-bsca-paper-architecture.md`；
6. `analysis/iscf_bsca_intro_evidence_full_search_20260730/result_selection_and_figure_report.md`；
7. `analysis/iscf_bsca_intro_evidence_full_search_20260730/design_and_figure_contract.md`；
8. `analysis/iscf_bsca_intro_concept_figure_20260730/figure_contract_and_qa.md`；
9. `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_bsca_v1_confirmation_step9_10_20260722/step9_10_three_seed_result_and_paper_handoff.md`；
10. `docs/code-explanation/stage-c-iscf-bsca-v1.md`；
11. `docs/paper-mainline.md`；
12. `docs/research-roadmap.md`；
13. `docs/stage-ledgers/stage-c-unified-forecasting-redesign.md`。

只有在下一任务确实需要追溯早期实验时，才按上述文件中的artifact links继续读取
历史analysis。不要从旧D22或旧SIFF路线重新开始研究。

## 1. Current authoritative state

| Field | Content |
| --- | --- |
| `project` | `R_2026_FATST` |
| `stage` | `StageC-UVHF paper consolidation` |
| `handoff_date` | `2026-07-31` |
| `last_updated` | `2026-08-04` |
| `source_commit_before_handoff` | `8e3eb99` |
| `paper_candidate` | exact frozen `ISCF-BSCA-v1` |
| `paper_core_status` | `passed_core_candidate_ready_for_paper_consolidation` |
| `active_workstream` | manuscript writing and paper-facing evidence consolidation |
| `active_section` | Section 3 v0.7 temporarily frozen；Section 4 Method pending author direction |
| `introduction_status` | `v0.9-author-refinement`=`temporarily_frozen_usable` |
| `section3_status` | `v0.7-author-risk-definition-refinement`=`temporarily_frozen_usable` |
| `active_method_search` | none |
| `new_method_implementation_from_writing_thread` | false |
| `new_remote_training_from_writing_thread` | false |
| `new_formal_test_from_writing_thread` | false |
| `parallel_experiment_authority` | use experiment handoff and current mainline；this writing handoff does not expand or revoke it |
| `next_action` | await author direction；when requested, design Method Figure 4 and draft Section 4 in computation-flow order |
| `conditional_next` | align Method and later result claims with the parallel paper-experiments workstream |

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

### Completed: Section 3

Section 3 v0.7已由用户确认并暂时冻结为论文可用版本。Canonical source：

`docs/paper-drafts/iscf-bsca-problem-formulation-initial-draft.md`

冻结范围包括正文、CHPC/CHPD/NCHPD与future-region prediction risk定义、
Figures 2--3 integration及captions。原naive-unified accuracy与Design Requirements
subsections不进入当前manuscript。后续默认不再修改Section 3；只有Section 4或
paper-facing evidence产生明确矛盾且用户显式批准时才解冻。

### Current next: Section 4 pending author direction

用户下一步明确授权后：

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
- 未经实验流Tier B2授权执行test-tuned HPO；
- 根据test结果做per-horizon、per-seed、per-metric或per-cell tuning；
- 恢复closed candidates；
- 为兑现P6而选择性报告favorable results。

如果Section 4或后续结果写作暴露必须修改Section 3或补充实验，先说明明确矛盾，
并分别请求Section 3解冻或实验授权；不得以writing handoff自动扩张scope。

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
3. docs/paper-drafts/iscf-bsca-problem-formulation-initial-draft.md
4. docs/iscf-bsca-paper-architecture.md
5. analysis/iscf_bsca_intro_evidence_full_search_20260730/result_selection_and_figure_report.md
6. analysis/iscf_bsca_intro_evidence_full_search_20260730/design_and_figure_contract.md
7. analysis/iscf_bsca_intro_concept_figure_20260730/figure_contract_and_qa.md
8. analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_bsca_v1_confirmation_step9_10_20260722/step9_10_three_seed_result_and_paper_handoff.md
9. docs/code-explanation/stage-c-iscf-bsca-v1.md
10. docs/paper-mainline.md
11. docs/research-roadmap.md
12. docs/stage-ledgers/stage-c-unified-forecasting-redesign.md

当前权威状态：
- exact ISCF-BSCA-v1 已冻结为 paper-core candidate；
- Introduction v0.9-author-refinement 已由用户确认，可作为暂时固定的可用版本；
- Section 3 v0.7-author-risk-definition-refinement 已由用户确认，可作为暂时固定的可用版本；
- Figure 1 已批准且只作 constructed conceptual illustration；
- Figures 2--3 已批准用于 Section 3，均为 validation-only illustrative evidence；
- Method Figure 4 仅完成设计规划，尚未生成；
- 当前没有 active method search；本writing thread不新增implementation、remote training或formal test授权，并行实验状态以experiment handoff和current mainline为准。

请不要重新讨论或改写Introduction与Section 3，除非后续章节产生明确矛盾并先向我说明、获得解冻同意。下一写作任务等待用户明确指定；若进入Section 4，则先冻结Method Figure 4 contract，再按真实forward tensor flow起草Method，并保持training-only BSCA与inference-time ISCF路径边界。

请使用专业时序预测研究员和高水平期刊作者的标准，严格区分problem、evidence、method和claim boundary。P6中“统一模型优于horizon-specific模型、组件有效、decoder可迁移”是待后续main/ablation/transfer tables兑现的provisional paper-facing claim，不得当作当前已完成证据。

完成任何后续章节后，同步更新paper architecture、paper-mainline、research-roadmap与Stage C ledger。完成最小诚实验证后，按AGENTS.md提交并推送。

请保留并忽略以下untracked目录，不要删除或提交：
- SRP-7C55/
- analysis/phase5_timealign_hss_h1c_capacity_preserving_gate_20260701_partial/
```
