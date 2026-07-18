# SC1-SIFF-v2-EQ-ATTR-v1 候选冻结与 Step 4 source-informed 改进审计

## 1. 当前研究记录

| Field | Content |
| --- | --- |
| `current_step` | Step 4 source-informed redesign audit |
| `problem` | healthy multi-arm SIFF为什么没有把conditional headroom转成超过`A6_MEASURE`与independent control的fused forecast？ |
| `existence_evidence` | v1相对A6_FULL `+1.6436%` MSE；internal health 7/7；但相对A6_MEASURE `-0.2366%`、相对independent仅`+0.2580%` |
| `idea` | 冻结v1为performance-near candidate，并定位fusion的information/calibration/geometry瓶颈后再定义v2 |
| `theory_check` | 现有router只看到history hidden与future coordinate；equal-skill只保证arms可用，并不使router知道哪一arm当前更可靠 |
| `design` | 复用已保存test probe与row-bin artifacts，做routing calibration及two-fold static fusion realizability audit；不训练、不读新test labels |
| `narrative_gate` | v1保留但仍为`performance_partial_pass_attribution_blocked`；provisional v2尚未通过Step5/6 |
| `effectiveness_gate` | v1原Step9结果不变；任何改动均形成新`test_informed` candidate并重新冻结完整矩阵 |
| `artifacts` | `diagnostics/routing_calibration.csv`、`diagnostics/probe_fusion_capacity.csv`、`diagnostics/summary.json` |
| `decision` | 保留v1；下一步进入以v1为parent的Step4→5 redesign，不做当前v1的盲调参 |

## 2. 候选冻结边界

`SC1-SIFF-v2-EQ-ATTR-v1`现在正式保留为`frozen_performance_near_candidate`。不可变清单见
`configs/stage_c_siff_equal_attribution_v1_candidate_freeze.json`，其中固定source commit、config/profile hash、
五个dataset checkpoints、模型contract与Step9结果。

这一portfolio decision不改写历史证据：v1仍未通过`A6_MEASURE` effectiveness comparison和independent
specificity margin，因此当前不能称为`passed_core_candidate`。保留它的理由是：它是本阶段正式公平矩阵中最接近
可发表performance的结构化decoder，也是下一候选必须超越的parent。

## 3. 代码与梯度路径审计

当前SIFF forward可简化为：

1. Encoder输出`hidden [B,C,R]`；
2. Q=2 scale field将constant component与normalized log-scale component组合成五个scope arms
   `arm_forecasts [B,C,S=5,T=720]`；
3. policy输入仅包含`hidden [B,C,R]`与future coordinate，输出`weights [B,C,T,S]`；
4. final forecast是五个arms的pointwise convex combination；
5. equal-skill objective同时训练fused forecast与五个arms，但没有给policy一个“当前target上哪一arm更可信”的
   direct calibration target。

[Strong Evidence] 这形成一个information mismatch：产生权重的网络在作决定时看不到arms之间的实际预测差异；
而equal-skill降低arm loss CV后，还会进一步减弱仅凭history state识别细微relative competence的信号。

## 4. Offline diagnostic：测试什么、为什么、如何构造

### 4.1 Routing calibration

从每个dataset的完整`arm_row_bin_mse [N,8,5]`与`policy_row_bin_usage [N,8,5]`计算：

- `policy_best_arm_match_rate`：最高权重arm是否等于最低MSE arm；随机五分类参考约为20%，但arms并非独立同分布，
  因此该参考只用于直觉，不作formal gate；
- `policy_skill_centered_alignment`：在arm轴上，centered policy weight与negative arm MSE的cosine alignment；
- `policy_allocation_gain_over_uniform_percent`：policy-weighted expected arm MSE相对uniform的收益。它不是实际
  fused MSE，只诊断allocation是否把权重放在更熟练的arm上。

### 4.2 Fusion realizability

每个dataset保存了256条`probe_arms [256,5,720]`、`probe_fused [256,720]`与targets。将rows对半分成两fold，
在一半拟合、另一半评估，然后交换：

- `static convex`：全局、非负、sum-to-one的五个weights；
- `bounded affine`：全局、sum-to-one且每个weight在`[-1,1]`；
- 与已训练的adaptive learned fusion、uniform与train-half选择的best fixed arm比较。

该实验只定位readout bottleneck。它使用test-derived probe，不能证明新模型有效，也不能作为选择dataset-specific
weights的依据。

## 5. 诊断结果

| Model | best-arm match | skill alignment | policy vs uniform allocation | static convex vs learned | bounded affine vs convex |
| --- | ---: | ---: | ---: | ---: | ---: |
| `SIFF_EQUAL` | 29.24% | 0.0277 | +0.0762% | **+2.2112%** | +0.1203% |
| `SIFF_INDEPENDENT_EQUAL` | 24.48% | 0.0125 | -0.1070% | **+0.7680%** | +0.7160% |

补充稳定性：`SIFF_EQUAL`的static convex在10个dataset-fold中8个优于learned fusion；bounded affine仅在6/10
fold优于convex，而且macro额外收益仅0.12%。Weather上affine有明显增量，但weights触及`±1`边界，属于可能的
dataset-specific几何线索，不足以支持立即放开signed weights。

[Strong Evidence] 当前首要瓶颈是`routing/readout calibration`，不是arms collapse，也不是convex hull本身太窄。
一个不看sample/target、只在另一半rows拟合的静态convex组合，已经能系统性超过当前adaptive fusion，说明当前
policy没有实现其理论上应有的conditional advantage。

[Speculative] `SIFF_EQUAL`相对independent control的微弱优势可能仍来自ordered field的regularization，而不是
成功的conditional scope selection。现有probe不能区分“router输入信息不足”与“end-to-end gradients无法稳定识别
relative competence”，两者需要Step5 theory和matched controls共同拆分。

## 6. 外部 primary-source audit

检索日期：2026-07-18。检索范围包括`multi-scale time-series MoE/router`、`adaptive multi-scale fusion`、
`differentiable routing`、`routing supervision/error signal`。本次按项目规则使用external-first primary sources；
没有把Zotero是否收录作为novelty证据。Pathformer与Soft MoE有正式会议页面/论文和官方实现或项目链接；AdaMixT、
AME-TS与FAME截至本次审计只核验到arXiv页面，未确认正式录用与官方实现完整性，因此其证据权重较低。

| Source | Discovery / Zotero | Primary evidence | 对本项目的边界 |
| --- | --- | --- | --- |
| Pathformer, ICLR 2024 | external；Zotero presence未检查 | https://openreview.net/forum?id=lJkOCMP2aW；https://github.com/decisionintelligence/pathformer | 已覆盖multi-scale temporal experts与input-dependent adaptive pathways；普通multi-scale gating不能作为本项目核心新意 |
| Soft MoE, ICLR 2024 | external；Zotero presence未检查 | https://proceedings.iclr.cc/paper_files/paper/2024/hash/79fea214543ba263952ac3f4e5452b14-Abstract-Conference.html | 已覆盖fully differentiable soft assignment；把softmax换成一般soft routing不足以形成贡献 |
| AdaMixT, arXiv 2025 | external；Zotero presence未检查 | https://arxiv.org/abs/2509.18107 | 已明确提出multi-scale time-series experts加adaptive weighted gating；更大的MLP或generic attention router重叠过高 |
| Spatial MoE, NeurIPS 2022 | external；Zotero presence未检查 | https://proceedings.neurips.cc/paper_files/paper/2022/file/4c5e2bcbf21bdf40d75fddad0bd43dc9-Paper-Conference.pdf | 证明可由error/gradient构造self-supervised routing signal；支持calibration loss可行，但也要求我们收紧novelty claim |
| AME-TS, arXiv 2026 | external；Zotero presence未检查 | https://arxiv.org/abs/2605.25166 | 以forecastability/seasonality/trend/sparsity形成soft structural prior；说明structure-guided routing已有直接邻近工作 |
| FAME, arXiv 2026 | external；Zotero presence未检查 | https://arxiv.org/abs/2606.08896 | 使用validation performance挖掘expert-suitability targets；与旧CCRL式offline label相近，也强化了本项目应坚持synchronous E2E的边界 |

因此以下改法不值得单独成为v2：加深policy MLP、普通attention、top-k、entropy/balance loss、仅换softmax为sigmoid、
直接用offline validation risk labels。它们或被prior art覆盖，或没有针对本次information mismatch。

## 7. 改进方向排序

### Route A：Contrast-Calibrated Scope Fusion（CCSF，优先进入Step5，名称暂定）

核心不是“更多experts”，而是让fusion显式回答：在同一fixed-past、同一future target上，五种coupling scopes为何
给出不同预测，以及这些差异是否可预测地对应relative competence。

候选机制由两部分组成：

1. `arm-contrast-aware policy`：policy除history state与future coordinate外，读取每个target上的arm consensus、
   centered disagreement或低维contrast descriptor。requested horizon仍不进入网络；完整T=720上一次计算后只crop，
   projectivity contract不变；
2. `synchronous competence calibration`：训练时由同一batch的detached per-arm target error构造soft competence
   distribution，给policy一个轻量calibration loss；不预训练另一个model，不丢掉前m samples，不生成offline labels。
   equal-skill仍负责“每个arm会预测”，calibration负责“policy知道何时信谁”。

这是一条`source-informed redesign`，不是v1局部调参。它改变了policy information set与训练contract，必须创建
新candidate version，并在Step5证明不会产生winner-take-all、label leakage、arm-policy collusion或破坏
projectivity。

### Route B：A6_MEASURE anchor containment（Step5并行理论control，不立即实现）

下一候选必须说明为什么复杂decoder不会丢掉simple A6_MEASURE的优势。可研究将A6_MEASURE global forecast作为
明确anchor arm，而其他scopes只在有可校准证据时获得权重。关键control包括：A6_MEASURE + parameter-matched generic
head、anchor-only、contrast router without scale order。若只能靠anchor恢复性能，说明SIFF本体仍未贡献。

该route与AME-TS的“anchor”概念邻近，claim必须落在projective multi-horizon output-coupling scopes与matched
containment，而不能泛称anchored MoE。

### Route C：signed/affine fusion（暂缓）

bounded affine相对convex仅多0.12% macro，且主要由Weather驱动。当前证据不足以把signed fusion升为核心架构；
它只保留为Route A的matched diagnostic control，用于确认softmax geometry是否在新policy下仍构成瓶颈。

### Route D：扩展Q/rank/scale set（拒绝作为下一步）

independent control几乎打平说明shared field可能受约束，但当前更直接的失败发生在fusion calibration。此时增加Q、
rank或scales会同时增加capacity与router难度，无法回答mechanism问题。只有Route A在matched setting下仍不能利用
headroom，才回到scale-field representation redesign。

## 8. 决策与下一步

最终判断不是“继续v1”与“回Step4”二选一：

- **模型层面**：v1固定保留，不删除、不改名、不事后修改；
- **研究层面**：正式回到Step4，以v1作为parent进行source-informed redesign；
- **下一步**：进入provisional Route A的Step5 theory feasibility，并把Route B作为containment/control问题一起
  审计；在Step5/6通过前，不实现、不启动远程训练。

Rollback规则：若arm-contrast在不使用target的inference feature下无法提供可辨识的competence signal，或同步
calibration必然退化为arm winner collapse，则回Step2/3重审“conditional scope selection是否可由fixed-past预测”；
若calibration理论成立但matched local gate失败，归因为`intervention_point_wrong`或`optimization`，不得直接否定
coupling-scope问题。
