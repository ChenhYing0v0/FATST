# ISCF-BSCA Related Work 文献设计与来源审计

## 1. 审计范围

| Field | Content |
| --- | --- |
| `search_date` | `2026-08-10` |
| `last_refined` | `2026-08-10` author review round 1 |
| `task` | 重构Section 2，澄清fixed-horizon strategies、ElasTST边界、forecast-generation gap与output-side multi-scale allocation |
| `source_policy` | 优先PMLR、NeurIPS Proceedings、OpenReview、AAAI Proceedings与publisher official page；项目既有笔记只作线索 |
| `zotero_status` | 本轮未核验用户Zotero `FSA` subset中是否已有各条目，因此不对馆藏覆盖率作结论 |
| `writing_boundary` | Introduction v0.9、Section 3 v0.7与Section 4 v0.7正文不改；Related Work不提前使用method effectiveness claims |
| `experiment_change` | None |

本轮采用由远及近的四段结构：先说明multi-step forecasting长期由fixed-horizon strategy组织，再进入unified/varied-horizon work；随后把讨论焦点从Encoder转向forecast generation，最后审计multi-scale与adaptive-allocation primitive overlap。该结构使Section 2既承接Introduction中的system gap，又能把读者自然送入Section 3的`CHPC → future-region sharing-demand heterogeneity`问题链。

## 2. 推荐subsections

| Subsection | 应回答的问题 | 进入Section 3前的落点 |
| --- | --- | --- |
| 2.1 Fixed-horizon multi-step forecasting | recursive、direct、MIMO与DIRMO如何区分？ | 既有strategy在一个预设$H$内部组织预测；本文转向one-model varied-horizon requests与nested trajectory |
| 2.2 Unified and varied-horizon forecasting | foundation models与ElasTST已经解决了什么？ | ElasTST已系统研究horizon invariance；本文的互补问题是future-region output-side sharing extent |
| 2.3 Forecast generation and output-side modeling | 既有decoder如何从history representation生成future trajectory？ | structured forecast generation已有prior，但cross-step sharing topology通常预先固定 |
| 2.4 Multi-scale forecasting and adaptive allocation | multi-scale与expert allocation如何区别于ISCF？ | ISCF分配显式state-reuse granularity，而非选择complete predictor、frequency expert或generic sparse expert |

## 3. Subsection-wise prior-work map

### 3.1 Fixed-horizon multi-step forecasting

本轮首先修正`direct`与`multiple-output`可能被混用的问题。Classical direct strategy为每个future step训练一个single-output model；MIMO用一个multi-output model同时生成完整$H$-step vector；DIRMO则将$H$划分为长度$sigma$的blocks，并为不同blocks训练multi-output models。现代LTSF的one-head fixed-window mapping在计算形式上更接近MIMO，而不是classical direct strategy。

| Work | Venue | 已有工作内容 | 与本文的边界 | Primary source |
| --- | --- | --- | --- | --- |
| Ben Taieb and Hyndman, *Boosting Multi-Step Autoregressive Forecasts* | ICML 2014 | 系统对比recursive与direct strategies，并以bias--estimation-variance trade-off解释二者差异 | 解决strategy selection与error behavior，不讨论多个requested horizons的shared-prefix contract | <https://proceedings.mlr.press/v32/taieb14.html> |
| Bontempi, *Long Term Time Series Prediction with Multi-Input Multi-Output Local Learning* | ESTSP 2008 | 提出LL-MIMO，由一个multi-output predictor同时返回完整future vector，以利用future outputs之间的关系 | 是MIMO代表工作；其输出维度仍绑定预设forecast horizon | <https://difusion.ulb.ac.be/vufind/Record/ULB-DIPOT%3Aoai%3Adipot.ulb.ac.be%3A2013/72311/Details> |
| Ben Taieb et al., *A Review and Comparison of Strategies for Multi-Step Ahead Time Series Forecasting* | ESWA 2012 | 系统比较recursive、direct、MIMO及multi-output strategies，并在NN5上进行大规模评估 | 支撑strategy taxonomy；不涉及one-model varied request endpoints或CHPC | <https://www.sciencedirect.com/science/article/pii/S0957417412000528> |
| Green et al., *Stratify* | DMKD 2025 | 用block size统一RecMO、DirMO与DirRecMO等single/multiple-output strategies | 统一的是multi-step strategy space，不是同一forecast origin下不同request endpoints的一致性 | <https://link.springer.com/article/10.1007/s10618-025-01135-1> |
| Zeng et al., DLinear | AAAI 2023 | 以temporal linear mapping直接生成固定prediction window；标准LTSF protocol报告96/192/336/720 | 可作为horizon-specific benchmark family与CHPD evidence carrier，不应描述为架构上绝对无法支持别的长度 | <https://ojs.aaai.org/index.php/AAAI/article/view/26317> |
| Nie et al., PatchTST | ICLR 2023 | patch-token Encoder结合target-window projection；标准实验按prediction length配置输出head | history representation强，但standard protocol仍是fixed-horizon optimization | <https://openreview.net/forum?id=Jbdc0vTOcol> |
| Liu et al., iTransformer | ICLR 2024 | 以variate tokens建模cross-variable correlation，再投影到固定future window | shared variate representation不等同于跨future steps可变sharing extent | <https://openreview.net/forum?id=JePfAI8fah> |

### 3.2 Unified and varied-horizon forecasting

| Work | Venue | 已有工作内容 | 与本文的边界 | Primary source |
| --- | --- | --- | --- | --- |
| Das et al., TimesFM | ICML 2024 | decoder-only foundation model跨datasets、granularities与forecasting horizons进行zero-shot forecasting | 主要目标是universal pretraining与cross-domain generalization，不等同于fixed-origin CHPC audit | <https://proceedings.mlr.press/v235/das24c.html> |
| Liu et al., Timer | ICML 2024 | GPT-style decoder进行next-token/segment generation，以统一generative forecasting tasks | autoregressive generation天然允许长度变化，但不能据此宣称其已系统评估CHPC | <https://openreview.net/forum?id=bYRYb7DMNo> |
| Gao et al., UniTS | NeurIPS 2024 | 通过task tokenization统一forecasting、imputation、classification等任务与不同task specifications | task unification范围更广，但不是面向future-region decoder sharing的工作 | <https://proceedings.neurips.cc/paper_files/paper/2024/hash/fe248e22b241ae5a9adf11493c8c12bc-Abstract-Conference.html> |
| Zhang et al., ElasTST | NeurIPS 2024 | structured masks保证shared future outputs对inference horizon不变；另含horizon reweighting、tunable RoPE与multi-scale patches | 是最接近的varied-horizon prior，已占据horizon-invariance property；本文不能claim CHPC思想首创 | <https://papers.nips.cc/paper_files/paper/2024/hash/d7aa002885ccbe68cf6880da583761b2-Abstract-Conference.html> |
| Shi et al., Time-MoE | ICLR 2025 | decoder-only autoregressive time-series foundation model支持flexible horizons，并以sparse MoE扩大capacity | flexible generation与generic expert specialization不是future-region sharing-demand的直接解法 | <https://openreview.net/forum?id=e1wDDFmlVu> |

### 3.3 Forecast generation and output-side modeling

| Work | Venue | 已有工作内容 | 与本文的边界 | Primary source |
| --- | --- | --- | --- | --- |
| DLinear / PatchTST / iTransformer output heads | AAAI 2023 / ICLR 2023 / ICLR 2024 | 分别使用temporal linear rows、flattened patch readout与shared variate-state projection生成future window | 说明shared history state与step-specific outputs可以并存，但其cross-step sharing topology是固定的 | official papers above; implementation pointers recorded in `docs/iscf-bsca-paper-architecture.md` |
| Zhang et al., TFFS | Information Sciences 2024 | 将common future features与step-specific information结合用于multistep forecasting | 已覆盖common-versus-specific fusion，不能把这一抽象作为ISCF novelty | <https://www.sciencedirect.com/science/article/pii/S0020025524010405> |
| Challu et al., N-HiTS | AAAI 2023 | 通过multi-rate sampling与hierarchical interpolation逐层合成不同frequency/scale的forecast components | 是structured multi-scale synthesis prior；ISCF不以basis/interpolation首创 | <https://ojs.aaai.org/index.php/AAAI/article/view/25854> |
| Li et al., Implicit Forecaster | NeurIPS 2025 | 预测frequency、amplitude与phase所表示的constituent waves，再组成global forecast pattern | 已明确把创新焦点放在forecasting phase；ISCF差异在region-local multi-scope state sharing | <https://proceedings.neurips.cc/paper_files/paper/2025/hash/0e82ef0c89df6a6eff8734ea7e27c42f-Abstract-Conference.html> |

### 3.4 Multi-scale forecasting and adaptive allocation

| Work | Venue | 已有工作内容 | 与本文的边界 | Primary source |
| --- | --- | --- | --- | --- |
| Chen et al., Pathformer | ICLR 2024 | 不同patch scales表示temporal resolutions，并通过adaptive pathways选择input-side multi-scale processing | adaptive scale selection已有prior；其scale主要作用于history modeling | <https://openreview.net/forum?id=lJkOCMP2aW> |
| Wang et al., TimeMixer | ICLR 2024 | Past-Decomposable-Mixing与Future-Multipredictor-Mixing联合多尺度历史与scale-specific predictors | 已覆盖generic multiscale predictor fusion；不等同于同一future target的latent sharing extent | <https://openreview.net/forum?id=7oLshfEIC2> |
| Ni et al., MoLE | AISTATS 2024 | 训练多个complete linear-centric experts，由router按sample混合outputs | multiple complete predictors与router output mixing已有明确prior；ISCF不应被写成简单model ensemble | <https://proceedings.mlr.press/v238/ni24a.html> |
| Liu, FreqMoE | AISTATS 2025 | frequency decomposition、frequency-band experts与dynamic gating | frequency-side expert specialization不等于future-region sharing scopes | <https://proceedings.mlr.press/v258/liu25i.html> |
| Liu et al., Moirai-MoE | ICML 2025 | 在time-series foundation model中以sparse MoE实现automatic token-level specialization | generic sparse expert specialization已有prior；不能claim conditional computation primitive | <https://proceedings.mlr.press/v267/liu25an.html> |
| Shi et al., Time-MoE | ICLR 2025 | sparse MoE与autoregressive decoding结合，面向大规模universal forecasting | 进一步压缩generic MoE与flexible-horizon叙事空间 | <https://openreview.net/forum?id=e1wDDFmlVu> |

## 4. 关键claim boundaries

1. **CHPC与ElasTST的定位。** ElasTST已经系统研究varied-horizon forecasting，并以structured masks、tunable RoPE、multi-patch representation和horizon reweighting处理horizon invariance与robustness。本文不能将其缩减为“只提出attention decoder”或声称其未分析varied-horizon challenges。可辩护差异是ElasTST改变history-patch resolution，而本文建模future-region output-side state-sharing extent。
2. **fixed-horizon strategy的定位。** Classical direct为每个future step训练一个model；MIMO用一个model返回完整future vector；DIRMO按blocks使用多个multi-output models。DLinear、PatchTST与iTransformer的standard LTSF heads更接近MIMO-like fixed-window prediction，不应笼统称为classical direct。
3. **decoder novelty的定位。** N-HiTS、TFFS与Implicit Forecaster已说明forecasting phase、shared/step-specific information和structured synthesis的重要性。ISCF的差异应具体落在`scope-indexed forecast field → region-local Scope-region State → step-specific synthesis → target-adaptive scope allocation`。
4. **multi-scale与MoE的定位。** Pathformer、TimeMixer、MoLE、FreqMoE、Time-MoE和Moirai-MoE已经覆盖adaptive scales、multi-predictor fusion、frequency experts、sparse routing与token specialization。本文不得以generic multi-scale、MoE、router或load balancing为component novelty。
5. **evidence边界。** Related Work只建立问题与机制位置。统一模型优于horizon-specific models、ISCF/BSCA组件有效、decoder可迁移等结论继续由Section 5的main、ablation与transfer tables兑现。

## 5. Citation-key ledger

| Provisional key | Work |
| --- | --- |
| `taieb2014boosting` | Boosting Multi-Step Autoregressive Forecasts |
| `bontempi2008mimo` | Long Term Time Series Prediction with Multi-Input Multi-Output Local Learning |
| `taieb2012review` | A Review and Comparison of Strategies for Multi-Step Ahead Time Series Forecasting |
| `green2025stratify` | Stratify |
| `zeng2023dlinear` | DLinear |
| `nie2023patchtst` | PatchTST |
| `liu2024itransformer` | iTransformer |
| `das2024timesfm` | TimesFM |
| `liu2024timer` | Timer |
| `gao2024units` | UniTS |
| `zhang2024elastst` | ElasTST |
| `shi2025timemoe` | Time-MoE |
| `zhang2024tffs` | TFFS |
| `challu2023nhits` | N-HiTS |
| `li2025implicit` | Implicit Forecaster |
| `chen2024pathformer` | Pathformer |
| `wang2024timemixer` | TimeMixer |
| `ni2024mole` | MoLE |
| `liu2025freqmoe` | FreqMoE |
| `liu2025moiraimoe` | Moirai-MoE |

这些keys当前用于draft可读性；最终组装LaTeX bibliography时必须与项目BibTeX库逐项对齐，避免作者年相同或key collision。

## 6. 决策

`Decision=related_work_v0_2_temporarily_frozen_usable`。Author已确认Section 2整体内容；2.3 opening以`Beyond shallow output projections`承接前段并移除对prior-work数量的主观判断。正文、subsection structure、citations与claim boundaries暂时冻结，只有后续章节或证据产生明确矛盾且author批准时才解冻。当前不需要新implementation、remote training、formal test或额外figure。
