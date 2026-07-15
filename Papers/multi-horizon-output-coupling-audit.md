# Multi-Horizon Output Coupling Prior-Art Audit

## Scope

- Search date: 2026-07-15
- Topic: multi-step forecasting strategy、Direct/MIMO/DIRMO、future-query decoder、joint future decoding、
  dynamic multi-output ensembles、expert selection与regret supervision
- Discovery policy: external primary-source search为主；Zotero只作seed/reference
- Source types: formal proceedings、publisher paper page、OpenReview、arXiv official page、official code page
- Confidence: classic strategy、CATS、MQF2、Implicit Forecaster与TimePerceiver为高；2026 preprint只作freshness
  pressure，不作为absence proof

## Research Question

一个fixed-past unified model是否应固定一种future-output coupling granularity，还是在同一个exact-prefix decoder
内部表示并选择target-wise、block-wise与global sharing scopes？

## Primary Sources And Adopted Boundaries

| Source | Primary finding used here | Boundary imposed on StageC |
| --- | --- | --- |
| [Stratify](https://link.springer.com/article/10.1007/s10618-025-01135-1) | 统一RecMO、DirMO、DirRecMO、RectifyMO；output dimension调节bias、variance、flexibility与computation；无单一strategy普遍最优 | 不能claim首次统一forecasting strategies或首次调block size |
| [Direct/MIMO/DIRMO review](https://www.sciencedirect.com/science/article/pii/S0957417412000528) | Direct逐horizon、MIMO联合输出、DIRMO以blocks折中 | point-to-global continuum是prior art，不是component novelty |
| [CATS, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/cf66f995883298c4db2f0dcba28fb211-Paper-Conference.pdf) | 每个future horizon作为independent query；query间独立、跨horizon共享parameters | future query、independent target readout与parameter sharing不能单独claim |
| [MQTransformer](https://arxiv.org/abs/2009.14799) | context-dependent decoder-encoder attention与decoder self-attention | context-aware future decoding已有直接先例 |
| [TimePerceiver, NeurIPS 2025](https://arxiv.org/abs/2512.22550) | target timestamp queries与decoder-training framework | target-coordinate query和generalized target placement已被占据 |
| [Implicit Forecaster, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0e82ef0c89df6a6eff8734ea7e27c42f-Abstract-Conference.html) | 批评逐点预测缺少global view，以frequency/amplitude/phase waves隐式生成future | global trajectory decoder与wave generation不能claim |
| [MQF2, AISTATS 2022](https://proceedings.mlr.press/v151/kan22a.html) | probabilistic multi-horizon中直接建模跨future-time dependency | 不claim首次联合future dependency；point-MSE与probabilistic dependency必须区分 |
| [Multi-output Ensembles](https://arxiv.org/abs/2306.14563) | 比较multi-output dynamic ensembles与horizon-wise/joint weighting；长horizon下dynamic methods常难胜equal mixture | dynamic weighting、regret与meta-learning是mandatory controls，不是SC2 novelty |
| [Two-Step Meta-Learning Ensemble](https://arxiv.org/abs/2011.10545) | 由time-series features预测model ranking与ensemble size | history-feature-based model/expert selection已有先例 |
| [TimeRouter](https://openreview.net/pdf?id=zwDRjyd0md) | 以context与forecast evidence选择TSFM expert或ensemble fallback | discriminative forecast-model routing已被占据；仅作2026 freshness pressure |
| [Temporal horizons trade-off](https://openreview.net/forum?id=BeudQIxT1R) | AR training horizon影响loss landscape与learnability | 只适用于AR-family evidence，不能直接外推到A6 direct decoder |

## Key Synthesis

[Strong Evidence] 文献已经证明forecasting strategy不是一个无关紧要的engineering choice。Direct、recursive、
MIMO及其multi-output variants通过不同的parameter sharing与prediction feedback，形成bias-variance-flexibility
trade-off；Stratify进一步说明最佳strategy依赖domain与function class。

[Strong Evidence] 现代neural decoders通常固定其中一个endpoint或机制：CATS强调independent future queries；
A6/linear-basis heads接近global multi-output mapping；AR foundation models采用sequential coupling；Implicit
Forecaster采用global waves。现有工作并未因为都叫“unified model”就消除forecasting-strategy选择。

[Theory Boundary] 对separable deterministic MSE，future covariance不是Bayes point predictor的必要输入。
StageC只能把output coupling解释为有限样本/有限capacity下的shared inductive bias，不能把probabilistic
dependency文献直接转换为point-MSE theorem。

## Novelty Opportunity

primitive-level novelty已被大量占据。当前仅保留以下complete-chain机会：

> one fixed-past projective neural decoder + explicit point-to-global coupling spectrum + sample/target-region
> coupling policy + train-only counterfactual regret supervision + no requested-H semantics + no external strategy
> search.

该链必须同时超过：

- A6/global MIMO-like arm；
- CATS-like independent query arm；
- fixed DIRMO block sizes；
- equal/static mixture；
- ordinary task-loss router；
- in-sample best-expert pseudo-label；
- dynamic ensemble/meta-learning controls；
- same-parameter generic capacity与random partition controls。

## Rejected Claims

- first adaptive multi-horizon forecasting strategy；
- first Direct/MIMO hybrid；
- first future-query decoder；
- first dynamic ensemble or expert router；
- first out-of-fold expert supervision；
- first multiscale/local-global forecast generation；
- first future dependency modeling。

## Search Gaps And Freshness Risk

- 2026 adaptive decoder/MoE work仍在快速增长；投稿前必须再次执行external search与citation chaining；
- TimeRouter本轮OpenReview PDF受challenge限制，搜索索引返回了方法摘要与公式片段；它只用于提高overlap风险，
  不作为absence/novelty结论的唯一证据；
- classic DIRMO原始论文与Stratify引用链应在formal Step 4下载全文逐项核对；
- 若发现within-model、per-target adaptive Direct-to-MIMO coupling的直接先例，PCSD/CCRL必须重新收窄或关闭；
- 最终采用的论文应回填Zotero，但Zotero缺失不能作为novelty evidence。

## Current Decision

`PCSD + CCRL`仅为`proposed_step2_3`。先运行D14验证coupling-scale crossing与regret predictability；D14通过
也只允许返回formal Step 4-6，不直接实现paper method。
