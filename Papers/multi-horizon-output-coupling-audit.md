# Multi-Horizon Output Coupling Prior-Art Audit

## Scope

- Search dates: 2026-07-15 initial audit；2026-07-16 D14-B and PCSD-native reset extensions
- Topic: multi-step forecasting strategy、Direct/MIMO/DIRMO、future-query decoder、joint future decoding、
  dynamic multi-output ensembles、expert selection、regret supervision、coordinate operator与local/global MoE
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
| [FFORMA](https://www.sciencedirect.com/science/article/pii/S0169207019300895) | 由series features学习forecast-method combination weights，并区分随机model selection objective与最终weighted average | feature-based weighting、meta-learning与soft combination不是SC2 novelty |
| [TimeFuse, ICML 2025](https://proceedings.mlr.press/v267/liu25cm.html) | sample-level meta-features驱动heterogeneous forecast models的adaptive fusion | sample-level fusion与direct fused loss必须成为matched control |
| [TimeRouter](https://arxiv.org/abs/2606.11625) | context/CV/forecast features、oracle-best labels、nonlinear router、selective fallback与OOF threshold selection | oracle labels、forecast snippets、nonlinear routing与fallback已被占据；仅作2026 freshness pressure |
| [AME-TS](https://arxiv.org/abs/2605.25166) | interpretable temporal descriptors形成soft expert prior，以KL alignment稳定specialization | structural-prior routing与KL supervision不能claim |
| [TimeExpert](https://arxiv.org/abs/2509.23145) | local timestamp experts、shared global expert与query-dependent routing | local/global expert mixture不是component novelty；其history-attention作用点与output coupling不同 |
| [Learning to Defer](https://proceedings.mlr.press/v119/mozannar20b.html) | cost-sensitive expert selection与consistent surrogate | regret/cost-sensitive reduction不是novelty |
| [Calibrated Learning to Defer](https://proceedings.mlr.press/v162/verma22c.html) | one-vs-all calibrated expert correctness | hard oracle/OvA router必须是control |
| [Temporal horizons trade-off](https://openreview.net/forum?id=BeudQIxT1R) | AR training horizon影响loss landscape与learnability | 只适用于AR-family evidence，不能直接外推到A6 direct decoder |
| [DeepONet](https://doi.org/10.1038/s42256-021-00302-5) | branch编码input function、trunk编码output coordinates并以内积合成operator | coordinate field、branch/trunk与separable synthesis不是PCSD novelty |
| [PoU-MoE DeepONet](https://arxiv.org/abs/2405.11907) | spatial partition-of-unity local experts与operator mixture | local/global operator mixture与spatial locality不是component novelty |
| [Soft MoE](https://openreview.net/forum?id=jxpsAj7ltE) | fully differentiable soft expert assignment | soft routing不是training contribution |
| [SMEAR](https://openreview.net/forum?id=7I199lc54z) | parameter-space soft expert merging并用standard gradients训练 | parameter merging不是PCSD novelty捷径 |
| [sMCL](https://proceedings.neurips.cc/paper/2016/hash/20d135f0f28185b84a4cf7aa51f29500-Abstract.html) | oracle loss驱动multiple predictors并行specialization | online best-arm/oracle target不是Contribution 2 novelty |

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

[D14-B Theory Correction] per-expert weighted risk不是weighted-average forecast的实际MSE；两者相差非负的
prediction-diversity项。因此`softmax(-regret)`不能被直接称为optimal fusion。cross-fitted regret只能作为
auxiliary conditional-risk supervision，actual fused forecast loss必须是primary objective。

[Official Code Boundary] TimeFuse官方`ModelFusor`为linear-softmax fusor并直接优化weighted forecasts；其feature
extractor覆盖statistics、ACF/stationarity、AR与spectrum。TimeRouter官方实现冻结四个TSFMs，使用XGBoost、
margin/diversity gate与CV-inverse fallback。StageC只吸收这些实现作为control/default证据，不直接复制模块：
`DIRECT_FUSION`与forecast-feature/hard-oracle arms必须运行，selective fallback不计novelty。本地不新增XGBoost
依赖，nonlinear sensitivity使用已有sklearn HistGradientBoosting。

[Training-Consistency Correction] CCRL虽可把预计算OOF risk作为最终auxiliary loss，但其teachers是独立scale
models，student是持续更新的shared PCSD arms；risk targets只覆盖部分training samples且会stale。该两阶段流程
没有形成与复杂度相称的novelty，故在Step7A前退出paper core。它只保留为未来严格secondary control。

[Native Architecture Reset] 新PCSD-CF不保存五个完整decoders。一个shared history-to-future mode field经
future-coordinate group pooling产生point/block/global states，并共享target synthesis rows。constant coordinate
mode给出A6 exact subspace，nonconstant zero-mean modes提供scope separation；direct task loss是首个control。

## Novelty Opportunity

primitive-level novelty已被大量占据。当前仅保留以下complete-chain机会：

> one fixed-past projective parameter field + scope pooling changes future-output state sharing + simultaneous
> point-to-global operators + exact A6 subspace + direct history/target allocation + no requested-H semantics.

该链必须同时超过：

- A6/global MIMO-like arm；
- CATS-like independent query arm；
- fixed DIRMO block sizes；
- equal/static mixture；
- ordinary task-loss router；
- TimeFuse-style matched direct fusion；
- TimeRouter-style hard oracle/forecast-feature router（只在未来SC2审计需要时）；
- in-sample best-expert pseudo-label与generic counterfactual credit（不得直接升级SC2）；
- dynamic ensemble/meta-learning、Soft MoE与SMEAR controls；
- same-parameter generic capacity与random partition controls。

## Rejected Claims

- first adaptive multi-horizon forecasting strategy；
- first Direct/MIMO hybrid；
- first future-query decoder；
- first dynamic ensemble or expert router；
- first out-of-fold expert supervision；
- first multiscale/local-global forecast generation；
- first future dependency modeling。

## Discovery And Freshness Risk

- FFORMA、TimeFuse、TimeRouter、AME-TS、TimeExpert与learning-to-defer sources均由2026-07-16 external search
  发现或重新核验；本轮未以Zotero presence作筛选，Zotero收录状态未核验；

- 2026 adaptive decoder/MoE work仍在快速增长；投稿前必须再次执行external search与citation chaining；
- TimeRouter本轮OpenReview PDF受challenge限制，搜索索引返回了方法摘要与公式片段；它只用于提高overlap风险，
  不作为absence/novelty结论的唯一证据；
- classic DIRMO原始论文与Stratify引用链应在formal Step 4下载全文逐项核对；
- 若发现within-model、per-target adaptive Direct-to-MIMO coupling field的直接先例，PCSD-CF必须重新收窄或关闭；
- 最终采用的论文应回填Zotero，但Zotero缺失不能作为novelty evidence。

## Current Decision

D14-A1 three-seed dual-carrier problem gate已通过。D14-B1/CCRL因training inconsistency与engineering-to-novelty
失衡在Step7A前关闭为paper core。PCSD-CF Step4-6对local implementation conditional pass；下一步只执行D15-A
A6 containment、projectivity、scope topology与accounting gates。Contribution 2 slot保持open，remote、test与
SC2 implementation仍false。
