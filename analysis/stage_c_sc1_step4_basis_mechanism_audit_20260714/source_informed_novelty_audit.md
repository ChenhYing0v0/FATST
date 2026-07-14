# SC1 Step 4 Balanced-Interval Generation: Source-Informed Novelty Audit

## Decision Summary

| Field | Decision |
| --- | --- |
| `search_date` | 2026-07-14 |
| `current_step` | Step 4 active |
| `basis_construction_novelty` | low：属于Haar-like interval multiresolution construction |
| `basis_forecast_generation_novelty` | moderate component novelty，不能声称first basis/wavelet forecast generation |
| `unified_horizon_operator_novelty` | potentially defensible：balanced future supports + domain-only H + native partial synthesis |
| `existence_evidence` | D3 basis main MSE +2.9174%，5/5 datasets |
| `next_gate` | SC1-D4 standard structured-basis/locality/specificity diagnostic |
| `method_training_authorized` | false |

## 1. Search Scope And Evidence Rule

本轮按项目规则使用外部primary sources，Zotero仅作seed library，不作为新颖性或检索完整性证据。queries覆盖：

- basis expansion / basis mapping for forecast generation；
- wavelet coefficient prediction and inverse reconstruction；
- fixed/learned future basis decoders；
- arbitrary prediction length / functional basis decoder；
- Haar、unbalanced Haar、interval wavelet与support-local synthesis。

核对范围以论文全文/official proceedings为主；2026 under-review工作仅作低置信度freshness pressure，不据其
未公开实现形成强结论。

## 2. Primary-Source Matrix

| Work | Forecast-generation mechanism | Pressure / remaining distinction |
| --- | --- | --- |
| [N-BEATS](https://arxiv.org/abs/1905.10437) | network预测forward expansion coefficients，并通过generic或polynomial/Fourier basis形成forecast | “basis coefficients -> forecast”早已存在；但模型按forecast horizon配置，不提供同一full-domain operator的support restriction |
| [N-HiTS](https://arxiv.org/abs/2201.12886) | multi-rate sampling与hierarchical interpolation生成coarse-to-fine additive forecast | hierarchical forecast generation已拥挤；其插值blocks不是orthogonal future support basis，也不解决domain-only H |
| [BasisFormer](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e150e6d0a1e5214740c39c6e4503ba7a-Abstract-Conference.html) | Coef module计算history-series与basis关系，Forecast module选择并聚合future-view bases | learned/interpretable future bases与coefficient-based generation不能单独成为claim；official scripts分别训练不同prediction lengths |
| [Fourier Basis Mapping](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0fd4ce94d29be88a5a262a2c77a18f47-Abstract-Conference.html) | Fourier basis expansion与mapping network组合，最后decoder生成future series | fixed structured basis用于LTSF generation已有强先例；DCT/Fourier是D4 mandatory control |
| [WaveToken](https://proceedings.mlr.press/v267/masserano25a.html), ICML 2025 | scale/decompose/quantize wavelet coefficients，autoregressively forecast coefficients for the forecast horizon | 直接否定“首次预测wavelet coefficients并inverse-generate future”；但它是foundation-model tokenization，不是continuous full-domain projective operator |
| [Implicit Forecaster](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0e82ef0c89df6a6eff8734ea7e27c42f-Abstract-Conference.html) | 预测frequency/amplitude/phase并隐式组成forecast waves | 从forecasting phase生成global constituent functions已被占据；本项目不能退化为另一种fixed functional decoder |
| [FlowState](https://arxiv.org/abs/2508.05287) | functional basis decoder按prediction length与sampling scale动态生成outputs | varied target length + functional basis已有直接压力；区别必须是H不进入learned path、只做support restriction |
| [Unbalanced Haar](https://doi.org/10.1198/016214507000000860) / [Lifting Scheme](https://doi.org/10.1137/S0036141095289051) | arbitrary interval splits、orthonormal Haar-like atoms与perfect reconstruction | balanced midpoint construction不是新的数学basis；其价值只能来自forecast-operator contract |

2025-2026 freshness补充：WaveTS等工作已把wavelet/derivative coefficients与inverse reconstruction用于深度
forecasting，但仍处于under-review状态，作为命名和机制拥挤度提示，不作为主要novelty判据。

## 3. Response To The Innovation Question

[Decision] “把balanced-interval basis用于预测生成”可以构成**组件级方法创新**，但不能仅凭应用场景就成为
完整SCI Contribution 1。原因不是要求所有数学组件从零发明，而是相邻工作已经明确覆盖了三件事：

1. 预测basis coefficients；
2. 用future bases重建forecast；
3. 用wavelet coefficients表示forecast horizon。

因此论文不能写：

> We are the first to use a multiresolution/wavelet basis to generate forecasts.

更可辩护的潜在claim是：

> We organize the future output domain with a balanced interval basis so that one horizon-agnostic coefficient
> operator supports native prefix-restricted synthesis across dense requested horizons, while retaining a
> cross-dataset readout advantage over global and data-adaptive structured bases.

这个claim包含四个不可拆开的边界：future-domain、one shared operator、H只裁剪support、structured-control
advantage。前三项形成叙事创新，第四项决定它不是只换坐标。

## 4. Why D3 Is Necessary But Insufficient

D3只证明balanced interval相对random orthogonal basis具有独立main effect。random orthogonal同时破坏：

- temporal smoothness ordering；
- local support；
- low-frequency energy compaction；
- target covariance alignment。

因此`+2.9174%`可能由任何一般structured coordinate解释。尤其D1-v2已经发现DCT rank-256 label capture为
`0.9029-0.9777`，显著高于A6 learned basis。若DCT/PCA/identity在相同GroupedNonlinearHead中匹配或超过
balanced interval，则D3只支持“random basis很差”，不能支持balanced-specific mechanism。

## 5. SC1-D4 Diagnostic Logic

D4固定random grouping，避免重开已失败的depth-grouping hypothesis。七类basis为：

1. `balanced_interval`：candidate geometry；
2. `identity`：time-point output control；
3. `dct2`：fixed global smooth basis；
4. `pca_fit`：fit-target-only data-adaptive global orthogonal control；
5. `permuted_interval`：保留atom values/support sizes，破坏contiguous temporal locality；
6. `random_interval_tree`：保留interval-local Haar family，改变exact balanced splits；
7. `random_orthogonal`：D3 replication anchor。

所有arms使用相同head capacity、random groups、initialization、optimizer与split，训练full H720，统一评估
`48/96/144/192/288/336/512/720`八个prefix horizons。同时记录fit-target covariance off-diagonal ratio、
variance compaction与exact-support active coefficient counts。

## 6. Narrative Outcomes

| Result | Interpretation | Next step |
| --- | --- | --- |
| balanced输给DCT/PCA/identity | standard structured basis解释performance | rollback Step 2；balanced只作control/component |
| balanced不胜permuted interval | contiguous locality不是收益来源 | Step 2 reformulation |
| balanced胜permuted但不胜random interval tree | interval-local family有效，exact balancing不特异 | Step 4 redesign；claim收紧为interval-local projective generation |
| balanced同时胜global、permuted与random interval tree | balanced specificity获得支持 | 仅授权Step 5 theory feasibility |

## 7. 11-Step Record

| Field | Record |
| --- | --- |
| `current_step` | Step 4 active |
| `problem` | D3 basis effect是否超越standard structure，并原生服务dense horizons |
| `existence_evidence` | D3 independent main effect +2.9174%，5/5 datasets |
| `idea` | balanced-interval projective generation，尚非candidate |
| `theory_check` | construction非新；组合novelty provisional |
| `design` | D4 seven-basis × dense-horizon diagnostic |
| `narrative_gate` | pending D4 mechanism attribution |
| `effectiveness_gate` | diagnostic only；end-to-end未授权 |
| `artifacts` | 本audit + D4 protocol/config pending |
| `decision` | run D4；pass only authorizes Step 5 |
