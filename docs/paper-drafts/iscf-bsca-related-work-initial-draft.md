# ISCF-BSCA Section 2: Related Work

## Draft status

| Field | Content |
| --- | --- |
| `document_role` | Clean manuscript-facing initial draft of Section 2 |
| `version` | `v0.1-initial-draft` |
| `date` | `2026-08-10` |
| `review_status` | `initial_draft_for_author_review` |
| `upstream_dependency` | Introduction v0.9, Section 3 v0.7 and Section 4 v0.7 remain temporarily frozen and unchanged |
| `literature_search` | Primary-source search refreshed on 2026-08-10 |
| `source_audit` | `analysis/iscf_bsca_related_work_research_20260810/literature_design_and_source_audit.md` |
| `implementation_change` | None |
| `experiment_change` | None |
| `claim_boundary` | CHPC is used to formalize a varied-horizon system contract, not claimed as the first horizon-invariance principle; multi-scale and MoE primitives are prior art |
| `narrative_spine` | horizon-specific multi-step forecasting → unified and flexible horizons → decoder-side future construction → output-side multi-scope allocation |

The status table and editorial audit are working metadata and are not part of the manuscript body submitted for review.

## Section design

| Subsection | Narrative role | Main boundary established |
| --- | --- | --- |
| 2.1 Multi-step forecasting under horizon-specific protocols | Connect the broad forecasting literature to the system limitation stated in the Introduction | Per-horizon accuracy does not impose agreement between independently optimized predictors |
| 2.2 Unified and flexible-horizon forecasting | Position the paper against the closest varied-horizon and foundation-model work | Horizon invariance is acknowledged as prior art; the open question is decoder-side sharing across the future domain |
| 2.3 Forecast decoders and output-side temporal modeling | Shift the review from history encoding to future construction | Existing decoders use useful shared or structured synthesis, but generally fix their sharing pattern architecturally |
| 2.4 Multi-scale forecasting and conditional mixtures | Separate ISCF-BSCA from generic multi-scale and MoE formulations | The contribution lies in target-adaptive output-side sharing scopes and their joint optimization, not generic routing or balancing |

## 2. Related Work

### 2.1 Multi-step forecasting under horizon-specific protocols

Multi-step forecasting has traditionally been organized around recursive, direct and multiple-output strategies. Recursive methods repeatedly apply a one-step predictor and can accumulate forecast errors, whereas direct methods fit target-specific predictors and trade lower recursive bias for higher estimation variance. Multiple-input multiple-output methods generate a future block jointly, and block-parameterized extensions such as DIRMO interpolate between recursive and direct constructions \citep{taieb2014boosting,green2025stratify}. These formulations primarily study error propagation, bias--variance trade-offs and dependencies among future outputs.

Modern long-term forecasting benchmarks predominantly adopt a fixed-horizon direct multi-output protocol. Architectures such as DLinear, PatchTST and iTransformer instantiate an output mapping for one prediction length, and their standard evaluations optimize separate models for horizons such as 96, 192, 336 and 720 \citep{zeng2023dlinear,nie2023patchtst,liu2024itransformer}. This protocol supports horizon-specific optimization, but independently trained predictors are not constrained to agree on targets shared by different requests. We study this relation as a property of the forecasting system rather than another per-horizon accuracy criterion.

### 2.2 Unified and flexible-horizon forecasting

Recent studies have moved from horizon-specific training toward models that operate across forecasting tasks or output lengths. TimesFM uses decoder-only pretraining to generalize across datasets, temporal granularities and horizons, while Timer and Time-MoE use autoregressive generation to support flexible forecast lengths \citep{das2024timesfm,liu2024timer,shi2025timemoe}. UniTS further unifies heterogeneous time-series tasks through task tokenization \citep{gao2024units}. These models broaden the operating range of one parameter set, although their primary objectives are universal pretraining, task unification or scalable generation rather than an explicit analysis of overlapping forecasts issued from one origin.

ElasTST is the work most directly related to our setting. It introduces non-autoregressive future placeholders and structured attention masks so that predictions for shared steps remain invariant to the inference horizon, together with horizon reweighting and multi-scale patches for varied-horizon training \citep{zhang2024elastst}. Building on this direction, we formalize shared-step invariance as cross-horizon prefix consistency and study a complementary decoder-side question: how broadly should a history-conditioned state be reused across different regions of the future domain? Section 3 examines this question through controlled empirical analysis and motivates region-adaptive sharing.

### 2.3 Forecast decoders and output-side temporal modeling

Most forecasting architectures emphasize history representation, yet their output heads impose distinct structures on future generation. DLinear applies temporal linear maps with future-step-specific output rows, while PatchTST and iTransformer project shared patch-level or variate-level representations to a fixed target window \citep{zeng2023dlinear,nie2023patchtst,liu2024itransformer}. TFFS explicitly combines common and step-specific future features for multi-step prediction \citep{zhang2024tffs}. These designs demonstrate that future steps can share historical information without receiving identical predictions.

Structured decoders further organize the forecast through explicit output-side priors. N-HiTS synthesizes a trajectory from hierarchically interpolated components at different sampling rates, whereas Implicit Forecaster predicts constituent waves and composes them into a global future pattern \citep{challu2023nhits,li2025implicit}. Such methods establish the importance of forecasting-phase design, but their cross-step sharing pattern is determined by a fixed decoder topology. ISCF instead represents sharing extent as an explicit scope and constructs region-local states under multiple scopes before step-specific synthesis.

### 2.4 Multi-scale forecasting and conditional mixtures

Multi-scale forecasting methods capture temporal dynamics at several resolutions. N-HiTS combines multi-rate sampling with hierarchical interpolation, Pathformer adaptively selects pathways over different input patch scales, and TimeMixer mixes decomposed histories and multiple scale-specific predictors \citep{challu2023nhits,chen2024pathformer,wang2024timemixer}. These approaches motivate multi-resolution modeling, but their scales primarily index input resolution, frequency structure or complete prediction branches. The sharing scope in ISCF instead specifies how many contiguous future targets reuse one history-conditioned state.

Conditional mixtures provide a related mechanism for adaptive specialization. MoLE routes among complete linear-centric forecasters, FreqMoE assigns frequency components to experts, and Time-MoE and Moirai-MoE introduce sparse expert specialization into time-series foundation models \citep{ni2024mole,liu2025freqmoe,shi2025timemoe,liu2025moiraimoe}. ISCF does not ensemble independently trained forecasters. It constructs a shared scope-indexed forecast field and evaluates each target's preference for different output-side sharing granularities. BSCA is consequently designed for the joint optimization of this field and its allocation process. Together, these distinctions lead to the two questions formalized in Section 3: consistency across horizon requests and heterogeneous sharing demand across future regions.

## Editorial citation and claim audit

| Item | Current status | Required follow-up |
| --- | --- | --- |
| Classical recursive/direct/MIMO taxonomy | Supported by primary sources | Confirm final BibTeX keys during manuscript assembly |
| Standard horizon-specific LTSF protocol | Supported by official papers and implementation contracts | Keep wording at protocol level; do not claim the architectures cannot be adapted |
| Flexible-horizon foundation models | Supported | Avoid equating autoregressive variable length with explicit CHPC evaluation |
| ElasTST overlap | Explicitly acknowledged | Do not claim first horizon-invariant varied-horizon model or first CHPC-like property |
| Decoder-side prior art | Supported by architecture papers and official implementations | Keep the distinction between shared history representation and cross-step latent-state sharing |
| Multi-scale and MoE prior art | Supported | Do not claim generic multi-scale routing, expert fusion or balancing as novel |
| ISCF-BSCA distinction | Architecture-level interpretation | Effectiveness, component value and transferability remain contingent on Sections 5.2--5.7 |
