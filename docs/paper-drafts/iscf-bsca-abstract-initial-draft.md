# HoriScope Abstract and Keywords

## Draft status

| Field | Content |
| --- | --- |
| `document_role` | Clean manuscript-facing initial draft of the Abstract and Keywords |
| `version` | `v0.4-uvhf-terminology-refinement` |
| `date` | `2026-08-25` |
| `review_status` | `pending_author_review` |
| `upstream_dependency` | Temporarily frozen Sections 1--7 and Appendices A--C |
| `evidence_scope` | CHPC construction; completed experiments in Sections 5.2--5.7 |
| `experiment_change` | None; no new implementation, training or formal test |
| `claim_boundary` | Validation-only motivation and selected internal diagnostics are not promoted to population-level or final-effectiveness claims |

The status table above is editorial metadata and is not part of the manuscript body submitted for review.

## Abstract logic

- Establish the practical need for forecasts of different lengths and the fragmentation of horizon-specific modeling.
- Define **unified varied-horizon forecasting (UVHF)** as one horizon-agnostic trajectory serving different request endpoints.
- Derive CHPC and heterogeneous output-side sharing as two forecast-generation requirements of UVHF, then address them from the decoder side.
- Present HoriScope and BSCA as the architectural and optimization responses to these two requirements.
- Close with aggregate accuracy, system consolidation, ablation and decoder-transfer evidence without enumerating individual baselines or values.

## Abstract

Most time-series forecasting methods optimize a model for a predefined prediction horizon, whereas practical applications often request forecasts of different lengths from the same history. Under the prevailing horizon-specific paradigm, separate models are optimized for different prediction lengths. This fragments multi-horizon service and can produce inconsistent forecasts for future steps shared across horizon requests. We instead formulate unified varied-horizon forecasting (UVHF), in which a single horizon-agnostic trajectory serves different request endpoints through its nested prefixes. This formulation imposes two coupled requirements on decoder design. First, predictions for shared future steps must remain invariant to the requested horizon, a property we term cross-horizon prefix consistency (CHPC). Second, jointly modeling short-, medium- and long-range futures requires the decoder to regulate how broadly history-conditioned information is shared across the future domain. We therefore propose HoriScope, an adaptive multi-scope decoder for UVHF. HoriScope constructs region-wise forecasts under multiple sharing scopes and integrates them through target-adaptive allocation into one trajectory, thereby satisfying CHPC by construction. We further introduce Balanced Scope Co-Adaptation (BSCA), which balances multi-scope optimization through scope-wise supervision and allocation regularization. Experiments across seven multivariate benchmarks show that HoriScope achieves state-of-the-art accuracy against recent horizon-specific and unified forecasters while serving all supported horizons with a single model. It also offers a favorable trade-off between checkpoint storage and inference memory. Controlled ablations and backbone transfer studies support the proposed components and decoder portability. These results establish output-side multi-scope generation as an effective foundation for UVHF.

## Keywords

Time series forecasting; unified varied-horizon forecasting; cross-horizon prefix consistency; multi-scope forecasting; adaptive forecast generation.

## Claim--evidence map

| Abstract claim | Evidence | Status |
| --- | --- | --- |
| Horizon-specific predictors leave overlapping predictions unconstrained | CHPC formulation and ETTh2 DLinear diagnostic in Section 3.2 | Supported as an objective-level limitation and audited diagnostic; not used as an accuracy claim |
| Different future regions may prefer different sharing extents | Controlled ETTm2 diagnostic in Section 3.3 and Figure 3 | Supported as motivation; modal wording avoids a population-level prevalence claim |
| HoriScope satisfies CHPC by construction | Future-step-indexed prediction and prefix-restricted inference in Sections 3.1 and 4.4 | Supported by architecture |
| State-of-the-art accuracy against recent horizon-specific and unified forecasters | Tables 1 and 2 over seven datasets and four horizons | Supported by the reported aggregate comparison matrices |
| Favorable checkpoint-storage and inference-memory trade-off | Figure 6 and Section 5.4 | Supported within the reported service-cost protocol |
| Component contribution and decoder portability | Table 3, Section 5.5 and Figure 7 | Supported within the evaluated ablation controls and two-backbone transfer study |

## Adversarial self-review

- **Contribution:** The abstract distinguishes the task reformulation, HoriScope architecture and BSCA optimization role without presenting the method as an incremental decoder patch.
- **Clarity and flow:** Each sentence advances one step in the sequence `need -> limitation -> insight -> method -> evidence -> implication`.
- **Terminology:** Unified varied-horizon forecasting (UVHF) is the formal task name; horizon-specific forecasting is its contrastive protocol, while multi-horizon forecasting remains a generic background description.
- **Evidence:** All quantitative claims map to the completed Section 5 tables or efficiency figure; selected validation diagnostics are used only to motivate the sharing-demand statement.
- **Boundary:** No claim of universal Encoder compatibility, causal scope specialization or population-level allocation behavior is made.
