# HoriScope Introduction: Initial Manuscript Draft

## Draft status

| Field | Content |
| --- | --- |
| `document_role` | Clean manuscript-facing draft of the Introduction |
| `version` | `v0.9-author-refinement` |
| `freeze_status` | `temporarily_frozen_usable` |
| `freeze_date` | `2026-07-31` |
| `highlighted_review` | `docs/paper-drafts/iscf-bsca-introduction-v0.9-highlighted-review.md` |
| `date` | `2026-07-30` |
| `paragraphs_1_6` | Compact manuscript-facing core narrative with one conceptual figure |
| `citation_status` | Provisional citation keys inserted; bibliography integration remains pending |
| `result_status` | Detailed problem evidence is assigned to Section 3; headline method results remain pending the formal main tables |
| `problem_evidence_status` | Introduction states the verified problems concisely; definitions, controls and real-data evidence are deferred to Section 3 |
| `figure_status` | Conceptual Figure 1 approved for the manuscript draft; empirical figures assigned to Section 3 |

The status table above is editorial metadata and is not part of the manuscript body submitted for review.

## 1. Introduction

Multi-horizon forecasts support decisions over multiple planning ranges, from short-term control to long-term scheduling \citep{zeng2023dlinear}. Yet most long-term time-series forecasting models and benchmark protocols remain horizon-specific: a separate model is trained for each forecast horizon $H$, such as 96, 192, 336, and 720 steps \citep{nie2023patchtst,liu2024itransformer}. Recent work, including ElasTST\citep{zhang2024elastst} and time-series foundation models such as TimesFM\citep{das2024timesfm} and Time-MoE\citep{shi2025timemoe}, has begun to support varied or flexible forecasting horizons through autoregressive and universal-generation designs \citep{liu2024timer,ansari2024chronos,woo2024moirai}. However, these efforts remain sparse relative to the extensive horizon-specific literature and have not yet established varied-horizon forecasting as an explicit task with systematic problem analysis and targeted decoder design. In this work, we investigate how the decoder should organize output-side representations across different parts of the future domain.

The horizon-specific protocol fragments forecasting into independent predictors. As illustrated in Figure~\ref{fig:conceptual-problems}a, models trained for different horizons can assign different values to the same future time step, despite identical history and forecast origin. Their outputs can therefore disagree on overlapping future targets rather than form nested views of a common trajectory. Supporting several horizons further entails separate training, checkpoint storage, deployment and maintenance.

To formalize a unified alternative, we define **unified varied-horizon forecasting (UVHF)** as the setting in which one model serves different request endpoints through a shared prediction trajectory. Formally, UVHF learns a single horizon-agnostic mapping from observed history and future-step index to prediction. Under this mapping, a future-step prediction depends on the history and its step index, but not on the requested horizon. For any $H_1<H_2$, predictions over steps $1{:}H_1$ therefore remain identical under both requests. We call this requirement **cross-horizon prefix consistency (CHPC)**.

CHPC specifies how forecasts at different horizons should relate, but not how a decoder should generate different future regions. Most architectural advances focus on history encoding, while the output stage often uses a uniform mechanism for all future steps \citep{zhou2021informer,wu2021autoformer,zhou2022fedformer,liu2022nonstationary}. Such a decoder fixes the sharing extent, that is, how broadly a history-conditioned latent state is reused before step-specific generation. Broad sharing can capture persistent trajectory structure but may smooth local variations. Finer sharing offers greater step-specific flexibility but weaker structural regularization. The preferred balance can vary across samples, variables and future regions, making a fixed extent inadequate for the full forecast domain. Figure~\ref{fig:conceptual-problems}b summarizes this output-side heterogeneity, which we term **future-region sharing-demand heterogeneity**. We examine this problem in greater detail in Section~3.

<a id="fig:conceptual-problems"></a>

![Conceptual illustration of cross-horizon forecast disagreement and future-region sharing-demand heterogeneity.](../../paper-figures/figure_intro_conceptual_problem.png)

**Figure 1 | Two challenges in unified varied-horizon forecasting.** **a**, Horizon-specific predictors may disagree at the same future time step $\tau^\star$ despite identical observed history. **b**, The sharing extent associated with the lowest risk can vary across future regions.

Motivated by this heterogeneity, we propose HoriScope, an adaptive multi-scope decoder that integrates forecasts generated under different sharing extents. HoriScope assigns each scope a dedicated history projection within one scope-indexed forecast field. Each scope specifies the number of future steps over which a history-conditioned latent state is reused, whereas a single-scope decoder fixes one such extent across the entire forecast domain. HoriScope instead allocates scope-conditioned forecasts for each sample, variable and future step. The resulting trajectory can adapt its sharing composition across future regions while retaining a single horizon-agnostic prediction function. To support joint learning, Balanced Scope Co-Adaptation (BSCA) supplies direct prediction signals to all scopes and discourages premature allocation concentration. BSCA operates only during training, adds no inference parameters or paths, and preserves CHPC across supported horizons.

The contributions of our work are summarized as follows:

1. We formulate UVHF as a forecasting system in which CHPC is a basic requirement, and identify future-region sharing-demand heterogeneity as an output-side challenge.
2. We introduce HoriScope, which integrates forecasts generated under multiple sharing scopes through target-adaptive allocation.
3. We develop BSCA to support balanced scope learning without increasing inference-time complexity.

Experiments across datasets from multiple application domains show that our unified model outperforms separately trained horizon-specific forecasters. Component-wise ablations support the utility of the proposed components, and generalization studies demonstrate decoder portability across the evaluated backbone families.
