# ISCF-BSCA Introduction: Initial Manuscript Draft

## Draft status

| Field | Content |
| --- | --- |
| `document_role` | Clean manuscript-facing draft of the Introduction |
| `version` | `v0.4-round1-full-search-evidence-selected` |
| `date` | `2026-07-30` |
| `paragraphs_1_6` | Provisional revision responding to the first author review-response decisions |
| `citation_status` | Provisional citation keys inserted; bibliography integration remains pending |
| `result_status` | Headline empirical results remain to be inserted after the main comparison tables are complete |
| `problem_evidence_status` | ETTh2 prefix and ETTm2 sharing validation figures selected after five-dataset search |

The status table above is editorial metadata and is not part of the manuscript
body submitted for review.

## 1. Introduction

Multi-horizon forecasts support decisions over multiple planning ranges, from
short-term control to long-term scheduling. Yet most long-term time-series
forecasting models and benchmark protocols remain horizon-specific: a separate
model is trained for each forecast horizon $H$, such as 96, 192, 336, and 720
steps \citep{zeng2023dlinear,nie2023patchtst,liu2024itransformer}. Recent work,
including ElasTST and time-series foundation models such as TimesFM and
Time-MoE, has begun to support varied or flexible forecasting horizons
\citep{zhang2024elastst,das2024timesfm,shi2025timemoe}. Nevertheless, such
efforts remain sparse relative to the extensive horizon-specific literature,
and varied-horizon forecasting is still insufficiently developed as a unified
problem with an explicit task definition, systematic problem analysis, and
targeted decoder design. In this work, we formulate these requirements
explicitly and investigate how a unified forecaster should organize
output-side representations across different parts of the future domain.

The prevailing horizon-specific protocol also leaves an important system-level
gap. For a fixed forecast origin and identical observed history, independently
trained models may produce different values for the same future time step. In
particular, the first $H_1$ predictions of an $H_2$-step model are not
guaranteed to agree with those of a separately trained $H_1$-step model when
$H_1<H_2$. Such horizon-dependent disagreement prevents the forecasts from
forming nested views of one future trajectory. Maintaining multiple independent
models also increases the total training, storage, deployment, and maintenance
burden required to serve a set of forecast horizons.

As illustrated in Fig.~\ref{fig:prefix-disagreement}, separately optimized
horizon-specific DLinear models produce visibly different overlapping
prefixes for the same observed history. In the selected ETTh2 validation
example, the 96-, 192-, and 336-step models differ from the 720-step model by
mean absolute values of 2.16--2.51 over their shared first 96 future steps.
This purposefully selected maximum-disagreement example is complemented by
pairwise disagreement statistics over all validation origins and variables.

<!-- Insert figure_intro_prefix_disagreement from
analysis/iscf_bsca_intro_evidence_full_search_20260730/selected_figures/. -->

We therefore study varied-horizon forecasting through a horizon-agnostic
prediction function indexed by future time step. Given an observed history, an
$H$-step forecast is instantiated by evaluating this function at the first $H$
future steps within the supported forecast domain. Because the prediction for
an overlapping future step depends on the history and its step index, rather
than on the requested horizon, it remains unchanged across horizon requests. We
call this basic property of a varied-horizon forecasting system
**cross-horizon prefix consistency (CHPC)**.

CHPC provides a consistent forecasting interface, but it does not determine how
a finite-capacity decoder should represent the future. Most architectural
advances have focused on history encoding or input-side temporal
representations, while direct multi-output decoders commonly apply one fixed
output-generation pattern across the forecast domain. Broadly sharing a
history-conditioned latent state across many future steps can regularize smooth
and persistent trajectory components, whereas finer sharing can provide the
step-specific flexibility needed for local variations. Their relative value
may change across samples, variables, and future regions, so the
bias--variance trade-off induced by a fixed sharing extent need not be uniform.
This is a finite-capacity modeling issue rather than a change in the
pointwise-MSE Bayes target. We refer to the resulting hypothesis as
**future-region sharing-demand heterogeneity**.

Consistent with this hypothesis, a capacity-matched neutral decoder family
exhibits strongly region-dependent risk ordering on ETTm2. In the selected
validation example, all five matched sharing extents become the best choice for
two or three of the twelve 60-step future regions, and all ten extent pairs
show margin-qualified bidirectional risk crossovers. The descriptive
region-wise minimum reduces average MSE by 8.1\% relative to the best fixed
extent $s=720$ (Fig.~\ref{fig:sharing-demand}).

<!-- Insert figure_intro_sharing_heterogeneity from
analysis/iscf_bsca_intro_evidence_full_search_20260730/selected_figures/. -->

To model these heterogeneous sharing demands, we propose ISCF-BSCA, an
output-side decoder for varied-horizon forecasting. Independent
Scope-Conditioned Forecasting (ISCF) organizes multiple latent-state sharing
scopes within a single scope-indexed forecast field. Each scope specifies how
broadly a history-conditioned latent state is reused across future steps before
step-specific synthesis, and a target-conditioned allocation softly aggregates
the resulting predictions for each sample, variable, and future step. ISCF
therefore adapts the decoder's cross-step sharing pattern while preserving a
single horizon-agnostic prediction function. Because the same allocation also
governs how forecasting gradients reach the different scopes, we further
introduce Balanced Scope Co-Adaptation (BSCA), a train-only objective designed
to provide direct prediction signals to all scopes and reduce premature
allocation concentration during joint learning. BSCA adds neither inference
parameters nor an additional inference path, and the complete decoder retains
CHPC for every supported horizon.

Our contributions are threefold. First, we formulate varied-horizon forecasting
as a unified forecasting-system problem in which CHPC is a basic requirement,
and we identify future-region sharing-demand heterogeneity as a testable
output-side, finite-capacity challenge. Second, we introduce ISCF, a decoder
that combines multiple cross-step latent-state sharing extents, step-specific
synthesis, and target-conditioned soft allocation within one horizon-agnostic
forecast function. Third, we develop BSCA to stabilize the joint learning of
these sharing scopes without increasing inference-time complexity. We evaluate
the complete framework against horizon-specific systems, matched unified
forecasters, and architecture and objective controls, examining its advantages
in unified deployment, predictive accuracy, cross-horizon consistency,
output-side adaptation, and transferability.
