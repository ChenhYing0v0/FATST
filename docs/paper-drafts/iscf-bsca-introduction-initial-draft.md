# ISCF-BSCA Introduction: Initial Manuscript Draft

## Draft status

| Field | Content |
| --- | --- |
| `document_role` | Clean manuscript-facing draft of the Introduction |
| `version` | `v0.5-problem-evidence-integrated` |
| `date` | `2026-07-30` |
| `paragraphs_1_6` | Complete manuscript-facing core narrative with two problem-evidence inserts |
| `citation_status` | Provisional citation keys inserted; bibliography integration remains pending |
| `result_status` | Problem-evidence results integrated; headline method results remain pending the formal main tables |
| `problem_evidence_status` | Complete: approved ETTh2 prefix and ETTm2 sharing figures embedded from `paper-figures/` |

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
mean absolute values of 2.51, 2.16, and 2.40, respectively, over their shared
first 96 future steps. After averaging over variables and all 2,161 validation
origins, the normalized disagreement remains non-zero for every horizon pair
(0.015--0.041), with the largest gap occurring between horizons 96 and 720.
The displayed origin--variable pair was intentionally selected from 15,127
candidates for maximum aggregate validation disagreement; it demonstrates
that the inconsistency can be substantial, rather than estimating its
prevalence or implying that horizon-specific models are less accurate.

<a id="fig:prefix-disagreement"></a>

![Predictions from four horizon-specific DLinear models and their
cross-horizon disagreement.](../../paper-figures/figure_intro_prefix_disagreement.png)

**Figure 1 | Independently optimized horizon-specific forecasts can disagree
on the same future steps.** **a**, Predictions from four DLinear models trained
separately for horizons 96, 192, 336, and 720 on the same ETTh2 history. The
panel shows the final 48 observed steps and the first 96 future steps shared by
all four requested horizons. Colors and sparse, staggered marker shapes
identify the four predictions; the inset reports their mean absolute
differences from the 720-step forecast on the common prefix. The displayed
validation origin--variable pair maximizes mean absolute disagreement
aggregated over all six horizon pairs among 15,127 candidates. **b**,
Normalized cross-horizon prefix disagreement (NCHPD) averaged over all ETTh2
validation origins ($n=2,161$) and variables. The selected example is
illustrative and is not a prevalence estimate. Source data are provided with
the figure package.

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
extent $s=720$ (Fig.~\ref{fig:sharing-demand}). Because both the example and
the region-wise choices are identified using validation labels, this value
quantifies descriptive finite-capacity headroom rather than the realized
out-of-sample gain of a learned allocation mechanism.

<a id="fig:sharing-demand"></a>

![Region-dependent risk ordering across five capacity-matched sharing
extents.](../../paper-figures/figure_intro_sharing_heterogeneity.png)

**Figure 2 | Preferred cross-step sharing extent varies across future
regions.** **a**, Percentage MSE excess of five capacity-matched neutral
decoders above the lowest-risk sharing extent within each 60-step future region
of one ETTm2 validation example. Outlined squares mark the region-wise best
extent. **b**, MSE reduction of each region winner relative to the best fixed
extent ($s=720$), with colors denoting the winning extent and the dashed line
showing the 12-region mean. All five extents win two or three regions, and the
descriptive region-wise minimum yields 8.1\% lower average MSE than the best
fixed extent. The example is selected on validation labels and does not
represent out-of-sample allocation performance. Source data are provided with
the figure package.

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
