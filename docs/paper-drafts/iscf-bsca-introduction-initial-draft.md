# ISCF-BSCA Introduction: Initial Manuscript Draft

## Draft status

| Field | Content |
| --- | --- |
| `document_role` | Clean manuscript-facing draft of the Introduction |
| `version` | `v0.1` |
| `date` | `2026-07-28` |
| `paragraphs_1_5` | Integrated from the current paragraph-level consensus |
| `paragraph_6` | Provisional contribution paragraph under the v0.6 ISCF framework |
| `citation_status` | Primary-source citations remain to be inserted in manuscript style |
| `result_status` | Headline empirical results remain to be inserted after the main comparison tables are complete |

The status table above is editorial metadata and is not part of the manuscript
body submitted for review.

## 1. Introduction

Multi-horizon forecasting is essential in applications that require predictions
over several planning ranges, from short-term control to long-term scheduling.
However, the standard long-term forecasting protocol typically trains a
separate model for each forecast horizon $H$, such as 96, 192, 336, and 720
steps. Although each model is individually optimized for its designated
horizon, the resulting collection of horizon-specific predictors does not
constitute a unified multi-horizon forecasting system.

For the same observed history, independently trained horizon-specific models
may produce different predictions for the same future time step. In particular,
the first $H_1$ steps predicted by an $H_2$-step model are not guaranteed to
agree with the output of a separately trained $H_1$-step model, even when
$H_1<H_2$. Such horizon-dependent disagreement prevents the forecasts from
being interpreted as nested views of one future trajectory. Moreover,
maintaining separate models for different horizons multiplies training,
storage, and deployment costs.

We therefore formulate unified multi-horizon forecasting through a
horizon-agnostic prediction function indexed by future time step. Given an
observed history, the model directly defines a prediction for each future step,
and an $H$-step forecast is instantiated by evaluating the corresponding
sequence of future steps. Since the prediction at each step is determined by
the observed history and its future-step index, rather than by the requested
horizon, predictions at overlapping future steps remain identical across
horizons. We refer to this property as cross-horizon prefix consistency. Our
architecture realizes this step-indexed interface through future-step-specific
synthesis vectors, allowing arbitrary horizons to be instantiated without
horizon-specific prediction heads.

Although a horizon-agnostic prediction field provides a consistent interface
across horizons, it does not make unified forecasting inherently easy. Many
direct multi-output forecasters generate all future steps either from a broadly
shared latent representation or through one fixed output-generation pattern.
Such a fixed pattern imposes the same extent of cross-step sharing throughout
the forecast domain. However, fine-scale variations may require greater
step-specific flexibility, whereas smoother and broader trajectory components
may benefit from reusing a common history-conditioned latent state across
multiple steps. Because their relative importance can vary across samples,
variables, and future regions, no single sharing extent is guaranteed to be
uniformly suitable. We refer to this challenge as **future-region
sharing-demand heterogeneity**.

To address these heterogeneous sharing demands, we propose ISCF-BSCA, a
scope-adaptive decoder for unified multi-horizon forecasting. Rather than
treating different scopes as independent predictors, Independent
Scope-Conditioned Forecasting (ISCF) defines a single scope-indexed forecast
field over future time steps and latent-state sharing scopes. For each scope, an
independent history projection provides scope-specific latent modes, from which
history-conditioned, region-indexed latent states are constructed under a
different cross-step sharing extent. Shared future-step-specific synthesis
vectors then map these states to a scope-conditioned slice of the forecast
field. At each forecast target, a target-conditioned scope allocation assigns
normalized weights across the slices, and a weighted contraction along the
scope axis produces the final prediction. This formulation allows multiple
latent-state sharing extents to contribute within one horizon-agnostic,
future-step-indexed prediction function. Because the same allocation also
determines how the forecasting loss distributes gradients across scope slices,
we further introduce Balanced Scope Co-Adaptation (BSCA), a train-only objective
that maintains broad learning access across the field and mitigates premature
allocation concentration during joint optimization. BSCA changes neither the
inference parameters nor the inference path. The resulting forecaster directly
instantiates arbitrary forecast horizons while retaining cross-horizon prefix
consistency.

Our contributions are threefold. First, we formulate prefix-consistent unified
multi-horizon forecasting as a forecasting-system problem and formalize
future-region sharing-demand heterogeneity as a testable finite-capacity
challenge, separating evidence for the task problem from the scope-based
mechanism used to address it. Second, we develop ISCF, an output-side
forecasting architecture that organizes independent history projections,
multiple latent-state sharing extents, shared future-step-specific synthesis,
and target-conditioned scope allocation as a single scope-indexed forecast
field. The resulting decoder adapts its output-side sharing pattern at each
forecast target while retaining cross-horizon prefix consistency without
horizon-specific prediction heads. Third, we introduce BSCA, an ISCF-specific,
train-only objective for stabilizing the co-adaptation of scope-conditioned
slices and their allocation without changing the inference graph. We evaluate
the complete framework against horizon-specific systems, matched unified
forecasters, and architecture and objective controls in terms of predictive
accuracy, system efficiency, cross-horizon consistency, mechanism attribution,
and decoder transferability.
