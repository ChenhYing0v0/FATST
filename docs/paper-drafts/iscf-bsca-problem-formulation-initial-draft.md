# ISCF-BSCA Section 3: Problem Formulation and Empirical Motivation

## Draft status

| Field | Content |
| --- | --- |
| `document_role` | Clean manuscript-facing initial draft of Section 3 |
| `version` | `v0.1-initial-draft` |
| `date` | `2026-07-31` |
| `review_status` | `pending_author_review` |
| `introduction_dependency` | Introduction `v0.9-author-refinement` remains unchanged |
| `figure_2_status` | Approved validation-only illustrative evidence; integrated below |
| `figure_3_status` | Approved validation-only illustrative evidence; integrated below |
| `method_figure_4_status` | Planned only; not generated or referenced as completed |
| `result_boundary` | No main-table, ablation or transfer claim is treated as established |

The status table and the editorial audit after Section 3 are not part of the
manuscript body submitted for review.

## Terminology ledger

| Term | Symbol | Meaning in this section |
| --- | --- | --- |
| Forecast horizon | $H$ | The requested maximum number of future steps |
| Future time step | $\tau$ | A position within the forecast domain, $1\leq\tau\leq H$ |
| Forecast target | $(\tau,c)$ | Future step $\tau$ of variable $c$ |
| Cross-horizon prefix consistency | CHPC | Invariance of shared-prefix predictions to the requested horizon |
| Cross-horizon prefix disagreement | CHPD | Raw-scale disagreement between overlapping horizon-specific forecasts |
| Normalized CHPD | NCHPD | CHPD normalized by train-split variable scale |
| Future region | $\mathcal B_b$ | A contiguous subset of future steps, not a requested horizon |
| Sharing extent | $s$ | The number of future steps that reuse one history-conditioned latent state |
| Unified penalty | $\operatorname{UP}_H$ | Relative MSE difference between matched unified and horizon-specific predictors |
| Cross-fitted headroom | CFH | Test risk reduction of a validation-selected region schedule over a validation-selected fixed extent |

## 3. Problem Formulation and Empirical Motivation

### 3.1 Varied-horizon forecasting and cross-horizon prefix consistency

Let

$$
\mathbf X_o
=
\left[\mathbf x_{o-L+1},\ldots,\mathbf x_o\right]
\in
\mathbb R^{L\times C}
$$

denote a length-$L$ multivariate history at forecast origin $o$, where $C$ is
the number of variables. For a requested forecast horizon $H$, the associated
future target is

$$
\mathbf Y_o^{(H)}
=
\left[y_{o+\tau,c}\right]_{\tau=1,\ldots,H;\ c=1,\ldots,C}
\in
\mathbb R^{H\times C}.
$$

We consider a supported horizon set
$\mathcal H=\{H_1,\ldots,H_M\}$ and let
$T=\max\mathcal H$ denote the largest supported future domain. Here, $H$
specifies the requested support, whereas $\tau$ identifies an individual
future time step. This distinction is central: the same target
$(\tau,c)$ belongs to every request with $H\geq\tau$.

The conventional horizon-specific protocol learns an independent predictor
for each $H$:

$$
\widehat{\mathbf Y}_o^{(H)}
=
f_{\theta_H}(\mathbf X_o),
\qquad
H\in\mathcal H.
$$

Because the parameters and optimization problems are independent, this
formulation imposes no relation between predictions made for overlapping
future steps. In contrast, varied-horizon forecasting seeks one predictor
shared by all supported requests. We formulate it as a horizon-agnostic,
future-step-indexed mapping

$$
g_\theta:
(\mathbf X_o,\tau,c)
\mapsto
\widehat y_{o+\tau,c},
$$

from which any requested forecast is assembled as

$$
\widehat{\mathbf Y}_o^{(H)}
=
\left[
g_\theta(\mathbf X_o,\tau,c)
\right]_{\tau=1,\ldots,H;\ c=1,\ldots,C}.
$$

The requested horizon therefore determines which future steps are returned,
but does not alter the prediction semantics of a step already contained in a
shorter request. Let
$\Pi_{H_i}:\mathbb R^{H_j\times C}\rightarrow\mathbb R^{H_i\times C}$
retain the first $H_i$ steps. A forecasting system satisfies
**cross-horizon prefix consistency (CHPC)** if, for an identical history,
forecast origin and preprocessing state,

$$
\Pi_{H_i}
\widehat{\mathbf Y}_o^{(H_j)}
=
\widehat{\mathbf Y}_o^{(H_i)},
\qquad
\forall\,H_i,H_j\in\mathcal H,\ H_i<H_j.
$$

The step-indexed formulation satisfies CHPC by construction because both
sides evaluate the same function $g_\theta(\mathbf X_o,\tau,c)$ for
$1\leq\tau\leq H_i$. CHPC is a systems contract rather than an accuracy
guarantee: a horizon-specific family may happen to produce similar prefixes,
and a CHPC-compliant predictor may still be inaccurate.

### 3.2 Horizon-specific prefix disagreement

We first examine the inconsistency permitted by independently optimized
horizon-specific systems. For the same forecast origin $o$, let
$\widehat{\mathbf Y}_{o}^{(H_i)}\in\mathbb R^{H_i\times C}$ and
$\widehat{\mathbf Y}_{o}^{(H_j)}\in\mathbb R^{H_j\times C}$ be predictions
from two models trained separately for $H_i<H_j$. Their
**cross-horizon prefix disagreement (CHPD)** is

$$
\operatorname{CHPD}_o(H_i,H_j)
=
\frac{1}{H_iC}
\sum_{\tau=1}^{H_i}
\sum_{c=1}^{C}
\left|
\widehat y_{o+\tau,c}^{(H_i)}
-
\widehat y_{o+\tau,c}^{(H_j)}
\right|.
$$

CHPD retains the physical scale of the data. To aggregate variables with
different magnitudes, we additionally define normalized CHPD:

$$
\operatorname{NCHPD}(H_i,H_j)
=
\frac{1}{|\mathcal O|C}
\sum_{o\in\mathcal O}
\sum_{c=1}^{C}
\frac{
\frac{1}{H_i}\sum_{\tau=1}^{H_i}
\left|
\widehat y_{o+\tau,c}^{(H_i)}
-
\widehat y_{o+\tau,c}^{(H_j)}
\right|
}{
\sigma_c^{\mathrm{train}}+\epsilon
},
$$

where $\mathcal O$ is the set of aligned evaluation origins,
$\sigma_c^{\mathrm{train}}$ is the standard deviation of variable $c$
estimated from the training split, and $\epsilon>0$ prevents division by zero.
Both statistics compare predictions generated from the same numerical history,
forecast origin, scaler and overlapping target indices. Self-replay of one
horizon-specific checkpoint and replay of one unified checkpoint must yield
exactly zero; these controls separate genuine cross-model disagreement from
alignment or serialization errors.

Figure 2 provides a validation-only illustration using DLinear on ETTh2. The
displayed origin-variable pair maximizes disagreement aggregated over all six
horizon pairs among 15,127 validation candidates. Within the first 96 future
steps, the forecasts requested at horizons 96, 192 and 336 differ from the
720-step forecast by mean absolute raw-scale values of 2.51, 2.16 and 2.40,
respectively. Across 2,161 aligned validation origins and all variables, NCHPD
remains non-zero for every pair, ranging from 0.0148 to 0.0406. The selected
trajectory is intentionally a strong example rather than a prevalence
estimate; the aggregate heatmap establishes dataset-level average disagreement
only for the audited DLinear family. Accordingly, this evidence shows that
independent horizon-specific optimization provides no CHPC guarantee. It does
not show that horizon-specific forecasts are less accurate, nor does it
characterize varied-horizon methods that enforce invariance by design.

<a id="fig:prefix-disagreement"></a>

![Validation-only illustration of horizon-specific prefix disagreement.](../../analysis/iscf_bsca_intro_evidence_full_search_20260730/selected_figures/figure_intro_prefix_disagreement.png)

**Figure 2 | Independently optimized horizon-specific forecasts can disagree
on the same future steps.** **a**, Predictions from four DLinear models trained
separately for horizons 96, 192, 336 and 720 on the same ETTh2 history. The
panel shows the final 48 observed steps and the first 96 future steps shared by
all four requested horizons. Colors and sparse, staggered marker shapes
identify the four predictions; the inset reports their mean absolute
differences from the 720-step forecast on the common prefix. The displayed
validation origin-variable pair maximizes mean absolute disagreement
aggregated over all six horizon pairs among 15,127 candidates. **b**,
Normalized cross-horizon prefix disagreement (NCHPD) averaged over all ETTh2
validation origins ($n=2{,}161$) and variables. The selected example is
illustrative and is not a prevalence estimate.

### 3.3 Evidence boundary for naive unified forecasting

System consolidation and CHPC do not imply an accuracy advantage. To evaluate
whether a naive unified adaptation compromises predictive performance, a
horizon-specific predictor and its unified counterpart must be matched in
encoder class, effective capacity, training data, objective, optimization,
checkpoint selection and evaluation origins. For such a comparison, define

$$
\operatorname{UP}_H
=
\frac{
\operatorname{MSE}^{\mathrm{unified}}_H
-
\operatorname{MSE}^{\mathrm{specific}}_H
}{
\operatorname{MSE}^{\mathrm{specific}}_H
}.
$$

A positive $\operatorname{UP}_H$ indicates a penalty for the particular
unified adaptation under the matched protocol; it does not establish that
unified forecasting is intrinsically harder. Conversely, a non-positive value
does not by itself attribute an advantage to horizon unification.

The evidence currently available does not establish a stable positive unified
penalty. The closest diagnostic comparison is small and heterogeneous across
dataset-horizon cells, while a separate training-protocol contrast produces a
larger and more consistent effect. The observed difference therefore cannot be
attributed to horizon unification. We do not use an assumed accuracy
compromise as problem evidence: the motivation for a unified formulation in
this section is limited to single-model service and the explicit CHPC
contract. Relative accuracy is left to the complete matched scorecards in the
experimental section.

### 3.4 Future-region sharing-demand heterogeneity

CHPC determines how overlapping requests should agree, but it does not
determine how a finite-capacity decoder should share predictive states across
future steps. Let a **future region**
$\mathcal B_b\subseteq\{1,\ldots,T\}$ be a contiguous subset of the maximum
future domain. A future region is an analysis unit inside the prediction
domain, not a requested forecast horizon. We use **sharing extent** $s$ to
describe how many future steps reuse one history-conditioned latent state
before step-specific synthesis.

The problem hypothesis is a finite-capacity bias--variance trade-off. Broad
sharing constrains many future steps through a common state and can reduce
estimation variance, but it may increase approximation bias when local
forecast structure changes. Fine sharing provides greater local freedom, but
offers weaker cross-step regularization and may be harder to estimate. We call
variation in this balance across samples, variables and future regions
**future-region sharing-demand heterogeneity**. It is neither a claim about
probabilistic dependence among future targets nor a claim that the requested
horizon changes the Bayes-optimal prediction of an already specified target.

To isolate this problem from a proposed method, consider capacity-matched
single-extent diagnostic predictors. A common encoder first maps a batch of
$N$ examples as

$$
\mathbf X:[N,L,C]\rightarrow\mathbf R:[N,C,D],
$$

after which the same generator constructs a candidate state for every future
step:

$$
\mathbf U_{n,c,\tau}
=
G_\omega\!\left(
\left[\mathbf R_{n,c},\boldsymbol\phi_\tau\right]
\right)
\in
\mathbb R^{D_z}.
$$

For extent $s$, contiguous groups
$\{\mathcal G_{s,g}\}_g$ partition the future domain and define

$$
\mathbf Z_{n,c,g}^{(s)}
=
\operatorname{LayerNorm}\!\left(
\frac{1}{|\mathcal G_{s,g}|}
\sum_{\tau\in\mathcal G_{s,g}}
\mathbf U_{n,c,\tau}
\right),
$$

$$
\widehat y_{n,c,\tau}^{(s)}
=
\left\langle
\mathbf a_\tau,
\mathbf Z_{n,c,g_s(\tau)}^{(s)}
\right\rangle
+\beta_\tau.
$$

Across settings of $s$, the encoder, generator, future-step descriptors,
synthesis vectors and biases have identical shapes and parameter counts, and
every predictor computes the full tensor $\mathbf U$. Only the parameter-free
pooling topology changes. Each diagnostic predictor contains one extent only;
there is no multi-extent fusion, target-conditioned allocation or auxiliary
balancing objective.

For aligned origin $o$, future region $\mathcal B_b$ and extent $s$, define
the matched region risk

$$
R_{o,b,s}
=
\frac{1}{|\mathcal B_b|C}
\sum_{\tau\in\mathcal B_b}
\sum_{c=1}^{C}
\left(
\widehat y_{o+\tau,c}^{(s)}
-
y_{o+\tau,c}
\right)^2,
$$

with population counterpart
$R_{b,s}=\mathbb E_o[R_{o,b,s}]$. Region-dependent sharing-scale preference is
supported descriptively when the matched risk curves cross and
$s_{o,b}^{\star}=\arg\min_sR_{o,b,s}$ changes across regions. For visualization,
the percentage excess risk above the region minimum is

$$
E_{o,b,s}
=
100
\frac{
R_{o,b,s}-\min_{s'}R_{o,b,s'}
}{
\min_{s'}R_{o,b,s'}
}.
$$

Figure 3 uses five single-extent predictors with
$s\in\{1,8,32,128,720\}$. All contain 111,312 parameters, are trained
end-to-end with the same uniform full-domain pointwise MSE objective, and are
evaluated over 12 contiguous 60-step regions. The diagnostic extent grid and
region boundaries are intentionally distinct from the production decoder and
from requested horizons. For the selected ETTm2 validation origin, the region
winners are
$[128,128,128,8,8,1,32,1,32,720,720,720]$: all five extents win two or three
regions, all ten extent pairs exhibit bidirectional crossings beyond the
predefined 0.5% margin, and the mean best-versus-second-best region margin is
10.266%. Relative to the best fixed extent for this sample ($s=720$), using
the label-selected region minimum yields a descriptive 8.112% average
headroom.

<a id="fig:sharing-heterogeneity"></a>

![Validation-only illustration of future-region sharing-demand heterogeneity.](../../analysis/iscf_bsca_intro_evidence_full_search_20260730/selected_figures/figure_intro_sharing_heterogeneity.png)

**Figure 3 | Preferred cross-step sharing extent varies across future
regions.** **a**, Percentage MSE excess of five capacity-matched neutral
decoders above the lowest-risk sharing extent within each 60-step future
region of one ETTm2 validation example. Outlined squares mark the region-wise
best extent. **b**, MSE reduction of each region winner relative to the best
fixed extent ($s=720$), with colors denoting the winning extent and the dashed
line showing the 12-region mean. All five extents win two or three regions,
and the descriptive region-wise minimum yields 8.1% lower average MSE than the
best fixed extent. The example was selected using validation labels and does
not represent out-of-sample allocation performance.

This result is illustrative existence evidence, not a learned-method result.
The same validation labels select the displayed origin, the region winners and
the descriptive oracle, so neither the winner sequence nor its headroom can be
interpreted as out-of-sample performance. A formal population claim requires
matched end-to-end training across datasets and seeds, validation-only
selection of both the fixed extent and the region schedule, and one frozen
official-test evaluation. Let

$$
s_{\mathrm{fixed}}^{\mathrm{val}}
=
\arg\min_s
\sum_b w_bR_{b,s}^{\mathrm{val}},
\qquad
s_b^{\mathrm{val}}
=
\arg\min_sR_{b,s}^{\mathrm{val}}.
$$

The corresponding **cross-fitted headroom (CFH)** is

$$
\operatorname{CFH}
=
\frac{
R_{\mathrm{fixed}}^{\mathrm{test}}
-
R_{\mathrm{region\ schedule}}^{\mathrm{test}}
}{
R_{\mathrm{fixed}}^{\mathrm{test}}
}.
$$

Such a schedule would splice predictions from independently trained
single-extent models and would therefore remain a diagnostic upper bound, not
a deployable forecaster. CFH has not yet been established by Figure 3.
Controls for any formal claim must keep the encoder, parameter count, data,
objective, optimizer, initialization class, checkpoint selector and evaluation
origins matched; verify target and scaler alignment; and report all datasets,
seeds and negative crossings. Forecast origins are the paired comparison
units; because neighboring origins have strongly overlapping targets,
uncertainty intervals should use a moving-block bootstrap with a frozen block
length of 720 origins. A group-size-matched random grouping control is
additionally required before attributing an effect specifically to temporal
contiguity.

### 3.5 Design requirements

The preceding formulation and evidence imply five requirements for a
varied-horizon decoder. First, one parameterized predictor should serve every
supported horizon. Second, the requested horizon should select the returned
support without changing predictions for shared future steps, thereby
satisfying CHPC by construction. Third, the output stage should not impose one
fixed cross-step sharing extent on the entire future domain. Fourth, it should
make several sharing extents available within one unified prediction function
and integrate them at the granularity of sample, variable and future step,
while retaining step-specific synthesis. Fifth, the resulting components must
be jointly trainable without allowing the integration mechanism to suppress
useful predictive paths before they have learned.

These requirements motivate **Independent Scope-Conditioned Forecasting
(ISCF)**, which constructs a unified forecast field over multiple output-side
sharing extents, together with **Balanced Scope Co-Adaptation (BSCA)**, a
training-only mechanism for supporting their joint learning. Section 4
develops this design; no method-effectiveness claim follows from the
illustrative evidence in Figures 2 and 3.

## Editorial evidence and claim audit

| Item | Evidence status | Permitted Section 3 claim | Prohibited promotion |
| --- | --- | --- | --- |
| Task formulation and CHPC | Mathematical definition | A horizon-agnostic step-indexed mapping satisfies CHPC by construction | CHPC implies lower forecast error |
| Figure 2 | Validation-only DLinear illustration plus all-validation ETTh2 NCHPD | Independently trained horizon-specific systems need not satisfy CHPC | Disagreement is prevalent across all model families; horizon-specific accuracy is worse |
| Naive unified penalty | Insufficient | $\operatorname{UP}_H$ is the required matched statistic; current evidence does not establish a stable positive penalty | Unified forecasting is intrinsically harder or currently proven superior |
| Figure 3 | Validation-selected, capacity-matched, single-sample illustration | Fixed sharing demand can vary across future regions in a finite-capacity diagnostic family | Oracle headroom is learnable, out-of-sample or attributable to ISCF-BSCA |
| CFH | Defined but not measured formally | States the future matched control needed for a population-level headroom claim | Figure 3 establishes formal test headroom |
| ISCF-BSCA | Introduced only after requirements | The problem analysis motivates the design requirements | Figures 2--3 establish component effectiveness, unified superiority or decoder portability |

The closest existing audit for the naive-unified question compares
horizon-specific specialists with `A6_MEASURE`: the aggregate MSE difference is
only 0.1659%, with specialists ahead in 7/15 dataset-horizon cells and 2/5
datasets. By contrast, changing the measure-training condition from
`A6_MEASURE` to `A6_FULL` produces a 1.7980% aggregate difference across 15/15
cells. Because the first contrast is small and the second is a larger,
systematic confound, the audit does not support a manuscript claim that naive
unification causes a stable performance penalty.

The Introduction P6 statements that one unified model outperforms
horizon-specific models, that the components are effective, and that the
decoder transfers across backbones remain provisional paper-facing claims.
They require the complete main, ablation and transfer tables and are not used
as evidence in this section.
