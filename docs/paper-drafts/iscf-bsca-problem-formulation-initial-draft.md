# ISCF-BSCA Section 3: Problem Formulation and Empirical Motivation

## Draft status

| Field | Content |
| --- | --- |
| `document_role` | Clean manuscript-facing initial draft of Section 3 |
| `version` | `v0.2-narrative-refinement` |
| `date` | `2026-07-31` |
| `review_status` | `author_feedback_round1_integrated_pending_review` |
| `introduction_dependency` | Introduction `v0.9-author-refinement` remains unchanged |
| `figure_2_status` | Approved validation-only illustrative evidence; integrated below |
| `figure_3_status` | Approved validation-only illustrative evidence; integrated below |
| `method_figure_4_status` | Planned only; not generated or referenced as completed |
| `result_boundary` | No main-table, ablation or transfer claim is treated as established |
| `narrative_spine` | Coherent varied-horizon task → observed prefix disagreement → accuracy boundary → heterogeneous sharing demand → design requirements |

The status table and the editorial audit after Section 3 are not part of the manuscript body submitted for review.

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

Related work shows that forecasting models are increasingly expected to serve multiple prediction ranges, yet it leaves two questions unresolved. First, what should a varied-horizon forecaster guarantee when different requests cover the same future steps? Second, once those requests are unified, how should the decoder organize predictive information across a future domain whose local structure may change with lead time? Without clear answers, a unified model is only a shared implementation, not a well-defined forecasting system.

This section addresses these questions in sequence. We first formulate varied-horizon forecasting as producing nested views of one future trajectory and define the consistency that this view requires. We then use independently trained horizon-specific models to show why conventional practice does not provide that consistency, while separating this systems issue from predictive accuracy. Finally, we examine how a finite-capacity decoder should share history-conditioned states across future regions. The resulting task and evidence boundaries lead to the design requirements stated in Section 3.5.

### 3.1 Varied-horizon forecasting and cross-horizon prefix consistency

The first requirement of varied-horizon forecasting is semantic coherence across requests. If two requests begin from the same history and both include future step $\tau$, they refer to the same forecast target; changing only the requested endpoint should not change what the system predicts at that shared step. We formalize the task around this observation.

At forecast origin $o$, let

$$
\mathbf X_o
=
\left[\mathbf x_{o-L+1},\ldots,\mathbf x_o\right]
\in\mathbb R^{L\times C},
\qquad
\mathbf Y_o^{(H)}
=
\left[y_{o+\tau,c}\right]_{\tau=1,\ldots,H;\ c=1,\ldots,C}
\in\mathbb R^{H\times C}.
$$

Here, $\mathbf X_o$ is a length-$L$ multivariate history, $C$ is the number of variables, and $\mathbf Y_o^{(H)}$ contains the next $H$ targets. We consider a supported horizon set $\mathcal H=\{H_1,\ldots,H_M\}$ and let $T=\max\mathcal H$. The forecast horizon $H$ specifies how far a request extends, whereas the future time step $\tau$ identifies one position within that request. Thus, target $(\tau,c)$ is shared by every request with $H\geq\tau$.

Conventional long-term forecasting treats each requested horizon as a separate learning problem:

$$
\widehat{\mathbf Y}_o^{(H)}
=
f_{\theta_H}(\mathbf X_o),
\qquad
H\in\mathcal H.
$$

Because both parameters and optimization are horizon-specific, the resulting models have no shared rule for their overlapping outputs. Varied-horizon forecasting instead seeks one model whose prediction for a target is defined independently of the requested endpoint. We express this idea through a **future-step-indexed prediction function**: one function is queried by the history, future time step and variable,

$$
g_\theta:
(\mathbf X_o,\tau,c)
\mapsto
\widehat y_{o+\tau,c},
$$

and a request of length $H$ simply collects its first $H$ future-step predictions:

$$
\widehat{\mathbf Y}_o^{(H)}
=
\left[
g_\theta(\mathbf X_o,\tau,c)
\right]_{\tau=1,\ldots,H;\ c=1,\ldots,C}.
$$

This formulation turns different horizon requests into nested views of the same predicted trajectory. We call the required nesting relation **cross-horizon prefix consistency (CHPC)**. For identical history, forecast origin and preprocessing, CHPC requires

$$
\widehat y_{o+\tau,c}^{(H_i)}
=
\widehat y_{o+\tau,c}^{(H_j)},
\qquad
\forall\,H_i,H_j\in\mathcal H,\ H_i<H_j,\quad
1\leq\tau\leq H_i,\quad
1\leq c\leq C.
$$

In other words, the shorter and longer requests must return the same value for every target they share. Our formulation of varied-horizon forecasting naturally satisfies CHPC because both requests evaluate the same function $g_\theta(\mathbf X_o,\tau,c)$ over their shared steps. Varied-horizon forecasting is therefore not merely parameter sharing across output lengths; it defines those outputs as coherent parts of one future trajectory.

### 3.2 Horizon-specific prefix disagreement

The task definition raises an empirical question: do independently trained horizon-specific systems already behave as coherent views of one trajectory, even though they are not required to do so? We answer this by measuring how their predictions differ on exactly the same history and future targets.

For the same forecast origin $o$, let $\widehat{\mathbf Y}_{o}^{(H_i)}\in\mathbb R^{H_i\times C}$ and $\widehat{\mathbf Y}_{o}^{(H_j)}\in\mathbb R^{H_j\times C}$ be predictions from two models trained separately for $H_i<H_j$. Their **cross-horizon prefix disagreement (CHPD)** is

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

CHPD reports the mean absolute difference in the original data scale. Because variables can have substantially different magnitudes, we also define normalized CHPD (NCHPD):

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

where $\mathcal O$ is the set of aligned evaluation origins, $\sigma_c^{\mathrm{train}}$ is the standard deviation of variable $c$ estimated from the training split, and $\epsilon>0$ prevents division by zero. The comparison holds the numerical history, origin, scaler and overlapping target indices fixed. As alignment controls, comparing one horizon-specific checkpoint with itself and comparing two horizon requests from one unified checkpoint both yield exactly zero, ruling out alignment or serialization artifacts.

Figure 2 shows that the absence of a consistency constraint is visible in practice. In the DLinear experiment on ETTh2, the displayed origin-variable pair maximizes disagreement aggregated over all six horizon pairs among 15,127 validation candidates. Within the first 96 future steps, the forecasts requested at horizons 96, 192 and 336 differ from the 720-step forecast by mean absolute raw-scale values of 2.51, 2.16 and 2.40, respectively. Across 2,161 aligned validation origins and all variables, NCHPD remains non-zero for every pair, ranging from 0.0148 to 0.0406. The selected trajectory is intentionally a strong example rather than a prevalence estimate; the aggregate heatmap establishes dataset-level average disagreement only for the audited DLinear family.

<a id="fig:prefix-disagreement"></a>

![Validation-only illustration of horizon-specific prefix disagreement.](../../analysis/iscf_bsca_intro_evidence_full_search_20260730/selected_figures/figure_intro_prefix_disagreement.png)

**Figure 2 | Independently optimized horizon-specific forecasts can disagree on the same future steps.** **a**, Predictions from four DLinear models trained separately for horizons 96, 192, 336 and 720 on the same ETTh2 history. The panel shows the final 48 observed steps and the first 96 future steps shared by all four requested horizons. Colors and sparse, staggered marker shapes identify the four predictions; the inset reports their mean absolute differences from the 720-step forecast on the common prefix. The displayed validation origin-variable pair maximizes mean absolute disagreement aggregated over all six horizon pairs among 15,127 candidates. **b**, Normalized cross-horizon prefix disagreement (NCHPD) averaged over all ETTh2 validation origins ($n=2{,}161$) and variables. The selected example is illustrative and is not a prevalence estimate.

The result turns the coherence requirement from an abstract preference into an observable systems issue: separately optimized horizon models can assign different values to a target that is identical in time and variable identity. The evidence is deliberately narrower than a performance comparison. It shows that independent horizon-specific optimization does not provide CHPC; it does not yet tell us whether either forecast is more accurate, nor does it characterize varied-horizon methods that enforce invariance by design.

### 3.3 Evidence boundary for naive unified forecasting

Coherence and predictive accuracy are distinct questions. Figure 2 motivates why a varied-horizon system should define overlapping outputs consistently, but it does not imply that unification improves forecast error. It is equally possible that a poorly designed unified model trades horizon-specific specialization for consistency.

Testing that possibility requires a matched comparison between a horizon-specific predictor and its unified counterpart, with encoder class, effective capacity, training data, objective, optimization, checkpoint selection and evaluation origins held fixed. For horizon $H$, define the relative unified penalty

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

A positive $\operatorname{UP}_H$ means that the evaluated unified adaptation has higher MSE under this protocol. Establishing a general compromise would require this effect to remain positive and stable across matched dataset-horizon comparisons.

The available evidence does not meet that standard. The closest diagnostic comparison is small and heterogeneous across dataset-horizon cells, while a separate training-protocol contrast produces a larger and more consistent effect. The observed difference therefore cannot be attributed to horizon unification. We consequently do not use a presumed accuracy penalty to motivate our method. At this stage, the case for varied-horizon forecasting is the need for a single deployable, coherent system; whether the proposed realization also improves accuracy remains a question for the complete matched scorecards in the experimental section.

### 3.4 Future-region sharing-demand heterogeneity

CHPC resolves how different requests should relate, but not how one unified decoder should construct the trajectory itself. A decoder must decide how broadly to reuse history-conditioned predictive states across future steps. Using one fixed reuse pattern throughout the future domain is a strong assumption: it treats early, middle and late regions as if they required the same balance between shared structure and step-specific flexibility.

We call the number of future steps that reuse one history-conditioned latent state the **sharing extent** $s$. A **future region** $\mathcal B_b\subseteq\{1,\ldots,T\}$ is a contiguous part of the maximum future domain, rather than a requested forecast horizon. Broad sharing can regularize a long trajectory through a common state, but may smooth local changes; fine sharing offers greater local freedom, but weaker cross-step regularization. We hypothesize that the preferred balance varies across samples, variables and future regions, a problem we term **future-region sharing-demand heterogeneity**.

To test whether this trade-off is observable without presupposing our method, we construct capacity-matched single-extent diagnostic predictors. Every predictor uses the same encoder, future-step generator, descriptors, step-specific synthesis, parameter count, data, objective and optimization; only parameter-free contiguous pooling changes how many future steps share a latent state. Each model contains exactly one sharing extent, so the diagnostic does not include multi-extent fusion, target-conditioned allocation or an auxiliary balancing objective.

For aligned origin $o$, we measure the risk of extent $s$ within region $\mathcal B_b$ as

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

where lower $R_{o,b,s}$ indicates that extent $s$ better matches the finite-capacity demand of that region under the controlled family. Preference is region-dependent when the matched risk curves cross and $s_{o,b}^{\star}=\arg\min_sR_{o,b,s}$ changes with $b$. Figure 3 reports the percentage excess of each extent above the minimum risk in each region.

For the selected validation example, the diagnostic reveals a markedly non-uniform risk landscape. Figure 3 uses five 111,312-parameter predictors with $s\in\{1,8,32,128,720\}$, trained end-to-end with the same uniform full-domain MSE objective and evaluated over 12 contiguous 60-step regions. For the selected ETTm2 validation origin, the region winners are $[128,128,128,8,8,1,32,1,32,720,720,720]$: all five extents win two or three regions, all ten extent pairs exhibit bidirectional crossings beyond the predefined 0.5% margin, and the mean best-versus-second-best region margin is 10.266%. Relative to the best fixed extent for this sample ($s=720$), using the label-selected region minimum yields a descriptive 8.112% average headroom.

<a id="fig:sharing-heterogeneity"></a>

![Validation-only illustration of future-region sharing-demand heterogeneity.](../../analysis/iscf_bsca_intro_evidence_full_search_20260730/selected_figures/figure_intro_sharing_heterogeneity.png)

**Figure 3 | Preferred cross-step sharing extent varies across future regions.** **a**, Percentage MSE excess of five capacity-matched neutral decoders above the lowest-risk sharing extent within each 60-step future region of one ETTm2 validation example. Outlined squares mark the region-wise best extent. **b**, MSE reduction of each region winner relative to the best fixed extent ($s=720$), with colors denoting the winning extent and the dashed line showing the 12-region mean. All five extents win two or three regions, and the descriptive region-wise minimum yields 8.1% lower average MSE than the best fixed extent. The example was selected using validation labels and does not represent out-of-sample allocation performance.

The value of this example lies in the change of winner identity, not in the oracle number itself. Within one controlled prediction problem, no single extent minimizes risk across all future regions. This supports the existence of finite-capacity sharing-demand heterogeneity and challenges the assumption that one fixed output-side sharing pattern is uniformly appropriate. Nevertheless, the same validation labels select the example, its region winners and the descriptive 8.112% headroom. Figure 3 is therefore illustrative problem evidence, not out-of-sample allocation performance or a result of the proposed method.

A population-level claim requires a stricter cross-fitted comparison. The fixed extent and the region-wise schedule must both be selected using validation risk,

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

and are then frozen for one official-test evaluation through **cross-fitted headroom (CFH)**:

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

Such a schedule would splice predictions from independently trained single-extent models and would therefore remain a diagnostic upper bound, not a deployable forecaster, and Figure 3 does not establish CFH. Any formal claim must additionally retain matched initialization and checkpoint selection, verify target and scaler alignment, report all datasets, seeds and negative crossings, and use forecast origin as the paired unit. Because neighboring origins share targets, uncertainty intervals require a moving-block bootstrap with a frozen block length of 720 origins. A group-size-matched random grouping control is further required before attributing an effect specifically to temporal contiguity.

### 3.5 Design requirements

The analysis above narrows the design target without prescribing a particular architecture. The task formulation yields two system-level requirements. One parameterized predictor should serve every supported horizon, and changing the requested endpoint should only change how much of its trajectory is returned. Together, these requirements make CHPC an architectural property rather than an agreement that must be recovered after training.

The sharing diagnostic adds an output-side requirement. A unified decoder should not impose one fixed sharing extent on the entire future domain; it should make multiple extents available and allow their integration to change across future steps within a sample, while retaining step-specific synthesis. The subsequent method can condition this integration more generally on sample and variable identity, but the value of that additional granularity must be tested by ablation rather than inferred from Figure 3. This remains an adaptive finite-capacity design requirement, not a claim that future targets have an intrinsic or horizon-dependent probabilistic coupling.

Finally, exposing several predictive paths creates an optimization concern: the paths must learn jointly before the integration mechanism concentrates on a subset of them. This requirement is not established by Figures 2 or 3; it is a design condition that the subsequent method must satisfy and that component ablations must test.

These requirements motivate **Independent Scope-Conditioned Forecasting (ISCF)**, which constructs a unified forecast field over multiple output-side sharing extents, together with **Balanced Scope Co-Adaptation (BSCA)**, a training-only mechanism for supporting their joint learning. Section 4 develops this design; no method-effectiveness claim follows from the illustrative evidence in Figures 2 and 3.

## Editorial evidence and claim audit

| Item | Evidence status | Permitted Section 3 claim | Prohibited promotion |
| --- | --- | --- | --- |
| Task formulation and CHPC | Mathematical definition | A horizon-agnostic future-step-indexed mapping satisfies CHPC by construction | CHPC implies lower forecast error |
| Figure 2 | Validation-only DLinear illustration plus all-validation ETTh2 NCHPD | Independently trained horizon-specific systems need not satisfy CHPC | Disagreement is prevalent across all model families; horizon-specific accuracy is worse |
| Naive unified penalty | Insufficient | $\operatorname{UP}_H$ is the required matched statistic; current evidence does not establish a stable positive penalty | Unified forecasting is intrinsically harder or currently proven superior |
| Figure 3 | Validation-selected, capacity-matched, single-sample illustration | Fixed sharing demand can vary across future regions in a finite-capacity diagnostic family | Oracle headroom is learnable, out-of-sample or attributable to ISCF-BSCA |
| CFH | Defined but not measured formally | States the future matched control needed for a population-level headroom claim | Figure 3 establishes formal test headroom |
| ISCF-BSCA | Introduced only after requirements | The problem analysis motivates the design requirements | Figures 2--3 establish component effectiveness, unified superiority or decoder portability |

The closest existing audit for the naive-unified question compares horizon-specific specialists with `A6_MEASURE`: the aggregate MSE difference is only 0.1659%, with specialists ahead in 7/15 dataset-horizon cells and 2/5 datasets. By contrast, changing the measure-training condition from `A6_MEASURE` to `A6_FULL` produces a 1.7980% aggregate difference across 15/15 cells. Because the first contrast is small and the second is a larger, systematic confound, the audit does not support a manuscript claim that naive unification causes a stable performance penalty.

The Introduction P6 statements that one unified model outperforms horizon-specific models, that the components are effective, and that the decoder transfers across backbones remain provisional paper-facing claims. They require the complete main, ablation and transfer tables and are not used as evidence in this section.
