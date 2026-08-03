# ISCF-BSCA Section 3: Problem Formulation and Empirical Motivation

## Draft status

| Field | Content |
| --- | --- |
| `document_role` | Clean manuscript-facing initial draft of Section 3 |
| `version` | `v0.3-concise-polish` |
| `date` | `2026-08-03` |
| `review_status` | `author_feedback_round2_integrated_pending_review` |
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

Varied-horizon forecasting raises two coupled questions. What should a forecaster guarantee when different requests contain the same future targets? How should a unified decoder organize predictive information when sharing needs may change across the future domain? Without clear answers, unification remains an implementation choice rather than a well-defined forecasting system.

This section formalizes varied-horizon forecasting as nested views of one future trajectory. It then measures whether independently trained horizon-specific models satisfy this view and separates consistency from accuracy. Finally, a controlled decoder diagnostic examines sharing demand across future regions and motivates the design requirements in Section 3.5.

### 3.1 Varied-horizon forecasting and cross-horizon prefix consistency

Varied-horizon forecasting should be coherent across request endpoints. Given the same history, a future target shared by two requests has one semantic identity. Changing only the endpoint should therefore change the returned range, not the prediction at that target.

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

Here, $\mathbf X_o$ is a length-$L$ multivariate history, and $\mathbf Y_o^{(H)}$ contains the next $H$ targets for $C$ variables. Let $\mathcal H=\{H_1,\ldots,H_M\}$ denote the supported horizons and $T=\max\mathcal H$. Horizon $H$ sets the request endpoint, whereas $\tau$ identifies a future position. Target $(\tau,c)$ is therefore shared by every request with $H\geq\tau$.

Conventional long-term forecasting commonly learns a separate predictor $f_{\theta_H}$ for each $H$. Independent parameters and optimization provide no rule linking their overlapping outputs. We instead define a **future-step-indexed prediction function** $g_\theta(\mathbf X_o,\tau,c)$. A request of length $H$ collects

$$
\widehat{\mathbf Y}_o^{(H)}
=
\left[
g_\theta(\mathbf X_o,\tau,c)
\right]_{\tau=1,\ldots,H;\ c=1,\ldots,C}.
$$

Different requests are thus nested views of one predicted trajectory. We call this relation **cross-horizon prefix consistency (CHPC)**. For identical history, origin and preprocessing, CHPC requires

$$
\widehat y_{o+\tau,c}^{(H_i)}
=
\widehat y_{o+\tau,c}^{(H_j)},
\qquad
\forall\,H_i,H_j\in\mathcal H,\ H_i<H_j,\quad
1\leq\tau\leq H_i,\quad
1\leq c\leq C.
$$

The formulation satisfies CHPC by construction because every shared target evaluates the same function. Varied-horizon forecasting therefore requires more than parameter sharing across output lengths. It defines all requested outputs as coherent parts of one trajectory.

### 3.2 Horizon-specific prefix disagreement

CHPC is guaranteed by the formulation, but independently trained predictors could still agree empirically. We test this possibility by comparing their outputs on identical histories and overlapping targets.

At origin $o$, let $\widehat{\mathbf Y}_{o}^{(H_i)}\in\mathbb R^{H_i\times C}$ and $\widehat{\mathbf Y}_{o}^{(H_j)}\in\mathbb R^{H_j\times C}$ denote forecasts at horizons $H_i$ and $H_j$, where $H_i<H_j$. The two models are trained independently. Their **cross-horizon prefix disagreement (CHPD)** is

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

CHPD measures mean absolute disagreement in the original scale. To compare variables with different magnitudes, we define normalized CHPD (NCHPD):

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
}.
$$

Here, $\mathcal O$ contains aligned evaluation origins, $\sigma_c^{\mathrm{train}}$ is the train-split standard deviation of variable $c$, and $\epsilon>0$ prevents division by zero. We hold the history, origin, scaler and overlapping indices fixed. Self-comparison and two requests from one unified checkpoint both yield zero, excluding alignment and serialization artifacts.

Figure 2 shows measurable disagreement in the audited DLinear family on ETTh2. Among 15,127 validation origin-variable candidates, the displayed pair maximizes aggregate disagreement across six horizon pairs. Over the first 96 steps, the $H\in\{96,192,336\}$ forecasts differ from $H=720$ by 2.51, 2.16 and 2.40 in mean absolute raw scale. Across 2,161 aligned origins and all variables, every horizon pair has non-zero NCHPD, ranging from 0.0148 to 0.0406. The selected trajectory is a strong illustration, whereas the heatmap reports dataset-level averages only for this model family.

<a id="fig:prefix-disagreement"></a>

![Validation-only illustration of horizon-specific prefix disagreement.](../../analysis/iscf_bsca_intro_evidence_full_search_20260730/selected_figures/figure_intro_prefix_disagreement.png)

**Figure 2 | Independently optimized horizon-specific forecasts can disagree on shared future steps.** **a**, Predictions from DLinear models trained separately for horizons 96, 192, 336 and 720 on the same ETTh2 history. The panel shows 48 observed steps and the first 96 shared future steps. The inset reports mean absolute differences from the 720-step forecast. The displayed validation pair maximizes disagreement across all six horizon pairs among 15,127 candidates. **b**, NCHPD averaged over all ETTh2 validation origins ($n=2{,}161$) and variables. The selected example is illustrative, not a prevalence estimate.

Thus, separately optimized models can assign different values to the same target. For the audited family, independent optimization did not enforce CHPC. This evidence concerns consistency, not comparative accuracy or methods that guarantee invariance by design.

### 3.3 Evidence boundary for naive unified forecasting

CHPC concerns consistency, not accuracy. Figure 2 does not show that unification improves forecast error. A poorly designed unified model could instead lose useful horizon-specific specialization.

Testing this possibility requires matched horizon-specific and unified predictors. Encoder class, effective capacity, data, objective, optimization, checkpoint selection and evaluation origins must remain fixed. For horizon $H$, define the relative unified penalty

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

A positive $\operatorname{UP}_H$ indicates higher MSE for the evaluated unified predictor. A general compromise would require a stable positive effect across matched datasets and horizons.

Current evidence does not meet this standard. The closest diagnostic produced only a 0.1659% aggregate gap, with specialists ahead in 7/15 cells and 2/5 datasets. A training-protocol contrast produced a larger 1.7980% gap across all 15 cells, revealing a systematic confound. We therefore do not use a presumed unified-forecasting penalty as method motivation. Accuracy claims await the complete matched scorecards in the experimental section.

### 3.4 Future-region sharing-demand heterogeneity

CHPC constrains relations among requests, but not how a unified decoder constructs the trajectory. The decoder must determine how broadly to reuse history-conditioned states across future steps. One fixed reuse pattern assumes that all future regions require the same balance between shared structure and local flexibility.

The **sharing extent** $s$ is the number of future steps that reuse one latent state. A **future region** $\mathcal B_b\subseteq\{1,\ldots,T\}$ is a contiguous subset of the maximum future domain, not a requested horizon. Broad sharing offers stronger cross-step regularization but may smooth local changes. Fine sharing offers more local freedom but less regularization. We call variation in the preferred extent across regions **future-region sharing-demand heterogeneity**.

We probe this trade-off using capacity-matched single-extent predictors. They share the encoder, future-step generator, descriptors, step-specific synthesis, parameter count, data, objective and optimization. Only parameter-free contiguous pooling changes the sharing extent. The diagnostic excludes multi-extent fusion, target-conditioned allocation and auxiliary balancing.

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
\right)^2.
$$

Lower $R_{o,b,s}$ indicates a better extent for that region within the controlled family. Region dependence appears when risk curves cross and $s_{o,b}^{\star}=\arg\min_sR_{o,b,s}$ changes with $b$. Figure 3 reports each extent's percentage excess above the regional minimum.

Figure 3 evaluates five 111,312-parameter predictors with $s\in\{1,8,32,128,720\}$ over 12 contiguous 60-step regions. For the selected ETTm2 validation origin, the winners are $[128,128,128,8,8,1,32,1,32,720,720,720]$. Every extent wins two or three regions, and all ten extent pairs cross bidirectionally beyond the predefined 0.5% margin. The mean best-versus-second-best margin is 10.266%. Against the best fixed extent ($s=720$), the label-selected regional minimum has a descriptive 8.112% average headroom.

<a id="fig:sharing-heterogeneity"></a>

![Validation-only illustration of future-region sharing-demand heterogeneity.](../../analysis/iscf_bsca_intro_evidence_full_search_20260730/selected_figures/figure_intro_sharing_heterogeneity.png)

**Figure 3 | Preferred cross-step sharing extent varies across future regions.** **a**, Percentage MSE excess of five capacity-matched neutral decoders above the regional minimum for one ETTm2 validation example. Outlined squares mark the best extent in each 60-step region. **b**, MSE reduction of each regional winner relative to the best fixed extent ($s=720$); the dashed line gives the 12-region mean. Every extent wins two or three regions, and the regional minimum reduces average MSE by 8.1%. Validation labels selected the example and winners, so this is not out-of-sample allocation performance.

The evidential signal is the changing winner, not the oracle magnitude. No single extent minimizes risk across all regions in this controlled example. Figure 3 therefore supports finite-capacity regional heterogeneity, but not learned allocation, population prevalence or effectiveness of the proposed method.

A population-level test must select both the fixed extent and regional schedule on validation data, then freeze them for official-test evaluation. We define **cross-fitted headroom (CFH)** as

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

The schedule splices independently trained predictors, so CFH remains a diagnostic upper bound rather than a deployable forecast. Figure 3 does not establish CFH. Formal evaluation also requires matched initialization and checkpoint selection, verified target and scaler alignment, complete datasets and seeds, and forecast-origin pairing. Overlapping targets require a moving-block bootstrap with a frozen 720-origin block length. A group-size-matched random grouping control is needed to attribute any effect to temporal contiguity.

### 3.5 Design requirements

The analysis yields three design requirements. First, one parameterized predictor should serve every supported horizon, and the endpoint should determine only how much of one trajectory is returned. This makes CHPC an architectural property.

Second, the decoder should expose multiple sharing extents, integrate them across future steps and retain step-specific synthesis. Conditioning that integration on samples and variables remains a method hypothesis requiring ablation. Figure 3 supports regional variation only, not intrinsic probabilistic coupling among future targets.

Third, the predictive paths must co-adapt before integration concentrates on a subset. This is an optimization requirement, not evidence from Figures 2 or 3. These requirements motivate **Independent Scope-Conditioned Forecasting (ISCF)** and its training-only **Balanced Scope Co-Adaptation (BSCA)** mechanism. Section 4 presents the design; method effectiveness remains to be established experimentally.

## Editorial evidence and claim audit

| Item | Evidence status | Permitted Section 3 claim | Prohibited promotion |
| --- | --- | --- | --- |
| Task formulation and CHPC | Mathematical definition | A horizon-agnostic future-step-indexed mapping satisfies CHPC by construction | CHPC implies lower forecast error |
| Figure 2 | Validation-only DLinear illustration plus all-validation ETTh2 NCHPD | Independently trained horizon-specific systems need not satisfy CHPC | Disagreement is prevalent across all model families; horizon-specific accuracy is worse |
| Naive unified penalty | Insufficient | $\operatorname{UP}_H$ is the required matched statistic; current evidence does not establish a stable positive penalty | Unified forecasting is intrinsically harder or currently proven superior |
| Figure 3 | Validation-selected, capacity-matched, single-sample illustration | Fixed sharing demand can vary across future regions in a finite-capacity diagnostic family | Oracle headroom is learnable, out-of-sample or attributable to ISCF-BSCA |
| CFH | Defined but not measured formally | States the future matched control needed for a population-level headroom claim | Figure 3 establishes formal test headroom |
| ISCF-BSCA | Introduced only after requirements | The problem analysis motivates the design requirements | Figures 2--3 establish component effectiveness, unified superiority or decoder portability |
