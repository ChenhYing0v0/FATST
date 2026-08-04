# ISCF-BSCA Section 3 v0.5: Highlighted Review

## Review status

| Field | Content |
| --- | --- |
| `document_role` | Review-only comparison for Section 3 v0.5 |
| `comparison_baseline` | Section 3 `v0.4-field-style-alignment` |
| `canonical_clean_draft` | `docs/paper-drafts/iscf-bsca-problem-formulation-initial-draft.md` |
| `highlight_rule` | Yellow highlight denotes added, replacement or moved text; strikethrough denotes removed text |
| `review_scope` | Manuscript body, subsection structure and figure captions |
| `manuscript_status` | This file is not the clean manuscript source |

## Structural changes

| Location | Highlighted change |
| --- | --- |
| 3.1 | CHPC is defined before contrasting horizon-specific and unified formulations |
| 3.2 | `inconsistency` names the CHPC violation; CHPD/NCHPD quantify prediction disagreement |
| Former 3.3 | `Accuracy under naive unified forecasting` is removed |
| Former 3.4 | Renumbered to 3.3 and reorganized around Figure 3a heterogeneity and Figure 3b oracle headroom |
| Former 3.5 | `Design requirements` is removed; method introduction begins in Section 4 |
| Supporting ledger | `Cross-horizon prefix inconsistency` is added; $\operatorname{UP}_H$ and CFH are removed from the manuscript-facing terminology |

## 3. Problem Formulation and Empirical Motivation

The preceding discussion shows that varied-horizon forecasting involves more than sharing parameters across prediction lengths. A unified forecaster should return coherent predictions for future steps shared by different horizon requests, while its decoder must determine how predictive information is shared across the future domain. We now formalize these two issues and examine them through controlled empirical analyses, which motivate the decoder design developed in Section 4.

### 3.1 Varied-horizon forecasting and cross-horizon prefix consistency

Consider two forecasting requests with horizons $H_i<H_j$ issued from the same observed history. Their first $H_i$ steps refer to identical future targets and should therefore receive identical predictions. We formalize this requirement below.

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

Here, $\mathbf X_o$ is a length-$L$ multivariate history, and $\mathbf Y_o^{(H)}$ contains the next $H$ targets for $C$ variables. Let $\mathcal H=\{H_1,\ldots,H_M\}$ denote the supported horizons and $T=\max\mathcal H$. The horizon $H$ specifies the request endpoint, whereas $\tau$ indexes a particular future step. Thus, target $(\tau,c)$ is shared by all requests with $H\geq\tau$.

<del>In v0.4, the horizon-specific and unified formulations were introduced here before CHPC was defined.</del>

<mark>This shared-target relation defines **cross-horizon prefix consistency (CHPC)**. Let $\widehat y_{o+\tau,c}^{(H)}$ denote the prediction returned for target $(\tau,c)$ under a request with horizon $H$. Given identical history, forecast origin and preprocessing, CHPC requires</mark>

$$
\widehat y_{o+\tau,c}^{(H_i)}
=
\widehat y_{o+\tau,c}^{(H_j)},
\qquad
\forall\,H_i,H_j\in\mathcal H,\ H_i<H_j,\quad
1\leq\tau\leq H_i,\quad
1\leq c\leq C.
$$

<del>Each horizon request is therefore a prefix of one predicted trajectory. We refer to this relation as **cross-horizon prefix consistency (CHPC)**.</del>

<mark>Conventional long-term forecasting learns an independent predictor $f_{\theta_H}$ for each horizon. Because both parameters and optimization are horizon-specific, these predictors are not constrained to satisfy CHPC on their overlapping outputs.</mark>

<mark>A varied-horizon forecaster should instead use a single **future-step-indexed prediction function** $g_\theta(\mathbf X_o,\tau,c)$ and construct an $H$-step forecast as</mark>

$$
\widehat{\mathbf Y}_o^{(H)}
=
\left[
g_\theta(\mathbf X_o,\tau,c)
\right]_{\tau=1,\ldots,H;\ c=1,\ldots,C}.
$$

<del>Because all requests query the same function at a shared target, CHPC holds by construction. Changing the requested horizon alters only the returned prefix length, not predictions already defined on that prefix.</del>

<mark>All horizon requests now query the same function at a shared target, so CHPC holds by construction. Changing the requested horizon alters only the returned prefix length, and each request becomes a nested view of one predicted trajectory.</mark>

### 3.2 <mark>Horizon-specific prefix inconsistency</mark>

<del>Although separately trained horizon-specific predictors could agree on their overlapping outputs, their training objectives do not enforce such agreement. We quantify the resulting inconsistency on aligned histories and shared future targets.</del>

<mark>Independently trained horizon-specific predictors may agree on some overlapping outputs, but their objectives do not enforce CHPC. We refer to a violation of CHPC as **cross-horizon prefix inconsistency** and quantify its magnitude through prediction disagreement on aligned histories and shared future targets.</mark>

At origin $o$, let $\widehat{\mathbf Y}_{o}^{(H_i)}\in\mathbb R^{H_i\times C}$ and $\widehat{\mathbf Y}_{o}^{(H_j)}\in\mathbb R^{H_j\times C}$ denote forecasts from independently trained models, where $H_i<H_j$. We define their **cross-horizon prefix disagreement (CHPD)** as

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

CHPD measures the mean absolute difference over the shared prefix in the original data scale. To aggregate variables with different magnitudes, we further define normalized CHPD (NCHPD):

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

<del>Here, $\mathcal O$ denotes the aligned evaluation origins, $\sigma_c^{\mathrm{train}}$ is the train-split standard deviation of variable $c$, and $\epsilon>0$ prevents division by zero. All comparisons use identical histories, forecast origins, scalers and overlapping target indices. As sanity checks, self-comparison and two requests from one unified checkpoint both yield zero disagreement.</del>

<mark>Here, $\mathcal O$ denotes the aligned evaluation origins, $\sigma_c^{\mathrm{train}}$ is the train-split standard deviation of variable $c$, and $\epsilon>0$ prevents division by zero. All comparisons use identical histories, forecast origins, scalers and overlapping target indices.</mark>

<del>As shown in Figure 2, independently optimized DLinear models exhibit clear prefix disagreement on ETTh2. The displayed origin-variable pair maximizes aggregate disagreement across six horizon pairs among 15,127 validation candidates. Over the first 96 future steps, forecasts for $H\in\{96,192,336\}$ differ from the $H=720$ forecast by 2.51, 2.16 and 2.40 in mean absolute raw scale, respectively. When averaged over 2,161 aligned origins and all variables, NCHPD remains non-zero for every horizon pair and ranges from 0.0148 to 0.0406. The trajectory provides a deliberately strong example, while the heatmap summarizes average disagreement within the audited model family.</del>

<mark>To visualize this inconsistency, we compare multi-horizon forecasts from DLinear models independently optimized on ETTh2. Figure 2a shows that their predictions diverge over the same 96 future steps: relative to the $H=720$ forecast, the forecasts for $H\in\{96,192,336\}$ differ by 2.51, 2.16 and 2.40 in mean absolute raw scale, respectively.</mark>

<mark>We further evaluate all 2,161 aligned ETTh2 validation origins and variables, with the resulting NCHPD matrix shown in Figure 2b. NCHPD is non-zero for every horizon pair and is more pronounced between the longest and shorter requested horizons: the $H=96$, $H=192$ and $H=336$ comparisons with $H=720$ yield 0.0406, 0.0365 and 0.0366, whereas pairs among the three shorter horizons range from 0.0148 to 0.0166. The independently optimized DLinear models evaluated here therefore do not form a single prefix-consistent prediction trajectory.</mark>

<a id="fig:prefix-disagreement"></a>

![Validation-only illustration of horizon-specific prefix disagreement.](../../analysis/iscf_bsca_intro_evidence_full_search_20260730/selected_figures/figure_intro_prefix_disagreement.png)

<del>**Figure 2 | Independently optimized horizon-specific forecasts can disagree on shared future steps.** **a**, Predictions from DLinear models trained separately for horizons 96, 192, 336 and 720 on the same ETTh2 history. The panel shows 48 observed steps and the first 96 shared future steps. The inset reports mean absolute differences from the 720-step forecast. The displayed validation pair maximizes disagreement across all six horizon pairs among 15,127 candidates. **b**, NCHPD averaged over all ETTh2 validation origins ($n=2{,}161$) and variables. The selected example is illustrative, not a prevalence estimate.</del>

<mark>**Figure 2 | Independently optimized horizon-specific forecasts are inconsistent on shared future steps.** **a**, An ETTh2 validation trajectory selected by aggregate CHPD, showing predictions from DLinear models trained separately for horizons 96, 192, 336 and 720. The panel includes 48 observed steps and the first 96 shared future steps; the inset reports mean absolute differences from the 720-step forecast. **b**, NCHPD averaged over all ETTh2 validation origins ($n=2{,}161$) and variables.</mark>

<del>These results show that independent horizon-specific optimization can produce different predictions for the same target. Importantly, CHPD measures cross-horizon consistency rather than forecast accuracy; it neither ranks the individual forecasts nor characterizes models that satisfy CHPC by design.</del>

<del>### 3.3 Accuracy under naive unified forecasting</del>

<del>Cross-horizon consistency alone does not guarantee accurate forecasting. A unified model may satisfy CHPC yet underperform horizon-specific predictors if its shared representation fails to preserve useful specialization.</del>

<del>Assessing this possibility requires matched horizon-specific and unified predictors with the same encoder class, effective capacity, data, objective, optimization, checkpoint selection and evaluation origins. The subsection defined the relative unified penalty as $\operatorname{UP}_H=(\operatorname{MSE}^{\mathrm{unified}}_H-\operatorname{MSE}^{\mathrm{specific}}_H)/\operatorname{MSE}^{\mathrm{specific}}_H$.</del>

<del>A positive $\operatorname{UP}_H$ indicates higher MSE for the unified predictor. A general accuracy trade-off would require this penalty to remain positive across matched datasets and horizons.</del>

<del>The available diagnostic evidence does not reveal such a stable penalty. The closest comparison yields only a 0.1659% aggregate gap, with specialists ahead in 7/15 cells and 2/5 datasets. By contrast, changing the training protocol produces a larger 1.7980% gap across all 15 cells. This confound prevents the observed difference from being attributed to unification alone. We therefore leave the accuracy effect of unified forecasting to the complete matched evaluation in the experimental section.</del>

### 3.3 <mark>Future-region sharing-demand heterogeneity</mark>

<del>CHPC specifies how different horizon requests should relate, but does not determine how a unified decoder should construct the future trajectory. The output stage must decide how broadly each history-conditioned state is reused before step-specific prediction. Applying one fixed sharing pattern assumes that all future regions require the same balance between shared structure and local flexibility.</del>

<mark>CHPC specifies how predictions from different horizon requests should relate, but it does not determine how a unified decoder should construct the shared future trajectory. Simply adapting an architecture designed for one fixed horizon leaves this output-side requirement unresolved: a varied-horizon forecaster must learn from requests spanning short to long endpoints while producing one prefix-consistent trajectory. The decoder is therefore central to how predictive information is organized across the future domain, particularly in determining how broadly each history-conditioned state is reused before step-specific prediction.</mark>

We use the **sharing extent** $s$ to denote the number of future steps that reuse one latent state. A **future region** $\mathcal B_b\subseteq\{1,\ldots,T\}$ is a contiguous subset of the maximum future domain rather than a requested horizon. Broad sharing can capture persistent trajectory structure but may smooth local variations, whereas fine sharing offers greater local flexibility with weaker cross-step regularization. We refer to variation in the preferred extent across regions as **future-region sharing-demand heterogeneity**.

<del>To examine whether one extent is sufficient throughout the future domain, we construct capacity-matched single-extent predictors. All predictors share the encoder, future-step generator, descriptors, step-specific synthesis, parameter count, data, objective and optimization. Only parameter-free contiguous pooling changes the sharing extent. This diagnostic deliberately excludes multi-extent fusion, target-conditioned allocation and auxiliary balancing.</del>

<mark>To isolate this factor, we construct capacity-matched single-extent predictors that differ only in sharing extent; all other architecture, training and evaluation settings remain identical.</mark>

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

<del>Lower $R_{o,b,s}$ indicates that extent $s$ better matches region $\mathcal B_b$ within the controlled family. A region-dependent preference appears when the matched risk curves cross and $s_{o,b}^{\star}=\arg\min_sR_{o,b,s}$ changes with $b$. Figure 3 reports the percentage excess risk of each extent above the regional minimum.</del>

<mark>Lower $R_{o,b,s}$ indicates that extent $s$ better matches region $\mathcal B_b$ within the controlled family. A region-dependent preference appears when $s_{o,b}^{\star}=\arg\min_sR_{o,b,s}$ changes with $b$.</mark>

<del>As shown in Figure 3, five 111,312-parameter predictors with $s\in\{1,8,32,128,720\}$ exhibit different preferences across 12 contiguous 60-step regions. For the selected ETTm2 validation origin, the winning extents are $[128,128,128,8,8,1,32,1,32,720,720,720]$. Each extent wins two or three regions, and all ten extent pairs exhibit bidirectional crossings beyond the predefined 0.5% margin. The mean best-versus-second-best margin is 10.266%. Relative to the best fixed extent ($s=720$), the label-selected regional minimum provides a descriptive average headroom of 8.112%.</del>

<mark>We evaluate the five predictors with $s\in\{1,8,32,128,720\}$ across 12 contiguous 60-step future regions on ETTm2. Figure 3a reports the percentage MSE excess of each extent above the lowest-risk extent within each region, with outlined squares marking the region-wise winners. The preferred extents vary across the future domain: each of the five extents wins two or three regions, all ten extent pairs exhibit bidirectional crossings beyond the predefined 0.5% margin, and the mean best-versus-second-best margin reaches 10.266%. No fixed extent minimizes risk throughout this controlled example, supporting heterogeneous sharing demand across future regions.</mark>

<a id="fig:sharing-heterogeneity"></a>

![Validation-only illustration of future-region sharing-demand heterogeneity.](../../analysis/iscf_bsca_intro_evidence_full_search_20260730/selected_figures/figure_intro_sharing_heterogeneity.png)

<del>**Figure 3 | Preferred cross-step sharing extent varies across future regions.** **a**, Percentage MSE excess of five capacity-matched neutral decoders above the regional minimum for one ETTm2 validation example. Outlined squares indicate the best extent in each 60-step region. **b**, MSE reduction of each regional winner relative to the best fixed extent ($s=720$); the dashed line denotes the mean over 12 regions. Every extent wins two or three regions, and the regional minimum reduces average MSE by 8.1%. Both the example and regional winners are selected using validation labels, so the figure does not measure out-of-sample allocation.</del>

<mark>**Figure 3 | Preferred sharing extent varies across future regions.** **a**, Percentage MSE excess of five capacity-matched single-extent predictors above the regional minimum for one selected ETTm2 validation example; outlined squares mark the best extent in each 60-step region. **b**, MSE reduction obtained by selecting the regional winner instead of the best fixed extent ($s=720$); the dashed line denotes the 12-region mean.</mark>

<del>The relevant observation is the change in winner identity across regions. No single extent minimizes risk throughout this controlled example, supporting heterogeneous sharing demand within a finite-capacity decoder family. Because validation labels select both the example and its regional winners, Figure 3 does not establish population prevalence, learned allocation or method effectiveness.</del>

<del>Extending this analysis beyond a selected example requires a cross-fitted comparison. Both the fixed extent and the regional schedule are selected on validation data and frozen before official-test evaluation. The subsection defined cross-fitted headroom as $\operatorname{CFH}=(R_{\mathrm{fixed}}^{\mathrm{test}}-R_{\mathrm{region\ schedule}}^{\mathrm{test}})/R_{\mathrm{fixed}}^{\mathrm{test}}$.</del>

<del>Since the regional schedule combines independently trained predictors, CFH remains a diagnostic upper bound rather than a deployable forecast. Figure 3 does not estimate this quantity. A formal comparison additionally requires matched initialization and checkpoint selection, verified target and scaler alignment, complete datasets and seeds, and forecast-origin pairing. Because neighboring origins share targets, uncertainty should be estimated with a moving-block bootstrap using a frozen block length of 720 origins. A group-size-matched random grouping control separates temporal contiguity from grouping alone.</del>

<mark>Figure 3b quantifies the performance upper bound associated with these region-wise preferences. Relative to the best fixed extent ($s=720$), selecting the lowest-risk extent separately for each region reduces average MSE by 8.112%. Because the regional winners are selected using validation labels across separately trained predictors, this value represents descriptive oracle headroom rather than the realized gain of a learned decoder. Within this controlled example, its magnitude nevertheless shows that one fixed extent can leave meaningful region-specific headroom, motivating a decoder that can adapt sharing across the future domain.</mark>

<del>### 3.5 Design requirements</del>

<del>The formulation and diagnostic observations above lead to three design requirements. First, one parameterized predictor should serve all supported horizons, with the requested endpoint controlling only the returned prefix length. This property makes CHPC intrinsic to the forecasting architecture. Second, the decoder should provide multiple sharing extents, integrate them across future steps and retain step-specific synthesis. Figure 3 motivates regional adaptation, while sample- and variable-conditioned allocation remains a method hypothesis to be evaluated by ablation.</del>

<del>Third, the predictive paths should learn jointly before the integration mechanism concentrates on a subset of them. This optimization consideration is not implied by Figures 2 or 3 and must be evaluated independently. Based on these requirements, we develop **Independent Scope-Conditioned Forecasting (ISCF)** together with the training-only **Balanced Scope Co-Adaptation (BSCA)** mechanism. Section 4 presents the resulting architecture, and the experimental section evaluates its accuracy, component contributions and transferability.</del>
