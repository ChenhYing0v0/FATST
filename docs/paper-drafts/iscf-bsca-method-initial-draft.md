# ISCF-BSCA Section 4: Method

## Draft status

| Field | Content |
| --- | --- |
| `document_role` | Clean manuscript-facing initial draft of Section 4 |
| `version` | `v0.3-author-refinement-4.1-4.3` |
| `date` | `2026-08-07` |
| `review_status` | `author_feedback_integrated_through_section_4.3` |
| `upstream_dependency` | Introduction v0.9 and Section 3 v0.7 remain frozen and unchanged |
| `method_contract` | Exact frozen ISCF-BSCA-v1 architecture and objective |
| `figure_4_status` | Visual design temporarily fixed by the author; stable vector-asset synchronization remains pending |
| `implementation_change` | None |
| `experiment_change` | None |
| `claim_boundary` | Structural properties are stated as construction facts; performance, ablation and transfer claims remain pending their paper-facing tables |
| `narrative_spine` | History State and Future Coordinate → Scope-conditioned Forecasts and Scope Probabilities → weighted contraction → Varied-Horizon Forecasting → balanced co-adaptation |

The status table, terminology ledger and editorial audit are working metadata and are not part of the manuscript body submitted for review.

## Terminology ledger

| Canonical term | Symbol | Meaning in Section 4 |
| --- | --- | --- |
| Maximum future domain | $T$ | Largest supported future-step index |
| Sharing scope | $s$ | Number of contiguous future steps that reuse one Scope-region State |
| Scope set | $\mathcal S$ | Supported latent-state sharing scopes, with $S=|\mathcal S|$ |
| History Series | $\mathbf X$ | Multivariate look-back window supplied to the forecaster |
| Patchify | -- | Division of each variable history into $P$ temporal patches |
| Encoder | $\mathcal E$ | Shared patch-processing backbone |
| History State | $\mathbf R=[\mathbf r_{b,c}]$ | Variable-wise representation produced by Patchify and the shared Encoder |
| Future Coordinate | $\boldsymbol\Phi=[\boldsymbol\phi_1,\ldots,\boldsymbol\phi_T]^\top$ | Fixed coordinate field identifying future steps |
| Scope Projection | $\mathbf W^{(s)},\mathbf b^{(s)}$ | Independent history projection assigned to sharing scope $s$ |
| Region Descriptor | $\overline{\boldsymbol\phi}_g^{(s)}$ | Mean Future Coordinate within region $\mathcal G_g^{(s)}$ |
| Scope Matrix | $\mathbf M_{b,c}^{(s)}$ | Scope-specific matrix produced from the History State by Scope Projection |
| Scope-region State | $\mathbf z_{b,c,g}^{(s)}$ | History-conditioned state shared within region $g$ under scope $s$ |
| Region-to-Step Forecast Generator | $\mathbf a_\tau,\mathbf n_\tau,\beta_\tau$ | Shared step-specific parameters that map a Scope-region State to future-step predictions |
| Scope-conditioned Forecast | $\mathcal F^{(s)}$ | Forecast slice generated under one sharing scope; not an independent forecasting model |
| Scope-indexed forecast field | $\mathcal F_\theta(\mathbf X)$ | Collection of Scope-conditioned Forecasts indexed by variable, future step and sharing scope |
| Condition Vector | $[\mathbf u_{b,c};\boldsymbol\phi_\tau]$ | Concatenation of a projected History State and the target Future Coordinate |
| Allocation MLP | $\mathbf W_p,\mathbf W_o$ | Network that maps the Condition Vector to scope logits |
| Scope Probabilities | $\boldsymbol\Pi=[\pi_{b,c,\tau,s}]$ | Normalized scope weights produced by the Allocation MLP |
| Weighted contraction | $\sum_s\pi_s\mathcal F_s$ | Integration of scope-conditioned forecasts along the scope axis |
| Varied-Horizon Forecasting | $\widehat{\mathbf Y}^{(H)}$ | Return of the first $H$ steps from one maximum-length prediction trajectory |
| Balanced Scope Co-Adaptation | BSCA | Training-only objective for direct slice supervision and broad allocation access |

## 4. ISCF-BSCA

Section 3 established two requirements for varied-horizon forecasting. Predictions for a shared future target should be invariant to the requested horizon, while the decoder should not impose one latent-state sharing extent on the entire future domain. We address these requirements with ISCF-BSCA. Independent Scope-Conditioned Forecasting (ISCF) is a decoder-side architecture that constructs one future-step-indexed trajectory from multiple sharing scopes, while Balanced Scope Co-Adaptation (BSCA) is the joint optimization method designed for it.

### 4.1 Architecture overview

Figure 4 summarizes the forward computation path of ISCF. The architecture consumes a variable-wise History State rather than relying on a particular Encoder. Any Encoder that models temporal patch tokens and returns the required tensor interface can therefore serve as its backbone. This interface covers the patch-based Encoder family commonly used in time-series forecasting. Given a History Series $\mathbf X\in\mathbb R^{B\times L\times C}$, Patchify and the Encoder produce $\mathbf R\in\mathbb R^{B\times C\times R}$. Scope Projection then maps this state to a dedicated Scope Matrix for each sharing scope. In parallel, the Future Coordinate is averaged within each scope region to form a Region Descriptor. Their contraction produces a Scope-region State, and the shared Region-to-Step Forecast Generator converts this state into step-wise predictions. Each future region is generated separately, and concatenating its predictions forms one Scope-conditioned Forecast. Collecting the forecasts over $S$ parallel scope lines yields the scope-indexed forecast field $\mathcal F_\theta(\mathbf X)\in\mathbb R^{B\times C\times T\times S}$.

The lower path determines how these forecasts are integrated at each target. A projected History State and the Future Coordinate $\boldsymbol\phi_\tau$ form the Condition Vector $[\mathbf u_{b,c};\boldsymbol\phi_\tau]$. The Allocation MLP maps this vector to the Scope Probabilities $\boldsymbol\Pi\in\mathbb R^{B\times C\times T\times S}$. Weighted contraction of the Scope-conditioned Forecasts with these probabilities produces one trajectory $\widehat{\mathbf Y}\in\mathbb R^{B\times T\times C}$. For an $H$-step request, the region-local construction permits the forward computation to be restricted to regions and targets that intersect the first $H$ steps. The requested horizon changes only the evaluated prefix; it neither changes the architecture nor the computation assigned to a shared future target. ISCF therefore produces variable-length outputs while satisfying CHPC by construction.

The Scope-conditioned Forecasts and Scope Probabilities are optimized jointly. Early probability concentration can weaken the forecasting gradients received by low-probability scopes. BSCA addresses this coupling through direct predictive supervision for every Scope-conditioned Forecast and a ramped uniform anchor on the Scope Probabilities. These terms are used only during training and are therefore described separately from the forward computation path in Figure 4.

<a id="fig:iscf-bsca-method"></a>

![Overview of the ISCF-BSCA architecture.](../../paper-figures/figure_iscf_bsca_method_overview.png)

**Figure 4 | ISCF integrates Scope-conditioned Forecasts into one trajectory for Varied-Horizon Forecasting.** The History Series is patchified and encoded into a shared History State. For each sharing scope, Scope Projection produces a Scope Matrix, while region-wise averaging of the Future Coordinate produces Region Descriptors. Their contraction forms Scope-region States, and the shared Region-to-Step Forecast Generator converts these states into Scope-conditioned Forecasts. In the parallel allocation path, the Condition Vector combines a projected History State with the target Future Coordinate, and the Allocation MLP produces target-specific Scope Probabilities. Weighted contraction along the scope axis yields one prediction trajectory, whose nested prefixes answer different horizon requests. Three representative scopes are displayed for visual clarity; the formulation uses $S$ scopes. The probability map and trajectories are schematic rather than empirical results.

### 4.2 History state and future coordinate

ISCF begins with the History State and Future Coordinate shown on the left of Figure 4. The History Series is normalized per variable, divided into $P$ patches by Patchify and processed by the shared Encoder. This produces $\mathbf Z\in\mathbb R^{B\times C\times P\times D_e}$, which is flattened over its patch and embedding dimensions:

$$
\mathbf Z
=
\mathcal E\!\left(\mathcal N(\mathbf X)\right),
\qquad
\mathbf r_{b,c}
=
\operatorname{vec}(\mathbf Z_{b,c})
\in\mathbb R^R,
\qquad
R=PD_e.
$$

Collecting $\mathbf r_{b,c}$ over samples and variables gives the History State $\mathbf R=[\mathbf r_{b,c}]\in\mathbb R^{B\times C\times R}$. The preserved variable axis provides one history representation for each sample-variable pair, which is used by both Scope Projection and the allocation path. ISCF requires only this tensor interface and does not otherwise constrain the internal design of the patch-token Encoder.

The History State summarizes the observed series, but a unified decoder must also identify where each prediction lies in the future domain. Using the requested horizon for this purpose would assign different conditioning contexts to the same target under different requests. ISCF instead introduces a **Future Coordinate** for every future step. This fixed coordinate supplies a horizon-independent target identity, provides a common positional basis from which regions of different scopes can be described, and allows the allocation path to vary its scope preference across future steps.

Formally, the Future Coordinate is the field $\boldsymbol\Phi=[\boldsymbol\phi_1,\ldots,\boldsymbol\phi_T]^\top\in\mathbb R^{T\times D_q}$. We construct it from low-order discrete cosine functions, which provide smooth coordinate channels at progressively finer temporal frequencies. For $d=0,\ldots,D_q-1$, we first define

$$
\widetilde\phi_{\tau,d}
=
\cos\!\left(
\frac{\pi(\tau-\tfrac12)d}{T}
\right).
$$

The constant coordinate is $\phi_{\tau,0}=1$. Each nonconstant coordinate is centered over the future domain and scaled as

$$
\phi_{\tau,d}
=
\sqrt{2}
\left(
\widetilde\phi_{\tau,d}
-
\frac{1}{T}
\sum_{j=1}^{T}
\widetilde\phi_{j,d}
\right),
\qquad d\geq 1.
$$

The constant channel preserves a global reference, while the centered nonconstant channels distinguish positions across the future domain. The resulting parameter-free field serves two roles. Region-wise averaging produces the Region Descriptors used to generate Scope-conditioned Forecasts, whereas the unpooled coordinate $\boldsymbol\phi_\tau$ identifies the individual target in the Condition Vector used for Scope Probabilities.

### 4.3 Generation of scope-conditioned forecasts

The upper path of Figure 4 contains $S$ parallel forecasting lines, one for each sharing scope in $\mathcal S=\{s_1,\ldots,s_S\}$. A single Scope Projection stage contains an independently parameterized projection for every scope, so each line receives a dedicated Scope Matrix as its history-conditioned information pool. The matrix form is important because it retains a Future-Coordinate axis and a latent-mode axis. Region Descriptors can therefore query the same scope-specific history information at different future locations, without assigning a separate prediction head to every region.

For a scope $s$, the future domain is divided into contiguous regions

$$
\mathcal G_g^{(s)}
=
\{(g-1)s+1,\ldots,gs\},
\qquad
g=1,\ldots,T/s.
$$

The Future Coordinate within each region is averaged to form the Region Descriptor:

$$
\overline{\boldsymbol\phi}_g^{(s)}
=
\frac{1}{s}
\sum_{\tau\in\mathcal G_g^{(s)}}
\boldsymbol\phi_\tau
\in\mathbb R^{D_q}.
$$

For each scope, Scope Projection maps $\mathbf r_{b,c}$ into its dedicated Scope Matrix

$$
\mathbf M_{b,c}^{(s)}
=
\operatorname{reshape}_{D_q\times K}
\left(
\mathbf r_{b,c}\mathbf W^{(s)}
+
\mathbf b^{(s)}
\right)
\in\mathbb R^{D_q\times K}.
$$

The Region Descriptor selects and combines information from this matrix through a coordinate contraction, producing the region-wise Scope-region State

$$
\mathbf z_{b,c,g}^{(s)}
=
\left(\overline{\boldsymbol\phi}_g^{(s)}\right)^\top
\mathbf M_{b,c}^{(s)}
\in\mathbb R^K.
$$

Once the Scope Matrix is available, each Scope-region State depends only on the descriptor of its own region. Regions can therefore be evaluated separately and in parallel, although they draw on the same scope-specific history information. All future steps in $\mathcal G_g^{(s)}$ reuse $\mathbf z_{b,c,g}^{(s)}$. A finer scope constructs more region states and limits reuse to shorter intervals, whereas a broader scope shares each state over a longer interval. Scope thus defines a cross-step reuse pattern rather than a requested prediction horizon.

The Region-to-Step Forecast Generator converts each shared Scope-region State into step-wise predictions, so region-wise reuse does not force identical outputs within that region. Let $g_s(\tau)$ denote the region containing step $\tau$. The generator is shared across scopes and regions, and uses step-specific vectors $\mathbf a_\tau,\mathbf n_\tau\in\mathbb R^K$ and bias $\beta_\tau$ to define

$$
\mathcal F_{b,c,\tau,s}
=
\left\langle
\mathbf a_\tau,
\mathbf z_{b,c,g_s(\tau)}^{(s)}
\right\rangle
+
\left\langle
\mathbf n_\tau,
\operatorname{GELU}\!\left(
\mathbf z_{b,c,g_s(\tau)}^{(s)}
\right)
\right\rangle
+
\beta_\tau.
$$

Concatenating the separately generated regions gives the Scope-conditioned Forecast $\mathcal F^{(s)}$ for scope $s$. Collecting these forecasts over samples, variables, future steps and scopes produces the scope-indexed forecast field $\mathcal F_\theta(\mathbf X)\in\mathbb R^{B\times C\times T\times S}$. Scope Projection is independently parameterized across scopes, whereas Patchify, the Encoder, Future Coordinate and Region-to-Step Forecast Generator are shared. Each $\mathcal F^{(s)}$ is therefore one scope-conditioned slice of a jointly constructed field, not a separately trained forecasting model.

### 4.4 Condition vector, scope probabilities and varied-horizon forecasting

The lower path of Figure 4 allows the contribution of each Scope-conditioned Forecast to vary across samples, variables and future steps. It begins by projecting the History State into a compact history summary

$$
\mathbf u_{b,c}
=
\mathbf W_h\mathbf r_{b,c}
+
\mathbf b_h
\in\mathbb R^{D_h}.
$$

The Condition Vector concatenates this summary with the Future Coordinate of target step $\tau$. The Allocation MLP then computes

$$
\boldsymbol\ell_{b,c,\tau}
=
\mathbf W_o
\operatorname{GELU}\!\left(
\mathbf W_p
[\mathbf u_{b,c};\boldsymbol\phi_\tau]
+
\mathbf b_p
\right)
+
\mathbf b_o
\in\mathbb R^S,
$$

$$
\pi_{b,c,\tau,s}
=
\frac{
\exp(\ell_{b,c,\tau,s})
}{
\sum_{s'=1}^{S}
\exp(\ell_{b,c,\tau,s'})
}.
$$

Collecting $\pi_{b,c,\tau,s}$ gives the Scope Probabilities $\boldsymbol\Pi\in\mathbb R^{B\times C\times T\times S}$, with $\sum_{s=1}^{S}\pi_{b,c,\tau,s}=1$ for every target. The probabilities are conditioned on the sample, variable and future-step identity; they do not observe the future target value. The final normalized prediction is obtained through weighted contraction along the scope axis:

$$
\widehat y_{b,\tau,c}
=
\sum_{s=1}^{S}
\pi_{b,c,\tau,s}
\mathcal F_{b,c,\tau,s}.
$$

The resulting normalized trajectory is transformed back to the original data scale. The Varied-Horizon Forecasting output for a request with horizon $H$ is the corresponding prefix

$$
\widehat{\mathbf Y}_b^{(H)}
=
\left[
\widehat y_{b,\tau,c}
\right]_{\tau=1,\ldots,H;\ c=1,\ldots,C}.
$$

The Scope Probabilities allow the sharing composition to vary over forecast targets, while every horizon request remains a nested view of the same trajectory. The architecture alone does not imply that the Allocation MLP recovers an oracle scope or forms universal specialization. Those questions require the component and internal-behavior analyses reported later.

### 4.5 Balanced Scope Co-Adaptation

The Scope Probabilities have two coupled roles during training. They combine the Scope-conditioned Forecasts and scale the fused-loss gradient received by each scope. For a fixed target, the contraction implies

$$
\frac{
\partial\mathcal L_{\mathrm{fuse}}
}{
\partial\mathcal F_{b,c,\tau,s}
}
=
\pi_{b,c,\tau,s}
\frac{
\partial\mathcal L_{\mathrm{fuse}}
}{
\partial\widehat y_{b,\tau,c}
}.
$$

Scope Probabilities that concentrate early can therefore restrict the forecasting gradients reaching low-probability scopes before their predictive paths have matured. BSCA addresses this optimization problem without modifying ISCF at inference.

The frozen training objective uses dense-prefix incidence weights. Averaging mean absolute error uniformly over all prefixes $h=1,\ldots,T$ assigns future step $\tau$ the weight

$$
\omega_\tau
=
\frac{1}{T}
\sum_{h=\tau}^{T}
\frac{1}{h},
\qquad
\sum_{\tau=1}^{T}\omega_\tau=1.
$$

During training, inverse normalization is applied to the fused forecast and every Scope-conditioned Forecast before evaluating the objective. For notational simplicity, the symbols below denote these raw-scale quantities. Let $y_{b,\tau,c}$ be the corresponding target. The fused prediction loss is

$$
\mathcal L_{\mathrm{fuse}}
=
\frac{1}{BC}
\sum_{b=1}^{B}
\sum_{c=1}^{C}
\sum_{\tau=1}^{T}
\omega_\tau
\left|
\widehat y_{b,\tau,c}-y_{b,\tau,c}
\right|.
$$

BSCA first supplies every Scope-conditioned Forecast with direct predictive supervision:

$$
\mathcal L_{\mathrm{skill}}
=
\frac{1}{BCS}
\sum_{b=1}^{B}
\sum_{c=1}^{C}
\sum_{s=1}^{S}
\sum_{\tau=1}^{T}
\omega_\tau
\left|
\mathcal F_{b,c,\tau,s}-y_{b,\tau,c}
\right|.
$$

It then discourages premature allocation concentration using a target-free uniform reference $q_s=1/S$:

$$
\mathcal L_{\mathrm{anchor}}
=
\frac{1}{BC}
\sum_{b=1}^{B}
\sum_{c=1}^{C}
\sum_{\tau=1}^{T}
\omega_\tau
\frac{
D_{\mathrm{KL}}\!\left(
\mathbf q\,\Vert\,\boldsymbol\pi_{b,c,\tau}
\right)
}{
\log S
}.
$$

The complete objective is

$$
\mathcal L_{\mathrm{BSCA}}
=
\mathcal L_{\mathrm{fuse}}
+
\lambda_{\mathrm{skill}}
\mathcal L_{\mathrm{skill}}
+
\lambda_{\mathrm{anchor}}(u)
\mathcal L_{\mathrm{anchor}},
$$

where $u\in[0,1]$ is optimizer progress. In the frozen configuration, $\lambda_{\mathrm{skill}}=1$ and

$$
\lambda_{\mathrm{anchor}}(u)
=
0.1
\min\!\left(
\frac{u}{0.25},1
\right).
$$

The direct skill term trains every Scope-conditioned Forecast even when its current probability is small. The anchor acts directly on Allocation-MLP logits and broadens probability-mediated access during early joint learning. Uniform Scope Probabilities are only an optimization proxy: they neither force equal inference-time usage nor guarantee semantically distinct scopes. Both BSCA terms are removed at inference, so the trained model retains Scope Projection, the Region-to-Step Forecast Generator, the Allocation MLP and weighted contraction defined by ISCF.

### 4.6 Structural properties and complexity

ISCF satisfies CHPC by construction. For a fixed target, the Scope-conditioned Forecast $\mathcal F_{b,c,\tau,s}$ and Scope Probability $\pi_{b,c,\tau,s}$ depend on the History State and target identity $(\tau,c)$. Neither depends on the requested horizon. Hence $\widehat y_{b,\tau,c}$ is identical under any requests $H_i,H_j\geq\tau$. Returning the indexed prefix $1{:}H$ changes only the exposed endpoint and yields the CHPC relation defined in Section 3.

The same construction separates sharing scope from horizon. Scope $s$ controls how many future steps reuse a latent state before step-specific synthesis. Requested horizon $H$ controls which already defined future-step predictions are returned. Changing one does not redefine the other.

For completeness, let $D_q$ be the coordinate dimension, $K$ the mode rank, $D_h$ the allocation history dimension and $D_a$ the allocation hidden dimension. Excluding the encoder, the frozen independent-scope decoder contains

$$
\begin{aligned}
N_{\mathrm{ISCF}}
={}&
S D_q R K
+
S D_q K
+
2TK
+
T
\\
&+
R D_h
+
D_h
+
(D_h+D_q)D_a
+
D_a
+
D_aS
+
S
\end{aligned}
$$

trainable parameters. The first line corresponds to Scope Projection, Scope Matrices and the Region-to-Step Forecast Generator; the second corresponds to the History-State projection and Allocation MLP used to produce Scope Probabilities. The Future Coordinate, contiguous region indices and Region-Descriptor averaging are parameter free. BSCA adds no trainable parameter.

Computing Scope-region States costs $\mathcal O\!\left(BC D_q K\sum_{s\in\mathcal S}T/s\right)$, while the Region-to-Step Forecast Generator costs $\mathcal O(BCSTK)$. The Allocation MLP is evaluated once per sample-variable History State and future target. Materializing the Scope-conditioned Forecasts and Scope Probabilities requires $\mathcal O(BCTS)$ working memory, and the implementation supports region and target chunking to reduce peak intermediates without changing the function. Only one encoder-decoder checkpoint is stored and served for all supported horizons; empirical latency, memory and forecasting accuracy are evaluated separately in Section 5.

## Editorial implementation and claim audit

| Manuscript element | Frozen implementation correspondence | Permitted claim | Deferred claim |
| --- | --- | --- | --- |
| History State | Normalization, Patchify, Encoder and flattened `hidden:[B,C,R]` | ISCF is compatible with patch-token Encoders that satisfy the stated tensor interface | Encoder superiority or empirical transfer across arbitrary backbones |
| Scope Projection and Scope Matrix | Independent SIFF scale basis with one component per scope | Each scope has an independent Scope Projection | Each Scope-conditioned Forecast is an independent forecasting model |
| Region Descriptor and Scope-region State | Contiguous group indices and pooled Future Coordinates | Scope controls state-reuse extent | Canonical partition is universally optimal |
| Region-to-Step Forecast Generator | Shared step-specific linear and nonlinear synthesis parameters | One shared state can produce distinct step-specific predictions | The generator is universally transferable |
| Scope-conditioned Forecasts | `arm_forecasts`, paper-facing shape `[B,C,T,S]` | One field contains multiple scope-conditioned slices | Learned specialization or oracle scope recovery |
| Condition Vector, Allocation MLP and Scope Probabilities | Direct History-State-plus-target-coordinate policy, `[B,C,T,S]` | Probabilities vary by sample, variable and future step | Allocation is label-conditioned or necessarily region optimal |
| Weighted contraction and prefix output | Scope-axis weighted sum yields `[B,T,C]`; the reference implementation materializes the full field before prefix slicing | One horizon-agnostic trajectory, CHPC and architecture-level support for prefix-bounded region evaluation | Realized latency gains from prefix-bounded execution or lower error than horizon-specific systems |
| BSCA | Uniform slice-skill loss plus ramped normalized `KL(uniform || allocation)` | Training-only balanced access; no inference path or parameter | Generic KL novelty, universal gain or semantic expert specialization |
| Section 5 interface | Pending main, ablation, transfer and efficiency tables | Structural and reproducibility statements only | Main-table superiority, component effectiveness and decoder portability |
