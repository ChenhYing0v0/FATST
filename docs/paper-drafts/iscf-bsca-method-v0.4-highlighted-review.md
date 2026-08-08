# ISCF-BSCA Section 4 v0.4: Highlighted Review

## Review status

| Field | Content |
| --- | --- |
| `document_role` | Review-only comparison for Section 4 v0.4 |
| `comparison_baseline` | Section 4 `v0.3-author-refinement-4.1-4.3` at commit `ba335c7` |
| `canonical_clean_draft` | `docs/paper-drafts/iscf-bsca-method-initial-draft.md` |
| `highlight_rule` | Yellow highlight denotes added or replacement text; strikethrough denotes removed text |
| `review_scope` | Path naming, concise Figure 4 caption, unified-field framing, target-adaptive allocation narrative and downstream consistency through Section 4.6 |
| `manuscript_status` | This file is not the clean manuscript source |

Only passages changed from v0.3 are reproduced below. Unchanged equations are retained where they are needed to review the revised narrative.

## Change summary

| Location | Highlighted change |
| --- | --- |
| 4.1 | Introduces `Scope Forecasting Path` and `Target-Adaptive Allocation Path`; simplifies the Figure 4 caption |
| 4.2 | Replaces generic allocation-path references with the two canonical path names |
| 4.3 | Removes the redundant identical-output clarification and reframes ISCF as one parameter-sharing forecast field rather than a model ensemble |
| 4.4 | Renames and restructures the subsection around target-wise information-granularity assignment and prefix-bounded execution |
| 4.5--4.6 | Synchronizes the optimization, CHPC and complexity explanations with the two named paths |

## Terminology additions

| Canonical term | Meaning in Section 4 |
| --- | --- |
| <mark>Scope Forecasting Path</mark> | <mark>Region-wise construction of the scope-indexed forecast field</mark> |
| <mark>Target-Adaptive Allocation Path</mark> | <mark>Target-wise assignment of scope-conditioned information using the History State and Future Coordinate</mark> |

## 4.1 Architecture overview

<del>Figure 4 summarizes the forward computation path of ISCF. The architecture consumes a variable-wise History State rather than relying on a particular Encoder. Any Encoder that models temporal patch tokens and returns the required tensor interface can therefore serve as its backbone. This interface covers the patch-based Encoder family commonly used in time-series forecasting. Given a History Series $\mathbf X\in\mathbb R^{B\times L\times C}$, Patchify and the Encoder produce $\mathbf R\in\mathbb R^{B\times C\times R}$. Scope Projection then maps this state to a dedicated Scope Matrix for each sharing scope. In parallel, the Future Coordinate is averaged within each scope region to form a Region Descriptor. Their contraction produces a Scope-region State, and the shared Region-to-Step Forecast Generator converts this state into step-wise predictions. Each future region is generated separately, and concatenating its predictions forms one Scope-conditioned Forecast. Collecting the forecasts over $S$ parallel scope lines yields the scope-indexed forecast field $\mathcal F_\theta(\mathbf X)\in\mathbb R^{B\times C\times T\times S}$.</del>

<mark>Figure 4 summarizes the forward computation of ISCF, which organizes its decoder into a **Scope Forecasting Path** and a **Target-Adaptive Allocation Path**. The architecture consumes a variable-wise History State rather than relying on a particular Encoder. Any Encoder that models temporal patch tokens and returns the required tensor interface can therefore serve as its backbone. This interface covers the patch-based Encoder family commonly used in time-series forecasting. Given a History Series $\mathbf X\in\mathbb R^{B\times L\times C}$, Patchify and the Encoder produce $\mathbf R\in\mathbb R^{B\times C\times R}$. The Scope Forecasting Path constructs region-wise predictions under each sharing scope and assembles them into $\mathcal F_\theta(\mathbf X)\in\mathbb R^{B\times C\times T\times S}$. The Target-Adaptive Allocation Path then determines how each future target draws on these sharing granularities.</mark>

<del>The lower path determines how these forecasts are integrated at each target. A projected History State and the Future Coordinate $\boldsymbol\phi_\tau$ form the Condition Vector $[\mathbf u_{b,c};\boldsymbol\phi_\tau]$. The Allocation MLP maps this vector to the Scope Probabilities $\boldsymbol\Pi\in\mathbb R^{B\times C\times T\times S}$. Weighted contraction of the Scope-conditioned Forecasts with these probabilities produces one trajectory $\widehat{\mathbf Y}\in\mathbb R^{B\times T\times C}$. For an $H$-step request, the region-local construction permits the forward computation to be restricted to regions and targets that intersect the first $H$ steps. The requested horizon changes only the evaluated prefix; it neither changes the architecture nor the computation assigned to a shared future target. ISCF therefore produces variable-length outputs while satisfying CHPC by construction.</del>

<mark>The Target-Adaptive Allocation Path combines a projected History State with the Future Coordinate $\boldsymbol\phi_\tau$ and produces Scope Probabilities $\boldsymbol\Pi\in\mathbb R^{B\times C\times T\times S}$. Weighted contraction with the scope-indexed forecast field yields one trajectory $\widehat{\mathbf Y}\in\mathbb R^{B\times T\times C}$. For an $H$-step request, the region-local construction permits computation to be restricted to regions and targets intersecting the first $H$ steps. The requested horizon changes only the evaluated prefix, not the computation assigned to a shared future target. ISCF therefore produces variable-length outputs while satisfying CHPC by construction.</mark>

### Figure 4 caption

<del>**Figure 4 | ISCF integrates Scope-conditioned Forecasts into one trajectory for Varied-Horizon Forecasting.** The History Series is patchified and encoded into a shared History State. For each sharing scope, Scope Projection produces a Scope Matrix, while region-wise averaging of the Future Coordinate produces Region Descriptors. Their contraction forms Scope-region States, and the shared Region-to-Step Forecast Generator converts these states into Scope-conditioned Forecasts. In the parallel allocation path, the Condition Vector combines a projected History State with the target Future Coordinate, and the Allocation MLP produces target-specific Scope Probabilities. Weighted contraction along the scope axis yields one prediction trajectory, whose nested prefixes answer different horizon requests. Three representative scopes are displayed for visual clarity; the formulation uses $S$ scopes. The probability map and trajectories are schematic rather than empirical results.</del>

<mark>**Figure 4 | ISCF constructs one trajectory for Varied-Horizon Forecasting.** The Scope Forecasting Path generates region-wise predictions under multiple sharing scopes. The Target-Adaptive Allocation Path assigns scope-conditioned information to each future target. Weighted contraction yields one trajectory whose nested prefixes answer different horizon requests. Three representative scopes are shown for clarity; the probability map and trajectories are schematic.</mark>

## 4.2 History state and future coordinate

Collecting $\mathbf r_{b,c}$ over samples and variables gives the History State $\mathbf R=[\mathbf r_{b,c}]\in\mathbb R^{B\times C\times R}$. The preserved variable axis provides one history representation for each sample-variable pair, which is used by <del>both Scope Projection and the allocation path</del><mark>both decoder paths</mark>. ISCF requires only this tensor interface and does not otherwise constrain the internal design of the patch-token Encoder.

The History State summarizes the observed series, but a unified decoder must also identify where each prediction lies in the future domain. Using the requested horizon for this purpose would assign different conditioning contexts to the same target under different requests. ISCF instead introduces a **Future Coordinate** for every future step. <del>This fixed coordinate supplies a horizon-independent target identity, provides a common positional basis from which regions of different scopes can be described, and allows the allocation path to vary its scope preference across future steps.</del> <mark>This fixed coordinate supplies a horizon-independent target identity and a common positional basis for regions at different scopes. It also allows the Target-Adaptive Allocation Path to vary its scope preference across future steps.</mark>

## 4.3 Generation of scope-conditioned forecasts

<del>The upper path of Figure 4</del><mark>The Scope Forecasting Path</mark> contains $S$ parallel <del>forecasting</del><mark>scope</mark> lines, one for each sharing scope in $\mathcal S=\{s_1,\ldots,s_S\}$. A single Scope Projection stage contains an independently parameterized projection for every scope, so each line receives a dedicated Scope Matrix as its history-conditioned information pool. The matrix form is important because it retains a Future-Coordinate axis and a latent-mode axis. Region Descriptors can therefore query the same scope-specific history information at different future locations, without assigning a separate prediction head to every region.

The Region-to-Step Forecast Generator converts each shared Scope-region State into step-wise predictions<del>, so region-wise reuse does not force identical outputs within that region</del>. Let $g_s(\tau)$ denote the region containing step $\tau$. The generator is shared across scopes and regions, and uses step-specific vectors $\mathbf a_\tau,\mathbf n_\tau\in\mathbb R^K$ and bias $\beta_\tau$ to define

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

<del>Concatenating the separately generated regions gives the Scope-conditioned Forecast $\mathcal F^{(s)}$ for scope $s$. Collecting these forecasts over samples, variables, future steps and scopes produces the scope-indexed forecast field $\mathcal F_\theta(\mathbf X)\in\mathbb R^{B\times C\times T\times S}$. Scope Projection is independently parameterized across scopes, whereas Patchify, the Encoder, Future Coordinate and Region-to-Step Forecast Generator are shared. Each $\mathcal F^{(s)}$ is therefore one scope-conditioned slice of a jointly constructed field, not a separately trained forecasting model.</del>

<mark>Concatenating the separately generated regions gives the Scope-conditioned Forecast $\mathcal F^{(s)}$ for scope $s$. Collecting these forecasts produces the scope-indexed field $\mathcal F_\theta(\mathbf X)\in\mathbb R^{B\times C\times T\times S}$. ISCF does not integrate $S$ independently trained forecasters. It constructs one unified forecasting framework with scope-specific history projections and shared representation and synthesis modules. The shared Encoder and Region-to-Step Forecast Generator couple representation learning and forecast synthesis across scopes, while dedicated Scope Matrices preserve granularity-specific information. Compared with deploying $S$ complete forecasters, ISCF evaluates the Encoder once and avoids duplicating the forecast generator, reducing redundant encoder computation and parameter storage.</mark>

## 4.4 <del>Condition vector, scope probabilities and varied-horizon forecasting</del><mark>Target-adaptive scope allocation</mark>

<del>The lower path of Figure 4 allows the contribution of each Scope-conditioned Forecast to vary across samples, variables and future steps. It begins by projecting the History State into a compact history summary</del>

<mark>Different future targets may require information organized at different sharing granularities. Estimating this preference requires both the dynamics encoded in the observed history and the target position in the future domain. The Target-Adaptive Allocation Path represents these two signals using a compact history summary and the Future Coordinate. It first projects the History State as</mark>

$$
\mathbf u_{b,c}
=
\mathbf W_h\mathbf r_{b,c}
+
\mathbf b_h
\in\mathbb R^{D_h}.
$$

<del>The Condition Vector concatenates this summary with the Future Coordinate of target step $\tau$. The Allocation MLP then computes</del>

<mark>The compact summary $\mathbf u_{b,c}$ retains the sample-variable context used for allocation. For future step $\tau$, the Condition Vector concatenates this summary with $\boldsymbol\phi_\tau$. The Allocation MLP maps the resulting vector to scope logits.</mark>

$$
\boldsymbol\ell_{b,c,\tau}
=
\mathbf W_o
\operatorname{GELU}\left(
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

<del>Collecting $\pi_{b,c,\tau,s}$ gives the Scope Probabilities $\boldsymbol\Pi\in\mathbb R^{B\times C\times T\times S}$, with $\sum_{s=1}^{S}\pi_{b,c,\tau,s}=1$ for every target. The probabilities are conditioned on the sample, variable and future-step identity; they do not observe the future target value. The final normalized prediction is obtained through weighted contraction along the scope axis:</del>

<mark>Softmax converts these logits into the Scope Probability $\pi_{b,c,\tau,s}$ assigned to scope $s$. Collecting all probabilities gives $\boldsymbol\Pi\in\mathbb R^{B\times C\times T\times S}$, with $\sum_{s=1}^{S}\pi_{b,c,\tau,s}=1$. Each probability vector parameterizes how strongly a future target draws on information organized at different sharing granularities. The final normalized prediction reads from the unified scope-indexed forecast field through weighted contraction.</mark>

$$
\widehat y_{b,\tau,c}
=
\sum_{s=1}^{S}
\pi_{b,c,\tau,s}
\mathcal F_{b,c,\tau,s}.
$$

<mark>This contraction is not an average over separately trained model outputs. The Scope Forecasting Path and Target-Adaptive Allocation Path are jointly learned within one decoder, allowing each target to receive a history-conditioned mixture of sharing granularities.</mark>

<del>The resulting normalized trajectory is transformed back to the original data scale. The Varied-Horizon Forecasting output for a request with horizon $H$ is the corresponding prefix</del>

<mark>For an $H$-step request, ISCF needs only the Scope-region States whose regions intersect $1{:}H$ and the allocation entries for $\tau\leq H$. The resulting normalized prefix is transformed back to the original data scale and returned as</mark>

$$
\widehat{\mathbf Y}_b^{(H)}
=
\left[
\widehat y_{b,\tau,c}
\right]_{\tau=1,\ldots,H;\ c=1,\ldots,C}.
$$

<del>The Scope Probabilities allow the sharing composition to vary over forecast targets, while every horizon request remains a nested view of the same trajectory. The architecture alone does not imply that the Allocation MLP recovers an oracle scope or forms universal specialization. Those questions require the component and internal-behavior analyses reported later.</del>

<mark>Changing $H$ restricts the active region and target computations but does not alter any prediction shared with a longer request. Every request is therefore a nested view of the same trajectory. The learned probabilities represent target-adaptive information allocation rather than an oracle scope selector. Later ablations and internal-behavior analyses evaluate whether this allocation yields useful scope differentiation.</mark>

## 4.5 Balanced Scope Co-Adaptation

<del>The Scope Probabilities have two coupled roles during training. They combine the Scope-conditioned Forecasts and scale the fused-loss gradient received by each scope.</del> <mark>The Target-Adaptive Allocation Path couples forecast integration with the optimization of individual scope lines. Scope Probabilities combine the Scope-conditioned Forecasts and scale the fused-loss gradient received by each scope.</mark>

<del>The direct skill term trains every Scope-conditioned Forecast even when its current probability is small. The anchor acts directly on Allocation-MLP logits and broadens probability-mediated access during early joint learning. Uniform Scope Probabilities are only an optimization proxy: they neither force equal inference-time usage nor guarantee semantically distinct scopes. Both BSCA terms are removed at inference, so the trained model retains Scope Projection, the Region-to-Step Forecast Generator, the Allocation MLP and weighted contraction defined by ISCF.</del>

<mark>The direct skill term trains every Scope-conditioned Forecast even when its current probability is small. The anchor acts directly on Allocation MLP logits and broadens probability-mediated access during early joint learning. Uniform Scope Probabilities are only an optimization proxy. They neither force equal inference-time usage nor guarantee semantically distinct scopes. Both BSCA terms are removed at inference, leaving the Scope Forecasting Path, Target-Adaptive Allocation Path and weighted contraction unchanged.</mark>

## 4.6 Structural properties and complexity

<del>ISCF satisfies CHPC by construction. For a fixed target, the Scope-conditioned Forecast $\mathcal F_{b,c,\tau,s}$ and Scope Probability $\pi_{b,c,\tau,s}$ depend on the History State and target identity $(\tau,c)$. Neither depends on the requested horizon. Hence $\widehat y_{b,\tau,c}$ is identical under any requests $H_i,H_j\geq\tau$. Returning the indexed prefix $1{:}H$ changes only the exposed endpoint and yields the CHPC relation defined in Section 3.</del>

<mark>ISCF satisfies CHPC by construction. For a fixed target, the Scope Forecasting Path produces $\mathcal F_{b,c,\tau,s}$ from the History State and future-step identity. The Target-Adaptive Allocation Path produces $\pi_{b,c,\tau,s}$ from the same history and target coordinate. Neither quantity depends on the requested horizon. Hence $\widehat y_{b,\tau,c}$ is identical under any requests $H_i,H_j\geq\tau$. Returning the indexed prefix $1{:}H$ yields the CHPC relation defined in Section 3.</mark>

<del>The same construction separates sharing scope from horizon. Scope $s$ controls how many future steps reuse a latent state before step-specific synthesis. Requested horizon $H$ controls which already defined future-step predictions are returned. Changing one does not redefine the other.</del>

<mark>The same construction separates sharing scope from requested horizon. Scope $s$ controls how many future steps reuse a latent state before step-specific synthesis. Horizon $H$ selects the target positions and intersecting regions evaluated for one request. Changing $H$ does not redefine the computations assigned to any shared target.</mark>

<del>Computing Scope-region States costs $\mathcal O\!\left(BC D_q K\sum_{s\in\mathcal S}T/s\right)$, while the Region-to-Step Forecast Generator costs $\mathcal O(BCSTK)$. The Allocation MLP is evaluated once per sample-variable History State and future target. Materializing the Scope-conditioned Forecasts and Scope Probabilities requires $\mathcal O(BCTS)$ working memory, and the implementation supports region and target chunking to reduce peak intermediates without changing the function. Only one encoder-decoder checkpoint is stored and served for all supported horizons; empirical latency, memory and forecasting accuracy are evaluated separately in Section 5.</del>

<mark>The full-domain costs below correspond to materializing all $T$ future steps. Computing Scope-region States costs $\mathcal O\!\left(BC D_q K\sum_{s\in\mathcal S}T/s\right)$, while the Region-to-Step Forecast Generator costs $\mathcal O(BCSTK)$. The Allocation MLP is evaluated once per sample-variable History State and future target. Materializing the Scope-conditioned Forecasts and Scope Probabilities requires $\mathcal O(BCTS)$ working memory. Region and target chunking reduces peak intermediate memory without changing the function. Prefix-bounded execution restricts the target-dependent operations to $\tau\leq H$ and the Scope-region States intersecting that prefix. Only one encoder-decoder checkpoint is stored and served for all supported horizons. Empirical latency, memory and forecasting accuracy are evaluated separately in Section 5.</mark>

## Editorial claim-audit updates

| v0.3 audit entry | v0.4 audit entry |
| --- | --- |
| <del>Scope-conditioned Forecasts: one field contains multiple scope-conditioned slices; learned specialization remains deferred</del> | <mark>Scope Forecasting Path: one Encoder evaluation, independent scope projections and shared synthesis produce one unified field; lower compute than every unified alternative remains deferred</mark> |
| <del>Condition Vector, Allocation MLP and Scope Probabilities: probabilities vary by sample, variable and future step; region-optimal allocation remains deferred</del> | <mark>Target-Adaptive Allocation Path: each target receives a learned allocation over sharing granularities; oracle recovery and region-optimal allocation remain deferred</mark> |
