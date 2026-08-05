# ISCF-BSCA Section 4: Method

## Draft status

| Field | Content |
| --- | --- |
| `document_role` | Clean manuscript-facing initial draft of Section 4 |
| `version` | `v0.1-architecture-and-objective-complete` |
| `date` | `2026-08-05` |
| `review_status` | `initial_draft_for_author_review` |
| `upstream_dependency` | Introduction v0.9 and Section 3 v0.7 remain frozen and unchanged |
| `method_contract` | Exact frozen ISCF-BSCA-v1 architecture and objective |
| `figure_4_status` | Initial architecture schematic generated for author review |
| `implementation_change` | None |
| `experiment_change` | None |
| `claim_boundary` | Structural properties are stated as construction facts; performance, ablation and transfer claims remain pending their paper-facing tables |
| `narrative_spine` | Requirements from Section 3 → scope-indexed field → target-conditioned contraction → balanced co-adaptation → structural properties |

The status table, terminology ledger and editorial audit are working metadata and are not part of the manuscript body submitted for review.

## Terminology ledger

| Canonical term | Symbol | Meaning in Section 4 |
| --- | --- | --- |
| Maximum future domain | $T$ | Largest supported future-step index |
| Sharing scope | $s$ | Number of contiguous future steps that reuse one scope-region latent state |
| Scope set | $\mathcal S$ | Supported latent-state sharing scopes, with $S=|\mathcal S|$ |
| Future-step coordinate | $\boldsymbol\phi_\tau$ | Fixed descriptor of future time step $\tau$ |
| Scope-region latent state | $\mathbf z_{b,c,g}^{(s)}$ | History-conditioned state shared within region $g$ under scope $s$ |
| Scope-indexed forecast field | $\mathcal F_\theta(\mathbf X)$ | Predictions indexed jointly by variable, future step and sharing scope |
| Scope-conditioned slice | $\mathcal F^{(s)}$ | The field restricted to one sharing scope; not an independent forecasting model |
| Target-conditioned scope allocation | $\boldsymbol\Pi$ | Scope weights conditioned on sample, variable and future-step identity |
| Weighted contraction | $\sum_s\pi_s\mathcal F_s$ | Integration of scope-conditioned forecasts along the scope axis |
| Balanced Scope Co-Adaptation | BSCA | Training-only objective for direct slice supervision and broad allocation access |

## 4. ISCF-BSCA

Section 3 established two requirements for varied-horizon forecasting. Predictions for a shared future target should be invariant to the requested horizon, while the decoder should not impose one latent-state sharing extent on the entire future domain. We address these requirements with ISCF-BSCA, an output-side architecture that constructs one future-step-indexed trajectory from multiple sharing scopes. Independent Scope-Conditioned Forecasting (ISCF) defines the inference graph, and Balanced Scope Co-Adaptation (BSCA) supports its joint optimization without changing that graph.

### 4.1 Architecture overview

Figure 4 summarizes the complete architecture. Given an observed history $\mathbf X\in\mathbb R^{B\times L\times C}$, a shared encoder first produces a variable-wise representation. We denote this history representation by $\mathbf R\in\mathbb R^{B\times C\times R}$. ISCF then applies independent history projections for $S$ sharing scopes. Each projection constructs latent states that are reused over a different number of future steps. Shared step-specific synthesis vectors convert these states into one scope-indexed forecast field $\mathcal F_\theta(\mathbf X)\in\mathbb R^{B\times C\times T\times S}$.

The field retains a forecast for every future step under every sharing scope. A target-conditioned allocation $\boldsymbol\Pi\in\mathbb R^{B\times C\times T\times S}$ assigns normalized scope weights to each sample, variable and future step. Contracting $\mathcal F_\theta(\mathbf X)$ with $\boldsymbol\Pi$ along the scope axis yields one trajectory $\widehat{\mathbf Y}\in\mathbb R^{B\times T\times C}$. A request with horizon $H$ returns its first $H$ future steps, but $H$ does not enter the history encoding, field construction or allocation functions.

The multiple scope-conditioned slices must learn jointly with the allocation that combines them. Early allocation concentration can weaken the forecasting gradients received by low-weight slices. BSCA addresses this optimization coupling through direct predictive supervision for every slice and a ramped uniform anchor on the allocation. Both terms are used only during training, leaving the ISCF inference path unchanged.

<a id="fig:iscf-bsca-method"></a>

![Overview of the ISCF-BSCA architecture.](../../paper-figures/figure_iscf_bsca_method_overview.png)

**Figure 4 | ISCF-BSCA constructs one prefix-consistent trajectory from multiple latent-state sharing scopes.** **a**, A single-scope decoder applies one fixed sharing extent throughout the future domain. **b**, ISCF maps a shared history representation through independent scope-specific projections. Scope-region states use different sharing extents, while shared step-specific synthesis produces one scope-indexed forecast field. **c**, An allocation conditioned on the history representation and future-step coordinate contracts the field along its scope axis. The resulting trajectory provides nested prefixes for different horizon requests. **d**, BSCA adds direct slice-skill supervision and a uniform allocation anchor during training only; the inference graph contains only the solid field, allocation and contraction paths. Scope size denotes latent-state sharing extent rather than requested horizon, and the displayed allocation is schematic rather than empirical.

### 4.2 History representation and future-step coordinates

ISCF operates on a variable-wise representation and can therefore be attached to an encoder that preserves the variable axis. In our realization, the input is normalized per variable, divided into $P$ patches and encoded with shared patch-processing layers. This produces $\mathbf Z\in\mathbb R^{B\times C\times P\times D_e}$, which is flattened over its patch and embedding dimensions:

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

The decoder identifies future targets through a fixed coordinate field $\boldsymbol\Phi=[\boldsymbol\phi_1,\ldots,\boldsymbol\phi_T]^\top\in\mathbb R^{T\times D_q}$. For $d=0,\ldots,D_q-1$, we first define

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

These parameter-free coordinates distinguish future time steps without introducing a requested-horizon input. They are used both to construct scope-region descriptors and to specify the target identity for scope allocation. No observed future value or label enters either path at inference.

### 4.3 Scope-indexed forecast field

The central operation in ISCF is to vary how broadly a history-conditioned state is reused before step-specific synthesis. Let $\mathcal S=\{s_1,\ldots,s_S\}$ denote the supported scopes. The frozen realization uses $\mathcal S=\{1,48,144,360,720\}$ with $T=720$. The formulation applies to any preregistered scope sizes that divide the future domain.

For a scope $s$, the future domain is divided into contiguous regions

$$
\mathcal G_g^{(s)}
=
\{(g-1)s+1,\ldots,gs\},
\qquad
g=1,\ldots,T/s.
$$

Each region is represented by the mean coordinate of its future steps:

$$
\overline{\boldsymbol\phi}_g^{(s)}
=
\frac{1}{s}
\sum_{\tau\in\mathcal G_g^{(s)}}
\boldsymbol\phi_\tau
\in\mathbb R^{D_q}.
$$

ISCF assigns an independent history projection to each scope. Projection $s$ maps $\mathbf r_{b,c}$ into a scope-specific mode matrix

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

The mode matrix interacts with the descriptor of region $g$ to produce a history-conditioned, region-indexed latent state:

$$
\mathbf z_{b,c,g}^{(s)}
=
\left(\overline{\boldsymbol\phi}_g^{(s)}\right)^\top
\mathbf M_{b,c}^{(s)}
\in\mathbb R^K.
$$

All future steps in $\mathcal G_g^{(s)}$ reuse this state. Scope $s=1$ therefore constructs one state per future step, whereas $s=T$ shares one state across the complete future domain. Intermediate scopes impose intermediate reuse patterns. Scope describes this cross-step relation and is not a property of an isolated future step or a requested horizon.

Sharing a region state does not force its future steps to receive identical predictions. Let $g_s(\tau)$ denote the region containing step $\tau$. ISCF uses shared step-specific synthesis vectors $\mathbf a_\tau,\mathbf n_\tau\in\mathbb R^K$ and bias $\beta_\tau$ to define

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

Collecting these values over samples, variables, future steps and scopes produces $\mathcal F_\theta(\mathbf X)\in\mathbb R^{B\times C\times T\times S}$. The history projections are independent across scopes, while the encoder, coordinate field and synthesis vectors are shared. Consequently, a fixed-$s$ slice is one view of a common output field rather than a separately trained forecasting model.

### 4.4 Target-conditioned scope allocation

No fixed scope is assumed to be preferable throughout the future domain. ISCF instead learns an allocation at each forecast target. A compact history projection first produces

$$
\mathbf u_{b,c}
=
\mathbf W_h\mathbf r_{b,c}
+
\mathbf b_h
\in\mathbb R^{D_h}.
$$

The history summary is paired with the coordinate of future step $\tau$. An allocation network then computes

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

Here, target-conditioned means conditioning on the sample, variable and future-step identity. It does not mean that the allocation observes the future target value. The final normalized prediction is obtained through weighted contraction along the scope axis:

$$
\widehat y_{b,\tau,c}
=
\sum_{s=1}^{S}
\pi_{b,c,\tau,s}
\mathcal F_{b,c,\tau,s}.
$$

The complete normalized trajectory is transformed back to the original data scale. A request with horizon $H$ returns

$$
\widehat{\mathbf Y}_b^{(H)}
=
\left[
\widehat y_{b,\tau,c}
\right]_{\tau=1,\ldots,H;\ c=1,\ldots,C}.
$$

The allocation allows the sharing composition to vary over future targets, but the architecture alone does not imply that it recovers an oracle scope or forms universal specialization. Those questions require the component and internal-behavior analyses reported later.

### 4.5 Balanced Scope Co-Adaptation

The allocation has two coupled roles during training. It combines the scope-conditioned predictions, and it scales the fused-loss gradient received by each slice. For a fixed target, the contraction implies

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

An allocation that concentrates early can therefore restrict the forecasting gradients reaching low-weight scopes before their predictive paths have matured. BSCA addresses this optimization problem without modifying ISCF at inference.

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

During training, inverse normalization is applied to the fused forecast and every scope-conditioned slice before evaluating the objective. For notational simplicity, the symbols below denote these raw-scale quantities. Let $y_{b,\tau,c}$ be the corresponding target. The fused prediction loss is

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

BSCA first supplies every scope-conditioned slice with direct predictive supervision:

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

The direct skill term trains each slice even when its current allocation is small. The anchor acts directly on allocation logits and broadens allocation-mediated access during early joint learning. Uniform allocation is only an optimization proxy: it neither forces equal inference-time usage nor guarantees semantically distinct scopes. Both BSCA terms are removed at inference, so the trained model retains the field, allocation and contraction defined by ISCF.

### 4.6 Structural properties and complexity

ISCF satisfies CHPC by construction. For a fixed target, $\mathcal F_{b,c,\tau,s}$ and $\pi_{b,c,\tau,s}$ depend on the history and target identity $(\tau,c)$. Neither depends on the requested horizon. Hence $\widehat y_{b,\tau,c}$ is identical under any requests $H_i,H_j\geq\tau$. Returning the indexed prefix $1{:}H$ changes only the exposed endpoint and yields the CHPC relation defined in Section 3.

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

trainable parameters. The first line corresponds to scope-specific history modes and shared step-specific synthesis; the second corresponds to target-conditioned allocation. Future-step coordinates, contiguous group indices and region averaging are parameter free. BSCA adds no trainable parameter.

Computing scope-region states costs $\mathcal O\!\left(BC D_q K\sum_{s\in\mathcal S}T/s\right)$, while step-specific synthesis costs $\mathcal O(BCSTK)$. The allocation network is evaluated once per sample-variable history state and future target. Materializing both the field and allocation requires $\mathcal O(BCTS)$ working memory, and the implementation supports region and target chunking to reduce peak intermediates without changing the function. Only one encoder-decoder checkpoint is stored and served for all supported horizons; empirical latency, memory and forecasting accuracy are evaluated separately in Section 5.

## Editorial implementation and claim audit

| Manuscript element | Frozen implementation correspondence | Permitted claim | Deferred claim |
| --- | --- | --- | --- |
| History representation | Normalization, patch embedding/encoder and flattened `hidden:[B,C,R]` | ISCF accepts a shared variable-wise history state | Encoder superiority or universal backbone compatibility |
| Independent scope projections | Independent SIFF scale basis with one component per scope | Each scope has an independent history projection | Each slice is an independent forecasting model |
| Scope-region state | Contiguous group indices and pooled future-step coordinates | Scope controls latent-state reuse extent | Canonical partition is universally optimal |
| Scope field | `arm_forecasts`, paper-facing shape `[B,C,T,S]` | One field contains multiple scope-conditioned slices | Learned specialization or oracle scope recovery |
| Allocation | Direct history-plus-target-coordinate policy, `[B,C,T,S]` | Weights vary by sample, variable and future step | Allocation is label-conditioned or necessarily region optimal |
| Weighted contraction | Scope-axis weighted sum yielding `[B,T,C]` | One horizon-agnostic trajectory and CHPC | Lower error than horizon-specific systems |
| BSCA | Uniform slice-skill loss plus ramped normalized `KL(uniform || allocation)` | Training-only balanced access; no inference path or parameter | Generic KL novelty, universal gain or semantic expert specialization |
| Section 5 interface | Pending main, ablation, transfer and efficiency tables | Structural and reproducibility statements only | Main-table superiority, component effectiveness and decoder portability |
