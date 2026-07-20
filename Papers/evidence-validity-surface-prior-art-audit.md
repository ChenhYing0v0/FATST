# Evidence-Validity Surface Prior-Art Audit

## Metadata

- Search date: `2026-07-20`
- Scope: fixed-past multi-horizon decoding, sample-adaptive model fusion,
  expert routing, multiscale prediction, target-coordinate retrieval, and
  varied-horizon projectivity
- Discovery: external primary-source search; Zotero is a seed library only
- Evidence status: proceedings, official arXiv/OpenReview pages, and official
  project pages; very recent submissions are treated as overlap pressure rather
  than settled results

## Why the old problem statement is insufficient

The provisional phrase `future-distance predictive support` is too broad.
Frequency robustness, future-coordinate queries, multiscale predictors, and
sample-level model arbitration can all be described with similar language.
The paper needs a problem that follows specifically from the contract

$$
x_{1:L}\longmapsto \hat y_{1:T},\qquad
F_H(x)=P_HF_T(x),quad H\le T,
$$

where requested horizon $H$ is not a semantic input.

D14-A already supplies the strongest local clue: independently trained
point/block/global coupling arms exhibit stable sample-by-region crossing on
both a neutral raw-history carrier and the A6-natural carrier. Three-seed
sample-over-bin oracle headroom is `6.7948%` and `8.5990%`, respectively.
However, an oracle using future truth does not show that this heterogeneity is
available at inference time.

D20 does not supply this evidence. Its SPEC path was important inside the
co-adapted model, but SPEC/RANDOM responsibility relocation explained the
importance and the complete SPEC model did not beat A6. Therefore D20 can only
motivate a question about conditional route validity; it cannot establish it.

## Primary-source boundary

| Work | What is already covered | Boundary for this project |
| --- | --- | --- |
| [TimeFuse](https://arxiv.org/abs/2505.18442) | input meta-features predict sample-level weights over heterogeneous forecasting models | sample-only external model fusion is not novel |
| [TimeRouter](https://arxiv.org/abs/2606.11625) | discriminative selection/gating/ensemble fallback over a pool of pretrained TSFMs | external expert arbitration is not the proposed problem |
| [Synapse](https://arxiv.org/abs/2511.05460) | context-dependent arbitration of TSFM outputs, including horizon-distribution effects | horizon-aware model-pool arbitration is a mandatory boundary |
| [TimeMixer](https://arxiv.org/abs/2405.14616) | multiscale past extraction and future multipredictor mixing | multiple scales or predictors alone are not a contribution |
| [MQTransformer](https://openreview.net/forum?id=rxF4IN3R2ml) | context-dependent decoder-to-history attention for each forecast context | target-dependent history retrieval is prior art |
| [TimePerceiver](https://arxiv.org/abs/2512.22550) | learnable target-timestamp queries retrieve from encoded latent states | future-coordinate queries alone are not novel |
| [ElasTST](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d7aa002885ccbe68cf6880da583761b2-Abstract-Conference.html) | varied-horizon invariance, structured masks, multi-patch input, horizon-reweighted training | projectivity and multiscale patches are required controls, not standalone claims |

## New problem definition

### Evidence-Validity Surface (EVS)

Let $e\in\mathcal E$ index an internal forecast-construction route. A route may
differ in output-coupling scope or in which history evidence it propagates, but
it is not a separately deployed forecasting model. Define its conditional
relative risk at future coordinate $\tau$ as

$$
R_e(x,\tau)=
\mathbb E[\ell(f_e(x,\tau),Y_\tau)\mid X=x].
$$

The **Evidence-Validity Surface** hypothesis is that route ordering is governed
by a non-separable function of both the observed past and the future coordinate:

$$
R_e(x,\tau)\ne a_e(x)+b_e(\tau)
$$

on a practically meaningful subset of the data. Equivalently, a policy that
uses a genuine $x\times\tau$ interaction should outperform all three simpler
explanations:

1. one globally fixed route;
2. a future-region-only fixed route;
3. a sample-adaptive but future-region-invariant route, including an additive
   sample-plus-region risk model.

This is the narrow point that makes the problem multi-horizon. If only item 2
is needed, the result is a static segmented decoder. If only item 3 is needed,
the result is generic TimeFuse-style adaptive fusion. The current paper needs
the interaction.

## Why this can support two contributions

The problem exposes two different missing contracts rather than two arbitrary
modules:

1. **Representation contract**: a single projective decoder must retain several
   internally useful construction routes and permit their influence to vary
   with $x$ and $\tau$ without reading requested $H$.
2. **Credit contract**: final fused-forecast loss must assign useful credit to
   those routes in the same end-to-end graph. Offline oracle labels,
   cross-fitted external teachers, and stale expert-risk targets are not
   acceptable as the core training principle.

Therefore a future architecture contribution and a future training
contribution can be causally linked. Neither is authorized until the EVS
problem diagnostic passes.

## Mandatory self-critique

- An expressive full-trajectory decoder can in principle represent
  coordinate-dependent behavior implicitly. EVS is an inductive-bias and
  trainability claim, not a universal function-class impossibility theorem.
- D14 arms are independently trained decoder strategies. They are suitable for
  an existence diagnostic but cannot be promoted directly into the final
  jointly trained architecture.
- Squared-error expert ranking is not the optimum convex-mixture weight because
  prediction-error cancellation matters. D21 tests predictable route selection
  as a conservative existence witness, not the final fusion objective.
- Failure of one handcrafted history descriptor/readout cannot reject EVS. It
  can reject only that identifiability probe unless a sufficiently expressive,
  leakage-free sensitivity model also fails.

## Decision

`SC-D21-EVS` replaces the underspecified `SC-D21-FDS` as the provisional
Step2/3 problem diagnostic. It is not a paper method. The problem may enter
Step4 only if validation-fitted policies transfer to official test and the
past-by-region interaction beats region-only, sample-only/additive, and
permuted-history controls.
