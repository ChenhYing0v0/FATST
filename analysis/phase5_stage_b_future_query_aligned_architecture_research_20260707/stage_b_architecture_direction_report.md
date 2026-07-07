# StageB Architecture Direction Research Report

## Decision

[Decision] `B8-FQA` is the preferred next StageB architecture candidate.

Full name: `Future-Query Aligned Basis Operator`.

Current status: `proposed_architecture_candidate`, not method-ready.

Next required action: run `B8-OCD`, a coefficient-space oracle capacity diagnostic, before implementation.

## User Constraint

[Fact] The user prefers StageB to establish a second main innovation point, ideally architecture-level, because
StageA mainly changed the decoder/head. Objective optimization, including B7-UPO, may remain a small contribution
but should not be the main StageB route.

[Decision] Channel-correlation modeling is excluded from this search.

## Why Not Simply Restore TimeAlign Future Align

Original TimeAlign is relevant but not sufficient as StageB:

- TimeAlign's contribution is training-time future reconstruction plus representation alignment.
- B4 dependency ablation already showed A6-LBF does not materially depend on inherited align/recon.
- Restoring a generic `w_align * align_loss` would make the paper look like a TimeAlign variant again.
- TimeAlign alignment is not designed around unified multi-horizon basis coefficients.

Therefore StageB should not be:

```text
A6-LBF + TimeAlign future branch + align loss
```

The usable idea is more abstract:

> history-only representations may not be sufficiently aligned with future target positions.

B8 turns this into an architecture problem rather than an auxiliary-loss problem.

## Literature Synthesis

### TimeAlign

[Fact] TimeAlign diagnoses a structural limitation of history-only forecasting: historical representations are mapped
directly to future targets, causing past/future distribution mismatch and low-frequency smoothing. It introduces a
training-only future reconstruction branch and local/global representation alignment.

Source: <https://arxiv.org/html/2509.14181v3>.

Implication for us:

- keep the problem framing: prediction representations need future alignment;
- reject the exact mechanism as StageB main route, because it is not unified-horizon-specific and was not necessary
  for A6-LBF.

### ElasTST

[Fact] ElasTST uses future placeholders and structured masks to make varied-horizon forecasts invariant to horizon
extension.

Source: <https://arxiv.org/html/2411.01842v1>.

Implication for us:

- future positions can be represented as tokens without future-value leakage;
- structured masks or independent queries are important to keep prefix invariance.

### TimePerceiver

[Fact] TimePerceiver uses target-position-aware decoder queries and shows that query-based decoding plus temporal
positional alignment is important in generalized forecasting.

Source: <https://arxiv.org/html/2512.22550v1>.

Implication for us:

- target queries are a credible architecture mechanism;
- we should not copy a full generalized forecasting framework, but can adapt the future-query idea to A6's basis
  coefficient interface.

### SRP++

[Fact] The local paper note `Papers/srp-step-specific-representation.md` records SRP++'s claim that multi-step
forecasting can need step/segment-specific representations rather than one shared representation for all future
steps.

Implication for us:

- A6-LBF's global coefficient vector may be a representation bottleneck;
- StageB can introduce future-position-specific representation without abandoning one unified model.

## Core Problem In A6

A6-LBF currently predicts:

```text
hidden = encoder(history)            # [B, C, R]
coeff = W(hidden)                    # [B, C, K]
y[t, c] = basis[t] @ coeff[c] + b[t]
```

This is clean and strong, but it has an architecture limitation:

- `basis[t]` is future-position-specific but global;
- `coeff[c]` is sample-specific but future-position-invariant;
- there is no `coeff[t, c]` or target-position-aware representation.

[Hypothesis] A second main innovation can target this exact missing interface:

> A unified multi-horizon model should align history representations to future positions before basis prediction,
> not only attach a prefix-native basis decoder after a history-only encoder.

## Recommended Candidate: B8-FQA

Minimal architecture:

```text
history tokens:
  x_tokens = encoder(history)                     # [B*C, Nx, D]

future queries:
  q_pos = future position / segment embeddings    # [Nf, D]

future-query alignment:
  z_f = CrossAttention(q_pos, x_tokens, x_tokens) # [B*C, Nf, D]

A6 base path:
  c_base = learned_basis_coeff(pool(x_tokens))    # [B, C, K]

position-aware coefficient modulation:
  c_s = c_base + alpha_s * DeltaCoeff(z_f[s])     # [B, C, K]

prediction:
  y_t = basis[t] @ c_s + bias[t], for t in segment s
```

Preservation rule:

- initialize `alpha_s = 0`;
- first forward pass exactly equals clean A6-LBF-r256;
- this satisfies the project rule that capacity-preservation claims must be code-theory checked.

## Candidate Comparison

| Candidate | Main idea | Narrative strength | Feasibility | Risk |
| --- | --- | --- | --- | --- |
| Restore TimeAlign align | Re-add future reconstruction/align branch | weak after B4 | high | looks inherited; not A6-specific |
| Basis-aware align | Align history/future coefficients | medium | medium | B6 showed learned basis not stronger than DCT; needs new evidence |
| ElasTST-style placeholders | Add future placeholders into encoder | medium-high | medium | may look like full ElasTST adaptation |
| TimePerceiver-style target queries | Query future positions from history | high | medium | can repeat old A5 target-query collapse if capacity not preserved |
| `B8-FQA` | Future queries modulate A6 basis coefficients | highest | medium | needs oracle evidence; implementation must remain lightweight |

## Why B8 Connects To StageA

StageA:

> Replace dense/fixed prediction head with a prefix-native learned-basis forecast operator.

StageB B8:

> Make the representation feeding that operator future-position-aware, so the unified operator receives different
> sample-conditioned states for different future regions.

This forms a coherent two-part architecture:

1. unified forecast operator;
2. future-query aligned representation interface.

## Difference From Prior Work

Different from TimeAlign:

- no future values at inference;
- not a generic hidden alignment loss;
- alignment target is future position/query state, not only reconstructed future value distribution.

Different from ElasTST:

- not a full placeholder Transformer;
- keeps A6 learned-basis operator;
- uses future queries to modulate coefficient space.

Different from TimePerceiver:

- not generalized interpolation/imputation;
- target queries are not the whole decoder;
- queries serve the learned-basis coefficient interface.

Different from B7:

- B7 optimizes loss weighting;
- B8 changes architecture.

## Required Next Diagnostic

`B8-OCD`: coefficient-space oracle capacity diagnostic.

Purpose:

> Determine whether future-segment-specific coefficients under the same learned basis can reduce A6 residuals.

Procedure:

1. Obtain clean A6 checkpoint or equivalent learned basis.
2. Use predictions/targets for ETTh2, ETTm1, Weather.
3. For each sample/channel/segment, solve ridge least-squares coefficients using the same learned basis rows.
4. Compare A6 global coefficient prediction with oracle segment-specific coefficient reconstruction.
5. If segment-specific oracle gains are meaningful, B8 has a real target.

Gate:

- Pass if oracle segment-specific coefficients reduce tail/segment residuals substantially on at least ETTh2 and
  ETTm1, without reducing to a generic DCT/frequency explanation.
- Fail if gains are tiny, only Weather-specific, or explained by a generic basis control.

## Next Research Step

Do not implement B8 yet.

Next action:

1. sync or locate clean A6 checkpoint containing `learned_temporal_basis`;
2. write `B8-OCD` diagnostic protocol and analyzer;
3. only after B8-OCD passes, enter Step 4-6 method design.
