# Phase5 StageB B8 Future-Query Aligned Basis Architecture

## Stage Record

| Field | Content |
| --- | --- |
| `candidate_id` | `B8-FQA` |
| `current_step` | Step 1-2: literature research and architecture problem proposal |
| `problem` | StageA A6-LBF-r256 已统一 decoder/head，但 sample-specific coefficient 是 horizon/position-invariant；future positions 只通过全局 learned basis 区分，缺少 target-position-aware representation |
| `existence_evidence` | code-theory evidence plus literature evidence; still needs coefficient-space oracle diagnostic |
| `idea` | Introduce future-position query/placeholder tokens that attend to history tokens and generate target-position-aware coefficient modulation before the learned-basis operator |
| `theory_check` | This is architecture-level and deepens unified prediction; it does not revive generic TimeAlign auxiliary loss and does not rely on channel modeling |
| `design` | Proposed below; no implementation yet |
| `narrative_gate` | promising but not passed; requires B8-OCD diagnostic |
| `effectiveness_gate` | not evaluated |
| `artifacts` | this protocol; `analysis/phase5_stage_b_future_query_aligned_architecture_research_20260707/` |
| `decision` | `proposed_architecture_candidate`; B7-UPO deferred as small objective contribution |

## Why StageB Should Not Stay On B7

[Inference] B7 unified prefix optimization is real and useful, but it is primarily an objective/training issue. It can
become a small contribution or a method refinement, but it is unlikely to carry the paper's second main innovation by
itself.

[Decision] StageB should first search for a second architecture-level contribution. Because StageA changes the
forecast operator/readout, StageB should change the representation interface that feeds this operator.

## Literature Evidence

[Fact] TimeAlign argues that history-only forecasting has past/future distribution mismatch and introduces a
training-only reconstruct branch plus local/global representation alignment. Its core claim is architectural: align
prediction representations with a target-distribution reference rather than relying only on point loss.
Source: <https://arxiv.org/html/2509.14181v3>.

[Fact] ElasTST introduces future placeholders and structured attention masks for varied-horizon forecasting. Its
placeholder design ensures that extending the inference horizon does not alter earlier future outputs.
Source: <https://arxiv.org/html/2411.01842v1>.

[Fact] TimePerceiver uses query-based decoding with temporal and channel positional embeddings in decoder queries.
It reports that decoder queries attending to inputs are important for generalized temporal prediction.
Source: <https://arxiv.org/html/2512.22550v1>.

[Fact] SRP++ argues that multi-step forecasting can suffer from step-invariant representation bottlenecks and uses
step/segment-specific representations. This is aligned with the weakness of a single horizon-invariant coefficient
vector in A6-LBF. Source: local paper note `Papers/srp-step-specific-representation.md`; external OpenReview access
was blocked by browser verification during this run.

## Code-Theory Problem In A6

Current A6-LBF-r256 computes:

```text
hidden = encoder(history)                     # [B, C, R]
coeff = learned_basis_coeff(hidden)           # [B, C, K]
prediction[t, c] = learned_temporal_basis[t] @ coeff[c] + bias[t]
```

Therefore:

- `coeff[c]` is sample-specific and channel-specific;
- `learned_temporal_basis[t]` is target-position-specific but global across samples;
- the model has no sample-specific future-position representation before the final dot product.

[Hypothesis] This creates a second-stage bottleneck:

> A unified forecast operator needs not only a prefix-native basis, but also target-position-aware predictive
> representations that can adapt the sample-specific coefficient state to different future regions.

This is distinct from B6-PLO. B6 asked whether labels/residuals need a basis/frequency objective. B8 asks whether the
architecture should expose future positions as query states before prediction.

## Proposed Architecture

Name: `Future-Query Aligned Basis Operator`.

Minimal tensor path:

```text
history tokens:
  x_tokens = encoder(history)                 # [B*C, Nx, D]

future queries:
  q_pos = future_position_embedding           # [Nf, D]
  q_tokens = repeat(q_pos, B*C)               # [B*C, Nf, D]

future-query alignment:
  z_f = CrossAttention(q_tokens, x_tokens)    # [B*C, Nf, D]

base A6 coefficient:
  c_base = learned_basis_coeff(pool(x_tokens)) # [B, C, K]

target-position modulation:
  delta_c_s = zero_init_mlp(z_f[s])           # [B, C, K] per future segment/query
  c_s = c_base + gate_s * delta_c_s

prediction:
  for t in segment s:
      y_t = learned_temporal_basis[t] @ c_s + bias[t]
```

Design constraints:

- no future values at inference;
- future queries contain only target positions or segment IDs;
- preserve StageA learned-basis forecast operator;
- initialize modulation gate to zero so the first forward pass is exactly A6-LBF-r256;
- use structured masking or independent future queries to preserve prefix/horizon invariance.

## Relation To TimeAlign

Do not restore TimeAlign as-is:

- original TimeAlign aligns history branch to a target reconstruction branch;
- B4 showed inherited align/recon are not required for A6 performance;
- reusing `w_align * align_loss` would weaken the paper boundary.

What to reuse conceptually:

- TimeAlign's problem framing: history-only representation is not sufficiently future-aligned;
- training-only future branch can serve as a diagnostic teacher, not as the first method;
- alignment should be evaluated in the representation space that the final predictor uses.

B8 changes the mechanism:

- from target-value reconstruction alignment to future-position query alignment;
- from generic hidden-state alignment to coefficient-space modulation before A6 basis prediction;
- from auxiliary-loss-first to architecture-first.

## Relation To Existing Future-Query / Placeholder Work

Compared with ElasTST:

- both use future placeholders/queries;
- ElasTST is a full Transformer architecture for varied horizon;
- B8 is a minimal future-query module attached to A6's learned-basis operator.

Compared with TimePerceiver:

- both use target-position-aware queries;
- TimePerceiver builds a generalized forecasting encoder-decoder;
- B8 keeps the standard LTSF task and asks whether target queries should modulate basis coefficients.

Compared with SRP++:

- both address step/segment-specific representation;
- SRP++ uses adapter/expert specialization;
- B8 uses future-position query states and preserves a single unified model/operator.

## Narrative Gate

Current judgment: `promising_not_passed`.

Strengths:

- directly deepens StageA: decoder/head is unified in StageA; representation-to-operator interface becomes
  future-position-aware in StageB;
- architecture-level contribution, not only objective optimization;
- can be initialized as function-preserving relative to A6;
- avoids channel-correlation route and generic frequency auxiliary losses;
- offers a cleaner alternative to TimeAlign's future alignment mechanism.

Risks:

- can collapse into the old A5 target-query failure if it replaces A6 capacity instead of modulating it;
- may overlap with TimePerceiver/ElasTST unless the basis-coefficient interface is central;
- needs evidence that segment-specific coefficient modulation can reduce A6 residuals;
- extra cross-attention must stay lightweight.

## Required Diagnostic Before Implementation

`B8-OCD`: coefficient-space oracle capacity diagnostic.

Goal:

> Test whether A6 errors can be reduced by allowing future-segment-specific coefficients under the same learned
> temporal basis.

Required artifacts:

- clean A6 checkpoint, or a checkpoint-equivalent state with `learned_temporal_basis`;
- predictions/targets for ETTh2, ETTm1, Weather.

Diagnostic outline:

1. Load A6 learned basis `B`.
2. For each sample/channel and each future segment, solve a small ridge least-squares coefficient `c_s^*` using only
   that segment's true values.
3. Compare reconstruction error of:
   - global A6 prediction;
   - oracle segment-specific coefficient under the same basis;
   - simple DCT/low-rank control if needed.
4. If oracle segment-specific coefficients significantly reduce tail/segment residuals, B8 has a real architectural
   target.
5. If oracle gains are small or purely DCT-like, do not implement B8.

Only after B8-OCD passes should we enter Step 4-6 method design and remote implementation.
