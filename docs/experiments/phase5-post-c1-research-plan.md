# Phase5 Post-C1 Research Plan

## Decision cursor

| Field | Content |
| --- | --- |
| `current_step` | C1 Step 9-10 failed；carrier rollback complete；StageB returns Step 2/3 |
| `active_carrier` | `A6-LBF-r256 + exact valid HPM [B,C,29,48]` |
| `paper_core` | Contribution 1 `A6-LBF-r256` only |
| `problem` | Contribution 1 still lacks an objective-only matched control；StageB architecture routes are exhausted while B7 supervision imbalance remains only partially supported |
| `idea` | First close the Contribution 1 protocol confound；then re-audit horizon-agnostic supervision allocation without adding another Encoder architecture |
| `narrative_gate` | matched control is confirmation-only；B7 must independently pass Step 2/3 before any method design |
| `effectiveness_gate` | defined separately below；no new model is authorized by this planning note |
| `rollback` | if B7 causal evidence is weak or dataset-specific, pause Contribution 2 and consolidate the paper around Contribution 1 |

## Why C1 is not repaired

[Fact] P16-S8/P48-S24 fail every shared and validation-selected gate. Best-val overall regressions are
`+3.75%/+4.73%`; neither is a near miss.

[Strong Evidence] C1 has two coupled design risks: early validation overfit and a global-only `D=256` readout that
compresses ETTh2/Weather legacy state widths by `83.33%/95.83%`. A dropout-only follow-up cannot isolate both.

[Decision] Do not run scale, mixer, width, global-readout or dropout sweeps. Such work would no longer be Encoder
normalization; it would be a new architecture search without a StageB innovation claim.

## Workstream A: Contribution 1 matched-supervision control

### Research question

> Under the same dataset-specific 720-step A6 architecture, initialization family, dropout, optimizer, learning rate,
> training epochs and checkpoint selector, does multi-prefix supervision outperform a single-prefix objective?

This control separates the learned-basis unified operator/objective from the source presets used by fixed-horizon
TimeAlign. It is not a new contribution and should precede further StageB claims.

### Design boundary

- freeze source-faithful A6 architecture separately for ETTh2/ETTm1/Weather；
- one 720-step output path and the same learned basis rank for every arm；
- arms: one `multi_prefix` objective over `{96,192,336,720}` and four `single_prefix_H` objectives；
- a `single_prefix_H` arm supervises only `prediction[:H]` but retains the same 720-step model；
- use the same official-last primary selector and record best-val as sensitivity；
- first run seed 2021；only a stable, material result authorizes seeds 2022/2023；
- report both target-horizon performance and disjoint future-region errors, so short-prefix gains cannot hide tail damage。

### Gate

This is an attribution gate, not a “multi-prefix must win everything” gate:

1. verify all non-objective effective configs are identical within each dataset；
2. compare multi-prefix to each matched single-prefix specialist at its own target horizon；
3. report dataset/horizon wins and mean MSE/MAE without mixing checkpoint selectors；
4. if the unified advantage disappears under matched architecture, narrow Contribution 1 to practical
   source-faithful unified forecasting rather than objective/architecture superiority；
5. if advantage remains consistent, promote the architecture/objective attribution from
   `controlled_confirmation_pending` to `controlled_supported`。

## Workstream B: B7 horizon-agnostic supervision re-audit

Workstream B begins only after Workstream A has fixed the contribution boundary. It returns to Step 2/3 and does not
revive the previous benchmark-horizon weighting proposal mechanically.

### Problem hypothesis

Current loss averages the mean losses of `{96,192,336,720}` prefixes. Its closed-form step exposure gives `0-96`
steps `14.39x` the weight of `336-720`. ETTh2/ETTm1 show weaker A6 gains in the tail, while Weather is a counterexample.
Therefore exposure imbalance is a plausible problem, not yet a causal cross-dataset fact.

### Diagnostic sequence

1. `B7-GTD`: on fixed train batches, decompose each prefix loss gradient on the shared A6 encoder and coefficient/basis
   path；report gradient norms, cosine matrices, conflict pairs and dataset stability。
2. `B7-Exposure`: replace benchmark-set sampling only in a diagnostic control with continuous random prefix lengths
   over a declared range；measure the empirical per-step exposure and gradient profile, not test performance cherry-picking。
3. Connect training evidence to disjoint future-region errors. A valid problem requires the exposure/conflict statistic
   to predict tail weakness across datasets or explain the Weather counterexample。

### Step 3 gate

B7 may enter Step 4-6 only if:

- imbalance/conflict is stable across batches and at least two datasets；
- the sign or magnitude aligns with disjoint-region generalization, not merely step distance；
- a continuous-prefix control changes the predicted gradient/exposure mechanism；
- the result cannot be explained only by total loss scale or longer-horizon intrinsic difficulty。

If the gate passes, Step 4-6 should compare horizon-agnostic sampling/normalization mechanisms and their exact gradient
semantics before implementation. If it fails, StageB is paused；do not reopen Encoder, retrieval, recurrent-unit,
stage-ID or basis-bank routes without new independent problem evidence。

## Execution order

1. Implement and locally verify Workstream A's matched-objective runner/analyzer。
2. Commit/push, then run the seed-2021 matched control remotely。
3. Decide Contribution 1 attribution from returned artifacts。
4. Implement B7-GTD as an offline/small-batch diagnostic only。
5. Pass or reject B7 Step 3；only then discuss a new StageB method。
