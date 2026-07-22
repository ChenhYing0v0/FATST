# FRSC Step7B Prelaunch Audit

## 1. What is tested and why

FRSC-v0 tests whether invertible scope conditioning can preserve the ISCF carrier while making canonical scope geometry useful. The candidate must improve on the frozen identity reference and cannot attribute a gain to scope structure unless it also beats same-alpha global, best-tuned global, and random-binding controls.

## 2. Matrix and artifact construction

- new training: four arms x five datasets x seed2021 = `20` from-scratch matched runs;
- effective audit: add five historical `sps_identity_canonical` checkpoints = `25` runs;
- checkpoint selector: mean validation MSE over H96/H192/H336/H720;
- required outputs: checkpoint, training log, four-horizon metrics, effective config, initialization contract, model diagnostics, validation diagnostic tensors, and trained invariants;
- split boundary: train optimizes the frozen full-H720 objective; validation selects checkpoints and screens the mechanism; official test is disabled.

## 3. Statistics and controls

`effectiveness_vs_identity` is the primary carrier-preservation comparison. `scope_specificity_vs_global` compares against alpha .45, while `same_alpha_scope_vs_global` isolates geometry at alpha .55. `canonical_binding_vs_random` is attribution-only. Internal health reports conditioned-arm pairwise RMS normalized by target RMS, oracle headroom, normalized policy entropy, future-bin winner count, and conditioning-delta/raw RMS. These diagnostics cannot override a negative effectiveness gate.

## 4. Prelaunch evidence

- decision: `step8_remote_validation_authorized`;
- checks: `37/37`;
- all model arms satisfy output shape, finite values, matched projection ranks/degrees, and strictly positive minimum operator eigenvalue;
- alpha0 reproduces the ISCF parent, and trainable initialization is paired within each dataset rank;
- runner dry-run enumerates 20 unique jobs, formal-test split exits with code 3, analyzer synthetic smoke passes, and GPU/log scanning is present.

## 5. Failure attribution and decision

Candidate below identity without pathology means exact FRSC-v0 is not supported and returns to Step4; global tie means scope specificity is unresolved; random tie/loss blocks canonical binding attribution but cannot reject ISCF; diversity/oracle collapse points to intervention/readout design; NaN/OOM/divergence is an optimization or numeric pathology and only blocks the exact execution. Current decision is `step8_remote_validation_authorized`: remote validation may start after commit-pinned pull, GPU/process preflight, and resource smoke. Formal test, confirmation seeds, modern baselines, new loss/router, and requested-H input remain unauthorized.
