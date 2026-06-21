# PatchEncoderFutureAwareAdapter Phase 1 Candidate

`PatchEncoderFutureAwareAdapter` extends `PatchEncoderFixedHeadAdapter` with a
training-only future teacher branch.

Purpose:

- keep the fixed flatten head and future-segment adapter interface from the
  Phase1-A.2 partial pass;
- use ground-truth future only during training to build a future-side teacher
  state;
- align the history-derived student adapter state to that teacher state without
  leaking future values into inference.

Training loss:

```text
loss = pred_mse
     + align_weight * alignment_loss
     + recon_weight * reconstruction_loss
```

Inference path:

```text
x -> patch encoder -> fixed head + adapter -> prediction
```

The `y` tensor is optional in `forward`. If `y` is absent, the teacher branch is
not executed. The trainer writes `future_alignment_stats.csv`, including
`prediction_leakage_max_abs`, to audit that predictions are unchanged when a
future tensor is supplied only for diagnostics.
