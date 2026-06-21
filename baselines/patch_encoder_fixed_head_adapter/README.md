# PatchEncoderFixedHeadAdapter Phase 1 Candidate

`PatchEncoderFixedHeadAdapter` keeps the Phase0 fixed flatten head as the main
readout path and adds a lightweight future-segment adapter.

Purpose:

- preserve the strong `Linear(n_patches * d_model, pred_len)` fixed-head
  readout capacity;
- test whether future segment states can improve forecast quality when used as
  conditioning instead of replacing the readout;
- provide a safer interface for later future-aware alignment and future-side
  MoE.

Default segment length is `48`. The adapter produces segment-wise affine terms
`gamma` and `beta` over the normalized fixed-head forecast:

```text
y = fixed_y * (1 + gamma) + beta
```

The final adapter projection is zero-initialized, so the initial forward path is
identical to `PatchEncoderFixedHead` before training updates the adapter branch.
