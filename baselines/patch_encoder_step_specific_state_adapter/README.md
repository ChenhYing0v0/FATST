# PatchEncoderStepSpecificStateAdapter Phase 1 Candidate

`PatchEncoderStepSpecificStateAdapter` keeps the Phase0 fixed flatten head rows
as the readout path, but moves the future-segment adapter before readout.

Purpose:

- preserve the strong `Linear(n_patches * d_model, pred_len)` fixed-head
  readout capacity;
- test whether future segment states can create step/segment-specific
  representations before the fixed-head rows are applied;
- provide a safer interface for later future-aware alignment and future-side
  MoE.

Default segment length is `48`. The adapter produces segment-wise affine terms
`gamma` and `beta` over the latent patch representation:

```text
z_j = z * (1 + gamma_j) + beta_j
y_j = fixed_head_rows_j(flatten(z_j))
```

The final adapter projection is zero-initialized, so the initial forward path is
identical to `PatchEncoderFixedHead` before training updates the adapter branch.
