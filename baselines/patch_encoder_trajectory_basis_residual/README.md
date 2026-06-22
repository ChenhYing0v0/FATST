# PatchEncoderTrajectoryBasisResidual

Phase1-A.6 candidate.

This baseline keeps the Phase0 patch encoder and fixed dense head as the base
trajectory readout, then adds a zero-initialized, future-position-aware
low-rank residual over output steps.

It tests the A.6 hypothesis:

- fixed head rows should not be removed;
- latent-state modulation can disturb short-horizon readout;
- the remaining decoder problem may be correlated output-trajectory residual
  structure.

Main diagnostics:

- `trajectory_residual_stats.csv`
- `metrics_by_segment.csv`
- `metrics_by_horizon.csv`

The first gate remains one-to-one horizon training. One-model compatibility,
future-aware alignment, and MoE are blocked until this candidate passes.
