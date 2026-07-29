# Neutral Sharing-Extent Introduction Diagnostic

This folder contains a diagnostic-only, channel-wise forecaster used to test
future-region sharing-demand heterogeneity before introducing ISCF-BSCA.

Each run contains exactly one sharing extent. All variants share the same
history encoder, future-step descriptors, state generator, step-specific
synthesis vectors, parameter count, and high-cost candidate-state path. The
only changed operation is parameter-free pooling of candidate states over
future-step blocks.

This is not a paper method, an ISCF ablation, or an inference-time component.
Validation artifacts are used only for exploratory visualization. The training
entrypoint never constructs or evaluates the test split.
