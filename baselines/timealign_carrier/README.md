# TimeAlign Carrier Baseline

This baseline is a repo-local, source-informed TimeAlign carrier for Phase5.
It is not a wholesale import of the upstream repository.

Purpose:

- test whether a TimeAlign-style fixed-horizon carrier is viable under this
  repository's data split and metrics;
- test whether a single `pred_len=720` TimeAlign carrier has a measurable
  unified multi-horizon gap against fixed-horizon TimeAlign;
- only if both are true should HSS be designed on top of this carrier.

Core mechanism:

- history branch predicts the future sequence from history patches;
- training-only future branch reconstructs the ground-truth future sequence;
- hidden states are aligned with local and global glocal alignment;
- future branch is not used at inference time.

Key outputs:

- `metrics_by_target_horizon.csv`;
- `h{H}/metrics_by_horizon.csv`;
- `h{H}/metrics_by_segment.csv`;
- `training_log.csv`;
- `checkpoint_selection_diagnostics.csv`;
- `effective_config.json`.
