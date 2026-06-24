# Phase4 Horizon-Decoupled Supervision Code Explanation

## Scope

This document explains the Phase4-R4.2 training-side implementation for
`Horizon Supervision Scheduling for Unified Multi-Horizon Forecasting`.

The implementation changes training supervision only. It does not add a model
architecture, decoder, future-aware module, or MoE module.

## Training Boundary

Evaluation remains fixed:

- `target_horizons = [96, 192, 336, 720]`;
- validation/test loaders are still built for those horizons;
- `metrics_by_target_horizon.csv` and prefix consistency keep the same meaning.

Training is split into two paths:

1. legacy baseline path:
   - `supervision_strategy in {"horizon_mixed", "r3_prefix_risk"}`;
   - train loaders still use `target_horizons`;
   - each step samples one target horizon as before.

2. horizon-decoupled path:
   - `supervision_strategy` is one of `full_time_mse`, `random_future_mask`,
     `interval_supervision`, `component_basis_top`,
     `component_basis_balanced`, `curriculum_units`;
   - train loader uses only `supervision_pred_len`, default `720`;
   - model training forward uses `pred_len=supervision_pred_len`;
   - loss is built from future supervision units, not evaluation horizons.

## Tensor Flow

For horizon-decoupled strategies, one training batch has:

```text
x:    [B, seq_len, C]
y:    [B, 720, C]
pred: [B, 720, C]
```

The model call is:

```text
pred = model(x, pred_len=supervision_pred_len)
```

The loss path then samples or constructs a supervision unit:

```text
unit = future mask / interval / component basis / curriculum phase
loss = supervision_loss(pred, y, unit)
```

This keeps the train-side tensor contract independent of
`target_horizons=[96,192,336,720]`.

## Supervision Strategies

`full_time_mse`:

- uses all 720 future steps;
- computes ordinary time-domain MSE.

`random_future_mask`:

- samples active future blocks with `supervision_mask_ratio`;
- computes MSE only on active time positions;
- records active step ratio in `supervision_trace.csv`.

`interval_supervision`:

- samples a contiguous future interval in block units;
- computes MSE only inside the sampled interval;
- records 1-based interval start/end.

`component_basis_top`:

- computes a train-split future-label covariance basis;
- projects `pred` and `y` onto the top `K` components;
- normalizes component residuals by target component energy to avoid a scale
  artifact;
- mixes time-domain loss and component loss using `supervision_component_alpha`.

`component_basis_balanced`:

- uses the same component basis;
- weights component residuals by inverse eigenvalue strength controlled by
  `supervision_component_beta`;
- clips and normalizes weights to avoid extreme noise amplification.

`curriculum_units`:

- uses a fixed three-stage schedule:
  - first 30% epochs: top component supervision;
  - middle 40% epochs: interval supervision;
  - final 30% epochs: dense full-time supervision.

## Logging

The implementation writes `supervision_trace.csv` with step-level supervision
metadata:

- strategy;
- train-side `supervision_pred_len`;
- unit type;
- active step count and ratio;
- interval range;
- component rank;
- curriculum phase;
- time loss, unit loss, and final loss.

`training_log.csv` adds epoch-level summaries:

- `train_supervision_strategy`;
- `train_supervision_pred_len`;
- `train_unit_loss`;
- `train_time_loss`;
- `train_active_step_ratio`;
- `train_component_rank`;
- `train_supervision_steps`;
- `train_curriculum_phase`.

`effective_config.json` records:

- `evaluation_target_horizons`;
- `train_horizons_effective`;
- `training_evaluation_decoupled`;
- `supervision_unit_config`;
- `component_basis_stats`.

## Remote Scripts

`scripts/remote/run_phase4_horizon_decoupled_gate.sh` runs the Phase4 candidate
matrix on `529_Lab-3090`.

`scripts/remote/check_phase4_horizon_decoupled_progress.sh` reports recent logs
and completed `metrics_by_target_horizon.csv` files.

`scripts/sync_phase4_horizon_decoupled_results.sh` syncs remote raw artifacts and
logs. It defaults to `SKIP_ANALYSIS=1` because the final analysis script should
be written after the remote artifact set exists.

## Consistency Evaluation

Intended theory:

- evaluation horizons are test points, not training units;
- future supervision units can be masks, intervals, components, or curriculum
  phases;
- better training strategy should improve full-horizon evaluation and provide a
  traceable optimization narrative.

How the code realizes it:

- only legacy baselines train by sampling `target_horizons`;
- horizon-decoupled candidates train on `supervision_pred_len=720`;
- validation/test still cover `96,192,336,720`;
- trace files expose what training units were actually used.

Remaining proxy:

- component basis is dataset-specific and computed from train labels only;
- random masks and intervals are simple first-pass supervision units;
- curriculum is fixed by epoch ratio rather than learned.

Falsification evidence:

- candidates fail against R.3 across full evaluation horizons;
- gains appear only in aggregate MSE without segment/component trace support;
- prefix consistency degrades;
- component strategies improve trend but systematically damage H96 local errors.
