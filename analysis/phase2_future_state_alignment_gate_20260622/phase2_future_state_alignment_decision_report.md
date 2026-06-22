# Phase2-A Future-State Alignment Gate Report

## Decision

[Decision] `PatchEncoderFutureStateAlignment` fails the Phase2 gate.

It is leakage-safe and shows a real positive signal on `ETTm1`, but it breaks the R.3 compatibility carrier on `ETTh2`. Therefore it is not a paper-core candidate and should not be used as the state carrier for MoE without a rollback diagnosis.

## Main Metrics vs FixedHead

- MSE wins vs horizon-specific `PatchEncoderFixedHead`: `6/12`.
- MAE wins vs horizon-specific `PatchEncoderFixedHead`: `6/12`.
- Mean relative MSE vs FixedHead: `+0.84%`.
- ETTh2 mean relative MSE: `+4.54%`.
- ETTm1 mean relative MSE: `-0.96%`.
- Weather mean relative MSE: `-1.04%`.

## Direct Comparison vs R.3 Carrier

- MSE wins vs `PatchEncoderPrefixRiskWeighted`: `7/12`.
- MAE wins vs `PatchEncoderPrefixRiskWeighted`: `7/12`.
- Mean relative MSE vs R.3: `+1.29%`.
- ETTh2 mean relative MSE vs R.3: `+5.25%`.
- ETTm1 mean relative MSE vs R.3: `-1.29%`.
- Weather mean relative MSE vs R.3: `-0.07%`.

| Dataset | Horizon | Relative MSE vs R.3 | Phase2 MSE | R.3 MSE |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | +7.64% | 0.328073 | 0.304796 |
| ETTh2 | 192 | +6.97% | 0.394779 | 0.369043 |
| ETTh2 | 336 | +2.73% | 0.393362 | 0.382910 |
| ETTh2 | 720 | +3.64% | 0.425418 | 0.410473 |
| ETTm1 | 96 | -1.21% | 0.295074 | 0.298685 |
| ETTm1 | 192 | -1.10% | 0.326025 | 0.329662 |
| ETTm1 | 336 | -1.37% | 0.355784 | 0.360729 |
| ETTm1 | 720 | -1.47% | 0.411167 | 0.417293 |
| Weather | 96 | +0.41% | 0.148634 | 0.148026 |
| Weather | 192 | -0.18% | 0.192054 | 0.192409 |
| Weather | 336 | -0.24% | 0.244214 | 0.244793 |
| Weather | 720 | -0.29% | 0.319923 | 0.320847 |

## Future Alignment And Leakage

- max prediction leakage abs: `0.00000000`.
- mean teacher/student cosine: `0.762778`.
- mean local alignment loss: `0.237222`.
- mean relation alignment loss: `0.074249`.

## Gate Assessment

Compatibility-preserving pass:

- mean relative MSE vs FixedHead remains <= R.3 `-0.43%`: `False` (`+0.84%`).
- no dataset average worse than R.3 by more than `+0.3%`: `False` (`ETTh2` is `+5.25%` vs R.3).
- prefix mismatch remains numerical zero-level: `True` (`4.727e-14`).
- prediction leakage max abs <= `1e-7`: `True`.

Paper-core candidate pass:

- wins vs R.3 >= `7/12`: `True` (`7/12`).
- mean relative MSE vs R.3 <= `-0.5%`: `False` (`+1.29%`).
- at least two weak settings improve: `True` (`2/4`).
- improvement is stable enough to tell a paper story: `False`, because ETTh2 degrades in all horizons.

## Interpretation

[Inference] The mechanism is not invalid in principle: it improves all `ETTm1` horizons vs R.3 and improves three of four `Weather` horizons. This suggests future-state alignment can inject useful information into the target-side state.

[Inference] The failure is dataset conflict, not leakage or prefix instability. `ETTh2` degrades by `+5.24%` vs R.3 on average and by `+7.64%` on h96, while prefix mismatch remains numerical-zero. The alignment objective is therefore steering the target state in a direction that helps some datasets but conflicts with ETTh2 dynamics.

[Decision] Do not proceed to MoE on top of this state. The rollback point is step 3-5: diagnose why future-state alignment conflicts on ETTh2. Candidate repairs should be conflict-aware, such as uncertainty-weighted alignment, stop-gradient schedule, or dataset/horizon-gated alignment, before any expert routing.
