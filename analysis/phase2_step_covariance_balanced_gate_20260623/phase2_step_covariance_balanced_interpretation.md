# Phase2-C.2 Step-Covariance Balanced Interpretation

## Decision

[Decision] `PatchEncoderStepCovarianceBalanced` fails the Phase2-C.2 gate.

[Fact] The candidate is better than uniform target-set training, but it does not
beat R.3 `PatchEncoderPrefixRiskWeighted`, which is the required primary
baseline.

## Main Result

| Comparison | Result |
| --- | ---: |
| MSE wins vs R.3 | `2/12` |
| MAE wins vs R.3 | `0/12` |
| Mean relative MSE vs R.3 | `+0.76%` |
| MSE wins vs uniform target-set | `12/12` |
| Mean relative MSE vs uniform target-set | `-0.28%` |
| MSE wins vs FixedHead | `6/12` |
| Mean relative MSE vs FixedHead | `+0.33%` |
| Max prefix mismatch MSE | `5.182444710459706e-14` |

Dataset mean relative MSE vs R.3:

| Dataset | Mean relative MSE vs R.3 |
| --- | ---: |
| `ETTh2` | `-0.09%` |
| `ETTm1` | `+1.35%` |
| `Weather` | `+1.03%` |

## What It Proves

[Strong Evidence] The QDF-style premise remains relevant: future-step objective
weights matter. A static novelty-aware diagonal objective improves over uniform
target-set in all `12/12` settings.

[Strong Evidence] However, the candidate is not strong enough as a paper-core
mechanism. R.3 remains clearly better:

- R.3 vs uniform mean MSE was `-1.03%`;
- this candidate vs uniform is only `-0.28%`;
- this candidate vs R.3 is `+0.76%`.

## Failure Pattern

The candidate under-preserves R.3's early-prefix advantage. Its weighted
pressure share for `1-96` is:

| Dataset | `1-96` weighted pressure share |
| --- | ---: |
| `ETTh2` | `0.4807` |
| `ETTm1` | `0.5501` |
| `Weather` | `0.4813` |

R.3's prefix-risk pressure share for `1-96` is `0.7217`. The new objective is
milder and more novelty-balanced, but this mildness loses the short-horizon gains
that made R.3 strong.

Specialist-gap settings all fail vs R.3:

| Dataset | Horizon | Relative MSE vs R.3 |
| --- | ---: | ---: |
| `ETTm1` | `96` | `+2.38%` |
| `ETTm1` | `720` | `+0.34%` |
| `ETTh2` | `720` | `+0.62%` |
| `Weather` | `96` | `+1.34%` |

H720 region diagnostics show partial local repairs but no stable story:

- `ETTh2 / 193-336`: `-1.25%` vs R.3;
- `ETTm1 / 337-720`: `-0.16%` vs R.3;
- `ETTh2 / 337-720`: `+1.24%` vs R.3;
- `Weather / 337-720`: `+0.88%` vs R.3.

## Relation To QDF

[Inference] QDF's broad claim is supported: identity MSE is not enough, and
heterogeneous future-step weights can help. But the current FATST simplification
only uses static diagonal region weights. It does not model QDF's off-diagonal
label autocorrelation effect and does not learn the weighting matrix with a
meta/bilevel procedure.

[Decision] Do not keep tuning `beta` / `eta` as a broad sweep. That would turn
the stage into manual loss search and would not fix the core issue: the
candidate cannot beat the already simple R.3 objective.

## Rollback

[Decision] The objective-only path has now failed in two forms:

1. `region_balanced`: coverage-only equal-region objective;
2. `step_covariance_balanced`: static novelty-aware diagonal objective.

Current rollback point: return to the 11-step loop step 2-3.

The next valid research question should be one of:

1. whether a true QDF-style off-diagonal / learned quadratic objective is worth
   reproducing as an external baseline or diagnostic, rather than folding it
   directly into FATST;
2. whether the project should pivot from objective-only modifications back to
   base architecture / external baseline selection.

Do not proceed to MoE on this objective carrier.
