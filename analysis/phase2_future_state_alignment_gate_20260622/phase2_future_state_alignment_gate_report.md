# Phase1-R PatchEncoderFutureStateAlignment Gate Report

## Decision

[Decision] `PatchEncoderFutureStateAlignment` does not reach compatibility pass.

It should not be treated as paper-core or as a passed carrier for future-aware / MoE mechanisms without a rollback assessment.

## Main Metrics

- MSE wins vs horizon-specific `PatchEncoderFixedHead`: `6/12`.
- MAE wins vs horizon-specific `PatchEncoderFixedHead`: `6/12`.
- Mean relative MSE: `+0.84%`.
- Relative MSE range: `-3.46%` to `+6.71%`.

| Dataset | Mean relative MSE |
| --- | ---: |
| ETTh2 | +4.54% |
| ETTm1 | -0.96% |
| Weather | -1.04% |

| Horizon | Mean relative MSE |
| --- | ---: |
| 96 | +3.11% |
| 192 | -0.15% |
| 336 | -0.60% |
| 720 | +1.01% |

## Compatibility Gate

- mean relative MSE <= +1.0%: `True`
- no dataset average degradation > +3.0%: `False`
- h96/h192 not worse than fixed H720-prefix: `False`
- prefix mismatch near zero: `True`
- target states non-identical: `True`
- H720-prefix MSE wins vs fixed H720-prefix: `8/12`
- H720-prefix h96/h192 mean relative MSE: `-0.61%`
- max prefix mismatch MSE: `4.72701e-14`
- mean target state cosine: `0.363533`
- mean |gamma| / |beta|: `0.687714` / `0.415917`

## Interpretation

[Inference] The first target-set implementation proves the prefix-stable interface works mechanically, because short-horizon predictions match the corresponding H=720 prefixes up to numerical noise.

[Inference] On the same H=720-aligned windows used by the Phase0 prefix diagnostic, the model improves h96/h192 prefixes over the fixed H720-prefix reference on average, but strict no-degradation still depends on the per-setting rows below.

[Inference] The accuracy side is the decisive issue. If the mean relative MSE is positive, the current dense target-conditioned readout is not yet a paper-core decoder; it can only remain as a carrier if the amortization gap is within the compatibility threshold.

## Per-Setting Relative MSE

| Dataset | Horizon | Relative MSE | Target MSE | Fixed MSE |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | +6.71% | 0.328073 | 0.307448 |
| ETTh2 | 192 | +4.62% | 0.394779 | 0.377340 |
| ETTh2 | 336 | +2.41% | 0.393362 | 0.384115 |
| ETTh2 | 720 | +4.42% | 0.425418 | 0.407403 |
| ETTm1 | 96 | +1.58% | 0.295074 | 0.290475 |
| ETTm1 | 192 | -3.46% | 0.326025 | 0.337701 |
| ETTm1 | 336 | -1.59% | 0.355784 | 0.361540 |
| ETTm1 | 720 | -0.39% | 0.411167 | 0.412788 |
| Weather | 96 | +1.05% | 0.148634 | 0.147087 |
| Weather | 192 | -1.62% | 0.192054 | 0.195208 |
| Weather | 336 | -2.62% | 0.244214 | 0.250787 |
| Weather | 720 | -0.99% | 0.319923 | 0.323127 |

## H720-Aligned Prefix Reference

| Dataset | Prefix horizon | Relative MSE vs fixed H720-prefix | Target prefix MSE | Fixed H720-prefix MSE |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | +4.50% | 0.265781 | 0.254338 |
| ETTh2 | 192 | +4.48% | 0.311090 | 0.297754 |
| ETTh2 | 336 | +5.53% | 0.340213 | 0.322379 |
| ETTh2 | 720 | +4.42% | 0.425418 | 0.407403 |
| ETTm1 | 96 | -3.42% | 0.282128 | 0.292111 |
| ETTm1 | 192 | -2.84% | 0.315996 | 0.325232 |
| ETTm1 | 336 | -2.76% | 0.351818 | 0.361814 |
| ETTm1 | 720 | -0.39% | 0.411167 | 0.412788 |
| Weather | 96 | -3.54% | 0.148233 | 0.153676 |
| Weather | 192 | -2.85% | 0.189879 | 0.195442 |
| Weather | 336 | -2.13% | 0.242171 | 0.247440 |
| Weather | 720 | -0.99% | 0.319923 | 0.323127 |
