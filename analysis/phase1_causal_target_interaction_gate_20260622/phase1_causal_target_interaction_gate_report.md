# Phase1-R PatchEncoderCausalTargetInteraction Gate Report

## Decision

[Decision] `PatchEncoderCausalTargetInteraction` does not reach compatibility pass.

It should not be treated as paper-core or as a passed carrier for future-aware / MoE mechanisms without a rollback assessment.

## Main Metrics

- MSE wins vs horizon-specific `PatchEncoderFixedHead`: `4/12`.
- MAE wins vs horizon-specific `PatchEncoderFixedHead`: `5/12`.
- Mean relative MSE: `+1.40%`.
- Relative MSE range: `-2.80%` to `+6.48%`.

| Dataset | Mean relative MSE |
| --- | ---: |
| ETTh2 | -0.03% |
| ETTm1 | -0.45% |
| Weather | +4.68% |

| Horizon | Mean relative MSE |
| --- | ---: |
| 96 | +3.62% |
| 192 | -0.14% |
| 336 | +0.90% |
| 720 | +1.23% |

## Compatibility Gate

- mean relative MSE <= +1.0%: `False`
- no dataset average degradation > +3.0%: `False`
- h96/h192 not worse than fixed H720-prefix: `False`
- prefix mismatch near zero: `True`
- target states non-identical: `True`
- H720-prefix MSE wins vs fixed H720-prefix: `5/12`
- H720-prefix h96/h192 mean relative MSE: `-0.14%`
- max prefix mismatch MSE: `4.52311e-14`
- mean target state cosine: `0.424291`
- mean |gamma| / |beta|: `0.677252` / `0.410253`

## Interpretation

[Inference] The first target-set implementation proves the prefix-stable interface works mechanically, because short-horizon predictions match the corresponding H=720 prefixes up to numerical noise.

[Inference] On the same H=720-aligned windows used by the Phase0 prefix diagnostic, the model improves h96/h192 prefixes over the fixed H720-prefix reference on average, but strict no-degradation still depends on the per-setting rows below.

[Inference] The accuracy side is the decisive issue. If the mean relative MSE is positive, the current dense target-conditioned readout is not yet a paper-core decoder; it can only remain as a carrier if the amortization gap is within the compatibility threshold.

## Per-Setting Relative MSE

| Dataset | Horizon | Relative MSE | Target MSE | Fixed MSE |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | +1.75% | 0.312832 | 0.307448 |
| ETTh2 | 192 | -2.08% | 0.369476 | 0.377340 |
| ETTh2 | 336 | +0.18% | 0.384825 | 0.384115 |
| ETTh2 | 720 | +0.03% | 0.407524 | 0.407403 |
| ETTm1 | 96 | +2.62% | 0.298093 | 0.290475 |
| ETTm1 | 192 | -2.80% | 0.328260 | 0.337701 |
| ETTm1 | 336 | -0.95% | 0.358102 | 0.361540 |
| ETTm1 | 720 | -0.66% | 0.410073 | 0.412788 |
| Weather | 96 | +6.48% | 0.156613 | 0.147087 |
| Weather | 192 | +4.45% | 0.203889 | 0.195208 |
| Weather | 336 | +3.46% | 0.259474 | 0.250787 |
| Weather | 720 | +4.32% | 0.337090 | 0.323127 |

## H720-Aligned Prefix Reference

| Dataset | Prefix horizon | Relative MSE vs fixed H720-prefix | Target prefix MSE | Fixed H720-prefix MSE |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | +1.06% | 0.257043 | 0.254338 |
| ETTh2 | 192 | -0.27% | 0.296963 | 0.297754 |
| ETTh2 | 336 | +1.43% | 0.326995 | 0.322379 |
| ETTh2 | 720 | +0.03% | 0.407524 | 0.407403 |
| ETTm1 | 96 | -2.59% | 0.284545 | 0.292111 |
| ETTm1 | 192 | -2.32% | 0.317700 | 0.325232 |
| ETTm1 | 336 | -2.34% | 0.353333 | 0.361814 |
| ETTm1 | 720 | -0.66% | 0.410073 | 0.412788 |
| Weather | 96 | +0.88% | 0.155022 | 0.153676 |
| Weather | 192 | +2.39% | 0.200105 | 0.195442 |
| Weather | 336 | +3.03% | 0.254945 | 0.247440 |
| Weather | 720 | +4.32% | 0.337090 | 0.323127 |
