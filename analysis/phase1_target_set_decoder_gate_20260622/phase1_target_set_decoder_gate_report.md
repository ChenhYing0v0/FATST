# Phase1-R PatchEncoderTargetSetDecoder Gate Report

## Decision

[Decision] `PatchEncoderTargetSetDecoder` does not reach compatibility pass.

It should not be treated as paper-core or as a passed carrier for future-aware / MoE mechanisms without a rollback assessment.

## Main Metrics

- MSE wins vs horizon-specific `PatchEncoderFixedHead`: `5/12`.
- MAE wins vs horizon-specific `PatchEncoderFixedHead`: `5/12`.
- Mean relative MSE: `+0.62%`.
- Relative MSE range: `-2.17%` to `+5.63%`.

| Dataset | Mean relative MSE |
| --- | ---: |
| ETTh2 | -0.27% |
| ETTm1 | +1.99% |
| Weather | +0.13% |

| Horizon | Mean relative MSE |
| --- | ---: |
| 96 | +2.74% |
| 192 | -0.95% |
| 336 | -0.41% |
| 720 | +1.09% |

## Compatibility Gate

- mean relative MSE <= +1.0%: `True`
- no dataset average degradation > +3.0%: `True`
- h96/h192 not worse than fixed H720-prefix: `False`
- prefix mismatch near zero: `True`
- target states non-identical: `True`
- H720-prefix MSE wins vs fixed H720-prefix: `6/12`
- H720-prefix h96/h192 mean relative MSE: `-0.85%`
- max prefix mismatch MSE: `5.11262e-14`
- mean target state cosine: `0.359800`
- mean |gamma| / |beta|: `0.653291` / `0.407875`

## Interpretation

[Inference] The first target-set implementation proves the prefix-stable interface works mechanically, because short-horizon predictions match the corresponding H=720 prefixes up to numerical noise.

[Inference] On the same H=720-aligned windows used by the Phase0 prefix diagnostic, the model improves h96/h192 prefixes over the fixed H720-prefix reference on average, but strict no-degradation still depends on the per-setting rows below.

[Inference] The accuracy side is the decisive issue. If the mean relative MSE is positive, the current dense target-conditioned readout is not yet a paper-core decoder; it can only remain as a carrier if the amortization gap is within the compatibility threshold.

## Per-Setting Relative MSE

| Dataset | Horizon | Relative MSE | Target MSE | Fixed MSE |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | +0.48% | 0.308910 | 0.307448 |
| ETTh2 | 192 | -2.17% | 0.369166 | 0.377340 |
| ETTh2 | 336 | -0.86% | 0.380801 | 0.384115 |
| ETTh2 | 720 | +1.47% | 0.413407 | 0.407403 |
| ETTm1 | 96 | +5.63% | 0.306820 | 0.290475 |
| ETTm1 | 192 | -0.33% | 0.336593 | 0.337701 |
| ETTm1 | 336 | +1.03% | 0.365282 | 0.361540 |
| ETTm1 | 720 | +1.62% | 0.419472 | 0.412788 |
| Weather | 96 | +2.12% | 0.150202 | 0.147087 |
| Weather | 192 | -0.35% | 0.194525 | 0.195208 |
| Weather | 336 | -1.41% | 0.247248 | 0.250787 |
| Weather | 720 | +0.16% | 0.323659 | 0.323127 |

## H720-Aligned Prefix Reference

| Dataset | Prefix horizon | Relative MSE vs fixed H720-prefix | Target prefix MSE | Fixed H720-prefix MSE |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | +0.19% | 0.254830 | 0.254338 |
| ETTh2 | 192 | -0.65% | 0.295809 | 0.297754 |
| ETTh2 | 336 | +1.12% | 0.325995 | 0.322379 |
| ETTh2 | 720 | +1.47% | 0.413407 | 0.407403 |
| ETTm1 | 96 | +0.17% | 0.292609 | 0.292111 |
| ETTm1 | 192 | -0.09% | 0.324952 | 0.325232 |
| ETTm1 | 336 | -0.41% | 0.360336 | 0.361814 |
| ETTm1 | 720 | +1.62% | 0.419472 | 0.412788 |
| Weather | 96 | -2.84% | 0.149316 | 0.153676 |
| Weather | 192 | -1.90% | 0.191726 | 0.195442 |
| Weather | 336 | -1.08% | 0.244765 | 0.247440 |
| Weather | 720 | +0.16% | 0.323659 | 0.323127 |
