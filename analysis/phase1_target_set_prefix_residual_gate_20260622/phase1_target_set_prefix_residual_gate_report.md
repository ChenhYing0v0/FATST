# Phase1-R PatchEncoderTargetSetPrefixResidual Gate Report

## Decision

[Decision] `PatchEncoderTargetSetPrefixResidual` does not reach compatibility pass.

It should not be treated as paper-core or as a passed carrier for future-aware / MoE mechanisms without a rollback assessment.

## Main Metrics

- MSE wins vs horizon-specific `PatchEncoderFixedHead`: `1/12`.
- MAE wins vs horizon-specific `PatchEncoderFixedHead`: `2/12`.
- Mean relative MSE: `+2.03%`.
- Relative MSE range: `-0.74%` to `+5.23%`.

| Dataset | Mean relative MSE |
| --- | ---: |
| ETTh2 | +2.21% |
| ETTm1 | +1.24% |
| Weather | +2.63% |

| Horizon | Mean relative MSE |
| --- | ---: |
| 96 | +4.52% |
| 192 | +1.87% |
| 336 | +0.82% |
| 720 | +0.90% |

## Compatibility Gate

- mean relative MSE <= +1.0%: `False`
- no dataset average degradation > +3.0%: `True`
- h96/h192 not worse than fixed H720-prefix: `False`
- prefix mismatch near zero: `True`
- target states non-identical: `True`
- H720-prefix MSE wins vs fixed H720-prefix: `4/12`
- H720-prefix h96/h192 mean relative MSE: `+1.20%`
- max prefix mismatch MSE: `1.54947e-14`
- mean target state cosine: `0.836250`
- mean |gamma| / |beta|: `0.568867` / `0.414790`

## Interpretation

[Inference] The first target-set implementation proves the prefix-stable interface works mechanically, because short-horizon predictions match the corresponding H=720 prefixes up to numerical noise.

[Inference] On the same H=720-aligned windows used by the Phase0 prefix diagnostic, the model is worse than the fixed H720-prefix reference on h96/h192 on average. This fails the prefix-reuse part of the compatibility gate.

[Inference] The accuracy side is the decisive issue. If the mean relative MSE is positive, the current dense target-conditioned readout is not yet a paper-core decoder; it can only remain as a carrier if the amortization gap is within the compatibility threshold.

## Per-Setting Relative MSE

| Dataset | Horizon | Relative MSE | Target MSE | Fixed MSE |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | +3.81% | 0.319171 | 0.307448 |
| ETTh2 | 192 | +2.80% | 0.387905 | 0.377340 |
| ETTh2 | 336 | +0.58% | 0.386358 | 0.384115 |
| ETTh2 | 720 | +1.64% | 0.414065 | 0.407403 |
| ETTm1 | 96 | +4.52% | 0.303602 | 0.290475 |
| ETTm1 | 192 | +0.29% | 0.338685 | 0.337701 |
| ETTm1 | 336 | +0.91% | 0.364819 | 0.361540 |
| ETTm1 | 720 | -0.74% | 0.409727 | 0.412788 |
| Weather | 96 | +5.23% | 0.154777 | 0.147087 |
| Weather | 192 | +2.52% | 0.200123 | 0.195208 |
| Weather | 336 | +0.96% | 0.253197 | 0.250787 |
| Weather | 720 | +1.81% | 0.328991 | 0.323127 |

## H720-Aligned Prefix Reference

| Dataset | Prefix horizon | Relative MSE vs fixed H720-prefix | Target prefix MSE | Fixed H720-prefix MSE |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | +2.71% | 0.261224 | 0.254338 |
| ETTh2 | 192 | +3.54% | 0.308297 | 0.297754 |
| ETTh2 | 336 | +3.17% | 0.332613 | 0.322379 |
| ETTh2 | 720 | +1.64% | 0.414065 | 0.407403 |
| ETTm1 | 96 | -0.23% | 0.291431 | 0.292111 |
| ETTm1 | 192 | +1.25% | 0.329311 | 0.325232 |
| ETTm1 | 336 | -0.26% | 0.360861 | 0.361814 |
| ETTm1 | 720 | -0.74% | 0.409727 | 0.412788 |
| Weather | 96 | -0.68% | 0.152630 | 0.153676 |
| Weather | 192 | +0.61% | 0.196643 | 0.195442 |
| Weather | 336 | +1.27% | 0.250590 | 0.247440 |
| Weather | 720 | +1.81% | 0.328991 | 0.323127 |
