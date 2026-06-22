# Phase1-R PatchEncoderPrefixRiskWeighted Gate Report

## Decision

[Decision] `PatchEncoderPrefixRiskWeighted` reaches compatibility pass.

It is not paper-core by itself unless follow-up mechanisms convert its target-side state into stable forecast gains, but it can remain a candidate carrier.

## Main Metrics

- MSE wins vs horizon-specific `PatchEncoderFixedHead`: `8/12`.
- MAE wins vs horizon-specific `PatchEncoderFixedHead`: `8/12`.
- Mean relative MSE: `-0.43%`.
- Relative MSE range: `-2.39%` to `+2.83%`.

| Dataset | Mean relative MSE |
| --- | ---: |
| ETTh2 | -0.66% |
| ETTm1 | +0.33% |
| Weather | -0.97% |

| Horizon | Mean relative MSE |
| --- | ---: |
| 96 | +0.87% |
| 192 | -2.00% |
| 336 | -0.98% |
| 720 | +0.38% |

## Compatibility Gate

- mean relative MSE <= +1.0%: `True`
- no dataset average degradation > +3.0%: `True`
- h96/h192 not worse than fixed H720-prefix: `True`
- prefix mismatch near zero: `True`
- target states non-identical: `True`
- H720-prefix MSE wins vs fixed H720-prefix: `9/12`
- H720-prefix h96/h192 mean relative MSE: `-2.46%`
- max prefix mismatch MSE: `5.3671e-14`
- mean target state cosine: `0.313276`
- mean |gamma| / |beta|: `0.672743` / `0.407658`

## Interpretation

[Inference] The first target-set implementation proves the prefix-stable interface works mechanically, because short-horizon predictions match the corresponding H=720 prefixes up to numerical noise.

[Inference] On the same H=720-aligned windows used by the Phase0 prefix diagnostic, the model improves h96/h192 prefixes over the fixed H720-prefix reference on average, but strict no-degradation still depends on the per-setting rows below.

[Inference] The accuracy side is the decisive issue. If the mean relative MSE is positive, the current dense target-conditioned readout is not yet a paper-core decoder; it can only remain as a carrier if the amortization gap is within the compatibility threshold.

## Per-Setting Relative MSE

| Dataset | Horizon | Relative MSE | Target MSE | Fixed MSE |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | -0.86% | 0.304796 | 0.307448 |
| ETTh2 | 192 | -2.20% | 0.369043 | 0.377340 |
| ETTh2 | 336 | -0.31% | 0.382910 | 0.384115 |
| ETTh2 | 720 | +0.75% | 0.410473 | 0.407403 |
| ETTm1 | 96 | +2.83% | 0.298685 | 0.290475 |
| ETTm1 | 192 | -2.38% | 0.329662 | 0.337701 |
| ETTm1 | 336 | -0.22% | 0.360729 | 0.361540 |
| ETTm1 | 720 | +1.09% | 0.417293 | 0.412788 |
| Weather | 96 | +0.64% | 0.148026 | 0.147087 |
| Weather | 192 | -1.43% | 0.192409 | 0.195208 |
| Weather | 336 | -2.39% | 0.244793 | 0.250787 |
| Weather | 720 | -0.71% | 0.320847 | 0.323127 |

## H720-Aligned Prefix Reference

| Dataset | Prefix horizon | Relative MSE vs fixed H720-prefix | Target prefix MSE | Fixed H720-prefix MSE |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | 96 | -1.80% | 0.249752 | 0.254338 |
| ETTh2 | 192 | -1.28% | 0.293936 | 0.297754 |
| ETTh2 | 336 | +1.25% | 0.326393 | 0.322379 |
| ETTh2 | 720 | +0.75% | 0.410473 | 0.407403 |
| ETTm1 | 96 | -2.72% | 0.284174 | 0.292111 |
| ETTm1 | 192 | -2.19% | 0.318111 | 0.325232 |
| ETTm1 | 336 | -1.71% | 0.355628 | 0.361814 |
| ETTm1 | 720 | +1.09% | 0.417293 | 0.412788 |
| Weather | 96 | -4.04% | 0.147463 | 0.153676 |
| Weather | 192 | -2.75% | 0.190075 | 0.195442 |
| Weather | 336 | -1.88% | 0.242797 | 0.247440 |
| Weather | 720 | -0.71% | 0.320847 | 0.323127 |
