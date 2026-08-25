# Appendix C visual-fidelity selection audit

## Selection objective

Appendix C is a qualitative validation-only illustration. Its selection rule
therefore targets curves that look faithful to the ground truth rather than
ranking samples by MSE alone. For every candidate and each supported prefix,
we compute four quantities in the train-split standardized scale:

1. level RMSE;
2. trajectory correlation loss, $(1-r)/2$;
3. first-difference correlation loss, which measures local movement agreement;
4. amplitude error, defined as the relative difference between prediction and
   target standard deviation.

The four-prefix visual-fidelity score is

$$
S_{\mathrm{vis}}
=0.70\,e_{\mathrm{level}}
 +0.15\,e_{\mathrm{corr}}
 +0.10\,e_{\Delta\mathrm{corr}}
 +0.05\,e_{\mathrm{amp}},
$$

where lower values are preferred. The relatively high level-error weight
prevents a visually smooth but substantially biased trajectory from outranking
an accurate one. The score is computed on validation labels only.

## Channel selection

Channel 0 is not uniformly representative of visual predictability, especially
for Weather and ECL. We therefore selected one fixed channel per dataset by
the lowest global validation visual-fidelity score, after excluding the lowest
20% of channels by validation target variance. This removes nearly flat
channels that would make the qualitative comparison visually uninformative.

| Dataset | Selected channel | Global visual score | Global RMSE | Global correlation |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | 2 | 0.6729 | 0.8594 | 0.773 |
| ETTh2 | 6 | 0.5129 | 0.6025 | 0.512 |
| ETTm1 | 0 | 0.6375 | 0.8114 | 0.768 |
| ETTm2 | 6 | 0.3698 | 0.4548 | 0.770 |
| Weather | 18 | 0.2501 | 0.3126 | 0.875 |
| ECL | 306 | 0.0964 | 0.1269 | 0.990 |
| Solar | 99 | 0.2506 | 0.3250 | 0.945 |

The selected channel is fixed before sample ranking within each dataset; it is
not changed separately for the two displayed samples.

## Sample selection and outcome

Within the selected channel, validation origins are ranked by
$S_{\mathrm{vis}}$, and the two lowest-scoring origins are retained subject to
a minimum raw-origin separation of 720 steps. The resulting pairs are:

| Dataset | Validation windows | Raw forecast origins |
| --- | --- | --- |
| ETTh1 | 640, 1361 | 9279, 10000 |
| ETTh2 | 1670, 519 | 10309, 9158 |
| ETTm1 | 10409, 2294 | 44968, 36853 |
| ETTm2 | 2076, 825 | 36635, 35384 |
| Weather | 673, 2649 | 37559, 39535 |
| ECL | 139, 908 | 18550, 19319 |
| Solar | 2309, 179 | 39100, 36970 |

The resulting Weather and ECL traces have materially smaller level and shape
deviations than the previous channel-0 selections. This remains an
illustrative validation diagnostic, not a prevalence estimate or a test-set
effectiveness claim.
