# Phase4 Residual-Stability Diagnostic

## 11-Step Record

| Field | Content |
| --- | --- |
| `current_step` | Step 5/6: redefine predictability proxy and dynamic router design evidence |
| `problem` | Fixed late adapter routing failed because it cannot tell learnable conflict from noisy conflict |
| `existence_evidence` | S1/S2/RG-A gates, gradient conflict diagnostic, train-only label residuals |
| `idea` | Use train-side baseline residual stability to decide whether a future unit should update adapter, shared path, or be shielded |
| `theory_check` | Seasonal/baseline residual gain plus smoothness can separate structured residuals from high-variation noisy residuals |
| `design` | Compare persistence with seasonal baselines over 48-step blocks; summarize selected high-novelty units by residual-stability buckets |
| `gate` | Weather late selected units should show a high noisy-conflict share; ETTh2 should retain more learnable-conflict units |
| `artifacts` | `analysis/phase4_residual_stability_diagnostic_20260625` |
| `decision` | Positive design evidence; advance to residual-stability dynamic routing rather than sweeping fixed late adapter |

## Selection Summary

| Dataset | Region | Selected share | Gain over persistence | Residual smoothness | Local variation |
| --- | --- | ---: | ---: | ---: | ---: |
| `ETTh2` | `early_1_96` | 6.0% | 2.346 | 0.453 | 0.103 |
| `ETTh2` | `middle_97_336` | 26.7% | 2.306 | 0.349 | 0.103 |
| `ETTh2` | `late_337_720` | 67.3% | 1.827 | 0.238 | 0.095 |
| `Weather` | `early_1_96` | 8.1% | 2.271 | 0.294 | 0.964 |
| `Weather` | `middle_97_336` | 26.1% | 2.411 | 0.274 | 0.783 |
| `Weather` | `late_337_720` | 65.8% | 1.957 | 0.199 | 0.526 |

## Bucket Summary

| Dataset | Region | Bucket | Share within region | Selected units | Gain | Smoothness | Variation | Baseline mode |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ETTh2` | `early_1_96` | `ambiguous_conflict` | 60.8% | 1115 | 2.246 | 0.367 | 0.092 | `seasonal_24` |
| `ETTh2` | `early_1_96` | `learnable_conflict` | 6.0% | 111 | 2.025 | 0.121 | 0.114 | `seasonal_48` |
| `ETTh2` | `early_1_96` | `noisy_conflict` | 33.2% | 609 | 2.587 | 0.672 | 0.122 | `seasonal_168` |
| `ETTh2` | `late_337_720` | `ambiguous_conflict` | 68.8% | 14040 | 1.527 | 0.180 | 0.094 | `persistence` |
| `ETTh2` | `late_337_720` | `learnable_conflict` | 9.5% | 1947 | 2.316 | 0.107 | 0.068 | `seasonal_168` |
| `ETTh2` | `late_337_720` | `noisy_conflict` | 21.7% | 4421 | 2.562 | 0.479 | 0.112 | `seasonal_168` |
| `ETTh2` | `middle_97_336` | `ambiguous_conflict` | 60.9% | 4932 | 1.953 | 0.271 | 0.090 | `persistence` |
| `ETTh2` | `middle_97_336` | `learnable_conflict` | 7.7% | 620 | 2.226 | 0.101 | 0.118 | `seasonal_96` |
| `ETTh2` | `middle_97_336` | `noisy_conflict` | 31.4% | 2545 | 3.011 | 0.561 | 0.125 | `seasonal_168` |
| `Weather` | `early_1_96` | `ambiguous_conflict` | 35.1% | 4083 | 2.984 | 0.187 | 0.082 | `seasonal_168` |
| `Weather` | `early_1_96` | `learnable_conflict` | 16.8% | 1956 | 2.373 | 0.079 | 0.038 | `seasonal_168` |
| `Weather` | `early_1_96` | `noisy_conflict` | 48.1% | 5593 | 1.716 | 0.448 | 1.931 | `seasonal_168` |
| `Weather` | `late_337_720` | `ambiguous_conflict` | 51.0% | 48070 | 1.817 | 0.110 | 0.068 | `persistence` |
| `Weather` | `late_337_720` | `learnable_conflict` | 16.2% | 15304 | 2.839 | 0.078 | 0.040 | `seasonal_48` |
| `Weather` | `late_337_720` | `noisy_conflict` | 32.8% | 30940 | 1.738 | 0.396 | 1.478 | `seasonal_48` |
| `Weather` | `middle_97_336` | `ambiguous_conflict` | 42.4% | 15846 | 2.605 | 0.156 | 0.072 | `persistence` |
| `Weather` | `middle_97_336` | `learnable_conflict` | 12.4% | 4619 | 2.941 | 0.087 | 0.039 | `seasonal_96` |
| `Weather` | `middle_97_336` | `noisy_conflict` | 45.3% | 16917 | 2.085 | 0.435 | 1.652 | `seasonal_96` |

## Interpretation

[Fact] Weather late selected units: noisy-conflict share `32.8%`, learnable-conflict share `16.2%`.
[Fact] ETTh2 late selected units: noisy-conflict share `21.7%`, learnable-conflict share `9.5%`.

[Decision Rule] If Weather late noisy-conflict share is high, a fixed late adapter is too broad; the next strategy should route only learnable-conflict units to adapter and suppress noisy-conflict units.

[Decision Rule] If ETTh2 retains a material learnable-conflict share, the router must not suppress all hard or late units; otherwise it will erase the positive S1/RG-A signal.

## Next Design Implication

[Decision] The next method should be `dynamic_residual_stability_routing`: dense base remains full 720; selected units are bucketed by residual stability; learnable-conflict units train adapter; noisy-conflict units do not update shared path and receive reduced or zero auxiliary pressure.
