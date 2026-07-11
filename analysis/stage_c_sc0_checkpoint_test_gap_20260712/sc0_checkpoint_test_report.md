# StageC SC0 Best-vs-Last Checkpoint Test Diagnostic

- `role`: diagnostic-only after carrier calibration; test metrics did not select hyperparameters.
- `source`: the original fixed-20 SC0 seed2021 best/last checkpoints; no retraining.
- `comparison`: `(last - best) / best`; negative means last is better.

| scope | comparisons | last MSE wins | mean val delta | mean test MSE delta | max test degradation | max test improvement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all_dense_horizons | 72 | 29 | 6.65% | 1.63% | 13.70% | -10.49% |
| full720_all_datasets_arms | 9 | 0 | 14.72% | 6.11% | 13.70% | 1.32% |
| full720_Weather | 3 | 0 | 1.33% | 2.19% | 3.49% | 1.32% |
| full720_ETTm1 | 3 | 0 | 3.06% | 6.80% | 9.06% | 3.20% |
| full720_ETTh2 | 3 | 0 | 39.75% | 9.35% | 13.70% | 4.69% |

This diagnostic distinguishes validation trajectory overfitting from actual test behavior. It cannot retroactively tune the carrier because test was opened only after the prior calibration decision.
