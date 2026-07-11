# StageC SC0 Standardized Carrier Calibration Report

- `current_step`: StageC Step 3 control calibration, seed `2021`.
- `role`: validation-only protocol control; not a paper-core method.
- `profile_hash`: `79a037f751c0c24eea98ff0b516cb0dfeaef950871b3bbc515904754f54fd900`.
- `decision`: `common_token_mlp_profile_not_supported`.
- `test_metrics_used_for_selection`: `false`.

## Selected Global Profiles

| selector | arm | macro_regret | max_dataset_regret | per_dataset_gate_pass | mean_epoch_seconds |
| --- | --- | --- | --- | --- | --- |
| best_val | sc0_p24_d64 | 0.004051 | 0.012153 | 1 | 9.965855 |
| last | sc0_p48_d32 | 0.008108 | 0.024324 | 1 | 10.667202 |

## Full Selection Table

| selector | arm | macro_regret | max_dataset_regret | per_dataset_gate_pass | selected |
| --- | --- | --- | --- | --- | --- |
| best_val | sc0_p24_d64 | 0.004051 | 0.012153 | 1 | 1 |
| best_val | sc0_p12_d128 | 0.013159 | 0.024185 | 1 | 0 |
| best_val | sc0_p48_d32 | 0.024859 | 0.055922 | 0 | 0 |
| last | sc0_p48_d32 | 0.008108 | 0.024324 | 1 | 1 |
| last | sc0_p24_d64 | 0.021828 | 0.042910 | 0 | 0 |
| last | sc0_p12_d128 | 0.024573 | 0.051386 | 0 | 0 |

## Gate Interpretation

- Complete run/config/numeric gate: `True`.
- Last/best global winner stability: `False`.
- Best-validation per-dataset regret gate: `True`.
- A preliminary pass only authorizes seeds 2022/2023 for the selected arm.
- A failure rolls StageC back to Step 2/3; it does not authorize dataset-specific presets.

## Failure Attribution Boundary

SC0 only tests whether one standardized token-MLP carrier profile is viable. A failure does not reject
unified forecasting, projective decoding, or horizon-measure learning. It rejects this common carrier
family as a sufficiently stable research instrument under the preregistered gate.
