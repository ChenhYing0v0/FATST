# SC-D21-EVS Result

## Protocol

- validation only fits the past-only risk probes; official test only evaluates transfer;
- requested horizon and future truth are absent from policy features;
- the result is a problem gate, not a forecasting method result.

## Macro comparisons

| Carrier | Readout | Comparison | Macro gain | Positive datasets |
| --- | --- | --- | ---: | ---: |
| neutral_raw | ridge | evs_interaction_vs_region_fixed | +0.1854% | 2/5 |
| neutral_raw | ridge | evs_interaction_vs_history_global | -0.0295% | 2/5 |
| neutral_raw | ridge | evs_interaction_vs_additive_history_region | -0.1242% | 2/5 |
| neutral_raw | ridge | evs_interaction_vs_permuted_history | +0.2762% | 4/5 |
| neutral_raw | ridge | oracle_vs_region_fixed | +7.6399% | 5/5 |
| a6_natural | ridge | evs_interaction_vs_region_fixed | +0.6576% | 3/5 |
| a6_natural | ridge | evs_interaction_vs_history_global | +0.3468% | 3/5 |
| a6_natural | ridge | evs_interaction_vs_additive_history_region | +0.2209% | 4/5 |
| a6_natural | ridge | evs_interaction_vs_permuted_history | +0.6076% | 5/5 |
| a6_natural | ridge | oracle_vs_region_fixed | +10.4053% | 5/5 |
| neutral_raw | hist_gradient_boosting | evs_interaction_vs_region_fixed | +0.6325% | 3/5 |
| neutral_raw | hist_gradient_boosting | evs_interaction_vs_history_global | +0.0755% | 3/5 |
| neutral_raw | hist_gradient_boosting | evs_interaction_vs_additive_history_region | +0.0347% | 4/5 |
| neutral_raw | hist_gradient_boosting | evs_interaction_vs_permuted_history | +0.6206% | 4/5 |
| neutral_raw | hist_gradient_boosting | oracle_vs_region_fixed | +7.6399% | 5/5 |
| a6_natural | hist_gradient_boosting | evs_interaction_vs_region_fixed | +1.1546% | 3/5 |
| a6_natural | hist_gradient_boosting | evs_interaction_vs_history_global | +0.1788% | 3/5 |
| a6_natural | hist_gradient_boosting | evs_interaction_vs_additive_history_region | -0.0069% | 2/5 |
| a6_natural | hist_gradient_boosting | evs_interaction_vs_permuted_history | +0.9895% | 5/5 |
| a6_natural | hist_gradient_boosting | oracle_vs_region_fixed | +10.4053% | 5/5 |

## Decision

`exact_descriptor_readout_probe_failed_direction_unresolved`.

A positive result authorizes only Step4 source-informed method design. A negative result rejects only this fixed descriptor/readout probe; it does not by itself reject every representation-level EVS design.
