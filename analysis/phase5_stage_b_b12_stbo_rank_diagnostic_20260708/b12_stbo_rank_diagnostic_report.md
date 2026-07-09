# Phase5 StageB B12-STBO Rank Diagnostic Report

## Scope

Valid configs: `L48-R32`, `L120-R64`, `L144-R128`, `L360-R256_capacity_probe`.
Invalid config `L96-R64` is excluded because `96` does not divide `720`.
A6 is not rerun; comparisons use the validated clean A6 anchor.

## Overall STBO vs A6

| config | arm | vs_a6_mse | vs_a6_wins |
| --- | --- | --- | --- |
| l48_r32 | stbo_shared | 0.474283 | 0/12 |
| l48_r32 | stbo_bank4 | 1.080749 | 1/12 |
| l48_r32 | stbo_dct | 0.892317 | 1/12 |
| l48_r32 | stbo_independent | 0.537772 | 1/12 |
| l120_r64 | stbo_shared | 0.424981 | 0/12 |
| l120_r64 | stbo_bank4 | 1.374876 | 0/12 |
| l120_r64 | stbo_dct | 1.142139 | 0/12 |
| l120_r64 | stbo_independent | 0.644363 | 0/12 |
| l144_r128 | stbo_shared | 0.841665 | 1/12 |
| l144_r128 | stbo_bank4 | 1.368317 | 2/12 |
| l144_r128 | stbo_dct | 0.968197 | 1/12 |
| l144_r128 | stbo_independent | 0.506654 | 1/12 |
| l360_r256_capacity_probe | stbo_shared | 0.327868 | 4/12 |
| l360_r256_capacity_probe | stbo_bank4 | 1.013015 | 3/12 |
| l360_r256_capacity_probe | stbo_dct | 1.313370 | 1/12 |
| l360_r256_capacity_probe | stbo_independent | 0.014466 | 4/12 |

## Learned STBO Mechanism Controls

| config | arm | vs_a6_mse | vs_a6_wins | vs_dct_mse | vs_dct_wins | vs_ind_mse | vs_ind_wins | bank_entropy_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| l48_r32 | stbo_shared | 0.474283 | 0/12 | -0.410110 | 10/12 | -0.063014 | 7/12 |  |
| l48_r32 | stbo_bank4 | 1.080749 | 1/12 | 0.184765 | 5/12 | 0.537548 | 3/12 | 0.999702 |
| l120_r64 | stbo_shared | 0.424981 | 0/12 | -0.697452 | 10/12 | -0.216991 | 9/12 |  |
| l120_r64 | stbo_bank4 | 1.374876 | 0/12 | 0.229073 | 6/12 | 0.722294 | 3/12 | 0.999908 |
| l144_r128 | stbo_shared | 0.841665 | 1/12 | -0.123221 | 6/12 | 0.331625 | 3/12 |  |
| l144_r128 | stbo_bank4 | 1.368317 | 2/12 | 0.393615 | 6/12 | 0.852432 | 3/12 | 0.999978 |
| l360_r256_capacity_probe | stbo_shared | 0.327868 | 4/12 | -0.961061 | 12/12 | 0.314127 | 7/12 |  |
| l360_r256_capacity_probe | stbo_bank4 | 1.013015 | 3/12 | -0.292865 | 10/12 | 0.999456 | 3/12 | 0.999988 |

## Gate Reading

[Decision] `rank_capacity_repair_insufficient`: increasing rank/tile length does not produce a learned shared/bank STBO candidate that matches or beats A6 overall.

- Best learned STBO vs A6: `l360_r256_capacity_probe:stbo_shared` with mean MSE +0.33%, wins 4/12.
- Learned-vs-DCT block: `False`.
- Bank specialization inactive: `True`.

## Failure Attribution

- `hypothesis_false`: not fully proven for all native multi-horizon architectures.
- `readout_or_head_design_wrong`: supported; the tested tiled readout does not preserve A6 performance even under higher local rank.
- `generic_basis_control_explains`: supported if learned shared/bank fail to beat same-rank DCT.
- `capacity_control_explains`: not sufficient if the 256-rank capacity probe still fails A6 or DCT.
- `direction_level_rejection`: no; reject the tested STBO family, not all future architecture search.

