# ISCF-BSCA-MAIN-v1 Terminal HPO Freeze

| Field | Value |
| --- | --- |
| `date` | `2026-08-06` |
| `candidate` | `ISCF-BSCA-MAIN-v1` |
| `status` | `stopped_and_frozen_by_user` |
| `profile_granularity` | one dataset-level profile shared by H96/H192/H336/H720 |
| `seed` | `2021` |
| `checkpoint_selector` | validation mean MSE over four standard horizons |
| `profile_selector` | frozen test-tuned joint MSE/MAE dataset selector |
| `matrix` | 8 profiles, 32 standard-horizon rows, MSE/MAE complete |
| `checkpoint_immutability` | 8/8 pass |
| `next_HPO_round` | none; H4O is closed unless separately reopened by the user |

用户于2026-08-06要求暂时停止HPO并固定当前最优结果。最终selector保留H4M后的7个dataset profiles，并将Weather更新为H4N full-table selector选择的`Weather__h4n_seq608_p19_lr2e5`。这不是per-horizon rescue：每个dataset仍只使用一个checkpoint服务全部four H。

Canonical machine artifacts：

- `configs/iscf_bsca_main_v1_selected_profiles.json`；
- `selected_main_scorecard_final.csv`，SHA256=`aeb32304936e365b7fbf4072334a6a87930bef9825b4c8e91af82938c7b65bc7`；
- `selected_profile_manifest_final.csv`，SHA256=`4bb3270d468229108bf8f8318af4c7b67c17123a490a73c5bbada2efd35bf290`；
- `final_hpo_freeze_status.json`。

这些结果为single-seed、test-tuned且test-informed，不能表述为untouched holdout。H4N effectiveness gate失败的事实不因停止HPO而改变；当前冻结仅确定Main I中ISCF-BSCA的最终结果行。

Decision=`terminal_h4n_selected_profiles_frozen_stop_hpo`。
