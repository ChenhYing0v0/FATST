# Solar H3B Test Result and ISCF-BSCA-MAIN-v1 HPO Freeze

## Decision summary

| Field | Value |
| --- | --- |
| `current_step` | H3B Step 9--10 complete；eight-dataset profile freeze |
| `H3B_test_matrix` | 4/4 checkpoints；16/16 cells；errors=0 |
| `Solar_selected` | `Solar__h3b_lr3e4_rank64` |
| `Solar_mean_MSE_MAE` | 0.191855 / 0.220518 |
| `Solar_target` | TimeAlign published mean MSE 0.192 |
| `decision` | target pass；stop HPO；freeze eight dataset profiles |

[Fact] H3B official test在commit `075c9bb`上完成。4/4 checkpoints、16/16 standard-horizon cells、dense metrics、candidate/trial/profile/seed provenance、invariants、NPZ和test前后checkpoint SHA256全部通过，errors为空。

## Solar terminal ranking

| Rank | Trial | Mean MSE | Mean MAE |
| ---: | --- | ---: | ---: |
| 1 | `lr3e4_rank64` | **0.191855** | 0.220518 |
| 2 | `lr3e4_dropout4` | 0.192723 | 0.217923 |
| 3 | `lr2e4` | 0.193616 | **0.217124** |
| 4 | `lr4e4` | 0.196418 | 0.221318 |

Selected profile的H96/H192/H336/H720 MSE为0.167700/0.189032/0.201780/0.208910。它相对H3A winner 0.193341改善0.768%，相对H1/H2 winner 0.196157改善2.193%，相对0.192 target低0.075%。因此aggregate target通过，但margin很小，不能宣称在每个horizon都优于TimeAlign；H720仍为弱项。

MSE selector与MAE排序不同：`lr2e4` MAE最低，但项目预冻结primary selector是four-H mean MSE，因此不得按MAE另选profile。Solar最终冻结`lr3e4_rank64`服务全部四个horizons。

## Eight-dataset freeze

最终single-seed main profiles冻结于`configs/iscf_bsca_main_v1_selected_profiles.json`。ECL使用`ECL__h3a_budget45`，Solar使用`Solar__h3b_lr3e4_rank64`；其余六datasets沿用H1/H2完整ranking winner。所有profiles均为test-tuned，ECL/Solar还是test-informed，不得描述为untouched holdout。

## Gates and next step

`paper_facing_effectiveness=performance_partial_pass_pending_complete_baseline_tables`。ECL与Solar当前target均通过，八dataset profile HPO完成，但SOTA paper claim还需Main I完整baselines及Main II matched unified controls。

`matched_mechanism_attribution=pending`；当前结果不能证明BSCA mechanism贡献。`internal_mechanism_health`沿用后续selected checkpoints diagnostics。`failure_attribution`无numeric pathology；Solar剩余H720 weakness不再通过per-horizon HPO修补。

HPO在当前same-architecture neighborhood停止。下一实验游标转向Main I baseline completion与Main II matched unified benchmark，而不是继续新增H4 hyperparameter cells。Additional seeds仍按用户先前决定保持optional。

Decision=`ISCF_BSCA_MAIN_v1_single_seed_HPO_complete_ECL_Solar_targets_pass_profiles_frozen`。
