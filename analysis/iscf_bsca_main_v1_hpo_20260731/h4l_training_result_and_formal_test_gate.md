# ISCF-BSCA-MAIN-v1 H4L Training Result and Formal-Test Gate

## 1. Decision summary

| Field | Value |
| --- | --- |
| `date` | 2026-08-04 |
| `current_step` | Step 9 train/validation artifact audit complete；formal test waiting explicit authorization |
| `training_matrix` | 48/48 complete；ETTm2 24 + Weather 24；test=0/48 |
| `numeric_health` | 48/48 pass |
| `provenance` | 48/48 effective configs match frozen config/search-space hashes and job fields |
| `checkpoint_hashes` | 48 present and unique SHA256 values frozen |
| `manifest` | `h4l_checkpoint_manifest.csv`；48 rows；SHA256 `c7ce6b6915dbe0323282140c0ed28ecad590b5ea256e8545a7f0fb3217c25584` |
| `official_test` | not authorized；0 target artifacts；0/48 |
| `decision` | `H4L_training_complete_manifest_frozen_formal_test_authorization_requested` |

## 2. Training and artifact audit

H4L于2026-08-04完成全部48个seed2021 jobs。Remote status为`complete=48/48, test=0/48`；结束后GPU0/1/2均为18 MiB、0% utilization，quota约198/220 GiB。远端仍保留三处与本实验无关的历史dirty CSV，本轮未修改或提交。

Generic analyzer与H4L-specific provenance checker共同确认：

- checkpoint、training log、four-H validation metrics、effective config、initialization contract、model diagnostics和environment artifacts均48/48存在；
- 48/48状态为`validation_complete`，official-test artifacts与`test_audit`目录均不存在；
- config SHA256=`3efb06bb9efeed0c51a44ddf74c007312eb6fb6260d64b6e65105e2b6ee84d81`；search-space SHA256=`90de181a070bdeba2e25e709a524a64c7c21fa9b9ccf7c71502da26778c1471d`；
- seed、trial/profile ID、four-H validation selector、dataset fields、training budget、all HPO fields、`official_test_mode=false`与`final_evaluation_split=val`均匹配冻结contract；
- 48个checkpoint SHA256互不重复，并与synced ledger逐项一致；
- train logs未检出Traceback、RuntimeError、OOM、NaN或Inf；best epoch范围为1--47；
- training-log epoch aggregate与best-checkpoint重新评估值的最大差异为`7.72e-8`，低于预注册audit tolerance `1e-7`，不改变best epoch或checkpoint identity。

## 3. Validation-only context

Validation结果只用于训练健康与是否值得完整test的解释，不用于H4L profile selection，也不允许筛掉其余checkpoints。

| Dataset | H4L validation-best | Four-H val MSE | Versus H4K selected profile | Versus H1--H4K validation frontier |
| --- | --- | ---: | ---: | ---: |
| ETTm2 | `ETTm2__h4l_patch1` | 0.178659 | +2.740% improvement | +1.622% improvement |
| Weather | `Weather__h4l_patch4` | 0.485544 | +0.486% improvement | -0.625% degradation |

ETTm2出现明确的validation frontier刷新，说明wide patch search产生了原邻域未覆盖的有效点；Weather只相对当前H4K test-selected profile改善，但未刷新历史validation frontier。这个不对称结果支持对完整48个checkpoints执行formal test，而不支持提前宣称H4L提高test排名，亦不支持只测试validation winners。

## 4. Frozen manifest and requested formal-test scope

`scripts/build_iscf_bsca_main_v1_h4l_test_manifest.py`从remote-audited ledger冻结48-row manifest。每行记录dataset、trial/profile、seed、best epoch、validation mean、parameter count、checkpoint SHA256、training artifact path和未来atomic test path。

请求授权的唯一scope为48 checkpoints × `{96,192,336,720}` = 192 standard-horizon rows，每row同时产生MSE/MAE。执行前仍需冻结H4L test-audit config与checker，并在remote exact-commit下重新验证48个checkpoint hashes、test target/tmp count=0、GPU occupancy与quota。禁止partial test、validation筛选、per-H/per-metric/per-cell selection、checkpoint retraining/mutation、H4M或3-seed扩展。

完整formal test后，必须把H4L与既有117 trials合并，继续使用冻结的dataset-level joint MSE/MAE guard与leading-cell selector；ETTm2和Weather各自只能选择一个profile共同服务四个horizons，所有negative trials必须保留。

## 5. Four-layer status

- `paper_facing_effectiveness=pending_complete_H4L_official_test`；
- `matched_mechanism_attribution=unchanged_pending_Main_II_and_ablations`；
- `internal_mechanism_health=48_of_48_training_numeric_provenance_and_artifact_pass`；
- `failure_attribution=no_training_pathology_performance_unknown_before_test`。

Decision=`H4L_training_complete_manifest_frozen_formal_test_authorization_requested`。
