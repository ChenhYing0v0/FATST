# ISCF-BSCA-MAIN-v1 H4K Training Result and Formal-Test Gate

## 1. Decision summary

| Field | Value |
| --- | --- |
| `date` | 2026-08-04 |
| `current_step` | Step 9 train/validation artifact audit complete；formal test waiting explicit authorization |
| `training_matrix` | 24/24 complete；test=0/24 |
| `datasets` | ETTh1 2、ETTh2 2、ETTm1 2、ETTm2 8、Weather 6、ECL 2、Solar 2 |
| `numeric_health` | 24/24 pass |
| `checkpoint_hashes` | 24 present and unique SHA256 values frozen |
| `manifest` | `h4k_checkpoint_manifest.csv`；24 rows；SHA256 `61446262b9ed645fb1e2ebd233a0dd88c7d1743ff80fa8ed0fc2a5a52e2747ab` |
| `official_test` | not authorized；0 target or temporary artifacts |
| `decision` | `H4K_training_complete_manifest_gate_pass_waiting_formal_test_authorization` |

## 2. Training and artifact audit

Remote train/validation于2026-08-04 01:33:35 +08:00完成，launch commit=`dc52471d585906f1f2251fdb682557f8e5686931`，wall time约8小时49分。24个jobs均产生checkpoint、training log、four-H validation metrics、effective config、initialization contract、model diagnostics和environment record；日志未检出OOM、NaN、Inf、Traceback或RuntimeError。结束后GPU0/1/2均为18 MiB、0% utilization，H4K root约1.2 GiB，用户quota约191/220 GiB。

Remote analyzer结果：

- `training_complete_trials=24/24`；
- `validation_complete=24/24`；
- `numeric_health_pass=24/24`；
- `test_complete_trials=0/24`；
- 24个checkpoint SHA256均存在且互不重复；
- 24/24 effective configs匹配冻结的config/search-space hashes；
- 24/24均为`official_test_mode=false`、`final_evaluation_split=val`和validation horizons `{96,192,336,720}`。

ECL `exact_budget60` profiles沿用冻结的60-epoch budget，因此best epoch 44/46并不违反protocol；其余profiles的best epoch范围为1--31。没有checkpoint或training contract pathology。

## 3. Validation-only context

下表仅用于训练健康与候选解释，比较H4K内最低four-H validation mean MSE与H4J完成后当前paper-row profile的validation值。它不用于选择H4K profile，也不替代official test。

| Dataset | H4K validation-best trial | Validation mean MSE | Relative to current paper-row profile |
| --- | --- | ---: | ---: |
| ECL | `ECL__h4k_exact_budget60_dropout4` | 0.131354 | +1.300% improvement |
| ETTh1 | `ETTh1__h4k_lr3e4_rank64` | 1.133715 | +0.049% improvement |
| ETTh2 | `ETTh2__h4k_lr5e4_rank64` | 0.398310 | +5.105% improvement |
| ETTm1 | `ETTm1__h4k_lr8e5` | 0.600780 | +0.076% improvement |
| ETTm2 | `ETTm2__h4k_rank64_capacity64` | 0.182041 | +0.663% improvement |
| Weather | `Weather__h4k_current_lr5e5_dropout0` | 0.485400 | +0.642% improvement |
| Solar | `Solar__h4k_patch3_lr3e4` | 0.129061 | -1.727% degradation |

这个结果给出有限的正面信号：6/7 datasets相对当前paper-row profile具有更低validation mean，且主要弱项ETTm2和Weather都出现改善。反向证据是：若与H1--H4J所有历史trials的dataset-wise最低validation mean比较，H4K在7/7 datasets均未刷新validation frontier，差距约0.158%--2.502%。因此不能从validation推断H4K已提高test排名，也不能提前筛掉任何H4K checkpoint；这轮搜索的目标本来就是test-informed weak-cell修复，完整official test仍是唯一performance gate。

## 4. Frozen manifest and pending test contract

`scripts/build_iscf_bsca_main_v1_h4k_test_manifest.py`从remote-audited ledger冻结24-row manifest。每行记录dataset、trial/profile、seed、best epoch、validation mean、parameter count、checkpoint SHA256、training artifact path和未来atomic test path。Local test contract已完成fail-closed dry-run：24 jobs、96 standard-horizon cells、`authorized=false`。

若用户授权，formal test的唯一允许scope为24 checkpoints × `{96,192,336,720}`，每cell同时报告MSE/MAE。必须在launch前再次验证remote commit、24个checkpoint hashes、test target/tmp file count=0、GPU occupancy和quota；禁止partial execution、per-H/per-metric/per-cell selection、checkpoint retraining或mutation。完成24/24后，H4K必须与既有93 trials合并执行冻结的dataset-level joint selector，并保留全部negative trials。

## 5. Four-layer status and next gate

- `paper_facing_effectiveness=pending_complete_H4K_official_test`；
- `matched_mechanism_attribution=unchanged_pending_Main_II_and_ablations`；
- `internal_mechanism_health=24_of_24_training_numeric_and_artifact_pass`；
- `failure_attribution=no_training_pathology_performance_unknown_before_test`。

Decision=`H4K_training_complete_manifest_gate_pass_waiting_formal_test_authorization`。Automatic H4L仍为false。
