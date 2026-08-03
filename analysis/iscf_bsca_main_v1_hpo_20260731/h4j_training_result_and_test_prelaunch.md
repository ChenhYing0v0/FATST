# ISCF-BSCA-MAIN-v1 H4J Training Result and Official-Test Prelaunch

## Decision summary

| Field | Value |
| --- | --- |
| `date` | 2026-08-03 |
| `current_step` | Step 9 train/validation artifact audit complete；direct complete official-test prelaunch |
| `training_matrix` | 40/40 complete；test=0/40 |
| `datasets` | ETTh1 2、ETTh2 2、ETTm1 2、ETTm2 9、Weather 9、ECL 4、Solar 10、Exchange 2 |
| `numeric_health` | 40/40 pass |
| `checkpoint_hashes` | 40 unique SHA256 values frozen |
| `manifest` | `h4j_checkpoint_manifest.csv`；40 rows；SHA256 `5b91d50040dc7bab6d822ed6f03fb4a878b643277d8fc473cc184bf5d094d00b` |
| `profile_ranking_before_test` | none |
| `official_test` | complete 40-checkpoint × four-H audit authorized |
| `decision` | `H4J_40_of_40_training_complete_direct_test_prelaunch` |

## 1. Training completion audit

Remote training finished at`2026-08-03T06:01:05+08:00` under commit`a7d23780311e75f05d04dd3843dd7b29ceef6b08`，output root=`/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h4j_r1`。Status reports40/40 complete and test=0/40；GPU 0/1/2 are idle。

`scripts/analyze_iscf_bsca_main_v1_hpo.py` audited every run for checkpoint、training log、four-H validation metrics、effective config、initialization contract and model diagnostics。Results：

- `training_complete_trials=40/40`；
- `validation_complete=40/40`；
- `numeric_health_pass=40/40`；
- best epochs range 1--45；
- 40 checkpoint hashes are present and unique；
- official-test artifacts remain absent。

The audit is stored under `analysis/iscf_bsca_main_v1_hpo_20260731/h4j_artifact_audit/`。No validation profile ranking was performed；validation only selected each trial checkpoint。

## 2. Frozen manifest and test contract

`scripts/build_iscf_bsca_main_v1_h4j_test_manifest.py` generated a 40-row manifest from the audited ledger。Each row records dataset、trial/profile ID、seed、best epoch、validation mean MSE、parameter count、checkpoint SHA256、training artifact path and future atomic test path。Manifest SHA256 is frozen in`configs/iscf_bsca_main_v1_hpo_joint_h4j_test_audit.json`。

The complete test matrix is40 checkpoints × H96/H192/H336/H720=160 standard-horizon rows，each withMSE andMAE。Partial test execution or partial profile selection is forbidden。Checkpoint re-training and mutation during test are forbidden；pre/post test SHA256 must be identical。

The phase-local test analyzer is configured as audit-only：it emits completeness、ledger、all-trial scorecard and aggregates, but deliberately does not perform its legacy MSE-only profile ranking。Final profile selection is deferred until all H4J test rows are combined with the existing53 trials and evaluated under the frozen joint selector。

## 3. Next action and gates

Before test launch：

1. focused commit/push containing manifest、test config、wrapper/checker and analyzer guard；
2. remote `git pull --ff-only` and exact commit match；
3. verify40 training checkpoint hashes against manifest；
4. verify test target/temporary file count is zero；
5. verify selected GPUs <=1024 MiB and <=20% utilization。

After those gates pass，execute the complete test once。Only40/40 complete test artifacts may advance to joint selection。The result gate remainsMSE>=20/28、MAE>=20/28、combined>=40/56；automatic H4K extension remains unauthorized。

Four-layer status：

- `paper_facing_effectiveness=pending_complete_H4J_test`；
- `matched_mechanism_attribution=unchanged_pending_Main_II_and_ablations`；
- `internal_mechanism_health=training_numeric_health_pass`；
- `failure_attribution=no_training_pathology`。

Decision=`H4J_40_of_40_training_complete_direct_test_prelaunch`。
