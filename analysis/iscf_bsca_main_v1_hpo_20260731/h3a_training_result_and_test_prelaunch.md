# ECL/Solar H3A Training Result and Direct-Test Prelaunch

## Decision summary

| Field | Value |
| --- | --- |
| `current_step` | H3A Step 9 training artifact audit complete；direct official-test prelaunch |
| `candidate_version` | `ISCF-BSCA-MAIN-v1-ecl-solar-h3a-test-informed-20260801` |
| `training_matrix` | ECL 1 + Solar 8 = 9/9 complete |
| `test_matrix` | 9 checkpoints × 4 horizons = 36 cells；prelaunch test=0 |
| `checkpoint_selector` | validation four-H mean MSE；best-val |
| `profile_selector` | official-test four-H mean MSE per dataset |
| `decision` | execute complete nine-checkpoint test audit immediately |

## Evidence and boundary

[Fact] Remote status at `2026-08-01T20:14:06+08:00` was `complete=9/9 test=0/9`. All jobs have checkpoint、training log、four-H validation metrics、effective config、initialization contract和model diagnostics。Analyzer reports 9/9 artifact completeness and numeric health pass。GPU 0/1/2均为18 MiB、0% utilization。

Validation只用于确认checkpoint selection与artifact health，不用于延迟test或选择profile。完整validation aggregate如下，用于provenance而非paper claim：

| Dataset | Trial | Validation mean MSE |
| --- | --- | ---: |
| ECL | `ECL__h3a_budget45` | 0.130614 |
| Solar | `Solar__h3a_budget45` | 0.131355 |
| Solar | `Solar__h3a_lr3e4` | 0.131994 |
| Solar | `Solar__h3a_dropout4` | 0.131977 |
| Solar | `Solar__h3a_wd5e2` | 0.131159 |
| Solar | `Solar__h3a_rank64` | 0.131338 |
| Solar | `Solar__h3a_effective_batch16` | 0.131603 |
| Solar | `Solar__h3a_patch2` | 0.129777 |
| Solar | `Solar__h3a_patch4` | 0.126802 |

这些validation值不决定profile winner。特别是H1/H2曾出现明显validation-test reversal，因此9个checkpoints全部直接进入test。

## Test contract

- manifest：`h3a_checkpoint_manifest.csv`，9个test前SHA256冻结；
- manifest hash：`0753d880fce934961e3989c979417865b74394ae4641162fcc95830d7ca1407b`；
- output root：`/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/ecl_solar_h3a/test_audit`；
- test publication：temporary directory完成后atomic move；
- completeness：每trial必须有720-row dense CSV、pass invariant和完整NPZ；
- provenance：candidate/trial/profile/seed/checkpoint hash逐artifact一致；
- selection：Solar八个H3A profiles完整比较；ECL H3A budget extension与既有H1/H2 winner在结果报告中合并比较；
- failure：missing/non-finite/stale/mixed/hash mutation均阻断ranking。

## Gates and rollback

`narrative_gate=passed_as_same_architecture_test_informed_HPO`。H3A不改变architecture或mechanism。

`effectiveness_gate=pending_complete_9_checkpoint_test`。Solar成功目标为H3A winner four-H mean MSE低于TimeAlign published target 0.192；ECL只判断budget45是否优于既有0.151191。若Solar仍未达到0.192，只允许基于完整one-factor response冻结一个新的、至多四profile H3B interaction batch；不得按单horizon组合。所有负结果保留。

Decision=`H3A_9_of_9_training_complete_direct_test_authorized_prelaunch`。
