# ISCF-BSCA-MAIN-v1 H2 Result and 40-Checkpoint Test Prelaunch

## 0. Decision summary

| Field | Value |
| --- | --- |
| `current_step` | Step 9 H2 artifact audit complete；Tier B2 official-test ranking prelaunch |
| `candidate_version` | `ISCF-BSCA-MAIN-v1-hpo-h1-h2-20260801` |
| `H1_status` | `16/16_validation_complete` |
| `H2_status` | `24/24_validation_complete` |
| `combined_status` | `40_checkpoints_hash_frozen_test_0` |
| `primary_seed` | 2021 |
| `checkpoint_rule` | validation mean MSE over H96/H192/H336/H720 |
| `profile_rule` | official-test mean MSE over H96/H192/H336/H720 per dataset |
| `decision` | execute the complete 40-checkpoint test-tuned ranking once |

## 1. Problem

H2需要判断冻结的ISCF-BSCA架构能否通过bounded dataset-level profile search改善H1，并为8 datasets的main-table profile selection提供完整候选集。H2 validation不能直接回答SOTA问题；最终profile必须在全部40个frozen checkpoints上按four-H official-test aggregate选择。

## 2. Existence evidence

[Fact] H2的24个jobs全部完成。逐trial audit确认24/24均具有checkpoint、training log、四-H validation MSE/MAE、effective config、initialization contract和model diagnostics；checkpoint SHA256与selector对应关系24/24一致，未发现Traceback、OOM、NaN或Inf，test artifacts为0。

| Dataset | H1 best validation MSE | H2 best profile | H2 validation MSE | Relative improvement |
| --- | ---: | --- | ---: | ---: |
| ETTh1 | 1.121983 | `h2_lookback336` | 1.106044 | 1.421% |
| ETTh2 | 0.400668 | `h2_lookback336` | 0.393310 | 1.836% |
| ETTm1 | 0.600672 | `h2_dropout5` | 0.599832 | 0.140% |
| ETTm2 | 0.183536 | `h2_lookback336` | 0.181867 | 0.909% |
| Weather | 0.484225 | `h2_capacity128` | 0.482510 | 0.354% |
| ECL | 0.132994 | `h2_dropout3` | 0.131686 | 0.984% |
| Solar | 0.133654 | `h2_dropout1` | 0.130607 | 2.279% |
| Exchange | 0.824208 | `h2_lr5e5` | 0.711429 | 13.683% |

[Strong Evidence] 8/8 datasets均出现优于H1的H2 validation profile，说明bounded search有效；但Exchange validation样本较少，13.683%不得外推为test收益。ECL `h2_dropout3`在epoch 29/30取最优，存在training-budget boundary；Solar最优在regularization方向，支持后续定向扩展预算与邻域。

## 3. Idea and theory check

当前40-checkpoint test audit不训练、不改变architecture、不修改checkpoint。Evaluator只读取best-validation checkpoint，在独立test artifact root输出dense H1--H720 metrics与diagnostics。每个dataset的五个profiles必须全部完成，才能按four-H mean test MSE生成排名。

[Theory check] Test feedback只选择一个dataset-level profile，并由同一profile覆盖H96、H192、H336、H720。这与unified forecasting claim一致。Test不得选择epoch、checkpoint、seed、MAE-specific profile或单个table cell。

## 4. Design

- matrix：8 datasets × 5 profiles × seed2021 = 40 checkpoints；
- standard scorecard：40 × 4 horizons = 160 MSE/MAE cells；
- checkpoint provenance：40个test前SHA256冻结；test前后必须一致；
- scheduler：ECL、Solar优先，其后Weather、ETTm1、ETTm2、ETTh1、ETTh2、Exchange；三GPU global shared queue；
- artifact root：`/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/test_audit`；
- selection：lower test mean MSE，依次以validation mean MSE、parameter count、lexical profile id打破平局；
- failure gate：任何missing/NaN/Inf/hash mismatch/invariant failure均阻断全部profile ranking；
- rollback：runner/evaluator failure回到local protocol repair；checkpoint mutation立即写ABORT sentinel并阻断新job。

用户于2026-08-01进一步授权ECL和Solar的test-tuned定向HPO，包括扩展training budget。该授权将在本轮40-checkpoint完整scorecard后建立独立H3 candidate version；可以减少validation对比成本，但每个trial仍由validation选择checkpoint，随后直接进入完整four-H test evaluation。禁止per-horizon、per-cell和selective reporting。

## 5. Narrative gate

`passed_as_same_architecture_hpo`。H2及后续ECL/Solar定向HPO只优化冻结architecture的dataset-level training/profile setting，不形成新mechanism claim。

## 6. Effectiveness gate

当前为`validation_search_pass_test_effectiveness_pending`。H2在8/8 datasets改善validation，但没有本轮official-test结果，不能宣称SOTA。40/40 test complete后，先判断ECL/Solar相对已审计published targets的差距，再冻结最小充分H3。

## 7. Failure attribution

当前没有numeric pathology。ECL budget-boundary暂记为`optimization_budget_possibly_insufficient`；Solar的regularization敏感性暂记为`profile_optimization_incomplete`。两者都不是`hypothesis_false`或architecture failure。若扩展budget和局部profile search仍无法达到目标，回到Step 6重估search freedom与SOTA claim，不得通过per-cell选择掩盖失败。

## 8. Artifacts

- `h2_artifact_audit/trial_ledger.jsonl`；
- `h2_artifact_audit/trial_scorecard.csv`；
- `h2_artifact_audit/profile_aggregates.csv`；
- `h2_artifact_audit/hpo_completeness.json`；
- `combined_checkpoint_manifest.csv`，SHA256=`ba12d47b1313dc435d6e9aa7f432c6ffb87a0753d980afa1bb7315f123fec879`；
- `configs/iscf_bsca_main_v1_hpo_test_audit.json`；
- `scripts/remote/run_iscf_bsca_main_v1_hpo_test_audit.sh`；
- `scripts/analyze_iscf_bsca_main_v1_hpo_test_audit.py`。

## 9. Decision

`H2_complete_40_checkpoint_test_audit_authorized_prelaunch`。先执行完整40-checkpoint test ranking；ECL和Solar结果优先查看但不做partial selection。其后依据完整four-H scorecard进入已授权的ECL/Solar定向扩展预算H3。Selected-profile confirmation、3-seed和final paper audit仍未在本阶段执行。
