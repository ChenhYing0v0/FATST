# H5A Formal-Test Result and Main II Selection

## 1. Formal-test completion

H5A once-only formal test于2026-08-13 00:34:37启动，01:00:46完成queue级全局复核。
冻结matrix的48个checkpoints均完成official-test evaluation：

- complete trials=`48/48`；standard-horizon rows=`192/192`；
- datasets=`ETTh1/ECL/Solar`，各16 profiles，seed=`2021`；
- 每个profile只使用validation four-H mean MSE选择checkpoint，test阶段没有重训或
  checkpoint mutation；
- 48个pre-test SHA256与post-test SHA256完全一致；manifest SHA256=
  `ee5940c8f66aceab5710f17a4bc8ce2efb9ae3c44fa9cec1459fcd9589fe6643`；
- 每个checkpoint的720-row dense metrics、test invariant和diagnostic NPZ均通过
  atomic publication validator；`ABORT`未出现；
- remote test root约22 GiB；结束时账户约206 GiB，超过200 GiB soft quota但低于
  220 GiB hard limit，未发生quota failure。

Canonical audit目录为
`analysis/iscf_bsca_main_v1_hpo_20260731/h5a_formal_test_result_20260813/`。
`test_audit_completeness.json` SHA256=
`61a4f96b39ceb584afef3c56da286773bd6cf3f679727c4071052f9e1a395f2f`；完整192-row
scorecard SHA256=`67d7b9c2d5d7ed4fa4ff3634521da6208b38012012119d0d65e19334bf5a4377`。
Generic analyzer附带生成的mean-MSE `profile_ranking/selected_profiles`只标记为
`excluded_generic_mse_only_diagnostic`；H5A权威selection仅使用下述frozen Main II analyzer。

## 2. Frozen Main II selector result

Selector合并189个H1--H4M historical profiles与48个H5A profiles，只在目标datasets
内重选一个shared four-horizon profile。Primary ranking为three-decimal best cells，
并要求four-H mean MSE和MAE均不超过current profile的`1.005x`；禁止per-H、
per-metric、per-seed或per-cell selection。

| Dataset | Selected profile | Mean MSE | vs current | Mean MAE | vs current | Best cells | Current |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | `ETTh1__h5a_lr3p5e4` | 0.392803 | -0.182% | 0.419707 | -0.315% | 2/8 | 1/8 |
| ECL | `ECL__h5a_seq336_p1` | 0.151848 | +0.147% | 0.244527 | -0.451% | 1/8 | 0/8 |
| Solar | `Solar__h5a_seq512_p4_lr2p5e4` | 0.188300 | -1.147% | 0.208895 | -0.900% | 6/8 | 4/8 |

所有三个selected profiles同时通过MSE与MAE 0.5% guard。逐cell完整结果为：

| Dataset | H | MSE | MAE | Three-decimal best cells |
| --- | ---: | ---: | ---: | --- |
| ETTh1 | 96 | 0.351003 | 0.389857 | -- |
| ETTh1 | 192 | 0.377530 | 0.405240 | -- |
| ETTh1 | 336 | 0.393247 | 0.418477 | MSE |
| ETTh1 | 720 | 0.449431 | 0.465254 | MSE |
| ECL | 96 | 0.115607 | 0.212988 | MAE |
| ECL | 192 | 0.136425 | 0.231490 | -- |
| ECL | 336 | 0.157400 | 0.250542 | -- |
| ECL | 720 | 0.197960 | 0.283089 | -- |
| Solar | 96 | 0.165931 | 0.195401 | MSE, MAE |
| Solar | 192 | 0.184838 | 0.206499 | MSE, MAE |
| Solar | 336 | 0.196278 | 0.213965 | MAE |
| Solar | 720 | 0.206153 | 0.219715 | MAE |

目标三dataset合计由`5/24`提高到`9/24` best cells；若四个非目标datasets保持不变，
Main II全局由`24/56`提高到`28/56`。冻结最低目标分别为ETTh1/ECL/Solar
`2/8,1/8,5/8`、目标总计至少8、全局至少27；实际为`2/8,1/8,6/8`、总计9、
全局28，全部通过。Solar MAE保持4/4 best cells。

Canonical selector output为
`h5a_formal_test_result_20260813/frozen_main_ii_selector/h5a_selection_result.json`，
SHA256=`0118b97ab30809ff731041824e1d49ccdb64aba8a73e1b23a8dd49302b26fae7`。

## 3. Evidence boundary and decision

`paper_facing_effectiveness`：H5A在预注册的目标、guard和single-profile contract上通过，
因此三个dataset的新profiles可标记为Main II replacement candidates。

`matched_mechanism_attribution`：H5A只比较同一ISCF-BSCA architecture内的
hyperparameters，不提供decoder mechanism attribution，也不能替代ablation evidence。

`internal_mechanism_health`：formal artifacts、finite metrics、provenance和checkpoint
immutability均通过；本轮没有新增routing/arm-skill机制结论。

`failure_attribution`：不存在runtime、numeric或artifact failure。仍保留负结果边界：
ETTh1没有MAE best cell，ECL只有H96 MAE达到best且mean MSE轻微上升，Solar的H336/H720
MSE仍非best。结果是targeted performance pass，不是所有cells SOTA。

Decision=`H5A_success_gate_pass_selection_frozen_table_mutation_not_authorized`。根据冻结授权，
本报告不自动修改Main I、Main II LaTeX/table data，不启动H5B、extra seeds或selected-profile
confirmation。若用户授权表格更新，下一步只替换Main II中ETTh1/ECL/Solar三个完整
four-H profile并重新计算全表style/count/hash；不得按单cell混用profiles。
