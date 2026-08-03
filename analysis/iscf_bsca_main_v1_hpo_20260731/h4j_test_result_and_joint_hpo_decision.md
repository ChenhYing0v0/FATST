# ISCF-BSCA-MAIN-v1 H4J Test Result and Joint-HPO Decision

## 1. Scope and decision

H4J完成40/40 frozen checkpoints的complete official-test audit，共160/160 standard-horizon cells。将H1、H2、H3A、H3B和H4J合并后，joint selector在93个retained trials上按冻结规则为每个dataset选择一个同时服务`{96, 192, 336, 720}`的profile。

Final decision=`H4J_complete_joint_HPO_material_partial_improvement_gate_fail_H4K_not_authorized`。H4J显著改善MAE coverage并使Solar成为强结果，但MSE `15/28`、MAE `15/28`、combined `30/56`均未达到预注册的`20/28`、`20/28`和`40/56`。因此当前结果是`performance_partial_pass`，不能据此宣称完整per-cell SOTA，也不能自动启动H4K。

由于terminal gate失败且H4K尚未决策，本轮不覆盖`configs/iscf_bsca_main_v1_selected_profiles.json`；该文件继续作为H4J前23/56 reference。当前30/56 selection由本报告及`joint_selected_profiles.csv`冻结为H4J decision artifact，而不是terminal Main I/II config。

## 2. Artifact and protocol audit

| Item | Result |
| --- | --- |
| H4J checkpoints | 40/40 complete |
| Standard-horizon cells | 160/160 complete |
| Test errors | 0 |
| Checkpoint hash immutable | 40/40 |
| `checkpoint_retrained=true` | 0/40 |
| `matrix_complete=true` | 40/40 |
| Test access date | 2026-08-03 |
| Seed | 2021 |
| Test role | test-tuned hyperparameter selection and paper benchmark |
| Selection granularity | one dataset-level profile shared by four horizons |

Canonical completeness SHA256=`232fd49f299c88f46efc96ab8f617c0b8f872de4c86b14091382eca07ac7a372`；40-row test ledger SHA256=`fe7c93cdc0e2fda4abc607941ace6c06fe611b32e30e84414867f0ebf76a122c`；H4J all-trial scorecard SHA256=`0d2680332581c2da6fbab2fde3e135207dec9eba19906d1d404b07c44263f40e`。

## 3. Joint selection result

| Dataset | Selected trial | Phase | Mean MSE | Mean MAE | MSE leads | MAE leads | Total |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ECL | `ECL__h2_intermediate_capacity` | H2 | 0.151625 | 0.245635 | 3 | 2 | 5 |
| ETTh1 | `ETTh1__h4j_lr3e4` | H4J | 0.393520 | 0.421034 | 3 | 2 | 5 |
| ETTh2 | `ETTh2__h2_lr5e4` | H2 | 0.307332 | 0.365116 | 3 | 3 | 6 |
| ETTm1 | `ETTm1__h2_table5_capacity` | H2 | 0.330699 | 0.363879 | 3 | 3 | 6 |
| ETTm2 | `ETTm2__h4j_rank64` | H4J | 0.248994 | 0.306669 | 0 | 0 | 0 |
| Solar | `Solar__h4j_patch4_lr3e4` | H4J | 0.190485 | 0.210792 | 2 | 4 | 6 |
| Weather | `Weather__h4j_timealign_dropout3` | H4J | 0.216581 | 0.248577 | 1 | 1 | 2 |
| Exchange | `Exchange__h2_lookback336` | H2 | 0.398836 | 0.425081 | — | — | — |

Exchange缺少冻结的同协议published target，不进入56-cell denominator；其profile按within-search equal-weight MSE/MAE regret选择。

相对H4J前selected profiles，共同7 datasets的macro mean MSE由0.263365降至0.262748，改善0.234%；macro mean MAE由0.310633降至0.308815，改善0.585%。Leading cells由MSE 14、MAE 9、combined 23提高到15、15、30，即`+1/+6/+7`。主要来源是Solar由1/8提高到6/8，Weather由1/8提高到2/8，以及ECL在1% joint-mean guard内重选后增加一个MAE cell。

## 4. Published-context competitiveness

在TimeAlign ICLR 2026 Table 6的共同7 datasets × 4 horizons上，joint-selected ISCF-BSCA的cell macro MSE为0.262748，TimeAlign为0.269286，ISCF-BSCA低2.428%；macro MAE为0.308815，TimeAlign为0.310429，低0.520%。相对TimeAlign本身，ISCF-BSCA分别领先16/28 MSE cells和16/28 MAE cells；相对Table 6五个published models逐cell最优值，则为15/28和15/28。

该结果支持“aggregate-level strong competitor”，但不支持无条件SOTA表述：ISCF-BSCA是single-seed且test-tuned，TimeAlign表值为three-run mean；published values经过三位小数舍入，并且native source protocol不是matched mechanism control。DLinear、PatchTST、TimeMixer和iTransformer上的明显优势只能作为published-context accuracy evidence。

## 5. Where the remaining losses are

| Horizon | MSE leads / 7 | MAE leads / 7 | Combined / 14 |
| ---: | ---: | ---: | ---: |
| 96 | 5 | 4 | 9 |
| 192 | 5 | 5 | 10 |
| 336 | 4 | 3 | 7 |
| 720 | 1 | 3 | 4 |

H720是最明显的shared-profile bottleneck。ECL、ETTh1和ETTh2在H720的MSE/MAE均落后published target；Solar的H720 MAE领先但MSE落后2.48%；ETTm1 H720 MSE只落后0.08%。

ETTm2是dataset-level最大缺口：selected profile四个horizons的MSE分别落后4.62%、4.04%、3.08%、0.36%，MAE分别落后2.86%、2.09%、1.82%、0.05%，合计0/8。Weather为2/8，其H96--H336差距大约为MSE 1.0%--1.6%、MAE 1.8%--3.0%，只有H720两项领先。

## 6. Selector and search-space diagnosis

合法joint selector已经达到93个trials的unrestricted single-profile upper bound 30/56。进一步允许每个dataset-horizon-metric从任意trial取最优值的diagnostic oracle仍只有30/56；每个dataset的selected lead count都等于该oracle count。因此当前10-cell gate gap不由1% joint guard、tie-break或shared profile selection造成，而是现有93个trial没有产生能够赢下这些cells的预测结果。

H4J内部，Solar的`patch4` family形成稳定的2 MSE + 4 MAE优势，证明本轮局部搜索在Solar上有效。ETTm2的`rank64`将旧selected mean MSE改善0.739%、mean MAE改善1.198%，但9个H4J profiles全部为0/8；Weather的`timealign_dropout3`只增加一个MAE win。ECL、ETTh2和ETTm1继续由旧H2 profiles保留，说明H4J新增arms没有提供更好的合法trade-off。

Validation与test ranking在ECL、Solar上方向一致，但在ETTm2和Weather上反转：ETTm2 H4J validation/test joint-score Spearman约为-0.583，Weather约为-0.667。该现象不违反当前test-tuned protocol，但说明下一轮不能把validation improvement当成paper-facing improvement的代理；validation仍只负责trial内checkpoint selection。

## 7. Four-layer interpretation and rollback

1. `paper_facing_effectiveness=performance_partial_pass_gate_fail`：aggregate表现很强，且相对TimeAlign macro MSE/MAE均更低，但30/56未达到40/56目标。
2. `matched_mechanism_attribution=not_evaluated_by_H4J`：H4J只比较同一ISCF-BSCA architecture内的hyperparameters，不能承担ISCF、allocation或BSCA component attribution。
3. `internal_mechanism_health=no_numeric_or_artifact_pathology_detected`：40/40 checkpoints和tests完整、hash immutable、无test error；本轮未新增router/component diagnostics。
4. `failure_attribution=search_space_performance_shortfall`：不是selector failure、checkpoint mutation、numeric pathology或architecture rejection。Rollback回到HPO Step 6，冻结minimal H4K search contract；architecture仍保持frozen。

如果继续H4K，minimal sufficient优先级应为：首先围绕ETTm2 `rank64` anchor探索尚未覆盖的rank/regularization/capacity interaction；其次围绕Weather `timealign_dropout3`和`table5_dropout3`搜索短horizon误差；最后只对ECL、ETTh1、ETTh2、ETTm1和Solar设计能够改善H720且保持four-H joint score在1% guard内的dataset-level profiles。任何H4K仍需新的frozen matrix和明确授权，不得按horizon选择profile或自动启动。

## 8. Canonical artifacts

- H4J test aggregation：`analysis/iscf_bsca_main_v1_hpo_20260731/h4j_test_result/`
- 93-trial joint analysis：`analysis/iscf_bsca_main_v1_hpo_20260731/joint_objective_h4j_result_20260803/`
- final status SHA256：`1818daafb237682a0fd5bbb5858737ba596aad9adba7eadc1156b2d991fd184f`
- selected profiles SHA256：`9ff12d8b98ccd5fe16bde409f16a1f31127651e068460549a4f50160714f8140`
- selected 32-cell scorecard SHA256：`51da4ea56342a76c8bedb06e2f9c0fce59cfd4e7a5df3979d1e66700a558a922`
