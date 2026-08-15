# iTransformer-style Decoder-Transfer v1 Training Result and Formal-Test Gate

Decision：`itransformer_transfer_v1_training_manifest_pass_validation_risk_negative_formal_test_pending_authorization`

## 1. Artifact gate

Remote train/validation于2026-08-15 19:06:46 +08:00完成。Driver记录15/15 `job_done`；修复只影响status/resume判定的completion predicate，未重训或覆盖任何run。独立checker确认：

- 15/15完整training artifact sets；
- 15/15 unique checkpoint SHA256；
- 5/5 dataset-level matched encoder-initialization triplets；
- four-H validation metrics全部finite，且其均值与training log中的best validation epoch在batch-reduction roundoff内一致；
- 每个dataset的三臂共享encoder profile与initialization class，全部from-scratch end-to-end joint training；
- formal-test artifacts和test jobs均为0。

Immutable manifest为`training_audit/immutable_training_manifest.csv`，SHA256=`062588a140ecd4fae385aa9d194c039355bef3c7d9f49f685d796779626eecc9`。远程manifest与hash sidecar已设为read-only。

## 2. Validation-only risk signal

该分析只读取checkpoint selector已产生的validation scorecards，不访问test loader或labels。五dataset macro如下：

| Arm | MSE | MAE |
| --- | ---: | ---: |
| Original Decoder | 0.588585 | 0.458550 |
| +ISCF | 0.599596 | 0.465834 |
| +ISCF-BSCA | 0.599834 | 0.467215 |

相对Original Decoder，+ISCF-BSCA的validation macro MSE/MAE为`-1.911%/-1.889%` gain，只赢`1/5` dataset MSE means、`4/20` MSE cells。相对matched +ISCF，其macro MSE/MAE为`-0.040%/-0.296%` gain；MSE虽赢`3/5` datasets，但只赢`9/20` cells且aggregate没有改善。

[Strong Evidence] 当前validation方向提示iTransformer-style replacement readout的formal gate风险很高，而且BSCA objective在此carrier上尚未呈现稳定aggregate utility。

[Boundary] Validation不能替代official-test effectiveness，也不能据此完成或拒绝Section 5.7 transfer结论。PatchTST曾出现validation/test reversal，因此若要对iTransformer carrier作方向级判断，仍必须执行冻结的完整formal surface，而不能选择性只测有利dataset或horizon。

## 3. Frozen next gate

如作者显式授权，下一阶段只能执行一次完整15-checkpoint × four-H=`60`-cell official-test audit。成功条件保持预注册不变：+ISCF-BSCA相对Original Decoder同时改善macro MSE与macro MAE，并赢至少`3/5` dataset-mean MSE comparisons。+ISCF-BSCA相对+ISCF必须单独完整报告。

Formal test、table mutation、extra HPO和extra seeds当前仍为false。Main Decoder-Transfer table继续使用已冻结的DLinear/PatchTST v2.1结果；iTransformer结果在formal matrix完整并经Step 9--10审计前不得进入论文表。

## 4. Failure-attribution boundary

若formal结果延续validation方向，claim-level应记为`hypothesis_false_for_cross_backbone_portability_after_two_failed_nonlinear_carriers`；design-level仍需区分`readout_or_head_design_wrong_for_itransformer_representation_compatibility`与BSCA objective本身。不得用该结果否定DLinear-style正向block，也不得删除PatchTST或iTransformer负向block。按冻结rollback，失败后不自动开启第四carrier、额外decoder HPO或多seed补救。
