# iTransformer-style Decoder-HPO v2 Training Result and Formal-Test Gate

Decision：`itransformer_decoder_hpo_v2_training_manifest_pass_280_cell_test_tuned_formal_authorized`。

70-run train/validation queue于`2026-08-16T07:15:14+08:00`完整结束，training期间formal test=`0`。Artifact checker确认：

- 70/70 required training artifacts完整；
- 70/70 checkpoint SHA256唯一；
- 5/5 datasets的14 profiles共享matched encoder initialization；
- four-H validation selector与effective config一致；
- logs中无Traceback、OOM、RuntimeError、NaN或Inf；
- immutable manifest SHA256=`ab3a4cd95e73d25540f5d21d4aabfef3221ac1ca7a4ab14e6c03e7ce8286c66a`。

用户已于2026-08-16明确允许按official-test结果反选decoder参数。本次formal access固定为70个immutable checkpoints × four standard horizons=`280` new cells，并复用v1 BSCA reference与Original Decoder各20个已有cells。完整candidate pool为14 new profiles + v1 reference，即`300` cells。

每个dataset只能按four-H mean test MSE选择一个共享profile；依次以mean test MAE、decoder parameter count和profile id打破完全相同的排序。禁止per-H、per-seed、per-metric或per-cell拼接；所有negative trials必须进入scorecard。Runner在test loader前后验证全部checkpoint hashes，任何partial failure均阻止selection。

本轮不授权checkpoint retraining、extra seeds或canonical paper-table mutation。若selected BSCA通过相对Original的macro MSE/MAE与3/5 dataset-win gate，下一步仍须补matched selected-profile `+ISCF` controls，不能直接宣称BSCA portability。

## Formal invariant compatibility repair

首次formal执行在16/70 complete时fail-fast。`p12/p13`的test prediction、metrics、prefix consistency与numeric health均正常，但旧evaluator将model scopes与全局default diagnostic scopes比较，错误产生`readout_contract_pass=false`。该失败归因为`formal_invariant_checker_protocol_defect`，不属于模型或checkpoint failure。

修复仅将scope invariant的expected value改为checkpoint effective config中的`adapter["pcsd_scales"]`。训练、模型forward、checkpoint、test loader与selection rule均不变；旧default-scope protocols保持相同行为。修复后先对一个failed alternate-scope checkpoint做hash-guarded rerun，确认invariant pass，再从缺失jobs恢复同一次完整formal audit。已完成的partial metrics不得用于selection。
