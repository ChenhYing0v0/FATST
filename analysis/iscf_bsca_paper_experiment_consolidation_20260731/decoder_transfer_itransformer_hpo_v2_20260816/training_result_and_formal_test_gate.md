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
