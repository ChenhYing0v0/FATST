# H4M training result与formal-test gate

日期：2026-08-05

## 结论

H4M已完成24/24 train/validation jobs，test仍为0/24。24个checkpoint、four-H validation metrics、training logs、effective configs、initialization contracts、model diagnostics和environment records均存在；24个checkpoint SHA256唯一。训练阶段使用`best-val`、four-H mean validation MSE selector、seed2021和`final_evaluation_split=val`，未启用official-test mode。

因此，H4M通过training artifact gate，可以按用户2026-08-05授权执行一次完整formal test：24 checkpoints × H96/H192/H336/H720 = 96 standard-horizon rows，同时报告MSE与MAE。不得checkpoint retraining/mutation，不得根据validation筛掉profiles，也不得per-H、per-metric或per-cell选择profile。

## Validation-only diagnostic

该排序只用于检查训练健康，不用于正式profile选择：

- ETTm2 validation best=`ETTm2__h4m_p2_lr5e5`，four-H mean MSE=`0.1795502`，best epoch=13；
- Weather validation best=`Weather__h4m_seq960_p30`，four-H mean MSE=`0.4828334`，best epoch=7；
- 全部best epoch范围为6--69。Weather出现epoch69的有效best checkpoint，说明统一扩展至90 epochs并非空耗，但不能据此预判test优劣。

## Frozen manifest

- rows=`24`；
- ETTm2/Weather=`12/12`；
- unique checkpoint hashes=`24`；
- manifest=`analysis/iscf_bsca_main_v1_hpo_20260731/h4m_checkpoint_manifest.csv`；
- manifest SHA256=`bd78782e70a911e8c5b118c67a4441cf512d7012b94fdda8a19cf14507e8adb3`；
- remote test root=`/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h4m/test_audit`；
- pre-freeze test artifacts=`0`。

Formal test结束后，H4M 24 trials必须与H1--H4L 165 trials合并为189-trial selector，再原样执行dataset-level joint MSE/MAE 1% guard、leading-cell和40/56 gate。H4N未授权。

Decision=`H4M_training_complete_24_checkpoint_manifest_frozen_formal_test_authorized`。
