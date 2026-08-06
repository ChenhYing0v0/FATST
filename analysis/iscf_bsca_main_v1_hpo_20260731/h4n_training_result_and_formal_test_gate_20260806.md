# H4N training result 与 formal-test gate

## 1. 结论

H4N 已完成 40/40 Weather train/validation jobs，训练阶段 official test 保持
0/40。40 个 full-run checkpoints、four-H validation metrics、training logs、
effective configs、initialization contracts、model diagnostics 与 environment records
均通过 artifact audit；40 个 checkpoint SHA256 全部唯一，best epoch 范围为 3--101。

因此 H4N 通过 training-artifact gate。按 2026-08-06 用户授权，下一步执行一次完整
formal test：40 checkpoints × `{96,192,336,720}` = 160 个 standard-horizon rows，
每个 row 同时报告 MSE 与 MAE。禁止 validation profile filtering、partial test、
checkpoint retraining/mutation，以及 per-H、per-metric、per-cell 或 seed-specific selection。

## 2. 冻结 provenance

- training config：`configs/iscf_bsca_main_v1_hpo_weather_h4n.json`；
- config SHA256：`cd20062d982fe6569758f79bbf57ca1c2a59d9fe8fe95570da4de1017e8c7164`；
- search-space SHA256：`0961d3197584c3f991b7d14e38279fe81895d268753fec3348e9adfafe1a226e`；
- checkpoint manifest：`analysis/iscf_bsca_main_v1_hpo_20260731/h4n_checkpoint_manifest.csv`；
- manifest SHA256：`a0f152f9172acc193fe512001123b71aeae6d6d3ab1028c915074f24d54c1ed4`；
- artifact ledger：`analysis/iscf_bsca_main_v1_hpo_20260731/h4n_artifact_audit/trial_ledger.jsonl`；
- formal-test contract：`configs/iscf_bsca_main_v1_hpo_weather_h4n_test_audit.json`；
- training commit：`ba17fc97e85cdd362611a5eaa58b088be85add54`；
- remote output：`/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h4n`。

Validation-best diagnostic 为
`Weather__h4n_seq640_p10_lr4e5`，four-H mean validation MSE=`0.4882119133`。
该值只证明 selector 可执行，不用于排除其余 39 个 profiles，也不作 formal-test
performance 预测。

## 3. Formal-test contract

Formal test 使用 frozen best-validation checkpoint；测试前后逐 checkpoint 复算 SHA256。
每个 trial 的 720 个 dense-prefix metrics 以 temporary directory 原子生成，只有 schema、
numeric health、test provenance、checkpoint hash 与 diagnostics 全部通过才发布到 target
directory。任一 failure 会生成 ABORT sentinel，并阻止 incomplete matrix 进入 selector。

完整 test 后，H4N 40 trials 与 H1--H4M 189 trials 合并为 229-trial pool；Weather 共有
96 个 profiles。仍只允许选择一个 dataset-level profile 共同服务 four H。Primary score
为冻结目标归一化的 four-H MSE/MAE relative mean；只有 relative score 差不超过 0.1%
时才使用 balanced lead cells tie-break。所有 negative trials 保留。

## 4. Resource gate 与 rollback

2026-08-06 preflight 前远程 quota=`171G/200G soft/220G hard`，GPU0--2 均为
18 MiB、0% utilization；预估 formal-test storage 为 1--2 GiB。若 manifest hash、remote
commit、checkpoint hash、zero-target/tmp 或 GPU gate 任一失败，则停止在 test access 前修复；
若 test 已开始后失败，只允许在 immutable contract 下 atomic resume，不修改 search space、
selector 或 checkpoint。

Decision=`H4N_training_complete_40_checkpoint_manifest_frozen_formal_test_authorized`。
