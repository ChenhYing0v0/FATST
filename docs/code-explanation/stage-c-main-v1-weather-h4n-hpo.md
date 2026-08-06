# Stage C ISCF-BSCA-MAIN-v1 Weather H4N HPO 代码说明

## 1. 功能模块

本轮不修改 model forward 或 loss。新增逻辑只包含 frozen HPO contract、prelaunch
checker、remote wrapper、post-training artifact checker 与 formal-test gate：

- `configs/iscf_bsca_main_v1_hpo_weather_h4n.json`：定义40个Weather profiles、
  dataset-level selector、训练预算、source hashes、test boundary 与 gates；
- `scripts/check_iscf_bsca_main_v1_hpo_weather_h4n.py`：在任何remote execution前检查
  search-space completeness、历史去重、target provenance 与 dry-run；
- `scripts/remote/run_iscf_bsca_main_v1_hpo_weather_h4n.sh`：只为generic HPO runner
  固定H4N config和repo-external output root；
- `scripts/check_iscf_bsca_main_v1_h4n_training_artifacts.py`：40个训练完成后逐checkpoint
  核对hash、validation metrics、effective config、numeric health、logs和test absence；
- `scripts/build_iscf_bsca_main_v1_h4n_test_manifest.py`：从完整ledger冻结40个full-run
  checkpoint paths、hashes、selector metadata与future test targets；
- `configs/iscf_bsca_main_v1_hpo_weather_h4n_test_audit.json`：固定160-row formal-test
  scope、用户授权、dataset-level selector、gates与rollback boundary；
- `scripts/remote/run_iscf_bsca_main_v1_hpo_weather_h4n_test_audit.sh`与对应checker：复用
  generic atomic evaluator，并检查manifest hash、dry-run job count与selection prohibitions。

## 2. Profile materialization

每个 job 先复制 `Weather_h4m_mae_frontier` base profile，再应用 `overrides`，最后加入
`trial_id/profile_id/source_prior`。产生的关键张量合同不变：输入仍为
`batch_x [B,L,21]`，encoder/readout仍输出最大未来域 `[B,720,21]`，四个标准
horizons通过full-prefix crop评估。`seq_len`、`patch_num`、encoder width和
`mode_rank`只改变同一计算图中的尺寸，不增加 requested-horizon input。

## 3. Prelaunch checker

`fingerprint()` 使用
`dataset/seq_len/patch_num/d_model/d_ff/dropout/learning_rate/weight_decay/`
`batch_size/gradient_accumulation_steps/mode_rank/layer_norm` 构成 effective profile
identity。Checker读取H1--H4M所有artifact ledgers，要求：

1. 历史189个IDs与fingerprints可完整重建，其中Weather为56个；
2. H4N 40个IDs/fingerprints内部唯一，且与历史集合交集为空；
3. target CSV hash固定，并从所有non-ISCF rows重新计算Weather逐H MSE/MAE minima；
4. 16/8/8/5/3五个search blocks数量正确；
5. context、LR、patch、rank和capacity边界均实际覆盖；
6. generic runner dry-run为40 jobs且training-time test jobs为0。

## 4. Training artifact checker

训练完成后的每个trial目录必须包含`checkpoint.pt`、`training_log.csv`、
`metrics_by_target_horizon.csv`、`effective_config.json`、initialization/model/environment
records。Checker从effective config逐字段验证：

- candidate/config/search-space hashes；
- seed2021、best-val checkpoint、full-crop与four-H validation；
- `seq_len/patch_num/d_model/d_ff/dropout/layer_norm`；
- optimizer、effective batch、mode rank、120 epochs与patience24；
- `official_test_mode=false`、`final_evaluation_split=val`。

随后重新计算four-H validation mean MSE，并要求与training log best row及artifact ledger
一致；checkpoint SHA256必须与pre-test ledger相同。任何Traceback/OOM/NaN/Inf、缺失
artifact、重复checkpoint hash或提前出现test artifact都会阻止formal test。

## 5. 代码—协议一致性

[Fact] H4N沿用generic HPO runner的同一ISCF-BSCA forward、L1 prediction objective、
four-H validation selector和test-disabled training path。

[Boundary] dataset-specific HPO gain只能支持Weather paper-facing effectiveness，不能
归因到ISCF、allocation或BSCA mechanism；正式归因仍需matched ablation/transfer。

[Falsification] 若dry-run不再显示`test_jobs=0`、任一effective profile重复、target hash
变化、40/40 manifest不完整或checkpoint hash在test前后变化，则本轮结果不可进入主表。

## 6. Formal-test artifact flow

Manifest仅从`h4n_artifact_audit/trial_ledger.jsonl`中的40个`validation_complete` rows
生成，不扫描`_resource_smoke`，因此不会把smoke checkpoints计入正式矩阵。Generic evaluator
对每个checkpoint先写`test_audit/_tmp/<trial>.worker-<gpu>`，验证720-row dense metrics、
test split provenance、candidate/trial/profile/seed identity、diagnostic NPZ与checkpoint hash后，
才原子移动至最终trial目录。40个published directories全部通过后，才允许生成160-row
standard-horizon scorecard并进入229-trial joint selector。
