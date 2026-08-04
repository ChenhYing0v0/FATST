# Stage C ISCF-BSCA-MAIN-v1 H4L HPO Tooling

## 1. Functional boundary

H4L没有修改model code。`configs/iscf_bsca_main_v1_hpo_wide_h4l.json`冻结48个ETTm2/Weather profiles；`scripts/check_iscf_bsca_main_v1_hpo_wide_h4l.py`验证matrix、source与历史noncollision；`scripts/remote/run_iscf_bsca_main_v1_hpo_wide_h4l.sh`仅向通用HPO runner固定config和repo-external output root。

训练完成后，`scripts/check_iscf_bsca_main_v1_h4l_training_artifacts.py`逐trial复核checkpoint、validation metrics、effective config、environment、initialization contract、diagnostics、日志与冻结hash；`scripts/build_iscf_bsca_main_v1_h4l_test_manifest.py`只在48/48 ledger均为`validation_complete`时生成formal-test前manifest。二者都不读取official-test labels。

## 2. Config materialization

每个job先复制dataset anchor，再应用`overrides`，最后加入`trial_id/profile_id/source_prior`。通用runner将resolved profile转换为：

`[B, L, C] -> TimeAlign-style token-MLP encoder -> ISCF scope field -> BSCA-trained fused forecast [B, 720, C]`。

H4L只改变`L`、patch token count、encoder widths/dropout/LN、optimizer regularization与ISCF decoder `mode_rank`；encoder mode、readout mode、scope scales、partition、objective与forecast tensor contract不变。每个trial用validation上四个prefix的mean MSE选择checkpoint，训练后只写validation artifacts，不构造official-test loader。

## 3. TimeAlign parameter inspiration

Checker对本地official ETTm2/Weather scripts与`train_repo.py`记录SHA256，并核对四个source-inspired jobs保留相应encoder tuple。其余rank、learning rate或weight decay属于H4L recombination。该约束防止把“借鉴TimeAlign参数”误写成复制TimeAlign head或alignment objective。

## 4. Safety checks

Checker验证：48个IDs与profiles唯一；ETTm2/Weather各24；与H1--H4K 117个effective profiles零重复；patch count整除context；effective batch恒为32；wide coverage达到冻结边界；architecture与selection invariants不变；dry-run显示`jobs=48, test_jobs=0, remote_authorized=true`。Formal test authorization必须保持false。

Post-training checker进一步要求config/search-space hash、trial/profile ID、seed、four-H validation selector、`official_test_mode=false`、`final_evaluation_split=val`、所有effective HPO fields与冻结job逐项一致；48个checkpoint SHA256必须唯一并与synced ledger一致，train logs不得出现Traceback、RuntimeError、OOM、NaN或Inf，`test_audit`目录必须不存在。

`training_log.csv`记录epoch内聚合的`val_mean_mse`，而最终`metrics_by_target_horizon.csv`由best checkpoint重新评估后聚合；两条数值路径允许`1e-7`绝对误差。H4L观察到的最大差异为`7.72e-8`，该容差只吸收floating-point/batch aggregation误差，不允许best-epoch或checkpoint identity改变。Ledger值直接从最终four-H metrics计算，仍要求`1e-12`一致。

用户于2026-08-04授权H4L完整formal test后，`configs/iscf_bsca_main_v1_hpo_wide_h4l_test_audit.json`冻结48 checkpoints、192个standard-horizon rows、MSE/MAE、一次完整访问及dataset-level shared-profile selection边界。`scripts/check_iscf_bsca_main_v1_hpo_wide_h4l_test_audit.py`核对manifest hash、phase/dataset/seed、authorization与禁止per-horizon/per-metric selection的约束；`scripts/remote/run_iscf_bsca_main_v1_hpo_wide_h4l_test_audit.sh`只为通用atomic test runner固定H4L config、manifest和repo-external test root。Runner在test loader构造前复核remote commit、GPU占用、48个checkpoint hashes与零target/temp artifacts；每个trial完成后再次核对checkpoint hash，再将完整临时目录atomic publish到`test_audit`。

Code-theory consistency：H4L实现的是frozen architecture内更宽的finite HPO search，而不是新的mechanism。若性能提升，只能归因于dataset-level hyperparameter selection，不能扩张ISCF/BSCA mechanism claim；若失败，也只说明该search contract未找到更优profile，不否定architecture方向。
