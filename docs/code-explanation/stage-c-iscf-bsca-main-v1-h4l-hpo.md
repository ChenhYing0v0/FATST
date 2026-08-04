# Stage C ISCF-BSCA-MAIN-v1 H4L HPO Tooling

## 1. Functional boundary

H4L没有修改model code。`configs/iscf_bsca_main_v1_hpo_wide_h4l.json`冻结48个ETTm2/Weather profiles；`scripts/check_iscf_bsca_main_v1_hpo_wide_h4l.py`验证matrix、source与历史noncollision；`scripts/remote/run_iscf_bsca_main_v1_hpo_wide_h4l.sh`仅向通用HPO runner固定config和repo-external output root。

## 2. Config materialization

每个job先复制dataset anchor，再应用`overrides`，最后加入`trial_id/profile_id/source_prior`。通用runner将resolved profile转换为：

`[B, L, C] -> TimeAlign-style token-MLP encoder -> ISCF scope field -> BSCA-trained fused forecast [B, 720, C]`。

H4L只改变`L`、patch token count、encoder widths/dropout/LN、optimizer regularization与ISCF decoder `mode_rank`；encoder mode、readout mode、scope scales、partition、objective与forecast tensor contract不变。每个trial用validation上四个prefix的mean MSE选择checkpoint，训练后只写validation artifacts，不构造official-test loader。

## 3. TimeAlign parameter inspiration

Checker对本地official ETTm2/Weather scripts与`train_repo.py`记录SHA256，并核对四个source-inspired jobs保留相应encoder tuple。其余rank、learning rate或weight decay属于H4L recombination。该约束防止把“借鉴TimeAlign参数”误写成复制TimeAlign head或alignment objective。

## 4. Safety checks

Checker验证：48个IDs与profiles唯一；ETTm2/Weather各24；与H1--H4K 117个effective profiles零重复；patch count整除context；effective batch恒为32；wide coverage达到冻结边界；architecture与selection invariants不变；dry-run显示`jobs=48, test_jobs=0, remote_authorized=true`。Formal test authorization必须保持false。

Code-theory consistency：H4L实现的是frozen architecture内更宽的finite HPO search，而不是新的mechanism。若性能提升，只能归因于dataset-level hyperparameter selection，不能扩张ISCF/BSCA mechanism claim；若失败，也只说明该search contract未找到更优profile，不否定architecture方向。
