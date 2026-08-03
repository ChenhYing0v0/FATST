# Stage C ISCF-BSCA-MAIN-v1 H4J HPO Tooling

## 1. Scope

本次更新只扩展HPO config materialization、既有trial审计和H4J prelaunch checker，不修改model forward、loss、ISCF/BSCA tensor path、scale set或inference graph。训练仍调用`baselines/timealign_official/train_repo.py`中的既有`ISCF-BSCA-MAIN-v1`路径。

## 2. Config materialization

`configs/iscf_bsca_main_v1_hpo_joint_h4j.json`使用`base_profiles`保存12个已解析的dataset profile anchors。每个job通过：

```text
base_profiles[base_profile_id]
  -> apply job.overrides
  -> apply trial_id/dataset/profile_id/source_prior
  -> resolved job
```

生成完整effective HPO job。`scripts/remote/run_iscf_bsca_main_v1_hpo.sh`和`scripts/analyze_iscf_bsca_main_v1_hpo.py`使用相同precedence；`base_profiles`也进入search-space SHA256，防止只改anchor但保持job overrides不变时hash漏检。旧config的`base_config + base_trial_id`路径保持兼容。

Resolved job继续映射到trainer CLI的`seq_len`、`patch_num`、`d_model`、`d_ff`、`dropout`、learning rate、weight decay、batch/accumulation、`mode_rank`、`layer_norm`、epoch与patience。所有runs输出一个长度720的unified forecast；validation按prefix H96/H192/H336/H720计算MSE/MAE，four-H mean validation MSE选择checkpoint。

## 3. Existing-evidence analyzer

`scripts/analyze_iscf_bsca_main_v1_joint_objective.py`读取H1/H2/H3A/H3B逐cell scorecards和frozen published targets。对每个`dataset × trial`验证四个standard horizons完整，再计算：

- `test_mean_mse_4h`、`test_mean_mae_4h`；
- MSE、MAE及combined leading-cell counts；
- equal-weight `dataset_joint_mean_score`；
- dataset MSE/MAE Pareto frontier；
- current selected count、unrestricted reselection upper bound与1% joint-mean guard内的upper bound。

Exchange没有published target，lead-cell列保持空；其joint score改由within-search per-metric minima归一化。该analyzer不会读取remote checkpoint二进制，也不会生成训练或test结果。

## 4. Prelaunch checker and manifest builder

`scripts/check_iscf_bsca_main_v1_hpo_joint_h4j.py`执行fail-closed static checks：

1. frozen target与existing-evidence SHA256一致；
2. 40个trial IDs唯一且与LPT queue完全相同；
3. dataset counts严格为2/2/2/9/9/4/10/2；
4. ETTm2/Weather/Solar合计28 jobs；
5. 每个resolved job满足`seq_len % patch_num == 0`的encoder protocol invariant；
6. 禁止per-horizon/per-metric selection；
7. success gates固定为MSE>=20/28、MAE>=20/28、combined>=40/56；
8. generic runner dry-run materializes 40 jobs且test jobs=0。

`scripts/build_iscf_bsca_main_v1_h4j_test_manifest.py`先把ledger path解析为absolute path，再记录repo-relative provenance；它只能在40-row training ledger完整后生成manifest。任一artifact/numeric-health失败、checkpoint hash缺失、dataset count不符或trial ID重复都会阻止manifest freeze。正式test runner随后还必须验证checkpoint pre/post hash不变。

H4J test config设置`defer_profile_selection_to_joint_analyzer=true`。因此generic test analyzer只写completeness、ledger、all-trial scorecard与aggregates，不生成legacy MSE-only winner；final selector必须合并H1--H4J全部retained trials后执行。

## 5. Code-theory consistency

Intended contract是“同一dataset只选一个profile服务四个horizons，并联合优化MSE/MAE平均质量与leading-cell coverage”。代码通过dataset-level job、four-H validation checkpoint selector、complete test manifest和禁止metric/H-specific selection落实该合同。

仍需正式结果验证的部分是H4J是否真的把MSE与MAE leading cells同时提高到gate；local dry-run只能证明protocol结构一致，不能证明performance。若40/40 complete test后未达标，正确结论是该frozen H4J search contract未达到目标，而不是选择性拼接profiles或拒绝ISCF-BSCA architecture。
