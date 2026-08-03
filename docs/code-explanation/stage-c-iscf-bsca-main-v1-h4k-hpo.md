# Stage C ISCF-BSCA-MAIN-v1 H4K HPO Tooling

## 1. Scope

H4K只增加targeted HPO config、static checker与remote wrapper，不修改model forward、loss、ISCF/BSCA tensor path、scale set或inference graph。全部training继续走`baselines/timealign_official/train_repo.py`现有的`ISCF-BSCA-MAIN-v1`路径。

## 2. Profile materialization

`configs/iscf_bsca_main_v1_hpo_targeted_h4k.json`以8个已完成winner profiles作为anchors。每个job按`base_profiles[base_profile_id] -> overrides -> trial metadata`解析；resolved job包含`seq_len`、`patch_num`、`d_model`、`d_ff`、dropout、optimizer、batch/accumulation、`mode_rank`与budget。

Checker递归解析H1/H2/H3A/H4J source configs，并逐字段验证8个H4K anchors与其来源trial一致，避免手工转录导致参数漂移。它还验证24个trial IDs不与既有93 trials重叠、dataset count=2/2/2/8/6/2/2、patch divisibility、source artifact SHA256、test jobs=0和formal-test未授权边界。

## 3. Training flow

`scripts/remote/run_iscf_bsca_main_v1_hpo_targeted_h4k.sh`只为generic HPO runner绑定H4K config和remote output root。每个trial的tensor路径仍为：

```text
history [B, 720, C]
  -> TimeAlign-token-MLP encoder
  -> ISCF scope forecasts
  -> BSCA allocation/fusion
  -> unified forecast [B, 720, C]
  -> crop H96/H192/H336/H720 for validation metrics
```

Trainer用four-H mean validation MSE进行trial内early stopping和checkpoint selection，`official_test_mode=false`，最终split为validation。Resource smoke只运行2个train batches、2个validation batches和1 epoch，不产生可用于performance结论的结果。

## 4. Code-theory consistency

Intended contract是“围绕已观测弱点搜索普通hyperparameters，同时保持一个dataset-level shared profile服务四个horizons”。代码通过frozen architecture flags、完整four-H checkpoint selector、禁止per-H/per-metric selection和test=0 training runner落实该合同。

ETTm2、Weather和H720只是search-prior标签，不改变loss权重或profile granularity；因此H4K不能通过针对单cell的训练目标或结果拼接兑现gate。Formal effectiveness仍需24/24 checkpoints后另行冻结manifest并完成完整official-test audit。
