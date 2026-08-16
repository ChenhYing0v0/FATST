# iTransformer-style Decoder-HPO v2 Formal Audit 代码说明

## 1. 功能边界

本次没有修改model forward或训练图，只新增manifest-gated official-test执行与完整test-tuned结果汇总。Formal runner不训练、不写checkpoint，也不允许partial profile selection。

## 2. Runner

`scripts/remote/run_iscf_bsca_decoder_transfer_itransformer_hpo_v2_formal.sh`首先验证formal protocol、training protocol、profile、diagnostic design和70-row manifest的SHA256。每个job读取一个validation-selected checkpoint，在test loader上生成dense H1--H720 metrics、diagnostics和invariants；论文scorecard只抽取H96/H192/H336/H720。Checkpoint在evaluation前后重新hash，任何mutation使完整queue失败。

三GPU worker按manifest索引交错分配70个jobs。`STATUS_ONLY=1`只统计完整formal artifacts；`DRY_RUN=1`只验证70个checkpoint与打印执行矩阵，不访问test loader。

Variable-scope HPO要求formal invariant使用checkpoint的effective config作为scope contract。Evaluator因此比较`model_diagnostics(model)["pcsd_scales"]`与`adapter["pcsd_scales"]`；旧实现比较全局diagnostic design中的默认scales，会错误拒绝已预注册的`p12/p13` alternate-scope profiles。对于早期PatchTST HPO checkpoints，legacy `effective_config.adapter`尚未序列化该字段，但同一次训练保存的`initialization_contract["pcsd_scales"]`包含exact instantiated scopes；checker仅在adapter字段缺失时回退到该值。该兼容路径不改变model、checkpoint、test loader或metrics。

## 3. Result builder

`scripts/build_iscf_bsca_decoder_transfer_itransformer_hpo_v2_results.py`读取：

1. 70-row immutable training manifest；
2. 70个formal artifact directories；
3. 已完成v1 formal block中的BSCA reference与Original Decoder cells。

它先形成280个new cells，再追加20个v1 BSCA reference cells得到300-cell candidate pool。每个dataset对15个profiles分别计算four-H mean MSE/MAE，并按`mean MSE -> mean MAE -> decoder parameters -> profile id`选一个shared profile。输出保留全部300 candidate cells、75个dataset-profile means、5个selected profiles和相对Original的40-cell comparison。

## 4. Claim boundary

该结果是显式`test_informed/test_tuned` evidence。一个profile必须同时服务同一dataset的四个horizons，禁止per-H/per-cell拼接。若selected BSCA超过Original Decoder，结果仍只构成performance evidence；BSCA attribution需要在相同selected profiles下补训matched `+ISCF` controls。
