# SC-ISCF-PSA-D1 v0.1 Diagnostic Protocol Repair

## 1. Decision

Decision=`diagnostic_protocol_fault_predecision_repair_frozen_training_continues`。

ETTh1 training与four-horizon validation metrics正常完成，但post-training validation diagnostic evaluator因D1 config缺少
`diagnostic_protocol.future_bins`触发`KeyError: 'step7b_protocol'`。该错误发生在checkpoint/metrics生成之后、probe forward
之前；没有partial H2/H3 result，也没有official-test access。

这是evaluator config compatibility fault，不是model、optimization或hypothesis failure。已冻结v0.1 evaluator-only repair；
existing checkpoints不重训、不修改，active Weather/ETTm1 jobs不受干扰。待所有training processes结束后才remote pull，
随后对five checkpoints补做validation diagnostic replay。

## 2. Failure boundary

| Layer | Status |
| --- | --- |
| model training | ETTh1 complete；others continue |
| checkpoint selector | mean validation MSE over four horizons，complete for ETTh1 |
| standard validation metrics | complete for ETTh1 |
| diagnostic forward | failed before output due missing config key |
| checkpoint mutation | none |
| official test | 0 |
| H2/H3 decision | not evaluated |

Failure attribution=`diagnostic_protocol_fault_predecision`，不是`optimization_or_numeric_pathology`，也不得用于research
direction decision。

## 3. Exact repair

Config version更新为`SC-ISCF-PSA-D1-control-v0.1-protocol-repair`，只增加：

1. evaluator使用的single `training_contracts` entry；
2. 与RSCC/SCC probes完全相同的eight `diagnostic_protocol.future_bins`；
3. `validation_diagnostic_replay_authorized=true`；
4. protocol revision说明。

未改变：datasets、seed、arm、architecture、objective、rank、optimizer、epochs、selector、training command、references、
comparisons、decision gates和test boundary。

## 4. Replay runner

新增`scripts/remote/run_stage_c_iscf_psa_d1_diagnostics.sh`：

- 只读取已存在的5个checkpoint与validation metrics；
- evaluator固定`evaluation-split=val`；
- three-GPU dataset scheduling；
- 每个checkpoint前后计算SHA256，必须完全相同；
- 输出exact policy/arms/targets probes与trained invariants；
- 不调用training entrypoint；
- authorization false或test true时exit 3。

## 5. Verification and execution boundary

Local v0.1 JSON parse、checker、analyzer synthetic branches、`py_compile`、runner `bash -n`与`git diff --check`通过。

Remote execution顺序冻结为：

1. 等待current supervisor与所有training child processes结束；
2. 确认5/5 checkpoints与20/20 standard validation cells存在；
3. remote fast-forward到repair commit；
4. 再次GPU preflight；
5. 执行five validation diagnostic replays；
6. 检查5/5 SHA nonmutation与no-test invariants；
7. 最后运行一次full analyzer。

在第7步前不读取partial metrics作attribution，不修改任何gate。
