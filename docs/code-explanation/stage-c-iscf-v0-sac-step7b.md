# Stage C ISCF-v0 SAC Step7B Code Explanation

## 1. 变更边界

本轮没有修改`TimeAlign.Model`、`SIFFCouplingFieldReadout`或ISCF-v0 parameters。现有SIFF路径已经把CLI的
`pcsd_partition`传入`PCSDCouplingFieldReadout`；`partition=random`只改变registered group-index buffers，不创建参数，也不消耗global model RNG。

新增代码只负责experiment orchestration与analysis：

- `configs/stage_c_iscf_v0_scope_attribution_confirmation.json`：冻结60-run effective matrix、25-run launch manifest、gates、sources与authorization；
- `scripts/check_stage_c_iscf_v0_sac_step7b.py`：local contract/prelaunch checker；
- `scripts/remote/run_stage_c_iscf_v0_sac.sh`：training与formal-test分离的remote runner；
- `scripts/analyze_stage_c_iscf_v0_sac.py`：three-seed source resolution、protocol audit、gain statistics与decision map。

## 2. Tensor and model contract

两条independent paths都接收Encoder输出`hidden [B,C,R]`，产生：

```text
mode_weight [S,D,R,K]
mode_bias   [S,D,K]
hidden      [B,C,R]
--------------------------------
scale_modes [B,C,S,D,K]
arms        [B,C,S,T]
weights     [B,C,T,S]
full        [B,C,T]
output      [B,H,C]
```

canonical与random只在每个scope的`group_indices [T/s,s]`不同。两者的`mode_weight`、synthesis tables、policy与其他parameters完全同形；scale1和720 indices一致，48/144/360不同。Q1-WIDE使用`S_component=1`的shared mode field和更宽rank，其余scope synthesis与policy contract不变。

## 3. Runner flow

runner从config的`launch_order`生成13-column TSV job：

```text
seed,dataset,arm,readout,policy,objective,partition,partition_seed,
rank,profile,patch_num,d_model,d_ff
```

训练flow为：

```text
frozen manifest
  -> validation-only end-to-end training
  -> best four-horizon validation checkpoint
  -> required artifact completeness
  -> wait until 25/25
  -> separately authorized formal-test process
  -> checkpoint hash before/after equality
  -> three-source analyzer
```

`DRY_RUN=1`只打印manifest并跑synthetic analyzer；`STATUS_ONLY=1`只读artifact counts；normal launch先读取config authorization，false时exit 3。`FORMAL_TEST_ONLY=1`还要求25/25 training complete。resource smoke使用two-batch/one-epoch、`final_evaluation_split=none`，并同时扫描`Traceback`、OOM、NaN与Inf；remote没有`rg`时回退`grep`。

## 4. Historical source resolver

analyzer对每个`(arm,dataset,seed)`显式选择root和source alias：

- seed2021 ISCF/Q1/A6：旧attribution root；
- seeds2022/2023 ISCF/A6：FCC root；
- Q1 seeds2022/2023和RANDOM all seeds：SAC new root。

每个run读取validation metrics、test metrics、`effective_config.json`、`test_audit_invariants.json`、`initialization_contract.json`与`model_diagnostics.json`。historical checkpoint hash必须与SHA256-frozen run audit一致；new runs则必须匹配SAC protocol、partition、objective、checkpoint selector和test authorization。

## 5. Statistics

`comparison_rows()`按dataset、seed、horizon构造MSE/MAE gain，并分别汇总dataset/horizon/seed方向。`decide()`不共享一个模糊threshold：Q1读取`+0.5%`，RANDOM读取`+0.3%`；两者再共同应用3/5 dataset、3/4 horizon、2/3 seed及MAE-positive gates。

`internal_health_rows()`逐dataset/seed核对：

- canonical/random Encoder initialization hash equality；
- canonical/random PCSD parameter initialization hash equality；
- partition hash inequality；
- active parameter equality；
- observed Q1 active-parameter gap与preregistered gap相等。

这些health checks只能阻止无效归因，不能救回negative MSE/MAE。

## 6. Code-theory consistency

预期理论是：不同future-output coupling extents可能需要各自的finite-capacity history projection，而canonical contiguous/nested grouping应比任意grouping提供更合适的output-sharing geometry。

代码实现了两个最小反事实：Q1移除独立maps但保留五scope；RANDOM保留独立maps与全部parameters但破坏temporal grouping。它没有实现requested-H adaptation、额外information access、router或second loss。

仍然只是proxy的部分：Q1无法exact match parameter count，RANDOM的optimization difficulty可能不同；因此结果只在当前training protocol与function family内归因。Q1或RANDOM任一primary gate失败都足以证伪对应paper claim；protocol pathology只能要求修复exact experiment，不能方向级拒绝。

## 7. Step8 validation-only extension

training-only阶段不存在`test_audit_*` artifacts，因此analyzer新增`--validation-only`。该模式仍解析完整60-run
three-source matrix并执行effective-config、initialization、parameter和partition audits，但只写validation metrics、
comparison summaries、internal health与`validation_readiness.json`；它不会调用`decide()`，也不会产生official-test
`decision.json`。

A6_FULL不经过PCSD，故其SAC role使用`partition=control`。checker现在只对这个non-PCSD control跳过
`pcsd_partition`比较；所有ISCF canonical/random arms仍要求effective partition与frozen config精确一致。该修正只移除
无语义字段造成的false failure，不改变model、metrics、gates或正式test审计。
