# PCSD-CF Milestone Test Audit 代码说明

## Functional Modules

本次不修改model forward与checkpoint。代码变化只扩展冻结checkpoint evaluation、aggregate analysis和remote
scheduling。

### Checkpoint evaluator

`scripts/evaluate_stage_c_pcsd_cf_checkpoint.py`新增`--evaluation-split {val,test}`。默认`val`保持历史行为；
`test`模式执行以下流程：

```text
effective_config + checkpoint.pt
  -> reconstruct frozen TimeAlign/PCSD model
  -> sequential official test loader
  -> fused [B,T,C]
  -> error [B,T,C]
  -> step SSE/SAE [T]
  -> cumulative prefix MSE/MAE for H=1..720
```

对于PCSD readouts，还生成denormalized arms`[B,T,S,C]`与policy`[B,C,T,S]`，按冻结八个future bins保存
row-level errors和usage。A6/M0/dense没有arm arrays，但仍保存fused与persistence diagnostics。

新CSV列：

- `target_horizon`：prefix长度$H$；
- `mse/mae`：所有test sample × channel × 前$H$ positions上的mean error；
- `num_rows_channels`：sample-channel row数量；
- `evaluation_split`：固定为`test`；
- `checkpoint_policy`：必须为`best-val`；
- `candidate_version`：固定`SC1-PCSD-CF-v1`。

`test_audit_invariants.json`记录checkpoint SHA-256、`checkpoint_retrained=false`、test授权、split、prefix equality、
finite/readout/protocol contracts。evaluator从不调用optimizer或`torch.save`。

### Aggregate analyzers

`analyze_stage_c_pcsd_cf_step7b.py`与`analyze_stage_c_pcsd_cf_step7b_deep_dive.py`新增
`--evaluation-split test-audit`。该mode只切换metrics、diagnostics和invariant文件名，复用validation访问前冻结的
arms、comparisons和thresholds；aggregate输出写入独立test audit目录，不覆盖validation结果。

deep dive继续生成逐H relative-gain curves、fixed-scope summary、same-run oracle及DIRECT-arm-vs-fixed表，
但所有source tensors均来自`pcsd_test_audit_diagnostics.npz`。

### Remote runner

`scripts/remote/run_stage_c_pcsd_cf_test_audit.sh`遍历12 × 5 frozen run directories。三个GPU只执行forward
evaluation；每个run前后用`sha256sum`检查`checkpoint.pt`未变化。runner支持`DRY_RUN=1`、`STATUS_ONLY=1`与
resume，aggregate完成前不产生research decision。

### Prelaunch checker

`scripts/check_stage_c_pcsd_cf_test_audit.py`读取machine-readable audit config，并检查60-run集合、test授权、
no-retraining/hash/complete-matrix gates、runner中不存在training entrypoint，以及cumulative prefix metric的精确性。
它输出`analysis/stage_c_pcsd_cf_test_audit_prelaunch_20260716/local_gate.json`；21个cases必须全部通过。

## Code-Theory Consistency

- Intended rule：test是milestone effectiveness primary gate，但不能选择checkpoint或触发隐式retraining。
- Code realization：training config仍声明`final_evaluation_split=val`，test audit使用独立CLI和artifacts；checkpoint
  hash前后一致且invariant固定`checkpoint_retrained=false`。
- Remaining proxy：一次seed2021 test audit仍不能证明multi-seed stability；positive只授权confirmation design。
- Falsification：任一checkpoint缺失/变更、matrix不足60、split不为test、best-val contract失效或artifact nonfinite，
  整个test audit不得形成method结论。
