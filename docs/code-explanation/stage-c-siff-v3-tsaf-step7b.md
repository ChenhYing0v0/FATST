# Stage C SIFF-v3 TSAF Step7B Tooling Explanation

## 1. Configuration contract

`configs/stage_c_siff_v3_tsaf_step7b.json`将formal comparison拆成9个effective arms：4个完整历史references与5个
new-training arms。runner只展开`launch_order`中的25个new runs，避免重训已hash-frozen的references。

每个job固定`pred_len=720`、pointwise full prediction loss、`equal_skill` training contract、validation
H96/H192/H336/H720 mean MSE selector与`final_evaluation_split=val`。runner不包含test evaluator调用。

## 2. Runner flow

`run_stage_c_siff_v3_tsaf_v1.sh`的job row为：

`dataset, arm, readout_mode, policy_mode, objective_mode, mode_rank, profile, patch_num, d_model, d_ff`。

该row进入`train_repo.py`：

1. profile fields构造history encoder；
2. `readout_mode`选择ordered或independent SIFF arm generator；
3. `policy_mode`选择categorical/TSAF/permuted/global allocation；
4. `mode_rank`只对independent control使用dataset-matched rank；
5. validation selector保存`checkpoint.pt`，训练结束只生成validation artifacts。

`DRY_RUN=1`始终允许并打印25 jobs。normal launch先读取config中的`remote_training_authorized`；当前为false时exit 3。
`RESOURCE_SMOKE=1`也受同一authorization保护。

## 3. Local prelaunch checker

`check_stage_c_siff_v3_tsaf_step7b.py`生成：

- `cases.csv`：15个contract cases；
- `jobs_seed2021.csv`：25个new jobs；
- `reference_runs.csv`：20个reused references及checkpoint hashes；
- `initialization_pairing.csv`：逐dataset encoder/TSAF parameter hash与capacity gap；
- `early_gradient.csv`：五个new arms的two-step gradient；
- `prelaunch_gate.json`：10-category summary与authorization boundary。

checker通过真实`TimeAlign.Model` production constructors计算`active_forward_parameters`。independent rank不沿用旧direct
parent ranks，而是最小化其与TSAF active parameters的relative gap。

## 4. Two-step gradient semantics

TSAF scalar output为zero-init，因此第一个backward中`target_allocation_projection`与
`scale_allocation_projection`可以为零，但scalar output必须有gradient。checker先执行一次SGD step，再执行第二个backward，
要求policy input projections为finite/nonzero。categorical与independent static-target controls采用同一逻辑检查
`policy_hidden`。

该检查证明zero-init没有永久阻断allocation path；它不证明训练稳定性或预测gain。

## 5. Four-layer analyzer

`analyze_stage_c_siff_v3_tsaf_v1.py`从两个roots组装effective matrix：

- `new_root`读取5个new arms；
- `reference_root`按`source_arm`读取4个historical arms；
- references逐checkpoint SHA256与config冻结值核对；
- 45/45 runs与180/180 test cells完整后才计算comparison；
- 输出run audit、cell/summary comparisons、mechanism health与four-layer decision。

internal health使用evaluator的`probe_arms`、`policy_row_bin_usage`、`scale_component_contribution`及prefix invariants。
这些诊断不能覆盖negative official-test effectiveness。

## 6. Code-theory consistency

- Intended theory：TSAF只改变shared target-scale allocation bias，history仍经SIFF arms决定forecast content；
- Realized tooling：candidate与controls共享profiles、seed、objective、selector和test scorecard；new arms全部from-scratch joint
  training；
- Capacity control：independent ranks为`109/115/115/106/115`，active-parameter gap不超过0.4%；
- Proxy：synthetic analyzer只验证gate logic，不是method evidence；
- Falsification：candidate必须同时超过A6_MEASURE、parent、categorical、permuted和global；否则按冻结failure map rollback；
- Held：remote resource smoke、25-run training、official test与confirmation均未执行。
