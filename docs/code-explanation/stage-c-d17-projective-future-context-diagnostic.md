# StageC D17 projective future-context diagnostic 代码说明

## 1. 功能边界

本次没有修改模型forward，也没有训练新参数。代码只完成两件事：

1. 从既有SIFF_EQUAL、PCSD_EQUAL checkpoints导出独立validation split的frozen forecast probes；
2. 在validation probes拟合固定ridge residual corrector，并在既有authorized test probes评估future-context
   problem hypothesis。

因此它是`diagnostic_only`，不是model implementation、formal ablation或paper-facing result。

## 2. Validation artifact export

`scripts/evaluate_stage_c_pcsd_cf_checkpoint.py`新增`--artifact-dir`。checkpoint、`effective_config.json`与
initialization contract仍从`--run-dir`读取；新增的validation diagnostics和invariants写到独立目录，避免修改历史
正式run artifacts。默认未传该参数时行为保持不变。

validation运行也会读取`--test-audit-config`中的`training.validation_horizons`与
`training.training_final_evaluation_split`作为protocol contract，但不会授权或访问test。checkpoint在运行前后由
remote runner计算SHA-256，必须完全一致。

## 3. D17 feature flow

输入为：

- validation/test `probe_fused [N,720]`；
- validation/test `probe_targets [N,720]`。

对每个future coordinate $\tau$构造：

1. `pointwise_wide`：当前forecast值，加固定full-domain
   $u_\tau=\tau/(T-1)$的polynomial/Fourier features及交互；
2. `causal_ordered`：追加lags 1/2/4/8/16/32/64/128的forecast值、current-minus-lag及availability mask；
3. `causal_row_shuffled`：feature数相同，但lag context来自另一row；
4. `symmetric_ordered`：追加future leads，只作non-projective upper control。

训练target是validation上的`probe_targets - probe_fused`。所有features只在validation上fit
`StandardScaler + Ridge(alpha=1)`；test labels只用于最终MSE。每个model都先形成完整`[N,720]` correction，再以
prefix crop报告H96/H192/H336/H720。

## 4. Projectivity invariant

coordinate始终以固定$T=720$归一化，不能按当前crop长度重新归一化。causal features分别从full draft与crop draft
重算，共享prefix最大绝对差记录为`prefix_invariance_max_abs_gap`，gate为`<=1e-10`。

首版代码曾按$H-1$归一化coordinate，导致crop gap约6–7；该结果已标记protocol invalid，不进入研究结论。

## 5. Artifacts

remote runner为`scripts/remote/run_stage_c_d17_projective_future_context.sh`。它只运行10个validation inference jobs
（2 carriers × 5 datasets），然后在remote直接产生：

- `transfer_metrics.csv`：每个carrier/dataset/model/horizon的validation-fit/test-eval MSE；
- `aggregate_metrics.csv`：同一cell聚合值；
- `comparison_cells.csv`：parent、pointwise、causal、shuffled与symmetric比较；
- `summary.json`：statistic定义、config hash、六项gate与conditional boundary。

local smoke只证明脚本、shape与projectivity计算可执行，不代表D17结果已通过。
