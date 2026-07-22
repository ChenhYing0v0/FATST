# Stage C ISCF-SCC Step9 Analyzer

## 1. Reader path

`scripts/analyze_stage_c_iscf_scc_step9.py`按`artifact audit -> validation effectiveness -> matched attribution ->
internal health -> failure attribution`顺序读取20个new runs与5个historical EQUAL parent。它只分析validation artifacts，
不加载test split、不选择checkpoint或修改模型。

## 2. Outputs and statistics

- `run_audit.csv`：artifact completeness、objective contract、checkpoint SHA256、initialization pairing与split role；
- `validation_metrics.csv`：来源于每run `metrics_by_target_horizon.csv`的100个arm/dataset/horizon cells；
- `comparison_cells.csv`：SCC相对EQUAL/FUSED/ARMERR/SHUFFLED的逐cell MSE/MAE relative gain；
- `comparison_summary.csv`：20-cell mean relative gain、cell/dataset/horizon wins；
- `internal_health.csv`：从`probe_arms/fused/targets/direct_policy`重算coalition oracle headroom、policy-credit Spearman、
  policy entropy、scope usage和positive contributors；
- `training_health.csv`：epochs、best validation score、five-scope gradient、credit-policy KL/alignment与entropy；
- `decision.json`：按冻结Step7B gates形成machine decision与failure attribution。

relative gain统一定义为`100 * (1 - candidate/reference)`；正值表示SCC更好。internal target-visible headroom只作
mechanism health，不能覆盖negative validation effectiveness。

## 3. Code–theory consistency

analyzer明确区分“credit signal存在”“policy对齐”“forecast gain兑现”。若SCC finite且gradient健康，但同时输给EQUAL及三个
matched controls，则归因为tested intervention point failure，而不是numeric pathology，也不能用oracle diagnostics救回候选。
