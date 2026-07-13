# Phase5 Clean A6 Rerun Analyzer Explanation

本文档解释 `scripts/analyze_phase5_a6_clean_operator_rerun.py`。该脚本只用于验证移除 A6
future reconstruction/alignment branch 后，active A6-LBF-r256 是否仍能作为 StageA paper-core evidence。

## Inputs

默认输入：

- clean rerun:
  `analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706/raw/`;
- historical A6-LBF-r256:
  `analysis/phase5_timealign_hss_a6_capacity_native_gate_20260703/raw/`;
- official TimeAlign reference:
  `analysis/phase5_timealign_hss_a3e_ettm1_replacement_gate_20260701/raw/`.

脚本只读取：

- `metrics_by_target_horizon.csv`;
- `training_log.csv`;
- `effective_config.json`.

它不读取 checkpoint，也不读取 prediction arrays。

## Metrics

每个 comparison row 定义为一个 `(dataset, target_horizon)` setting。

- `clean_mse` / `clean_mae`: clean A6 rerun 在该 setting 的 test metric；
- `reference_mse` / `reference_mae`: reference run 的 test metric；
- `relative_mse_pct = (clean_mse / reference_mse - 1) * 100`;
- `relative_mae_pct = (clean_mae / reference_mae - 1) * 100`;
- `clean_wins_mse`: `clean_mse < reference_mse`.

summary CSV 对每个 dataset 和 `ALL` 聚合：

- `settings`: 可比较 setting 数；
- `mse_wins`: clean A6 的 MSE wins 数；
- `mean_relative_mse_pct`: relative MSE 的算术平均；
- `mean_relative_mae_pct`: relative MAE 的算术平均。

## Gate

报告 `clean_a6_rerun_report.md` 的 decision 规则：

- `clean_a6_validated`: clean A6 相对 historical A6 的整体 mean relative MSE 绝对值不超过 `0.5%`，
  且相对 fixed TimeAlign、official unified TimeAlign 都有负 mean relative MSE 和至少 `7/12` MSE wins；
- `clean_a6_effective_but_not_identical_to_historical`: clean A6 仍超过 fixed/unified reference，但相对
  historical A6 漂移超过 `0.5%`;
- `clean_a6_needs_recheck_or_rollback`: clean A6 不再稳定超过 fixed/unified reference；
- `incomplete`: 任一 reference 缺失，不能完整判断。

## Code-Theory Consistency

Intended theory: clean A6-LBF-r256 的贡献应来自 learned-basis forecast operator，而不是 inherited
TimeAlign future reconstruction/alignment branch。

Code realization:

- `effective_config.json` 中读取 `official_args.w_recon` 和 `official_args.w_align`，确认实际 loss 权重；
- 比较 clean rerun 与 historical A6，判断 code cleanup 是否改变经验结论；
- 比较 clean rerun 与 fixed/unified TimeAlign，判断 Contribution 1 evidence 是否仍成立。

Remaining proxy:

- 单 seed rerun 不能证明跨 seed variance；
- historical A6 使用旧代码路径，clean rerun 与 historical A6 的小差异可能包含初始化顺序变化和训练随机性。

Falsification evidence:

- clean rerun 相对 fixed/unified reference 不再有负 mean relative MSE；
- clean rerun 相对 historical A6 漂移超过阈值且集中在多个 dataset/horizon；
- effective config 显示 A6 仍在使用 non-zero future auxiliary losses。
