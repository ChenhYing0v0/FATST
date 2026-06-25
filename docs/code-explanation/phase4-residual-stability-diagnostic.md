# Phase4 Residual-Stability Diagnostic

## Purpose

`scripts/analyze_phase4_residual_stability_diagnostic.py` 服务于 Phase4-RG 的回退步骤：
在 fixed late adapter routing 失败后，重新判断 selected future units 中哪些是
learnable conflict，哪些更像 noisy conflict。

它不是新的 training strategy，也不评价模型预测结果。它只使用 train split 的 label
sequence 构造 horizon-agnostic supervision router 的设计证据：

> 高 novelty 的 future unit 不一定都值得更强监督；如果 residual 对简单 baseline 仍高频震荡，
> 它更可能是会污染 shared representation 的 noisy conflict。

## Data Flow

脚本入口是 `main()`，默认分析 `ETTh2` 和 `Weather`，使用 `seq_len=336`、
`pred_len=720`、`block_size=48`。每个 dataset 只读取 train split：

1. `ForecastDataset` 加载 train windows。
2. `chunk_targets` 取每个 window 的 720-step future label。
3. `persistence_ref` 使用 history 最后一个时间点。
4. `seasonal_reference` 构造 seasonal baselines，默认 period 为 `24,48,96,168`。
5. 每个 48-step block 分别计算 residual metrics。

对应代码位置：

- [scripts/analyze_phase4_residual_stability_diagnostic.py](/Users/river/PaperResearch/Project/R_2026_FATST/scripts/analyze_phase4_residual_stability_diagnostic.py:152)
- [scripts/analyze_phase4_residual_stability_diagnostic.py](/Users/river/PaperResearch/Project/R_2026_FATST/scripts/analyze_phase4_residual_stability_diagnostic.py:189)

## Metric Definitions

`block_metrics` 对每个 block 构造 persistence 与 seasonal residual stack：

$$
r_{b,k}=y_b-\hat{y}_{b,k}^{baseline}.
$$

其中 baseline candidates 包括 persistence 和所有可用 seasonal periods。脚本选择 block 内
MSE 最低的 baseline 作为 `best_baseline`。

输出统计量含义：

- `novelty_mse`: target block 相对 persistence 的 MSE，用于近似 label novelty。
- `best_baseline_mse`: target block 相对最佳 persistence/seasonal baseline 的 MSE。
- `best_gain_over_persistence`: `novelty_mse / best_baseline_mse`。数值越大，说明比 persistence
  更强的 baseline 能解释更多结构。
- `local_variation`: target block 一阶差分能量。
- `residual_smoothness`: best residual 一阶差分能量除以 `best_baseline_mse`。数值越高，说明
  即使扣除简单 baseline，残差仍更高频。
- `residual_bias_share`: residual mean square 占 `best_baseline_mse` 的比例，用于观察 residual
  是偏置型还是波动型。

对应代码位置：

- [scripts/analyze_phase4_residual_stability_diagnostic.py](/Users/river/PaperResearch/Project/R_2026_FATST/scripts/analyze_phase4_residual_stability_diagnostic.py:69)

## Selection And Buckets

脚本先在每个 train window 内按 `novelty_mse` 选择 top `25%` blocks，模拟 S1/S2/RG-A
关注的 high-novelty future units。随后 `classify_selected_units` 只在 selected units 内做
dataset-relative bucket：

- `learnable_conflict`: `best_gain_over_persistence >= gain_q60` 且
  `residual_smoothness <= smooth_median`。
- `noisy_conflict`: `residual_smoothness > smooth_median` 且
  `local_variation >= variation_median`。
- `ambiguous_conflict`: 其余 selected units。

这个 bucket 是 diagnostic proxy。它的用途是判断 fixed late route 是否过粗，而不是直接定义
最终训练中的硬规则。

对应代码位置：

- [scripts/analyze_phase4_residual_stability_diagnostic.py](/Users/river/PaperResearch/Project/R_2026_FATST/scripts/analyze_phase4_residual_stability_diagnostic.py:125)
- [scripts/analyze_phase4_residual_stability_diagnostic.py](/Users/river/PaperResearch/Project/R_2026_FATST/scripts/analyze_phase4_residual_stability_diagnostic.py:215)

## Output Files

默认输出目录：

`analysis/phase4_residual_stability_diagnostic_20260625`

输出文件：

- `phase4_residual_stability_block_summary.csv`: block-level selected/nonselected residual statistics。
- `phase4_residual_stability_bucket_summary.csv`: dataset-region-bucket 聚合结果。
- `phase4_residual_stability_selection_summary.csv`: selected units 在 early/middle/late region 的分布。
- `phase4_residual_stability_diagnostic_report.md`: 11-step 记录、关键表格与下一步设计决策。

报告生成逻辑在：

- [scripts/analyze_phase4_residual_stability_diagnostic.py](/Users/river/PaperResearch/Project/R_2026_FATST/scripts/analyze_phase4_residual_stability_diagnostic.py:348)

## Code-Theory Consistency

[Theory] SRP 给出的关键启发是 multi-step forecasting 的不同 future supervision
可能需要不同 representation update path。Phase4 进一步把问题从 step-specific 参数扩展到
horizon-agnostic supervision scheduling：training 不应只决定监督强度，还应决定 gradient
允许更新哪里。

[Code] 本脚本不直接实现 gradient routing，而是用 train-side label residual stability 判断
selected high-novelty units 是否可被简单 baseline 解释为结构性 residual，或更像高频 noisy
residual。

[Proxy] `learnable_conflict` / `noisy_conflict` 依赖 dataset-relative quantiles。它适合做
diagnostic 和 method design evidence；进入 online training 时需要重新考虑 batch-level
threshold 的稳定性、trace 可解释性，以及是否应加入 warmup。

[Falsification] 如果 Weather late selected units 没有高 noisy-conflict share，或者 ETTh2
没有保留 material learnable-conflict subset，那么 RG-A 失败就不能被解释为 fixed late route
过粗，下一步应回退到其他 conflict proxy，而不是设计 residual-stability router。
