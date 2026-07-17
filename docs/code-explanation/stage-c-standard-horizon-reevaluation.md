# StageC 标准 Horizon 重评估脚本说明

## 1. Functional module

`scripts/analyze_stage_c_post_pcc_standard_horizons.py`把已经完成的SIFF/MCCA Step 7B validation artifacts转换为
新的paper-facing四horizon scorecard。脚本只读旧CSV，不训练模型、不选择checkpoint、不访问test。

项目默认协议同时以`configs/paper_facing_evaluation_protocol.json`固定horizon、MSE/MAE、split职责、
checkpoint default、dense evidence边界与test freeze fields。该JSON是后续runner/analyzer生成candidate-specific
config时的上层约束，不会改变历史artifacts。

输入：

```text
artifact root
  ├── <new arm>/<dataset>/h720_full/seed2021/metrics_by_target_horizon.csv
  └── references/{pcsd_cf,pcc}/<arm>/<dataset>/h720_full/seed2021/
      └── metrics_by_target_horizon.csv
```

每个source CSV必须完整包含H1..720。脚本验证完整性与finite values后，只抽取H96/H192/H336/H720。

## 2. Computation flow

```text
16 relevant arms × 5 datasets × dense horizon rows
  -> validate H1..720 completeness
  -> select H96/H192/H336/H720 MSE/MAE
  -> build lookup[(dataset, arm, horizon)]
  -> compute paired relative gains
  -> average matched factors for SIFF and MCCA main effects
  -> aggregate by dataset, horizon and full matrix
  -> apply original frozen gates
```

`gain_percent(candidate, reference)`计算：

$$
100\left(1-\frac{L_{\mathrm{candidate}}}{L_{\mathrm{reference}}}\right).
$$

`architecture_main_effect`对EQUAL/PCC/MCCA三组matched SIFF-vs-PCSD gains在每个
dataset-horizon cell内平均；`mcca_main_effect`对PCSD/SIFF两个carrier内的MCCA-vs-PCC gains平均。这样不会把
不同training mode或carrier的raw loss直接混合。

## 3. Output columns

### `standard_horizon_metrics.csv`

- `dataset/arm/horizon`：source artifact identity；
- `mse/mae`：原dense curve对应horizon的validation metric。

### `standard_horizon_effects.csv`

- `effect`：paired comparison或factorial main effect；
- `candidate/reference`：比较两侧；composite effect使用matched-factorial标记；
- `mse_gain_percent/mae_gain_percent`：该dataset-horizon cell的relative gain；
- `factor_count`：该cell平均的paired comparisons数，普通comparison为1。

### `standard_horizon_breakdown.csv`

- `aggregation_axis`：`dataset`或`horizon`；
- `group`：具体dataset或horizon；
- `macro_*_gain_percent`：该组cells的等权gain均值；
- `cell_wins/cells`：MSE gain大于0的cell数与总数。

### `standard_horizon_summary.csv`

- `macro_*_gain_percent`：完整paper-facing matrix的等权gain；
- `cell_wins/cells`：完整matrix正向cells；
- `dataset_wins/datasets`：先跨四horizon平均后为正的dataset数；
- `worst_dataset*`：最差dataset及其四horizon MSE gain均值。

### `standard_horizon_gate.json`

记录evaluation split、test访问、paper horizons、继承的checkpoint rule、是否重新选checkpoint、threshold来源、
cell wins职责和最终gate。它明确把本次结果标为`retrospective_development_screen`。

## 4. Consistency boundary

- Intended theory：常规开发与论文主表使用相同四horizon，dense metrics降为diagnostic。

- Code realization：脚本只对四个冻结horizon计算paired scorecard，并沿用原Step 7B gain/dataset-win thresholds。

- Proxy boundary：旧run只保存best-H720 checkpoint，因此本次没有实现新默认
`mean validation MSE over four horizons` checkpoint selection。它评估的是旧state在新scorecard下的表现。

- Falsification：若source curve缺horizon、包含non-finite value、arm/reference不完整，脚本直接失败；若未来按新
checkpoint rule重跑后排序反转，应记录`checkpoint_selection_reversal`，不能覆盖本次历史结果。
