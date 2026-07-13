# Phase5 StageB B12-STBO Rank Diagnostic Analyzer Explanation

对应文件：

- `scripts/analyze_phase5_stage_b_b12_stbo_rank_diagnostic.py`

## Purpose

该 analyzer 用于分析 B12-STBO rank/capacity diagnostic。它回答一个限定问题：

```text
第一轮 B12-STBO 失败是否主要来自 local basis rank 过低？
```

它不直接支持 paper-core claim。即便高 rank STBO 接近 A6，也必须同时满足：

1. learned shared/bank STBO 接近或超过 A6；
2. learned shared/bank STBO 超过 same-rank DCT；
3. `stbo_bank4` 出现非均匀 bank specialization；
4. 结果不能只由 `stbo_independent` capacity probe 解释。

## Inputs

Analyzer 需要两个 root：

```bash
--raw-root analysis/phase5_stage_b_b12_stbo_rank_diagnostic_20260708/raw
--a6-root analysis/phase5_stage_b_b12_stbo_small_gate_20260708/raw/official-last
```

`raw-root` 包含 rank diagnostic 的四个有效 configs：

- `l48_r32`;
- `l120_r64`;
- `l144_r128`;
- `l360_r256_capacity_probe`。

`a6-root` 提供 clean A6 anchor。A6 不在 rank diagnostic 中重复训练，避免浪费远程时间；该 anchor 已在
B12 small gate 中证明与 clean A6 rerun 完全一致。

## Metrics Source

每个 run 的主指标来自：

```text
${config}/official-last/
  TimeAlignOfficialUnified720_${arm}_official-last/
    ${dataset}/mixed_h96_h192_h336_h720/seed2021/
      metrics_by_target_horizon.csv
```

每行读取：

- `target_horizon`;
- `mse`;
- `mae`。

模型诊断来自：

```text
model_diagnostics.json
```

主要读取：

- `total_parameters`;
- `trainable_parameters`;
- `stbo_tile_len`;
- `stbo_tile_count`;
- `stbo_rank`;
- `stbo_tile_bank_entropy_mean`;
- basis / coeff norm diagnostics。

## Output Files

| File | Meaning |
| --- | --- |
| `b12_stbo_rank_metrics.csv` | A6 和所有 STBO config/arm/dataset/horizon 的原始 MSE/MAE |
| `b12_stbo_rank_comparisons.csv` | relative MSE/MAE comparisons |
| `b12_stbo_rank_summary.csv` | comparison-level dataset/ALL summary |
| `b12_stbo_rank_model_diagnostics.csv` | parameter count, STBO rank/tile config, bank entropy |
| `b12_stbo_rank_best_by_setting.csv` | 每个 dataset-horizon 的 best arm |
| `b12_stbo_rank_diagnostic_report.md` | 自动 gate report |

## Relative Metric

Relative MSE 使用：

$$
\mathrm{relMSE}(a,b)=\left(\frac{\mathrm{MSE}_a}{\mathrm{MSE}_b}-1\right)\times 100
$$

负数表示 candidate 好于 baseline。

## Comparisons

对每个 STBO config 和 arm，生成三类 comparison：

1. vs `a6_clean`；
2. vs same-config `stbo_dct`；
3. vs same-config `stbo_independent`。

`stbo_dct` 用于判断 learned local basis 是否超过 generic fixed smooth basis。

`stbo_independent` 用于判断 shared/bank 机制是否只是被 independent tile capacity 解释。

## Gate Logic

自动 report 的 gate 顺序是：

1. 若所有 learned shared/bank STBO 相对 A6 的 overall mean MSE 都大于 0，则输出
   `rank_capacity_repair_insufficient`；
2. 否则若 learned STBO 未能超过 same-rank DCT，则输出
   `generic_local_basis_control_explains`；
3. 否则输出 `requires_followup`，需要人工检查 dataset/horizon split。

这次结果触发第一种：提高 rank 后，best learned shared/bank 仍未 overall beat A6。

## Code-Theory Consistency

Intended theory:

- 如果 rank bottleneck 是主要原因，高 rank STBO 应明显接近 A6；
- 如果 B12 方法成立，shared/bank learned STBO 应接近或超过 A6，并超过 DCT/independent controls。

Code realization:

- Analyzer 分开统计 A6 gate、DCT gate、independent capacity gate；
- `L360-R256_capacity_probe` 被保留为 capacity probe，不被当作默认 stage-local method；
- bank entropy 单独进入 report。

Proxy limitation:

- Analyzer 只看 single seed；
- 不做显著性检验；
- 不检查 learned basis geometry；
- `L360-R256` 只有两个 tiles，其结果不能直接支持 stage-local narrative。

Falsification:

- 若高 rank shared/bank 仍输 A6，则当前 B12-STBO method blocked；
- 若 high-rank STBO 只靠 independent tile 接近 A6，则说明 capacity can recover performance but shared/bank mechanism remains unsupported；
- 若 bank entropy 仍接近 1，则 bank specialization 不成立。
