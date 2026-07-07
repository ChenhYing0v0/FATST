# Phase5 StageB B7 Unified Prefix Optimization Analyzer Explanation

本文档解释 `scripts/analyze_phase5_stage_b_unified_prefix_optimization.py`。

## Purpose

该脚本检验一个 StageB Step 2/3 problem candidate：

> A6-LBF-r256 已经是 unified forecast operator，但当前 multi-prefix objective 是否因为 nested prefix
> averaging 而造成 short-prefix over-supervision 与 long-tail under-supervision？

它不是 method implementation，不训练模型，不修改 loss。

## Inputs

默认输入：

- clean A6-LBF-r256 rerun:
  `analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706/raw/`;
- fixed-horizon TimeAlign references:
  `analysis/phase5_timealign_hss_a3e_ettm1_replacement_gate_20260701/raw/`.

读取文件：

- `metrics_by_segment.csv` from clean A6;
- `metrics_by_segment.csv` from `TimeAlignOfficialFixedH{96,192,336,720}_official-last`.

## Statistics

### Prefix Step Weight

For horizons $\mathcal{H}=\{96,192,336,720\}$, current multi-prefix loss is:

$$
\mathcal{L}=\frac{1}{|\mathcal{H}|}\sum_{H\in\mathcal{H}}\frac{1}{H}\sum_{t=1}^{H}\ell_t.
$$

Therefore each future step has effective scalar weight:

$$
w(t)=\frac{1}{|\mathcal{H}|}\sum_{H\in\mathcal{H},t\le H}\frac{1}{H}.
$$

`stage_b_b7_prefix_weight_profile.csv` stores:

- `segment_start`, `segment_end`: future-step interval in 0-indexed half-open form;
- `avg_prefix_weight`: mean of $w(t)$ inside the interval;
- `relative_to_tail_weight`: interval average divided by the `336-720` tail interval average.

### Segment Gap

`stage_b_b7_segment_gaps_vs_fixed.csv` stores one row per `(dataset, target_horizon, segment)`:

- `clean_mse`, `clean_mae`: clean A6 segment metric;
- `fixed_mse`, `fixed_mae`: matching fixed-horizon TimeAlign segment metric;
- `relative_mse_pct = (clean_mse / fixed_mse - 1) * 100`;
- `relative_mae_pct = (clean_mae / fixed_mae - 1) * 100`;
- `clean_wins_mse`: whether A6 beats fixed on MSE;
- `avg_prefix_weight`, `relative_to_tail_weight`: supervision exposure of that segment.

`stage_b_b7_segment_gap_summary.csv` groups rows by dataset and by four buckets:

- `early_0_96`;
- `mid_96_192`;
- `late_192_336`;
- `tail_336_720`;
- `ALL`.

## Gate

The report decision is:

- `prefix_imbalance_problem_candidate` if all rows exist and the overall tail-minus-early relative MSE gap is above
  `1.0%`;
- `prefix_imbalance_not_supported_yet` if segment evidence is weak;
- `incomplete` if required rows are missing.

This gate is intentionally weak: passing it only opens a stronger `B7-GTD` gradient/task decomposition diagnostic.
It does not justify implementing a new loss.

## Code-Theory Consistency

Intended theory:

- unified prediction should not only have a unified operator, but also avoid training-time horizon-task imbalance;
- nested prefix averaging may over-optimize short future steps and leave long-tail steps closer to fixed baseline.

Code realization:

- derive the exact scalar step weights from current `multi-prefix` loss;
- compare clean A6 and fixed TimeAlign at segment level;
- check whether low-weight tail segments are where A6 gains shrink.

Remaining proxy:

- segment-level evidence does not prove gradient conflict;
- fixed-horizon TimeAlign has a different objective and architecture head;
- Weather is a counterexample, so the result is not method-ready.

Falsification evidence:

- gradient/task diagnostic finds no prefix-loss gradient imbalance or conflict;
- tail weakness disappears under clean repeated seeds;
- weakness is entirely explained by generic horizon difficulty rather than A6's nested-prefix training.
