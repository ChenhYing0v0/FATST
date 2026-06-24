# Phase3-A Prefix Specialist Tradeoff Diagnostic

更新时间：2026-06-24

## 1. 目的

Phase2-E2 已否定 QDF learned precision direct transfer。Phase3-A 重新检查 R.3 的剩余问题是否来自
`prefix-consistent shared trajectory` 与 `horizon-specialist performance` 的 tradeoff。

脚本：

- `scripts/analyze_phase3_prefix_specialist_tradeoff.py`

输入：

- R.3 predictions:
  `analysis/phase2_qdf_alignment_diagnostic_20260623/raw/PatchEncoderPrefixRiskWeighted`;
- R.3 vs fixed aggregate:
  `analysis/phase1_prefix_risk_weighted_gate_20260622/phase1_prefix_risk_weighted_vs_fixed.csv`;
- H720 prefix reference:
  `analysis/phase1_prefix_risk_weighted_gate_20260622/phase1_prefix_risk_weighted_h720_prefix_reference.csv`;
- segment table:
  `analysis/phase1_prefix_risk_weighted_gate_20260622/phase1_prefix_risk_weighted_vs_fixed_segments.csv`。

## 2. Forward Analysis Flow

对每个 dataset 和 short horizon `H in {96,192,336}`：

1. 读取 standalone `hH/predictions_test.npz`：
   - `pred`: `[N_H, H, C]`;
   - `true`: `[N_H, H, C]`。
2. 读取 `h720/predictions_test.npz`：
   - `pred`: `[N_720, 720, C]`;
   - `true`: `[N_720, 720, C]`。
3. 对齐前 `N_720` 个窗口：
   - standalone prefix: `pred[:N_720, :H, :]`;
   - H720 prefix: `pred720[:N_720, :H, :]`。
4. 计算三组 MSE：
   - `full_mse`: standalone `hH` 的完整 test windows；
   - `aligned_mse`: 与 H720 可对齐的前 `N_720` windows；
   - `short_only_extra_mse`: standalone `hH` 额外拥有、H720 不可对齐的末端 windows。
5. 检查同输入 prefix identity：
   - `pred_prefix_mismatch_mse`;
   - `true_prefix_alignment_mse`;
   - `residual_prefix_mismatch_mse`;
   - `residual_prefix_cosine`。

对 H720：

1. 读取 segment-level R.3 vs fixed 表；
2. 筛出 `relative_mse_pct > 0` 的 segment gaps。

## 3. Output Definitions

输出目录：

- `analysis/phase3_prefix_specialist_tradeoff_20260624`

文件：

- `phase3_prefix_specialist_short_alignment.csv`
  - short horizon full/aligned/extra window 分解；
  - prefix identity metrics；
  - gap type。
- `phase3_prefix_specialist_h720_segments.csv`
  - H720 segment-level R.3 vs fixed gaps。
- `phase3_prefix_specialist_summary.json`
  - gate 和核心计数。
- `phase3_prefix_specialist_report.md`
  - 面向研究决策的摘要。

`gap_type` 定义：

- `short_extra_window_gap`: standalone full horizon 输 fixed，但 H720-aligned prefix 不输；
- `shared_prefix_gap`: standalone full horizon 和 H720-aligned prefix 都输；
- `h720_prefix_only_gap`: standalone full horizon 不输，但 H720-aligned prefix 输；
- `no_mse_gap`: 两者都不输。

## 4. Code-Theory Consistency

[Theory] 如果 R.3 的问题来自 prefix-consistent shared trajectory 本身，那么同一输入下 standalone
short horizon 与 H720 prefix 应出现 prediction 或 residual conflict。

[Code Realization] 代码直接比较同一批 `N_720` aligned windows 上的 prediction、truth 和 residual。
若 mismatch 接近零，则不能把 short-horizon gap 归因于同输入 prefix conflict。

[Finding] 当前结果显示 prefix identity pass，因此 short-horizon gaps 更像 test-window coverage /
regime effect；H720 gaps 则集中在 late segments。

[Falsification] 若未来候选声称修复 prefix/specialist tradeoff，但不能分别解释 short-only extra
windows 与 H720 late segments，则该候选不是针对当前证据的有效机制。
