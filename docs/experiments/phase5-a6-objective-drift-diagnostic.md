# Phase5-A6 Objective Drift Diagnostic

本文档记录 A6 partial-pass 后的下一轮 diagnostic-only 实验设计。该实验不改变主协议：
所有主判断仍使用 `official-last` / without early stop；`best-val` 不作为本轮实验目标。

## 11-Step Position

| Field | Content |
| --- | --- |
| `current_step` | Step 4/5/6 diagnostic design |
| `problem` | A6-LBF 已恢复 dense-capacity path，但 ETTh2 出现明显 official-last trajectory drift，且 A6-LBF 仍未超过 best stage controls |
| `existence_evidence` | A6 partial-pass diagnostic 显示 ETTh2 三个 A6 arms 的 last-vs-best validation MSE 平均漂移 `+11.81%`，ETTm1/Weather 平均仅 `+0.12%` |
| `idea` | 不改变 checkpoint protocol，而是检查 prefix-conditioned training objective 是否在 ETTh2 上造成 post-best drift 或 prefix conflict |
| `theory_check` | A6-DER 与 A6-LBF 已接近 dense-equivalent ceiling；若 `full` 或 stochastic/continuous prefix objective 能降低 official-last drift，说明剩余瓶颈更偏 optimization/objective，而不是 rank capacity |
| `design` | ETTh2 only；`A6-LBF-r256` 与 `A6-DER` 各跑 `full`、`stochastic-prefix`、`continuous-prefix` 三个 objective variants；所有 run 使用 `official-last` |
| `narrative_gate` | not_required：本轮是 diagnostic-only，不能直接升级为 paper-core method |
| `effectiveness_gate` | 比较 final test MSE、last-vs-best validation drift、是否缩小 ETTh2 gap；若只改变 selector 不改变 official-last final，则不算修复 |
| `artifacts` | `analysis/phase5_timealign_hss_a6_objective_drift_diagnostic_20260703/phase5_timealign_hss_a6_objective_drift_diagnostic_report.md` |
| `decision` | completed_failed_as_repair：objective switch 没有修复 A6 的 ETTh2 gap |

## Variants

| Variant | Readout | Rank | `pred_loss_mode` | Purpose |
| --- | --- | ---: | --- | --- |
| `lbf_r256_full` | `learned-basis-forecast-operator` | 256 | `full` | 检查只优化 720-step full trajectory 是否减少 ETTh2 drift |
| `lbf_r256_stochastic_p1` | `learned-basis-forecast-operator` | 256 | `stochastic-prefix` | 检查每 batch 单 prefix 随机监督是否降低 prefix conflict |
| `lbf_r256_continuous_p4` | `learned-basis-forecast-operator` | 256 | `continuous-prefix` | 检查更密集 prefix coverage 是否提升 official-last generalization |
| `der_full` | `prefix-native-dense-equivalent-row-bank` | 64 | `full` | dense-equivalent objective control |
| `der_stochastic_p1` | `prefix-native-dense-equivalent-row-bank` | 64 | `stochastic-prefix` | dense-equivalent stochastic-prefix control |
| `der_continuous_p4` | `prefix-native-dense-equivalent-row-bank` | 64 | `continuous-prefix` | dense-equivalent continuous-prefix control |

## Decision Rule

[Decision] 本轮不测试 `r512`，因为 A6 partial-pass diagnostic 已显示 `r512` 的 rank 扩张没有转化为
metric gain。若本轮 objective variants 均不能缩小 ETTh2 official-last gap，则应回 Step 4/5
重新设计 explicit anti-drift regularization 或 teacher/nested-style stability path，而不是继续调 rank。

## Result

[Strong Evidence] A6OD 不通过 repair gate。最佳 variant 是 `lbf_r256_stochastic_p1`，相对
ETTh2 best stage control 平均仍差 `+1.79%`，wins `0/4`，last-vs-best validation drift 仍为
`+6.25%`。

[Fact] `full` objective 明显更差：`lbf_r256_full` 相对 best control `+5.61%`，`der_full`
为 `+5.81%`。`continuous-prefix` 也未修复 gap。

[Decision] 下一步不继续 objective-sampling sweep；应回 Step 4/5 设计 explicit stability path，
例如 official-last-compatible regularization、teacher/nested stability control，或重新评估 best controls
的 regularization advantage。
