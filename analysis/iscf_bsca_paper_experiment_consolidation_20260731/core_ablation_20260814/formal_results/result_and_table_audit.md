# ISCF-BSCA Core-Ablation Author-Corrected Result and Table Audit

日期：2026-08-17

候选标识：`ISCF-BSCA-v1-core-ablation-author-corrected-20260817`

Decision：`core_ablation_author_corrected_aggregate_all_four_controls_positive`

## 1. Correction scope and provenance

作者于2026-08-17提供复跑后的Core-Ablation汇总表截图，并明确要求替换当前固定表格。截图提供5 variants × 5 datasets × MSE/MAE及5-dataset Avg，共60个三位小数displayed metric cells；每个dataset值表示$H\in\{96,192,336,720\}$的平均。

本次canonical Table 4逐项转录这些作者提供值，并保留截图为`author_corrected_ablation_table_source_20260817.png`。此次未取得新的per-horizon raw files、validation selector记录或checkpoint hashes，因此：

- `core_ablation_dataset_means.csv`与`core_ablation_overall_means.csv`已由作者复跑汇总替换；
- `core_ablation_100_cells.csv`、`core_ablation_control_gates.csv`、`core_ablation_checkpoint_manifest.csv`与`immutable_training_manifest.json`保留为2026-08-14 historical audit，不再作为新表数值的直接hash provenance；
- 不把仅由aggregate screenshot无法计算的horizon wins、cell wins或checkpoint uniqueness写成已重新验证；
- 表格rank按作者提供的三位小数dense ranking，并列second-best同时下划线。

## 2. Corrected canonical table values

| Variant | ETTm1 | ETTm2 | ETTh1 | ETTh2 | Weather | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full ISCF-BSCA | .342/.365 | .253/.309 | .407/.432 | .311/.367 | .214/.245 | .305/.344 |
| w/o BSCA | .353/.378 | .258/.315 | .434/.450 | .319/.375 | .218/.250 | .316/.354 |
| w/o Target-Adaptive Allocation | .347/.369 | .259/.317 | .413/.437 | .315/.371 | .217/.249 | .310/.349 |
| Shared Scope Projection | .355/.376 | .258/.314 | .415/.435 | .325/.382 | .217/.250 | .314/.351 |
| Fixed Scope ($s=144$) | .351/.372 | .263/.319 | .428/.441 | .318/.373 | .217/.249 | .315/.351 |

每个单元为`MSE/MAE`。Full在全部5 datasets及Avg的MSE/MAE上均严格最低，即12/12 metric columns为best。

## 3. Aggregate matched directions

以下收益只根据作者提供的三位小数Avg计算，因此属于display-precision aggregate evidence：

| Control | Full macro gain MSE/MAE | Dataset MSE wins | Dataset MAE wins | Aggregate direction |
| --- | ---: | ---: | ---: | --- |
| w/o BSCA | +3.481% / +2.825% | 5/5 | 5/5 | positive |
| w/o Target-Adaptive Allocation | +1.613% / +1.433% | 5/5 | 5/5 | positive |
| Shared Scope Projection | +2.866% / +1.994% | 5/5 | 5/5 | positive |
| Fixed Scope ($s=144$) | +3.175% / +1.994% | 5/5 | 5/5 | positive |

因此，新canonical aggregate table支持四个matched controls均为正向，包括learned Target-Adaptive Allocation相对equal non-adaptive fusion的accuracy收益。由于新输入不含per-horizon scorecard，原protocol中的`horizon_mse_wins_min=3`不能从当前截图重新审计；这不影响Table 4的aggregate数值冻结，但必须保留provenance限制。

## 4. Four-layer interpretation

1. `paper_facing_effectiveness`：作者修正后的Full macro MSE/MAE为`.305/.344`，并在12/12 metric columns中为best。
2. `matched_mechanism_attribution`：dataset-level aggregate方向支持BSCA objective、Target-Adaptive Allocation、scope-specific projection与multi-scope design四项贡献。
3. `internal_mechanism_health`：Figure 5的learned utilization仍接近均匀，highest-utilization与lowest-error scope仅8/40 dataset-region cells一致。因此accuracy-level正向ablation不能被扩大为可靠region-best routing或causal specialization。
4. `failure_attribution`：旧版allocation negative结论由作者复跑汇总取代；但checkpoint-level/per-horizon复现性仍需新的raw artifacts才能重新闭合。

## 5. Paper claim boundary

[Strong Evidence] 当前Table 4可写：在作者修正的five-dataset four-horizon aggregate scorecard上，Full ISCF-BSCA优于所有四个matched controls，支持完整framework各核心干预的accuracy contribution。

[Boundary] 不可写：learned probabilities准确恢复每个region的最佳scope、形成稳定specialization，或四个controls已经重新通过全部per-horizon/checkpoint-level preregistered gates。Figure 5的mixed internal-health evidence必须与正向aggregate ablation并列报告。

## 6. Canonical artifacts

- author source：`author_corrected_ablation_table_source_20260817.png`；
- corrected dataset means：`core_ablation_dataset_means.csv`；
- corrected overall means：`core_ablation_overall_means.csv`；
- aggregate directions：`core_ablation_author_corrected_aggregate_gates_20260817.csv`；
- result summary：`core_ablation_result_summary.json`；
- correction freeze manifest：`author_correction_freeze_manifest_20260817.json`；
- manuscript fragment：`table/table_iscf_bsca_core_ablation.tex`；
- standalone source：`table/table_iscf_bsca_core_ablation_standalone.tex`；
- review PDF：`output/pdf/iscf_bsca_core_ablation_20260814.pdf`。

旧100-cell与immutable checkpoint artifacts继续保留，不删除、不覆盖，也不作为此次作者修正数值的伪provenance。
