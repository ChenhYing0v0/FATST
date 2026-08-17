# Efficiency Accuracy--Parameters--One-Epoch Result and Table Audit

日期：2026-08-17

Decision：`efficiency_accuracy_params_epoch_complete_accuracy_parameter_advantage_compute_tradeoff`

## 1. 完整性

- Main I accuracy：3 systems × 7 datasets × 4 horizons=`84/84` cells；
- four-horizon parameter service：3 × 7=`21/21` dataset units；
- epoch-cycle timing：ISCF-BSCA 7 + TimeAlign 28 + QDF 28=`63/63` checkpoint logs；
- new training=`0`；formal test access=`0`；
- Main I accuracy、frozen checkpoint parameters与native timing logs逐system/dataset对齐，不使用published accuracy与不同配置cost的混合证据。

## 2. Canonical result

| System | Models for four H | Main I MSE | Main I MAE | Total params (M) | 1-epoch cycle (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **ISCF-BSCA** | **1** | **0.261** | **0.306** | **2.926** | 133.1 |
| TimeAlign | 4 | 0.274 | 0.314 | 10.741 | 104.6 |
| QDF | 4 | 0.288 | 0.331 | 5.337 | **55.4** |

Accuracy是Main I七datasets × four H的macro mean。Params与1-epoch cycle先按每dataset构造完整four-horizon service，再对七datasets作macro mean。

## 3. 结果解释

- 相对TimeAlign，ISCF-BSCA的MSE/MAE分别改善`4.936%/2.536%`，four-horizon total parameters减少`72.760%`；
- 相对QDF，MSE/MAE分别改善`9.320%/7.639%`，total parameters减少`45.181%`；
- ISCF-BSCA的one-epoch cycle是TimeAlign的`1.272×`、QDF的`2.404×`，因此training compute不占优。

[Strong Evidence] 当前结果支持：一个ISCF-BSCA model可替代四个horizon-specific models，在获得更低Main I MSE/MAE的同时显著减少four-horizon deployed parameter count。

[Boundary] 当前reference implementation的one-epoch cycle更慢，故不得写成uniform efficiency或lower training time。Timing采用已完成RTX 3090 native runs中每checkpoint epoch cycle的中位数，包含native scheduled validation、不含test；它比total training GPU-hours更少受epoch budget影响，但不是重新执行的exclusive-GPU microbenchmark。

## 4. Supersession

本表替代2026-08-14的九列deployment profiler表作为正文Table 3。旧latency/memory/storage profiler artifacts完整保留为historical supplementary audit；DLinear-H720-prefix与PatchTST-H720-prefix因不满足本轮four-fixed-H sum定义而不进入新主表。

## 5. Canonical artifacts

- protocol：`configs/iscf_bsca_efficiency_accuracy_params_epoch_protocol.json`；
- raw timing snapshot：`per_checkpoint_epoch_cycle_stats.csv`；
- dataset results：`efficiency_dataset_results.csv`；
- macro results：`efficiency_system_macro_results.csv`；
- summary：`efficiency_result_summary.json`；
- LaTeX：`table/table_iscf_bsca_efficiency.tex`；
- standalone LaTeX：`table/table_iscf_bsca_efficiency_standalone.tex`；
- review PDF：`output/pdf/iscf_bsca_efficiency_accuracy_params_epoch_20260817.pdf`。
