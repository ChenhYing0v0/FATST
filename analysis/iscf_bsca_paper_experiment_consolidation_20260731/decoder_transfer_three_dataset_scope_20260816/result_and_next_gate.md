# Decoder-Transfer Author-Corrected Result and Freeze Audit

日期：2026-08-17

Candidate：`ISCF-BSCA-decoder-transfer-author-corrected-20260817`

Decision：`decoder_transfer_author_corrected_aggregate_both_backbones_all_columns_positive`

## 1. Correction scope and provenance

作者于2026-08-17提供Decoder-Transfer复跑后的汇总表截图，并要求替换当前固定表。正文范围保持Weather、ETTm1、ETTm2；每个dataset值为$H\in\{96,192,336,720\}$的MSE/MAE平均。截图覆盖2 backbones × 2 decoders × 3 datasets + Avg，共32个三位小数displayed metric cells。

本次逐项转录作者提供值，并将截图保存为`author_corrected_decoder_transfer_source_20260817.png`。新输入未包含per-horizon raw files、validation selectors、profile IDs或checkpoint hashes，因此：

- `framework_portability_dataset_means.csv`、canonical framework LaTeX与review PDF更新为作者复跑汇总；
- `framework_portability_48_cells.csv`及此前five-dataset/three-dataset checkpoint audits保留为historical artifacts，不再作为新表值的直接hash provenance；
- 不把无法从aggregate screenshot计算的per-horizon/cell wins或checkpoint uniqueness写成已重新验证；
- ETTh1/ETTh2与iTransformer-style历史负向结果继续保留，不据此扩大为universal transferability。

## 2. Corrected canonical values

| Backbone | Decoder | Weather | ETTm1 | ETTm2 | Avg. |
| --- | --- | ---: | ---: | ---: | ---: |
| DLinear-style | Original Decoder | .246/.274 | .353/.372 | .311/.354 | .303/.333 |
| DLinear-style | ISCF-BSCA (ours) | .232/.259 | .347/.370 | .279/.333 | .286/.321 |
| PatchTST-style | Original Decoder | .229/.253 | .358/.375 | .259/.314 | .282/.314 |
| PatchTST-style | ISCF-BSCA (ours) | .226/.251 | .349/.369 | .254/.310 | .276/.310 |

每个单元为`MSE/MAE`。ISCF-BSCA在两个backbones的Weather、ETTm1、ETTm2与Avg上均同时取得更低MSE/MAE，即16/16 comparator metric columns正向。

## 3. Display-precision aggregate gains

| Backbone | Original Avg. | ISCF-BSCA Avg. | MSE/MAE gain | Dataset MSE/MAE wins |
| --- | ---: | ---: | ---: | ---: |
| DLinear-style | .303/.333 | .286/.321 | +5.611% / +3.604% | 3/3 / 3/3 |
| PatchTST-style | .282/.314 | .276/.310 | +2.128% / +1.274% | 3/3 / 3/3 |

这些百分比由作者提供的三位小数Avg计算，是display-precision aggregate evidence，不是新的unrounded per-horizon audit。

## 4. Claim boundary

[Strong Evidence] 在作者指定的Weather、ETTm1、ETTm2正文范围内，完整ISCF-BSCA framework相对对应Original Decoder在DLinear-style和PatchTST-style两类backbones上均取得MSE/MAE aggregate改善，并覆盖3/3 datasets。

[Boundary] 本表检验完整framework portability，不区分ISCF与BSCA内部贡献。不得写成对任意backbone、dataset或domain都有效；必须保留author-refined posthoc scope、test-tuned history、ETTh1/ETTh2与iTransformer-style negative evidence。新的per-horizon/checkpoint reproducibility仍需作者提供raw rerun artifacts后才能重新闭合。

## 5. Canonical artifacts

- author source：`author_corrected_decoder_transfer_source_20260817.png`；
- corrected means：`framework_portability_dataset_means.csv`；
- aggregate gains：`author_corrected_transfer_aggregate_gains_20260817.csv`；
- result summary：`result_summary.json`；
- correction manifest：`author_correction_freeze_manifest_20260817.json`；
- manuscript fragment：`table_decoder_transfer_three_dataset_framework.tex`；
- standalone source：`table_decoder_transfer_three_dataset_framework_standalone.tex`；
- review PDF：`output/pdf/iscf_bsca_decoder_transfer_three_dataset_framework_20260816.pdf`。

旧48-cell、72-cell与checkpoint-linked artifacts不删除、不覆盖，并明确降级为historical audit。
