# Main I TimeAlign Table-6-Style 结果表与来源审计

## 1. 交付内容

本目录给出一张可直接进入论文排版迭代的 Main Results I 草表。表格沿用
TimeAlign Table 6 的 dataset block 结构：每个 dataset 包含
`H={96,192,336,720}` 与一个重新计算的 `Avg.` 行，每个模型报告 MSE/MAE，
红色粗体和蓝色下划线分别标记按三位小数显示后的 best 与 second-best。

当前共同比较面为 7 datasets：`ETTm1`、`ETTm2`、`ETTh1`、`ETTh2`、
`Weather`、`ECL`、`Solar`。表中共有 13 个模型：`ISCF-BSCA`、
`TimeAlign`、`CMoS`、`TimeBase`、`TVNet`、`iTransformer`、`TimeMixer`、
`Leddam`、`ModernTCN`、`PatchTST`、`Crossformer`、`TimesNet` 和
`DLinear`。标准 horizon 矩阵为 $13\times7\times4=364$ 行；加入重新计算的
dataset-model averages 后，machine-readable long table 共 455 行。

## 2. 数据来源

| 表中系统 | 数据来源 | 证据角色 |
| --- | --- | --- |
| ISCF-BSCA | H1--H4M 共 189 trials 的 frozen dataset-level joint selector；seed2021；official-test tuned | 一个 unified model / dataset，single-seed test-tuned effectiveness result |
| TimeAlign：ETTm2、Weather | official-native local reproduction，seed2021，共 8 个 fixed-H systems | artifact-complete native external baseline |
| TimeAlign：其余 5 datasets | TimeAlign ICLR 2026 Table 6 published three-run mean | unmatched published context |
| 其余 11 个模型 | TimeAlign ICLR 2026 Table 6 published values | unmatched published context |

关键输入及 SHA256：

- TimeAlign paper PDF：`tmp/pdfs/timealign/timealign_iclr2026.pdf`，
  `0a0d7deba16c5b3025902f897dddc70fd1f6efaeb3737c4a751d3b2440670b08`；
- ISCF-BSCA selected scorecard：
  `analysis/iscf_bsca_main_v1_hpo_20260731/joint_objective_h4m_result_20260805/joint_selected_cells.csv`，
  `b7478154f17669a57c6a226ef72b01c138a87e4b635052f8d689263d4802ffa7`；
- TimeAlign reproduction scorecard：
  `analysis/iscf_bsca_main_v1_hpo_20260731/timealign_official_reproduction_20260804/reproduced_metrics_and_comparison.csv`，
  `197b0486438f70527e113c1be14cf6b748fbbb68f4dc4a4716bc335ac5f7f109`；
- 已审计的五模型 published subset：
  `analysis/iscf_bsca_paper_experiment_consolidation_20260731/timealign_table6_main_i_published.csv`，
  `c9286468a4ca5977bac40f18635d387eafda0fd14794cabaac9b6f659ae82d8c`。

构建脚本从 PDF 第 22 页按坐标抽取 12 个 published baseline，并要求其中
TimeAlign、TimeMixer、DLinear、iTransformer、PatchTST 的 140 个目标 rows
与此前人工渲染核验后的 CSV 精确一致，否则拒绝生成表格。

## 3. 排名口径与 33/56 边界

原 `33/56` 是 frozen H4M gate 在五模型 comparator subset 上的结果，即
7 datasets × 4 horizons × 2 metrics；它不是这张 13-model full table 的
best-cell 数量。将 TimeAlign Table 6 中全部模型加入后，ISCF-BSCA 在 56 个标准
MSE/MAE cells 中为：

- best：29/56；
- second-best：16/56。

由于 published baselines 只有三位小数，排名在三位小数显示值上计算，并允许
并列 best/second；因此不能把 `29+16` 解读为 45 个互斥排名 cells。论文正文若引用
`33/56`，必须同时写明其五模型 frozen comparator scope；若引用本表，则应使用完整
13-model scope 下的结果。

## 4. 诚实边界

- ISCF-BSCA 是 single-seed、dataset-level test-tuned 结果，不是 untouched-holdout
  estimate，也不是 three-seed mean。
- TimeAlign 列是 mixed-source：仅 ETTm2/Weather 为本地复现，其余 5 datasets
  仍是 published mean；表头以 `TimeAlign*` 和脚注明示。
- 其余 published baselines 未在本项目中按统一环境复跑，不能承担 matched mechanism
  attribution。
- `Avg.` 全部由四个逐 horizon rows 算术平均重新计算，不沿用原 Table 6 中已知的
  reported-average inconsistencies。
- Traffic 因缺少当前 ISCF-BSCA scorecard 而不进入共同表面；Exchange 因 TimeAlign
  Table 6 不提供对应结果而不进入本表。二者都不是选择性删除弱项。
- 本表是当前 Main I 的宽比较草表，不改变 H4M effectiveness gate fail / rollback
  Step 6，也不自动授权新的 baseline training、HPO、formal test 或 3-seed。

## 5. 生成物

- `table_data_long.csv`：455-row machine-readable long table；
- `table_iscf_bsca_vs_timealign_table6.tex`：论文 LaTeX 草表；
- `table_iscf_bsca_vs_timealign_table6.png`：视觉预览；
- `table_build_summary.json`：矩阵规模、排名口径、来源 hash 与 claim boundary；
- `output/pdf/iscf_bsca_main_i_timealign_table6_style.pdf`：单页 PDF。
