# Main I：HPO 终止冻结与 TimeAlign 全数据集本地复跑结果

## 1. 结论

- `ISCF-BSCA-MAIN-v1` HPO 已按用户指令终止；H4O 不启动。最终固定 8 个
  dataset-level profiles、32 个标准 horizon cells 和 8 个 immutable checkpoint
  hashes。Weather 使用 H4N full-table selector 选择的
  `Weather__h4n_seq608_p19_lr2e5`，其余数据集保持既有最终 profile。
- TimeAlign 已完成 Main I 全部 8 datasets × 4 horizons × seed2021，共 32/32
  fixed-H runs。其中 ETTm2/Weather 8 个通过同一 contract 复用，其他 24 个本轮
  从 official source 重新训练并在训练结束后各访问 official test 一次。
- 32 个 TimeAlign checkpoint hashes 全部唯一；32/32 metrics、effective config、
  environment、initialization contract、training log 与 run log 通过审计；无
  OOM、NaN、Inf 或 Traceback。
- 审计完成后删除本轮24-job `_resource_smoke`临时目录，释放524 MiB；formal
  checkpoints、metrics、logs和hash manifest均保留，remote quota为175/220 GiB。
- 论文主表中 TimeAlign 的 7 个 shared-dataset blocks 已全部由本地复跑值替换，
  不再混用论文 TimeAlign 数值。Exchange 另列 companion block，因为 TimeAlign
  没有 official Exchange script，且 TimeAlign Table 6 也没有其他 baseline 的
  Exchange 数值。

Decision：`HPO_terminally_frozen_TimeAlign_32_of_32_local_reproduction_complete`。

## 2. 冻结 contract 与 provenance

| Item | Value |
| --- | --- |
| TimeAlign protocol | `TIMEALIGN-OFFICIAL-MAIN-I-8DATASET-REPRODUCTION-20260806` |
| Source/runner commit | `46b5bc5568d55df8f9fdf10d7ad2e93325e157e6` |
| Config SHA256 | `64cca166f38d8e29b4f5cfca5ab854cc9dcb1aa57949673d4d74a8fd1d27d567` |
| Local metrics SHA256 | `6fccaeddec410ff2eda2c2533afacb1ce28459261eae6566dd6eb3f5b390d9f8` |
| Artifact manifest SHA256 | `f09e4634e5dee543291225d2fe3fbcedc0aac75805a7e925ffb40a769519c461` |
| Combined checkpoint-hash file SHA256 | `fbbe37c19738ed36537ca218f670ad5d604fd56220f462c7e220dc643a4321cb` |
| Seed | `2021` |
| Horizons | `{96, 192, 336, 720}` |
| Metrics | `MSE`, `MAE` |
| Checkpoint rule | TimeAlign official-last；test only once after training |
| Source role | 7 datasets=`official preset`；Exchange=`ETTh1-derived source-informed bootstrap` |

ETTh1 H96 保留 official script 的 1 epoch 设置，其余新 runs 使用 10 epochs；
ECL batch size 为 16，其余为 32。为了遵守远程 220 GiB hard quota，新增 24 个
runs 不保存 prediction arrays，但保留 checkpoint、metrics 和全部复现实验 provenance；
既有 8 个 ETTm2/Weather runs 的 prediction arrays 已在先前 artifact audit 中保留。

## 3. TimeAlign 本地复跑结果

下表为 four-horizon arithmetic mean；逐 horizon 原始值见
`timealign_main_i_local_metrics.csv`。

| Dataset | Mean MSE | Mean MAE | Preset role |
| --- | ---: | ---: | --- |
| ETTh1 | 0.417990 | 0.429396 | official |
| ETTh2 | 0.346665 | 0.386517 | official |
| ETTm1 | 0.339704 | 0.366959 | official |
| ETTm2 | 0.242889 | 0.302523 | official |
| Weather | 0.215800 | 0.244725 | official |
| ECL | 0.154704 | 0.243853 | official |
| Solar | 0.195970 | 0.216647 | official |
| Exchange | 0.512558 | 0.459692 | source-informed bootstrap, not official |

在有 TimeAlign Table 6 published results 的 28 个 cells 中，本地 single-seed
复跑相对 published three-run mean：MSE 6 个更低、22 个更高，平均相对差
`+1.323%`；MAE 7 个更低、21 个更高，平均相对差 `+0.717%`。最大偏差出现在
ETTh1 H96（MSE `+4.852%`、MAE `+5.085%`）；因此论文中必须标记为本地
single-seed reproduction，不能把它描述为 published three-run mean 的精确复刻。

## 4. ISCF-BSCA 与本地 TimeAlign

| Scope | ISCF mean MSE | TimeAlign mean MSE | ISCF mean MAE | TimeAlign mean MAE | ISCF lead cells (MSE/MAE) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 shared datasets, 28 cells | 0.262469 | 0.273389 | 0.308281 | 0.312946 | 20/28, 17/28 |
| 8 datasets, 32 cells | 0.279514 | 0.303285 | 0.322881 | 0.331289 | 22/32, 19/32 |

在 7-dataset shared surface 上，ISCF-BSCA 的 macro MSE/MAE 分别低
`3.994%/1.491%`；加入 Exchange companion 后分别低 `7.838%/2.538%`。
这是一个 single-seed、test-tuned ISCF-BSCA 与 native fixed-H TimeAlign 的
performance comparison，不是 matched mechanism attribution。

## 5. Main I table artifacts

主表采用 TimeAlign Table-6 layout，覆盖 7 个共同 datasets、13 models 和
four horizons：ISCF-BSCA、TimeAlign、CMoS、TimeBase、TVNet、iTransformer、
TimeMixer、Leddam、ModernTCN、PatchTST、Crossformer、TimesNet、DLinear。
TimeAlign 的 28 个 cells 全部使用本地复跑值；其他 11 个 baselines 保持
TimeAlign Table 6 的 published context。按三位小数 displayed-value ranking，
ISCF-BSCA 在完整 13-model 表中为 27/56 best、19/56 second（允许并列）。

Artifacts：

- `table_iscf_bsca_vs_timealign_table6.tex`：7-dataset dense LaTeX table；
- `table_data_long.csv`：455-row source/rank-aware long table；
- `table_exchange_iscf_vs_timealign.tex`：Exchange companion block；
- `table_exchange_companion_long.csv`：Exchange 10-row long table；
- `output/pdf/iscf_bsca_main_i_local_timealign_20260806.pdf`：rendered table。

## 6. 诚实边界与剩余缺口

当前已完成的是 **TimeAlign Table-6-style shared-surface Main I numeric table**，
以及 Exchange 上 ISCF-BSCA/TimeAlign 的 companion block。它尚不等于最初规划的
8 datasets × 9 named baseline families 全数闭合：AMD、SimpleTM、TimePerceiver、
SRSNet 等不在 TimeAlign Table 6 中，其他 published baselines 的 Exchange 数值也
缺失，且本轮没有授权复跑这些模型。因此这些缺口不得填入推测值、跨协议值或
选择性结果；若要把它们纳入最终投稿主表，需要另行冻结 source/protocol matrix
并授权相应 official reproduction。

HPO 继续保持停止；本报告不授权 selected-profile three-seed confirmation、Main II、
ablation、transfer 或其他 baseline training/formal test。
