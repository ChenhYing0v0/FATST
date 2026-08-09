# ISCF-BSCA 论文实验表格总账

## 1. 总账边界

本文件是当前 **paper-facing experiment tables** 的统一入口。它汇总已经完成的
正式表、正文中已经确定但尚待实验的表，以及明确只放 Appendix 的表。历史中间版、
source transcription audit、checkpoint manifest 和 smoke 统计不作为论文表重复列入。
机器可读契约见 `configs/iscf_bsca_paper_table_registry.json`。

状态含义：

- `complete_hash_frozen`：结果、来源角色与输出 hash 已冻结；未经用户显式解冻不得改写；
- `complete_presentation_aligned`：完整结果已通过审计，并已按 Main I 视觉契约生成；
- `complete_limited_surface`：结果完整，但只覆盖明确列出的部分 systems；
- `partially_reusable_prelaunch_pending`：部分 checkpoint/metric evidence 可复用，正式表仍不完整；
- `source_patch_and_retrain_required_prelaunch_pending`：需要先完成 source/protocol gate，尚无正式结果；
- `planned_grid_not_yet_frozen`：只冻结了研究问题，具体矩阵仍未冻结。

## 2. Main I / Main II 统一展示契约

Main I 与 Main II 现在使用同一套表格语言：

| Field | Frozen presentation |
| --- | --- |
| Dataset order | ETTm1, ETTm2, ETTh1, ETTh2, Weather, ECL, Solar |
| Horizon rows | 96, 192, 336, 720, Avg. |
| Metrics | MSE / MAE |
| Avg. | 四个 displayed horizons 的 arithmetic mean |
| Precision | 三位小数 |
| Ranking | 先统一舍入到三位小数，再按 distinct displayed values 取 best/second；允许 ties |
| Best | red + bold |
| Second | blue + underline |
| LaTeX | `table*`、`resizebox{\textwidth}{!}`、`tabcolsep=1.2pt` |
| Required packages | `booktabs`, `multirow`, `graphicx`, `xcolor` |

“相同形式”只约束展示结构和排名规则，不意味着两张表必须包含相同 systems。Main I
比较一个 unified model 与 horizon-specific systems；Main II 比较各 system 的一个
H720 model 所形成的 four-horizon prefix forecasts。

## 3. 当前论文表格清单

| ID | 论文位置 | 状态 | 当前规模 | Canonical artifact | 能支持的结论 |
| --- | --- | --- | --- | --- | --- |
| `Main-I` | 正文主结果 | `complete_hash_frozen` | 14 systems × 7 datasets × 4 H；392 system–dataset–H rows | `analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_i_final_amd_simpletm_20260808/table_iscf_bsca_main_i_qdf.tex` | ISCF-BSCA 与 separately optimized horizon-specific systems 的 system-level accuracy competitiveness；不作 matched mechanism attribution |
| `Main-II` | 正文主结果 | `complete_presentation_aligned` | 8 systems × 7 datasets × 4 H；224 cells，70 checkpoint evaluations | `analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_ii_h720_prefix_20260808/formal_results_20260809/table/table_iscf_bsca_main_ii.tex` | one-model-for-all-horizons system competitiveness；不作 BSCA/decoder attribution |
| `Main-I-Exchange` | Supplementary companion | `complete_limited_surface` | ISCF-BSCA / TimeAlign / QDF × Exchange × 4 H | `analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_i_final_amd_simpletm_20260808/table_exchange_companion.tex` | Exchange 的部分系统背景；不能表述为完整 Main I extension |
| `Efficiency` | 正文 supporting result | `measurement_pending_baseline_subset_not_yet_frozen` | 指标已定，baseline subset 与 profiler contract 待冻结 | 尚无正式表 | trained-model/storage/training/inference/CHPC trade-off；测量完成前不作 efficiency claim |
| `Core-Ablation` | 正文 mechanism attribution | `partially_reusable_prelaunch_pending` | 5 variants × 5 datasets × 4 H；seed2021 共有100 cells，其中40 reusable、60需source patch/retrain | 尚无正式表 | Independent Fields、Target-Wise Fusion、Multiple Coupling Scopes 与 BSCA 的 matched with/without attribution |
| `Decoder-Transfer` | 正文 transfer evidence | `source_patch_and_retrain_required_prelaunch_pending` | 2 backbones × 3 decoder columns × 5 datasets × 4 H；120 cells | 尚无正式表 | decoder 是否可在 DLinear-style 与 PatchTST-style backbones 上 end-to-end transfer |
| `Ablation-Sensitivity` | Appendix | `planned_grid_not_yet_frozen` | random partition、scope count、$\lambda$ sensitivity | 尚无正式表 | 只作 sensitivity/robustness；canonical grouping 暂不作为正向核心结论 |

## 4. 已完成主表结果摘要

### Main I

- ISCF-BSCA：29/56 best、19/56 second；
- TimeAlign、QDF、AMD、SimpleTM 为本地 official-source/native reproduction；
- 其余 columns 为 published context，来源与 protocol 差异必须保留在 caption/Methods；
- Main I 已由 `main_i_freeze_manifest.json` hash 冻结，Main II 或后续实验不得回写。

### Main II

- 完整性：70 checkpoint evaluations、280 raw prefix rows、224 aggregate cells、448 MSE/MAE scalars；
- ISCF-BSCA macro MSE/MAE：0.262469 / 0.308281，均为八 systems rank 1；
- 三位小数显示口径：24/56 best、27/56 second，共51/56 metric cells 位于 top-2；
- Main II 现已与 Main I 对齐 dataset order、year label、best/second emphasis、column spacing 与 required packages；
- external source contracts 不 matched，因此该表不能兑现 component effectiveness 或 decoder portability claims。

## 5. 尚未完成表格的最小闭合顺序

1. `Core-Ablation`：先冻结三个缺失 controls 的 exact end-to-end identity、预算与完整100-cell single-seed matrix；不得复跑 contract 匹配的现有 Full / w/o BSCA evidence。
2. `Decoder-Transfer`：冻结 DLinear-style 与 PatchTST-style 的 backbone-specific profiles，并完成30 checkpoints / 120 cells；frozen replacement 只可作为 diagnostic。
3. `Efficiency`：在所比较 checkpoints 全部冻结后，再冻结同机 profiler、warm-up/repetition 和统计规则；避免先测后选 baseline subset。
4. `Ablation-Sensitivity`：只有在正文核心表闭合后才冻结 Appendix grid；不因已见结果选择性扩张。

以上新实验仍分别需要 local protocol patch、remote training 与 formal test 的分级授权；
本次表格整理不新增任何实验授权。

## 6. 不作为论文表的现有材料

- TimeAlign Table 6 transcription/source inconsistency audit；
- Main II H720 continuity 与 published-mean signed-deviation audit；
- resource smoke、checkpoint hash 与 completion manifests；
- unified penalty、NCHPD/CHPD、scope utilization、diversity/oracle headroom 等 mechanism diagnostics：当前仍按 figures/analysis blocks 组织，尚未冻结为表；
- case studies：按 figures 组织。

这些材料用于 provenance、protocol disclosure 或图形证据，不应为了增加表格数量而重复进入正文。
