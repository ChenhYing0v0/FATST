# ISCF-BSCA 论文实验表格总账

## 1. 总账边界

本文件是当前 **paper-facing experiment tables** 的统一入口。它汇总已经完成的
正式表、正文中已经确定但尚待实验的表，以及明确只放 Appendix 的表。历史中间版、
source transcription audit、checkpoint manifest 和 smoke 统计不作为论文表重复列入。
机器可读契约见 `configs/iscf_bsca_paper_table_registry.json`。

状态含义：

- `complete_hash_frozen`：结果、来源角色与输出 hash 已冻结；未经用户显式解冻不得改写；
- `complete_hash_frozen_h5a_synced`：H5A选择的完整dataset-level profiles已获授权替换，结果与输出hash重新冻结；
- `complete_hash_frozen_h5d_bs16_lr2p4_synced`：用户指定eligible H5D profile作为当前ETTh1 paper row，原H5D gate历史保留，结果与输出hash重新冻结；
- `complete_hash_frozen_horizon_loader_reaudit`：Main II已使用既有H720 checkpoints在各fixed-H official test loaders上完整重算，continuity、origin-count与输出hash均已冻结；
- `complete_hash_frozen_partial_attribution_3_of_4_controls`：Core-Ablation完整矩阵与hash已冻结，但仅3/4 matched controls通过，结论必须按control收窄；
- `complete_presentation_aligned`：完整结果已通过审计，并已按 Main I 视觉契约生成；
- `complete_limited_surface`：结果完整，但只覆盖明确列出的部分 systems；
- `author_fixed_controls_prelaunch_pending`：author已固定control identities，但implementation、预算与formal matrix仍待独立prelaunch；
- `source_patch_and_retrain_required_prelaunch_pending`：需要先完成 source/protocol gate，尚无正式结果；
- `deferred_outside_current_core_closure`：不属于当前正文闭合范围，恢复前需要新的设计与授权。

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
H720 checkpoint在四个source-native fixed-H official test loaders上的prefix forecasts。

## 3. 当前论文表格清单

| ID | 论文位置 | 状态 | 当前规模 | Canonical artifact | 能支持的结论 |
| --- | --- | --- | --- | --- | --- |
| `Main-I` | 正文主结果 | `complete_hash_frozen_h5d_bs16_lr2p4_synced` | 14 systems × 7 datasets × 4 H；392 system–dataset–H rows | `analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_i_h5d_bs16_lr2p4_synced_20260813/table_iscf_bsca_main_i_qdf.tex` | ISCF-BSCA 与 separately optimized horizon-specific systems 的 system-level accuracy competitiveness；不作 matched mechanism attribution |
| `Main-II` | 正文主结果 | `complete_hash_frozen_horizon_loader_reaudit` | 8 systems × 7 datasets × 4 H；224 cells，63 external checkpoint objects / 252 formal evaluations | `analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_ii_horizon_loader_reaudit_20260813/formal_results/table/table_iscf_bsca_main_ii.tex` | one-model-for-all-horizons system competitiveness；不作 BSCA/decoder attribution |
| `Main-I-Exchange` | Supplementary companion | `complete_limited_surface` | ISCF-BSCA / TimeAlign / QDF × Exchange × 4 H | `analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_i_h5d_bs16_lr2p4_synced_20260813/table_exchange_companion.tex` | Exchange 的部分系统背景；不能表述为完整 Main I extension |
| `Efficiency` | 正文 supporting result | `complete_hash_frozen_tradeoff_supported_no_uniform_compute_advantage` | 5 systems × 7 datasets；35/35 service units / 77 checkpoint objects | `analysis/iscf_bsca_paper_experiment_consolidation_20260731/efficiency_20260814/formal_results/table/table_iscf_bsca_efficiency.tex` | 支持one-model consolidation、architectural CHPC及相对four-model TimeAlign/QDF的parameter/storage reduction；不支持uniform training/latency/computation advantage |
| `Core-Ablation` | 正文 mechanism attribution | `complete_hash_frozen_partial_attribution_3_of_4_controls` | 5 variants × 5 datasets × 4 H；100/100 cells，25 checkpoints（5 reused + 20 new） | `analysis/iscf_bsca_paper_experiment_consolidation_20260731/core_ablation_20260814/formal_results/table/table_iscf_bsca_core_ablation.tex` | 支持BSCA objective、scope-specific projections与multi-scope design；不支持learned Target-Adaptive Allocation相对equal fusion的独立accuracy utility |
| `Decoder-Transfer` | 正文 transfer evidence | `complete_hash_frozen_cross_backbone_portability_gate_failed` | 2 backbones × 3 decoder columns × 5 datasets × 4 H；30 checkpoints / 120/120 cells | `analysis/iscf_bsca_paper_experiment_consolidation_20260731/decoder_transfer_20260814/formal_results/table/table_iscf_bsca_decoder_transfer.tex` | DLinear-style相对gate通过但PatchTST-style失败，因此不支持总体cross-backbone portability claim |
| `Ablation-Sensitivity` | Appendix | `deferred_outside_current_core_closure` | random partition、scope count、$\lambda$ sensitivity | 尚无正式表 | 当前不纳入核心闭合；若恢复仅作sensitivity/robustness，canonical grouping不作为既定正向结论 |

## 4. 已完成主表结果摘要

### Main I

- ISCF-BSCA：31/56 best、18/56 second；
- TimeAlign、QDF、AMD、SimpleTM 为本地 official-source/native reproduction；
- 其余 columns 为 published context，来源与 protocol 差异必须保留在 caption/Methods；
- 当前ETTh1采用用户指定的eligible H5D profile `h5d_bs16_lr2p4`；ECL、Solar及其余datasets保持H5A同步版不变；
- 当前版由 `main_i_h5d_bs16_lr2p4_freeze_manifest.json` hash冻结；standalone LaTeX与A3 review PDF分别为`table_iscf_bsca_main_i_standalone.tex`和`output/pdf/iscf_bsca_main_i_h5d_bs16_lr2p4_20260813.pdf`。

### Main II

- 完整性：63个external H720 checkpoint objects、63个unique hashes、252/252 fixed-H evaluations、196个external aggregate cells；连同28个ISCF-BSCA cells形成224-cell主表；
- 当前ISCF-BSCA仍采用已冻结dataset-level profiles，其中ETTh1为用户指定的eligible H5D profile `h5d_bs16_lr2p4`；本轮只重算全部external baselines，不改Main I；
- 49/49个H720 same-checkpoint continuity checks与63/63个origin-count monotonicity checks均通过；
- 三位小数显示口径：41/56 best、13/56 second，共54/56 metric cells 位于 top-2；
- Main II 现已与 Main I 对齐 dataset order、year label、best/second emphasis、column spacing 与 required packages；
- 可直接编译的完整source为`table_iscf_bsca_main_ii_standalone.tex`，A3 landscape
  review PDF为`output/pdf/iscf_bsca_main_ii_horizon_loader_20260813.pdf`；正式manuscript仍使用
  原始table fragment；
- external source contracts 不 matched，因此该表不能兑现 component effectiveness 或 decoder portability claims。

### Core-Ablation

- 完整性：5 variants × 5 datasets × 4 horizons = 100/100 cells；Full复用5个exact checkpoints，四个controls完成20/20 matched end-to-end runs与20个unique hashes；
- Full macro MSE/MAE=`0.308549/0.346278`；
- `w/o BSCA`、`Shared Scope Projection`、`Fixed Scope (s=144)`分别通过预注册gate，Full的macro MSE收益为`2.401%`、`1.416%`、`1.796%`；
- `w/o Target-Adaptive Allocation`未通过：Full macro MSE反而高`0.039%`，dataset/horizon MSE wins仅`2/5`与`0/4`；
- 因此证据状态是3/4 controls通过的`performance_partial_pass`，不得写成“all core components are effective”；
- canonical audit=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/core_ablation_20260814/formal_results/result_and_table_audit.md`；standalone LaTeX=`table/table_iscf_bsca_core_ablation_standalone.tex`；review PDF=`output/pdf/iscf_bsca_core_ablation_20260814.pdf`。

### Decoder-Transfer

- 完整性：30/30 from-scratch end-to-end checkpoints、30 unique immutable hashes、120/120 official-test cells；
- DLinear-style中+ISCF-BSCA相对Original Decoder的macro MSE/MAE改善`15.702%/8.184%`，赢4/5 dataset MSE means，预注册gate通过；
- PatchTST-style中+ISCF-BSCA相对Original Decoder为`-0.733%/-0.062%`，只赢2/5 dataset MSE means，预注册gate失败；
- +ISCF-BSCA相对+ISCF在PatchTST-style改善1.385% MSE并赢16/20 MSE cells，但仍未超过Original Decoder，不能转写为decoder portability；
- DLinear-style的ETTh1/ETTh2绝对结果存在profile/optimization风险，其正向相对gate只作限定证据；
- canonical audit=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/decoder_transfer_20260814/formal_results/result_and_table_audit.md`；review PDF=`output/pdf/iscf_bsca_decoder_transfer_20260814.pdf`。

### Efficiency

- 完整性：35/35 service units、77/77 immutable checkpoint objects；所有finite/CV gates通过，最大all-H round CV=`0.0368`；测量后checkpoint hash复核通过；
- ISCF-BSCA以1个model提供architectural CHPC；七dataset macro为`2.926M` parameters、`17.68 MiB` checkpoints、`2.028 logged GPU h`、single/all-H latency=`10.306/10.318 ms`、peak memory=`38.8 MiB`；
- 相对TimeAlign/QDF four-model services，ISCF-BSCA分别减少`72.8%/45.2%` deployed parameters和`81.5%/13.3%` checkpoint storage；
- 负向边界：ISCF-BSCA all-H latency分别为TimeAlign/QDF/DLinear-prefix/PatchTST-prefix的`2.88×/2.33×/24.04×/2.66×`，logged training time也不领先，因此不得声称uniform compute efficiency；
- canonical audit=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/efficiency_20260814/formal_results/result_and_table_audit.md`；review PDF=`output/pdf/iscf_bsca_efficiency_20260814.pdf`。

## 5. 尚未完成表格的最小闭合顺序

1. `Figure 5`：冻结Scope Probability、aggregate utilization、regional preference/error与illustrative trajectory的统计和selection contract；不得用该图补救allocation control的failed effectiveness。
2. `Ablation-Sensitivity`：当前deferred，不属于正文核心闭合；若后续恢复，须单独冻结Appendix grid且不得因已见结果选择性扩张。

以上未完成实验仍分别需要独立prelaunch与相应授权；Core-Ablation scope已闭合，不自动追加seeds、control或allocation redesign。

## 6. 不作为论文表的现有材料

- TimeAlign Table 6 transcription/source inconsistency audit；
- Main II H720 continuity 与 published-mean signed-deviation audit；
- resource smoke、checkpoint hash 与 completion manifests；
- exact CHPC/CHPD、scope utilization与scope-wise regional preference/error等mechanism diagnostics：当前仍按Figure 5/analysis blocks组织，尚未冻结为表；realized allocation value明确不进入当前计划；
- qualitative trajectory：并入Figure 5，从Full相对frozen matched control提升最清晰的样本之一选择，caption披露comparator、split和selection rule；只作illustrative evidence，不作representative或prevalence claim；不设置独立failure-case figure。

这些材料用于 provenance、protocol disclosure 或图形证据，不应为了增加表格数量而重复进入正文。
