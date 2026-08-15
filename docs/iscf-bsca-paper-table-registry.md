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
- `complete_hash_frozen_author_corrected_20260815`：作者提供的修正复跑值已按指定scope替换，其余cells保持上一冻结版；Avg.、排名、LaTeX、PDF与输出hash已重新冻结；
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
| `Main-I` | 正文主结果 | `complete_hash_frozen_author_corrected_20260815` | 14 systems × 7 datasets × 4 H；392 system–dataset–H rows | `analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_tables_author_corrected_20260815/main_i/table_iscf_bsca_main_i_qdf.tex` | ISCF-BSCA 与 separately optimized horizon-specific systems 的 system-level accuracy competitiveness；不作 matched mechanism attribution |
| `Main-II` | 正文主结果 | `complete_hash_frozen_author_corrected_20260815` | 8 systems × 7 datasets × 4 H；224 cells | `analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_tables_author_corrected_20260815/main_ii/table_iscf_bsca_main_ii.tex` | one-model-for-all-horizons system competitiveness；不作 BSCA/decoder attribution |
| `Main-I-Exchange` | Supplementary companion | `complete_limited_surface` | ISCF-BSCA / TimeAlign / QDF × Exchange × 4 H | `analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_i_h5d_bs16_lr2p4_synced_20260813/table_exchange_companion.tex` | Exchange 的部分系统背景；不能表述为完整 Main I extension |
| `Efficiency` | 正文 supporting result | `complete_hash_frozen_tradeoff_supported_no_uniform_compute_advantage` | 5 systems × 7 datasets；35/35 service units / 77 checkpoint objects | `analysis/iscf_bsca_paper_experiment_consolidation_20260731/efficiency_20260814/formal_results/table/table_iscf_bsca_efficiency.tex` | 支持one-model consolidation、architectural CHPC及相对four-model TimeAlign/QDF的parameter/storage reduction；不支持uniform training/latency/computation advantage |
| `Core-Ablation` | 正文 mechanism attribution | `complete_hash_frozen_partial_attribution_3_of_4_controls` | 5 variants × 5 datasets × 4 H；100/100 cells，25 checkpoints（5 reused + 20 new） | `analysis/iscf_bsca_paper_experiment_consolidation_20260731/core_ablation_20260814/formal_results/table/table_iscf_bsca_core_ablation.tex` | 支持BSCA objective、scope-specific projections与multi-scope design；不支持learned Target-Adaptive Allocation相对equal fusion的独立accuracy utility |
| `Decoder-Transfer` | 正文 transfer evidence | `complete_hash_frozen_v2p1_patchtst_hpo_portability_gate_failed` | 2 backbones × 3 decoder columns × 5 datasets × 4 H；30 checkpoint objects / 120/120 cells | `analysis/iscf_bsca_paper_experiment_consolidation_20260731/decoder_transfer_patchtst_v2p1_20260815/formal_results/table/table_iscf_bsca_decoder_transfer.tex` | DLinear-style相对gate通过；validation-selected PatchTST-style仍失败，因此不支持总体cross-backbone portability claim |
| `Ablation-Sensitivity` | Appendix | `deferred_outside_current_core_closure` | random partition、scope count、$\lambda$ sensitivity | 尚无正式表 | 当前不纳入核心闭合；若恢复仅作sensitivity/robustness，canonical grouping不作为既定正向结论 |

## 4. 已完成主表结果摘要

### Main I

- ISCF-BSCA：44/56 best、9/56 second；seven-dataset macro MSE/MAE=`0.260714/0.306107`；
- 作者修正scope为ISCF与TimeAlign全部28 cells、SimpleTM Solar 4 cells、**TVNet ETTh2 4 cells**；共64个standard rows；
- 修正值按作者提供的三位小数直接冻结，不推断额外精度或checkpoint hash；未列出的baseline cells保持上一冻结版的official-local / published-context role；
- 当前canonical table fragment=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_tables_author_corrected_20260815/main_i/table_iscf_bsca_main_i_qdf.tex`；standalone LaTeX=`main_i/table_iscf_bsca_main_i_standalone.tex`；A3 review PDF=`output/pdf/iscf_bsca_main_i_author_corrected_20260815.pdf`。

### Main II

- 完整性：8 systems × 7 datasets × 4 H=`224` cells；上一版63 external H720 objects / 252 formal evaluations的审计保留为未修正cells的来源证据；
- 作者修正scope为ISCF与TimeAlign全部28 cells、SimpleTM Solar 4 cells、PatchTST ETTh2 4 cells；共64个standard rows；修正cells不沿用被替换版本的checkpoint hashes；
- 三位小数显示口径：ISCF-BSCA 50/56 best、6/56 second，共56/56 metric cells位于top-2；seven-dataset macro MSE/MAE=`0.260714/0.306107`；
- Avg.采用显式decimal half-up rounding，已修复旧builder在`.xxx5`上的banker's-rounding差异；
- Main II 现已与 Main I 对齐 dataset order、year label、best/second emphasis、column spacing 与 required packages；
- 可直接编译的完整source为`analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_tables_author_corrected_20260815/main_ii/table_iscf_bsca_main_ii_standalone.tex`，A3 landscape review PDF为`output/pdf/iscf_bsca_main_ii_author_corrected_20260815.pdf`；正式manuscript使用同目录table fragment；
- external source contracts 不 matched，因此该表不能兑现 component effectiveness 或 decoder portability claims。

两张主表的共同freeze audit与hash manifest分别为
`analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_tables_author_corrected_20260815/result_and_freeze_audit.md`
和`analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_tables_author_corrected_20260815/freeze_manifest.json`。旧Main I/Main II冻结目录保留为historical snapshots，不再是canonical manuscript source。

### Core-Ablation

- 完整性：5 variants × 5 datasets × 4 horizons = 100/100 cells；Full复用5个exact checkpoints，四个controls完成20/20 matched end-to-end runs与20个unique hashes；
- Full macro MSE/MAE=`0.308549/0.346278`；
- `w/o BSCA`、`Shared Scope Projection`、`Fixed Scope (s=144)`分别通过预注册gate，Full的macro MSE收益为`2.401%`、`1.416%`、`1.796%`；
- `w/o Target-Adaptive Allocation`未通过：Full macro MSE反而高`0.039%`，dataset/horizon MSE wins仅`2/5`与`0/4`；
- 因此证据状态是3/4 controls通过的`performance_partial_pass`，不得写成“all core components are effective”；
- canonical audit=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/core_ablation_20260814/formal_results/result_and_table_audit.md`；standalone LaTeX=`table/table_iscf_bsca_core_ablation_standalone.tex`；review PDF=`output/pdf/iscf_bsca_core_ablation_20260814.pdf`。

### Decoder-Transfer

- v2.1完整性：5个validation-selected BSCA checkpoints + 5个matched ISCF checkpoints形成10/10 unique hashes与5/5 matched initialization pairs；新增formal test为40/40 cells，并复用v1 DLinear三arms与PatchTST Original的80 cells，combined table为120/120 cells；
- parent v2的50-run validation HPO只有40/50 unique hashes，因此parent artifact gate保持FAIL；v2.1只冻结五个互异selected checkpoints，未删除negative trials或追认parent gate；
- DLinear-style中+ISCF-BSCA相对Original Decoder的macro MSE/MAE改善`15.702%/8.184%`，赢4/5 dataset MSE means，预注册gate通过；
- HPO后PatchTST-style +ISCF-BSCA相对Original Decoder仍为`-0.436%/-0.568%`，只赢2/5 dataset means与9/20 MSE cells，预注册gate失败；
- +ISCF-BSCA相对matched +ISCF改善`0.912%` MSE、`0.448%` MAE并赢16/20 MSE cells，说明BSCA objective在replacement head内仍有作用；但两种replacement heads均未超过Original Decoder，不能转写为decoder portability；
- 相对v1，BSCA macro MSE改善0.295%，但MAE恶化0.506%；HPO缩小MSE deficit却未改变方向级结论；
- DLinear-style的ETTh1/ETTh2绝对结果存在profile/optimization风险，其正向相对gate只作限定证据；
- failure attribution：claim-level=`hypothesis_false_for_cross_backbone_portability_after_decoder_HPO`；design-level=`readout_or_head_design_wrong_for_PatchTST_representation_compatibility`，不否定BSCA objective本身；
- canonical audit=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/decoder_transfer_patchtst_v2p1_20260815/formal_results/result_and_table_audit.md`；review PDF=`output/pdf/iscf_bsca_decoder_transfer_v2p1_20260815.pdf`。

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
