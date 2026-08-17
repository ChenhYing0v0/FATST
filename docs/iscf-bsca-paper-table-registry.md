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
- `complete_author_corrected_aggregate_all_controls_positive_provenance_partial`：Core-Ablation作者复跑aggregate表已冻结且四项方向均正；新的per-horizon/checkpoint artifacts未同步，必须保留provenance限制；
- `complete_presentation_aligned`：完整结果已通过审计，并已按 Main I 视觉契约生成；
- `complete_limited_surface`：结果完整，但只覆盖明确列出的部分 systems；
- `author_fixed_controls_prelaunch_pending`：author已固定control identities，但implementation、预算与formal matrix仍待独立prelaunch；
- `source_patch_and_retrain_required_prelaunch_pending`：需要先完成 source/protocol gate，尚无正式结果；
- `deferred_outside_current_core_closure`：不属于当前正文闭合范围，恢复前需要新的设计与授权。
- `complete_mixed_mechanism_evidence`：diagnostic matrix与Figure已完成，但正负证据并存，结论必须保留失败边界。

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
| `Core-Ablation` | 正文 mechanism attribution | `complete_author_corrected_aggregate_all_controls_positive_provenance_partial` | 5 variants × 5 datasets × MSE/MAE + Avg；60 displayed metric cells；historical 100-cell audit retained | `analysis/iscf_bsca_paper_experiment_consolidation_20260731/core_ablation_20260814/formal_results/table/table_iscf_bsca_core_ablation.tex` | 作者修正aggregate表支持四项matched accuracy directions；per-horizon/checkpoint rerun provenance待补；Figure 5不支持reliable routing或causal specialization |
| `Decoder-Transfer` | 正文 transfer evidence | `complete_author_corrected_aggregate_both_backbones_all_columns_positive_provenance_partial` | 2 backbones × 2 systems × 3 datasets + Avg；32 displayed metric cells；historical 48-cell audit retained | `analysis/iscf_bsca_paper_experiment_consolidation_20260731/decoder_transfer_three_dataset_scope_20260816/table_decoder_transfer_three_dataset_framework.tex` | 作者修正aggregate表中两类backbones均为3/3 dataset MSE/MAE wins且16/16 comparator columns正向；不作内部ISCF-vs-BSCA attribution，per-H/hash provenance与five-dataset negative audit保留 |
| `Figure-5-Diagnostics` | 正文 mechanism behavior | `complete_mixed_mechanism_evidence` | 5 datasets × Full/Fixed artifacts；20 CHPC cells；全部validation rows + 1,280-row qualitative pool | `paper-figures/figure_iscf_bsca_mechanism.svg` | exact CHPC与scope-arm regional error heterogeneity；learned allocation接近均匀且仅8/40 utilization/error-preference cells一致；Figure 5自身不建立accuracy attribution、reliable routing或causal specialization |
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

- 2026-08-17作者提供复跑后的dataset-level four-horizon means；canonical table为5 variants × 5 datasets × MSE/MAE + Avg，共60 displayed metric cells；
- Full macro MSE/MAE=`0.305/0.344`，并在12/12 metric columns中为best；
- 相对`w/o BSCA`、`w/o Target-Adaptive Allocation`、`Shared Scope Projection`、`Fixed Scope (s=144)`，Full的display-precision macro MSE收益分别为`3.481%`、`1.613%`、`2.866%`、`3.175%`，五数据集MSE/MAE方向均为5/5；
- 新截图未附per-horizon raw files、selector与checkpoint hashes；旧100-cell及immutable manifest保留为historical audit，不冒充新表provenance；
- Table 4可支持四项aggregate accuracy contribution，但Figure 5仍不支持reliable region-best routing或causal specialization；
- canonical audit=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/core_ablation_20260814/formal_results/result_and_table_audit.md`；correction manifest=`author_correction_freeze_manifest_20260817.json`；standalone LaTeX=`table/table_iscf_bsca_core_ablation_standalone.tex`；review PDF=`output/pdf/iscf_bsca_core_ablation_20260814.pdf`。

### Decoder-Transfer

- 正文范围按作者要求收窄为Weather、ETTm1、ETTm2；该范围在观察five-dataset结果后确定，因此标为`author_refined_posthoc_scope`，ETTh1/ETTh2负向证据移至完整audit/limitations而非删除；
- 正文不再区分ISCF与BSCA内部贡献，只比较完整ISCF-BSCA framework与各backbone的native Original Decoder；Core-Ablation独立承担component attribution；
- 作者修正aggregate表中，DLinear-style与PatchTST-style完整framework相对Original分别为`+5.611%/+3.604%`与`+2.128%/+1.274%` display-precision macro MSE/MAE gains；两者均为3/3 dataset MSE/MAE wins，合计16/16 comparator metric columns正向；
- correction manifest=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/decoder_transfer_three_dataset_scope_20260816/author_correction_freeze_manifest_20260817.json`；新的per-horizon raw files、selectors、profiles与checkpoint hashes仍未同步；
- 新截图未附per-horizon raw files、selectors、profiles或checkpoint hashes；旧48-cell与five-dataset artifacts保留为historical audit，不冒充修正表直接provenance；
- 当前结果已闭合framework-level portability claim，不增加BSCA HPO，也不补Weather/ETTm1 `+ISCF` controls；
- three-dataset audit=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/decoder_transfer_three_dataset_scope_20260816/result_and_next_gate.md`；canonical PDF=`output/pdf/iscf_bsca_decoder_transfer_three_dataset_framework_20260816.pdf`；

以下为five-dataset historical audit，继续作为supplementary/limitations evidence保留，不覆盖上述三数据集paper candidate：

- v2.1完整性：5个validation-selected BSCA checkpoints + 5个matched ISCF checkpoints形成10/10 unique hashes与5/5 matched initialization pairs；新增formal test为40/40 cells，并复用v1 DLinear三arms与PatchTST Original的80 cells，combined table为120/120 cells；
- parent v2的50-run validation HPO只有40/50 unique hashes，因此parent artifact gate保持FAIL；v2.1只冻结五个互异selected checkpoints，未删除negative trials或追认parent gate；
- DLinear-style中+ISCF-BSCA相对Original Decoder的macro MSE/MAE改善`15.702%/8.184%`，赢4/5 dataset MSE means，预注册gate通过；
- HPO后PatchTST-style +ISCF-BSCA相对Original Decoder仍为`-0.436%/-0.568%`，只赢2/5 dataset means与9/20 MSE cells，预注册gate失败；
- +ISCF-BSCA相对matched +ISCF改善`0.912%` MSE、`0.448%` MAE并赢16/20 MSE cells，说明BSCA objective在replacement head内仍有作用；但两种replacement heads均未超过Original Decoder，不能转写为decoder portability；
- 相对v1，BSCA macro MSE改善0.295%，但MAE恶化0.506%；HPO缩小MSE deficit却未改变方向级结论；
- DLinear-style的ETTh1/ETTh2绝对结果存在profile/optimization风险，其正向相对gate只作限定证据；
- failure attribution：claim-level=`hypothesis_false_for_cross_backbone_portability_after_decoder_HPO`；design-level=`readout_or_head_design_wrong_for_PatchTST_representation_compatibility`，不否定BSCA objective本身；
- auxiliary iTransformer rescue也为负：14-profile × 5-dataset test-tuned HPO相对旧iTransformer BSCA改善macro MSE/MAE `2.128%/1.719%`，但相对native Original仍为`-0.505%/-0.750%`；因此canonical two-backbone table不改，且不得用4/5 dataset MSE wins掩盖ETTh1与macro fail；
- iTransformer auxiliary audit=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/decoder_transfer_itransformer_hpo_v2_20260816/formal_results/result_and_decision.md`；该证据为test-tuned negative result，不是新增paper table column；
- PatchTST parent-HPO full unique audit补测35个未测unique checkpoints并闭合40 unique/160 cells、200 expanded cells与220 candidate cells；mean-MSE selector相对Original为`+0.134% MSE / -0.253% MAE`，MSE-only转正但joint gate失败；161,051种dataset-profile组合中没有macro MSE/MAE双正向方案；
- PatchTST full-audit result=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/decoder_transfer_patchtst_test_tuned_full_20260816/formal_results/result_and_decision.md`；v2.1三臂表与PDF保留为historical diagnostic，正文canonical已由48-cell framework table supersede；
- canonical audit=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/decoder_transfer_patchtst_v2p1_20260815/formal_results/result_and_table_audit.md`；review PDF=`output/pdf/iscf_bsca_decoder_transfer_v2p1_20260815.pdf`。

### Efficiency

- 完整性：35/35 service units、77/77 immutable checkpoint objects；所有finite/CV gates通过，最大all-H round CV=`0.0368`；测量后checkpoint hash复核通过；
- ISCF-BSCA以1个model提供architectural CHPC；七dataset macro为`2.926M` parameters、`17.68 MiB` checkpoints、`2.028 logged GPU h`、single/all-H latency=`10.306/10.318 ms`、peak memory=`38.8 MiB`；
- 相对TimeAlign/QDF four-model services，ISCF-BSCA分别减少`72.8%/45.2%` deployed parameters和`81.5%/13.3%` checkpoint storage；
- 负向边界：ISCF-BSCA all-H latency分别为TimeAlign/QDF/DLinear-prefix/PatchTST-prefix的`2.88×/2.33×/24.04×/2.66×`，logged training time也不领先，因此不得声称uniform compute efficiency；
- canonical audit=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/efficiency_20260814/formal_results/result_and_table_audit.md`；review PDF=`output/pdf/iscf_bsca_efficiency_20260814.pdf`。

### Figure 5 Diagnostics

- 完整性：5 datasets × 2 frozen validation artifact roles=`10/10` objects；training/test=`0/0`；
- numerical CHPC：20/20 paper-facing cells的maximum absolute CHPD均为0；
- aggregate learned probabilities范围为`0.18258--0.21479`，总体接近均匀；
- macro region-best scope随future region变化，最大best-to-worst excess MSE=`6.123%`；
- highest-utilization scope仅在`8/40` dataset-region cells等于lowest-MSE scope，因此只支持descriptive arm heterogeneity，不支持successful allocation；
- qualitative row从完整1,280-row validation pool按预冻结MSE rule选择，只作illustrative evidence；
- canonical audit=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/figure5_mechanism_diagnostics_20260816/result_and_decision.md`；PDF=`output/pdf/iscf_bsca_figure5_20260816.pdf`。

## 5. 当前正文实验闭合状态

Figure 5完成后，Main I、Main II、Efficiency、Core-Ablation、Figure 5与Decoder-Transfer均已有canonical artifact。`Ablation-Sensitivity`继续deferred，不属于当前正文核心闭合；若后续恢复，须单独冻结Appendix grid且不得因已见结果选择性扩张。

当前不自动追加seeds、controls、allocation redesign或新的formal test。

## 6. 不作为论文表的现有材料

- TimeAlign Table 6 transcription/source inconsistency audit；
- Main II H720 continuity 与 published-mean signed-deviation audit；
- resource smoke、checkpoint hash 与 completion manifests；
- exact CHPC/CHPD、scope utilization与scope-wise regional preference/error等mechanism diagnostics：已按Figure 5/analysis blocks完成，不另建论文表；realized allocation value明确排除；
- qualitative trajectory：已并入Figure 5，并按冻结规则从Full相对`Fixed Scope (s=144)`的完整1,280-row validation pool选择；只作illustrative evidence，不作representative或prevalence claim；不设置独立failure-case figure。

这些材料用于 provenance、protocol disclosure 或图形证据，不应为了增加表格数量而重复进入正文。
