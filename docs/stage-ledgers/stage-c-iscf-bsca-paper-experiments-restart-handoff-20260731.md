# Stage C ISCF-BSCA Paper-Experiments Restart Handoff

## 0. Authority and use

本文件是2026-07-31之后新对话并行推进ISCF-BSCA论文实验工作的current首读入口。
它与
`docs/stage-ledgers/stage-c-iscf-bsca-paper-writing-restart-handoff-20260731.md`
分别管理experiment workstream与manuscript-writing workstream。两者共享同一冻结
paper candidate与claim boundary，但互不替代。

如果本文件与旧聊天、旧D22 handoff、archive或历史实验计划冲突，以本文件、
`configs/paper_facing_evaluation_protocol.json`及三份主线文档顶部的最新cursor
为准。

新实验对话必须严格按以下顺序读取：

1. `AGENTS.md`；
2. 本handoff；
3. `docs/iscf-bsca-paper-architecture.md`；
4. `configs/paper_facing_evaluation_protocol.json`；
5. `docs/paper-drafts/iscf-bsca-introduction-initial-draft.md`；
6. `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_bsca_v1_confirmation_step9_10_20260722/step9_10_three_seed_result_and_paper_handoff.md`；
7. `docs/code-explanation/stage-c-iscf-bsca-v1.md`；
8. `configs/stage_c_iscf_bsca_v1.json`；
9. `configs/stage_c_iscf_bsca_v1_confirmation.json`；
10. `analysis/stage_c_post_d21_unconstrained_reset_20260720/post_d24_paper_story_and_modern_baseline_gap_audit.md`；
11. `analysis/stage_c_post_d21_unconstrained_reset_20260720/sc_mnb_step13_source_and_protocol_audit.md`；
12. `configs/stage_c_modern_native_baseline_protocol.json`；
13. `docs/paper-mainline.md`；
14. `docs/research-roadmap.md`；
15. `docs/stage-ledgers/stage-c-unified-forecasting-redesign.md`；
16. `docs/stage-ledgers/stage-c-iscf-bsca-paper-writing-restart-handoff-20260731.md`。

只有在artifact inventory确认确有需要时，才沿上述报告中的链接追溯更早analysis。
不要从D17--D24或SIFF路线重新开始method search。

## 1. Current authoritative state

| Field | Content |
| --- | --- |
| `project` | `R_2026_FATST` |
| `stage` | `StageC-UVHF paper consolidation` |
| `handoff_date` | `2026-07-31` |
| `paper_candidate` | exact frozen `ISCF-BSCA-v1` |
| `paper_core_status` | `passed_core_candidate_ready_for_paper_consolidation` |
| `active_workstream` | paper-facing experiment execution |
| `active_experiment_step` | Main I/Main II H5A synced and hash frozen；H5B ETTh1 Step 6 frozen -> Step 8 remote resource gate |
| `introduction_status` | `v0.9-author-refinement`=`temporarily_frozen_usable` |
| `active_method_search` | none |
| `local_audit_and_design_authorized` | true |
| `local_protocol_patch_authorized` | true；限H5B ETTh1 config/runner/checker与table sync chain |
| `remote_training_authorized` | true；限H5B 36 profiles、三GPU resource smoke及train/validation |
| `test_tuned_hpo_authorized` | true；限36/36 manifest后的ETTh1 dataset-level shared-profile selection |
| `formal_test_authorized` | conditional true；36/36 immutable manifest后一次完整144-row formal test，禁止partial execution |
| `next_action` | exact commit push -> remote disk/GPU audit -> 36/36 resource smoke -> three-GPU full training；manifest前test=0 |
| `conditional_next` | 任何新baseline仍需独立source/protocol gate；不得改写AMD/SimpleTM native role为matched attribution |

2026-08-13用户授权H5A table mutation并继续ETTh1-only HPO。H5A selected profiles已替换Main I/Main II中ETTh1/ECL/Solar的12个ISCF cells；Main I=`31/56` best、`17/56` second，Main II=`28/56` best、`25/56` second，standalone LaTeX/PDF与hash均重新冻结。H5B审计25个ETTh1 profiles后固定36个expanded-range trials：重点为LR fine grid、`L576--960` context/patch、regularization interactions与rank，统一120 epochs/patience24；capacity仅保留一个moderate probe，LayerNorm保持开启。Canonical prelaunch=`analysis/iscf_bsca_main_v1_hpo_20260731/h5b_etth1_expanded_search_20260813/design_and_prelaunch_gate.md`。Decision=`H5B_frozen_authorized_remote_resource_gate_next`。

2026-08-08用户要求暂时冻结Main I，并把Main II改为H720-trained one-model-all-horizons benchmark。Main I冻结manifest记录14 models × 7 dense datasets × four H、392 rows、29/56 best、19/56 second及全部关键hash。Main II v1包含ISCF-BSCA、TimeAlign、QDF、AMD、SimpleTM、iTransformer、PatchTST、DLinear；每个external baseline逐dataset训练/复用一个H720 model，并从同一H720 test tensor裁剪H96/H192/H336。49个checkpoint objects可复用，21个iTransformer/PatchTST/DLinear H720 jobs需新训练；PatchTST/DLinear Solar无official script/loader，固定为source-patch-required。Exchange因Main I H720 anchors不完整而deferred。Canonical protocol=`configs/iscf_bsca_main_ii_h720_prefix_protocol.json`；prelaunch=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_ii_h720_prefix_20260808/design_and_prelaunch_gate.md`。当前只完成design/source audit，Tier A/B/C仍为false。

2026-08-10用户判断Main II中ETTh1/ECL/Solar优势不足并显式重启targeted HPO。H5A冻结
48个seed2021 profiles，每dataset 16个；不改architecture/objective/scales/inference graph。
当前三dataset best counts=`1/8,0/8,4/8`，最低目标=`2/8,1/8,5/8`，同时要求
four-H mean MSE/MAE都不比当前profile退化超过0.5%。训练阶段test=0；48/48 immutable
manifest后完整formal test已授权。H5B、extra seeds、architecture redesign及自动表格mutation
未授权。Canonical prelaunch=`analysis/iscf_bsca_main_v1_hpo_20260731/h5a_main_ii_weak_dataset_search_20260810/design_and_prelaunch_gate.md`。Decision=`H5A_48_profile_targeted_HPO_frozen_remote_authorized`。

H5A exact commit=`7544f76d`已通过48/48 resource smoke，48 checkpoints/metrics/logs、
48 unique hashes、test=0且failure token=0。Full queue于2026-08-10 15:13:32在GPU0--2
启动，PID=`2375625`；前三个ECL jobs均进入epoch1，memory约1.5--1.9 GiB/GPU。
Canonical launch=`analysis/iscf_bsca_main_v1_hpo_20260731/h5a_main_ii_weak_dataset_search_20260810/remote_launch.md`。Decision=`H5A_resource_gate_pass_training_active_test_zero`。

H5A full queue于2026-08-12 20:24:03完成48/48，training test=0且failure token=0。
Artifact/provenance/four-H validation selector/numeric-health audit通过，48个checkpoint hashes
全部唯一。Immutable manifest SHA256=`ee5940c8f66aceab5710f17a4bc8ce2efb9ae3c44fa9cec1459fcd9589fe6643`；
用户于2026-08-13授权继续完整formal test。Canonical gate=
`analysis/iscf_bsca_main_v1_hpo_20260731/h5a_main_ii_weak_dataset_search_20260810/training_result_and_formal_test_gate.md`。Decision=`H5A_training_complete_manifest_frozen_formal_test_authorized`。

Exact commit=`cb496f31`通过formal-test preflight；once-only 48-checkpoint/192-standard-row
queue于2026-08-13 00:34:37启动，PID=`4169962`。Atomic publication和ABORT gate启用，
48/48前不得selector或partial table mutation。Canonical launch=
`analysis/iscf_bsca_main_v1_hpo_20260731/h5a_main_ii_weak_dataset_search_20260810/formal_test_launch.md`。Decision=`H5A_formal_test_active`。

H5A formal test于2026-08-13 01:00:46完成全局复核：48/48 checkpoints、192/192
standard rows、48 immutable hashes、errors=0。Frozen selector选择ETTh1
`h5a_lr3p5e4`、ECL `h5a_seq336_p1`、Solar `h5a_seq512_p4_lr2p5e4`，best cells
由`1/0/4`提高到`2/1/6`；target total=`9/24`，projected global Main II=`28/56`。
所有mean guards、dataset targets、Solar MAE与global target通过。Canonical result=
`analysis/iscf_bsca_main_v1_hpo_20260731/h5a_main_ii_weak_dataset_search_20260810/formal_test_result_and_main_ii_selection.md`。
Decision=`H5A_success_gate_pass_selection_frozen_table_mutation_not_authorized`。

H4K train/validation与formal test均已完成，但terminal effectiveness gate失败。H4L只扩大ETTm2/Weather
dataset-level HPO范围：48个profiles与117个历史profiles零重复，其中四项借鉴TimeAlign official encoder
parameter coupling，但不复制其head/loss。H4L 48/48 training artifacts与unique checkpoint hashes已通过audit，
manifest冻结后执行的一次完整formal test已48/48、192/192完成，checkpoint hashes全部immutable。165-trial joint selector只把ETTm2从0/8提高到1/8，Weather保持2/8；global MSE/MAE/combined=`15/28,16/28,31/56`，未通过`20/28,20/28,40/56` gates。Failure attribution=`search_space_performance_shortfall`，rollback=Step 6 strategy decision。
不得为了兑现Introduction P6而跳过controls、补选有利数据集，或把historical test结果当成untouched holdout。

2026-08-04用户进一步授权：先清理remote result root，再同时推进ETTm2/Weather高影响参数HPO和TimeAlign official reproduction。清理已删除7个resource-smoke目录与157个nonselected diagnostic NPZ，保留全部165个metrics/invariants/checkpoints/manifests/logs和8个selected NPZ；精确释放约36.51 GiB，quota由201G降至165G。H4M冻结24个历史effective-profile nonduplicates，主轴为ETTm2 `patch_num × learning_rate`和Weather `seq_len × patch_num / learning_rate / mode_rank`；TimeAlign冻结8个seed2021 fixed-H jobs并标记为`official-source model/config + FATST test-hygiene/artifact adapter`。Canonical prelaunch=`analysis/iscf_bsca_main_v1_hpo_20260731/h4m_parameter_impact_and_parallel_timealign_prelaunch_20260804.md`。

2026-08-05 H4M formal test已24/24 checkpoints、96/96 standard cells完整结束，checkpoint hashes全部immutable。H1--H4M累计189 trials的冻结selector得到MSE/MAE/combined=`17/28,16/28,33/56`：ETTm2保持1/8，Weather提高到4/8；legal selector、unrestricted single-profile upper bound与per-cell diagnostic oracle均为33/56，因此40/56目标在当前search pool内不可达。TimeAlign ETTm2/Weather 8/8 official-native reproduction也通过artifact audit，数值与paper three-run mean非常接近，但仅作native external role。Failure attribution=`search_space_performance_shortfall`，rollback=Step 6；automatic H4N=false。Canonical result=`analysis/iscf_bsca_main_v1_hpo_20260731/h4m_test_result_and_joint_hpo_decision_20260805.md`。

2026-08-05用户随后显式要求重点优化Weather并扩大超参数范围。H4N冻结40个Weather-only seed2021 profiles，按16个context×LR interpolation、8个LR wide boundary、8个patch geometry、5个rank与3个capacity profiles组成；与H1--H4M 189个历史effective profiles零重复。主目标改为Weather four-H MSE/MAE normalized relative mean最小，只有0.1% near-tie才使用lead cells，保持one profile shared by four H。训练统一120 epochs/patience24，training test=0；40/40 immutable manifest后一次完整160-cell formal test已授权。Canonical prelaunch=`analysis/iscf_bsca_main_v1_hpo_20260731/h4n_weather_wide_matrix_and_prelaunch_20260805.md`。H4O/extra seeds/architecture redesign=false。

H4N exact commit=`ba17fc9`已在remote通过40/40 resource smoke：40 checkpoints、40 metrics、40 logs，test=0且无OOM/NaN/Inf/Traceback。Full three-GPU queue于2026-08-05 10:47:16启动，PID=`1397808`；前三个jobs均进入epoch2，observed memory约1.6--1.7 GiB/GPU。预计15--26 wall-hours。Canonical launch=`analysis/iscf_bsca_main_v1_hpo_20260731/h4n_weather_remote_launch_20260805.md`。

2026-08-06 H4N 40/40 full train/validation artifacts已通过audit：40个checkpoint SHA256唯一、best epoch范围3--101、training test=0/40。Immutable manifest SHA256=`a0f152f9172acc193fe512001123b71aeae6d6d3ab1028c915074f24d54c1ed4`；formal-test config冻结40 checkpoints × four H=160 rows。Canonical gate=`analysis/iscf_bsca_main_v1_hpo_20260731/h4n_training_result_and_formal_test_gate_20260806.md`。Decision=`H4N_training_complete_40_checkpoint_manifest_frozen_formal_test_authorized`。

H4N formal test于2026-08-06 14:22:25完成：40/40 checkpoints、160/160 rows与hash immutability全部通过。Full-table selector选择`Weather__h4n_seq608_p19_lr2e5`，mean MSE/MAE=`0.214887/0.245821`，相对H4M current为+0.063%/-0.608%；full-table leads仍4/8，四项gates全fail。Frozen prelaunch historical score复用了legacy 5-baseline normalization；same-target conservative recomputation得到0.205% improvement并判fail。Legacy selector frontier仍33/56，wide-table displayed count仍29/56。Failure=`search_space_performance_shortfall + prelaunch_historical_score_target_mismatch`，rollback=Step 6，automatic H4O=false。Canonical report=`analysis/iscf_bsca_main_v1_hpo_20260731/h4n_test_result_and_weather_decision_20260806.md`。

2026-08-06 TimeAlign Main I reproduction已32/32完成：24个new fixed-H runs与8个reused ETTm2/Weather runs组成完整8-dataset matrix；32个checkpoint hashes唯一，artifact/provenance/numeric-health audit全部通过。7个shared datasets的TimeAlign 28/28 cells已由本地seed2021结果替换，ISCF-BSCA相对本地TimeAlign为MSE 20/28、MAE 17/28 cells领先，macro MSE/MAE低3.994%/1.491%；完整13-model displayed table中ISCF-BSCA为27/56 best、19/56 second。Exchange只形成ISCF/TimeAlign companion，因为TimeAlign无official Exchange script且Table 6无其他baseline Exchange evidence。Canonical report=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_i_final_local_timealign_20260806/main_i_result_and_source_audit.md`。HPO保持停止，other baseline training未授权。

2026-08-06用户进一步授权QDF在`seq_len=336`下完整本地复跑8 datasets × four H × seed2023。Exact rule为六个official scripts只改lookback、其余逐Hprofiles不变；Solar采用ECL-derived profile，Exchange采用ETTh1-derived profile且均标记source-informed。Formal matrix=32 fixed-H systems，先通过8个H720/test=0 resource smokes，再后台启动并停止驻守；32/32 artifact audit前当前L96 QDF table不变，禁止partial mix。Canonical prelaunch=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/qdf_main_i_l336_20260806/design_and_launch_gate.md`。

2026-08-06 QDF L336 formal queue已32/32完成：32 checkpoints、32 learned losses、32 metrics/configs/logs组成160-row manifest，checkpoint/loss hashes均32 unique，逐字段config与numeric/log gates通过。七个dense datasets的QDF macro MSE/MAE=`0.287511/0.331426`，相对ISCF-BSCA高`9.541%/7.508%`且仅领先1/56 cells。Main I现使用28个本地L336 dense cells，并在Exchange companion加入4个QDF cells；ISCF完整14-model best/second保持27/56、19/56。Canonical report=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/qdf_main_i_l336_20260806/result_and_table_audit.md`。QDF scope关闭，remote/formal authorization复位false。

2026-08-06用户授权 AMD 与 SimpleTM 的 Main I official-source reproduction。Matrix固定为7 dense datasets × four H × two baselines=56 table cells；AMD official scripts为`L512/seed2024/1 repeat`共28 checkpoints，SimpleTM为`L96/fix_seed2025/native itr`共82 checkpoints。SimpleTM upstream每epoch读取test，本轮adapter只移除该pass，保留validation early stopping并在每个selected checkpoint后一次formal test；`num_workers=0`只作runtime safety。AMD commit=`000d377...`且MIT；SimpleTM commit=`3c77d82...`但无upstream license，故不vendoring，只在repo-external exact checkout研究执行。先通过14个H720 one-epoch/test=0 smokes与35 GiB storage gate，再后台启动formal queue；110/110前不得替换published cells。Canonical prelaunch=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/amd_simpletm_main_i_reproduction_20260806/design_and_launch_gate.md`。

AMD/SimpleTM final resource gate于22:13:19通过：14/14 units、14 checkpoints、formal test=0；checkpoint 2× safety projection=4.470 GiB <35 GiB。Remote GitHub HTTPS两次超时后采用exact audited checkout rsync fallback，remote commit/clean/source hashes重新核验。Formal queue于22:14:19后台启动，PID=`4100426`，experiment commit=`014b068`；first wave为SimpleTM ECL/Solar/Weather，initial failure tokens=0。按用户要求不驻守，完成通知前禁止partial results读取或table replacement。

2026-08-07完成审计确认上述queue未完成：只创建3个SimpleTM incomplete units且在H96后终止。Upstream `setting` format遗漏repeat index `ii`，native `itr`的metrics虽产生但selected checkpoints覆盖；collector以`expected 3 checkpoints, found 1`拒绝，行为正确。Failure=`artifact_collection_defect`，不是method/optimization failure。Recovery adapter只追加checkpoint目录的repeat identity，保持official training与test协议不变；旧partial units保留但excluded。新root需重做7个SimpleTM no-test smokes并通过resource gate后重启全部110 repetitions；110/110前禁止替换CMoS/TimeBase。

Recovery commit=`b09c6e8`在new root完成7个新SimpleTM no-test smokes，并与7个同contract AMD smokes形成14/14 units、14 unique hashes、test=0、failure tokens=0的combined gate。Repaired formal queue于2026-08-07 10:27:10启动，PID=`825838`，first wave=SimpleTM ECL/Solar/Weather；old-root partial rows永久excluded。按用户既定要求不驻守，new-root 110/110前不修改Main I。

2026-08-08 recovery root已14/14 units完整结束。Hash-aware analyzer确认AMD 28/28、SimpleTM 82/82 raw metrics、110/110 unique checkpoint hashes与56/56 aggregated cells；旧root三份partial units未混入。Main I已原子移除CMoS/TimeBase并加入AMD/SimpleTM，保持14 models × 7 datasets × four H；ISCF-BSCA更新为29/56 best、19/56 second。AMD/SimpleTM仅作official-native fixed-H accuracy context。Canonical result=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_i_final_amd_simpletm_20260808/result_and_table_audit.md`。该授权scope已关闭。

## 2. Paper claims to be discharged

Introduction P6当前包含三项provisional paper-facing claims。实验工作必须逐项
建立可追溯的claim-to-evidence mapping：

| Claim | Primary evidence | Required control |
| --- | --- | --- |
| 一个unified ISCF-BSCA模型可优于分别训练的horizon-specific forecasters | Main Results I | 明确`#models`、相同dataset/horizon/metric定义、native或matched protocol role |
| ISCF-BSCA在同一unified setting下具有更强forecasting能力 | Main Results II | matched unified baselines、相同validation selector与test matrix |
| ISCF、target-conditioned allocation与BSCA各自有效，decoder具有可迁移性 | with/without ablations + backbone transfer | end-to-end matched training；不得以frozen replacement作方向级结论 |

现有ISCF-BSCA-v1 three-seed official-test confirmation只支持BSCA相对
ISCF-EQUAL的小幅、方向稳定改进：

- macro MSE gain `+0.3541%`；
- macro MAE gain `+0.3073%`；
- MSE-positive cells `41/60`；
- positive seed means `3/3`；
- positive dataset means `4/5`；
- positive horizon means `4/4`；
- ETTm2 mean `-0.6506%`，必须保留；
- cluster-bootstrap interval跨0。

它不证明horizon-specific superiority、完整component attribution或decoder
portability，因此不得把已完成的BSCA confirmation重命名为完整main result。

## 3. Frozen paper-facing protocol

除非在观察新结果前完成书面变更并获得授权，正式矩阵遵循：

- Main I/II datasets：`ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `Weather`,
  `ECL`, `Solar`, `Exchange`；core ablation/transfer仍为原5 datasets；
- horizons：`{96, 192, 336, 720}`；
- primary metric：MSE；
- secondary metric：MAE；
- validation只用于每个trial的early stopping、checkpoint selection和
  implementation debugging；
- unified candidate的trial内checkpoint selector为四个standard horizons的
  mean validation MSE；
- official test用于按dataset的joint four-H MSE/MAE relative mean 1% guard与
  leading-cell count选择一个shared hyperparameter profile，并作为formal
  effectiveness surface；
- 结果明确标记`test_tuned/test_informed`；禁止按horizon、seed、metric或cell
  选择配置及选择性报告；
- historical H720-only selector checkpoint若进入matched comparison，必须重训；
- 每次formal test前冻结candidate version、source/config、profiles、seeds、
  checkpoint rule、完整matrix、gates与rollback consequences。

正式机制结论必须分开报告：

1. `paper_facing_effectiveness`；
2. `matched_mechanism_attribution`；
3. `internal_mechanism_health`；
4. `failure_attribution`。

## 4. E0: Existing-artifact and matrix audit

在新训练前建立逐cell manifest，至少回答：

1. 每个已有checkpoint的candidate、dataset、seed、training horizon contract、
   validation selector、checkpoint hash、test access date与artifact completeness；
2. 哪些ISCF-BSCA、ISCF-EQUAL、A6_FULL或其他checkpoint可以直接复用；
3. 哪些历史checkpoint因H720-only selector、缺少MAE、缺少标准horizon或protocol
   不匹配而必须重训；
4. 哪些结果只是diagnostic、validation-only或frozen replacement，不能进入正式表；
5. 每个planned table cell的状态：
   `reusable / retrain_required / source_patch_required / missing / excluded_with_reason`；
6. 预计runs、GPU-hours、storage与最慢dataset的调度风险。

不要默认复跑已经完成且contract匹配的ISCF-BSCA-v1 three-seed confirmation。
也不要仅因artifact存在就视为可比较；先核对selector、split、metric、seed与test role。

## 5. Baseline consolidation before launch

当前论文架构草案与modern native baseline audit包含两套尚未完全对齐的baseline
集合。实验对话的首个设计任务是合并并冻结其角色，而不是把所有候选直接相加：

### 5.1 Candidate baseline roles

1. `horizon_specific_standard`
   - 论文架构草案中的DLinear、PatchTST、iTransformer、TimeMixer等；
   - 每个horizon独立训练，用于Main Results I的多模型系统比较。
2. `matched_unified_adaptation`
   - 相同backbone一次服务全部supported horizons；
   - 使用相同four-horizon validation selector；
   - 用于Main Results II的matched unified comparison。
3. `native_single_weight_varied_horizon`
   - 当前P0外部候选为ElasTST；
   - 保留其native selector与source-faithful contract；
   - 如果增加matched-selector版本，必须作为单独control标识。
4. `modern_native_fixed_h_accuracy`
   - CATS、TimePerceiver、SRSNet；
   - 用于accuracy context，不得冒充matched architecture attribution。

### 5.2 Current source/protocol blockers

`SC-MNB`已有source set已冻结，但尚未通过prelaunch：

- ElasTST：确认`limit_train_batches=10`语义；native selector不是FATST
  four-horizon mean MSE；
- CATS：移除training-epoch test access；修正ETTm2 H96 dataset identifier typo；
- TimePerceiver：移除training-epoch test access；
- SRSNet：完成executed-file license trace、metric-equivalence与prediction artifact
  export audit；
- 全部baseline：证明dataset/split/channel/metric边界，冻结artifact schema、checker、
  resource estimate与failure gate。

旧SC-MNB草案的外部总量为65 runs / 80 cells，但这不是自动授权的final matrix。
必须结合paper narrative、计算成本和表格角色选择minimal sufficient baseline set，并
记录保留或删除每个baseline的理由。

## 6. Planned experiment blocks

### 6.1 Main Results I: one unified model versus horizon-specific systems

目标：回答一个ISCF-BSCA unified model能否与每个horizon分别训练的多个models
竞争。

至少冻结：

- 每个baseline的native source与training contract；
- `#models`和支持全部horizons所需总参数存储；
- 每个dataset × horizon的MSE/MAE；
- macro average的计算方式；
- native-protocol differences与可比性边界；
- 不得因某个baseline结果弱而事后删除。

### 6.2 Main Results II: matched unified benchmark

目标：隔离“统一训练协议”与“ISCF-BSCA decoder设计”的影响。

matched unified baselines必须：

- 一次服务完整supported horizon set；
- 使用相同train/validation/test split；
- 使用相同four-horizon validation checkpoint rule；
- 报告全部5 datasets × 4 horizons；
- 不引入ISCF/BSCA；
- 在参数量或训练预算明显不匹配时同时报告差异。

### 6.3 Core with/without ablations

主消融保持紧凑，只检验核心部件：

1. Full ISCF-BSCA；
2. w/o BSCA：ISCF-EQUAL；
3. w/o Independent Scope-Conditioned Forecasting；
4. w/o Target-Conditioned Scope Allocation；
5. w/o Multiple Sharing Scopes：single-scope decoder。

在冻结矩阵前，必须逐项核对exact code identity、tensor path、parameter difference、
objective difference与expected claim。所有paper-core controls默认从同一
initialization class end-to-end joint training。frozen replacement、cross-swap和
warm-start只能标为secondary diagnostic。

random partition、scope数量、route-weight sensitivity等不进入主with/without表，
除非主结果暴露必须解释的机制风险；不得为凑表格事后扩张。

### 6.4 Decoder transferability

优先选择两个结构不同且成本可控的backbones：

- 一个linear/MLP-style backbone，如DLinear；
- 一个patch/Transformer-style backbone，如PatchTST或iTransformer。

每个backbone至少比较：

- Original Decoder；
- +ISCF；
- +ISCF-BSCA。

迁移实验必须是matched end-to-end training，不使用从另一共同训练模型中冻结替换
decoder的结果证明portability。若算力不足，应先减少backbone数量或seed深度，并在
结果前冻结，而不是选择性减少dataset/horizon cells。

### 6.5 Efficiency and system properties

同时记录：

- model/checkpoint count；
- parameter count与支持全部horizons的total stored parameters；
- training GPU-hours；
- single-request latency与服务全部horizons的system cost；
- peak memory；
- CHPC是否由architecture contract保证；
- BSCA的train-only性质。

效率比较必须区分单次forward成本与维护多个horizon-specific systems的总成本。

### 6.6 Problem alleviation and mechanism diagnostics

这些结果用于解释，不替代MSE/MAE effectiveness：

- naive unified penalty before/after；
- horizon-specific CHPD/NCHPD versus ISCF-BSCA的architectural CHPC；
- target-conditioned allocation map；
- per-scope standalone error、fused error与descriptive oracle headroom；
- policy usage/entropy、prediction diversity、gradient/component contribution。

不要把entropy、diversity或oracle headroom本身写成有效机制的证明。

## 7. Required prelaunch deliverables

在任何remote launch前，实验对话应先落地：

1. `analysis/iscf_bsca_paper_experiment_consolidation_20260731/design_and_prelaunch_gate.md`
   - claim-to-table mapping；
   - existing-artifact inventory；
   - final baseline roles；
   - complete run/cell matrix；
   - controls、seeds、selector、metrics、test role；
   - resource estimate与workload-aware GPU schedule；
   - narrative/effectiveness/failure gates；
   - rollback and stop rules。
2. `configs/iscf_bsca_paper_experiment_protocol.json`
   - 上述冻结内容的machine-readable版本。
3. 对新增或修改runner/checker的最小诚实验证计划。
4. 对三份主线文档和Stage C ledger的同步更新。

如果只是完成E0--E2 audit，允许不修改model code。若需要patch external baseline
protocol或实现matched unified/transfer controls，必须先记录source-informed design
与code-theory contract，再获得相应授权。

## 8. Authorization and remote sequence

当前允许：

- 读取与盘点已有artifacts；
- source/protocol audit；
- 设计final matrix与controls；
- 估算资源并准备prelaunch documents；
- 运行非破坏性local inspection和format checks。

当前不允许：

- 修改paper-core method；
- 修改external baseline protocol；
- 启动remote training；
- 访问新的official-test labels；
- 在未冻结search space/budget/selector且未获得Tier B2授权时执行test-tuned HPO；
- 根据test结果做per-horizon、per-seed、per-metric或per-cell tuning。

只有prelaunch gate通过并获得用户明确授权后，才能：

1. 完成必要的local protocol implementation；
2. 运行targeted smoke/checker；
3. focused commit并push；
4. SSH到`529_Lab-3090`，先执行`nvidia-smi`；
5. 选择有安全余量的GPU并采用workload-aware scheduling；
6. 冻结并记录command、environment与repo-external output path；
7. 启动remote validation/training；
8. 在另行授权Tier B2 test-tuned HPO后访问official test作dataset-profile ranking；
9. 在另行授权Tier C后完成完整test-tuned reporting audit。

## 9. Reporting and failure attribution

返回结果必须完整报告冻结matrix，包括negative cells。不得选择最佳seed、dataset或
horizon作为正式结论。

若结果不足以支持claim，必须区分：

- `hypothesis_false`；
- `intervention_point_wrong`；
- `readout_or_head_design_wrong`；
- `optimization_or_numeric_pathology`；
- `capacity_control_explains`。

positive test result但缺matched attribution只能是
`performance_partial_pass`。validation/test reversal、baseline protocol mismatch或
numeric pathology不得隐藏，也不得直接否定固定ISCF架构。

## 10. Working-tree preservation

以下untracked目录与当前实验工作无关，必须原样保留、忽略，不得删除或提交：

- `SRP-7C55/`；
- `analysis/phase5_timealign_hss_h1c_capacity_preserving_gate_20260701_partial/`。

## 11. Copy-ready startup prompt

```text
请在 /Users/river/PaperResearch/Project/R_2026_FATST 中并行推进 ISCF-BSCA 论文实验工作。

首先严格阅读并遵守仓库 AGENTS.md，然后按顺序完整阅读：
1. docs/stage-ledgers/stage-c-iscf-bsca-paper-experiments-restart-handoff-20260731.md
2. docs/iscf-bsca-paper-architecture.md
3. configs/paper_facing_evaluation_protocol.json
4. docs/paper-drafts/iscf-bsca-introduction-initial-draft.md
5. analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_bsca_v1_confirmation_step9_10_20260722/step9_10_three_seed_result_and_paper_handoff.md
6. docs/code-explanation/stage-c-iscf-bsca-v1.md
7. configs/stage_c_iscf_bsca_v1.json
8. configs/stage_c_iscf_bsca_v1_confirmation.json
9. analysis/stage_c_post_d21_unconstrained_reset_20260720/post_d24_paper_story_and_modern_baseline_gap_audit.md
10. analysis/stage_c_post_d21_unconstrained_reset_20260720/sc_mnb_step13_source_and_protocol_audit.md
11. configs/stage_c_modern_native_baseline_protocol.json
12. docs/paper-mainline.md
13. docs/research-roadmap.md
14. docs/stage-ledgers/stage-c-unified-forecasting-redesign.md
15. docs/stage-ledgers/stage-c-iscf-bsca-paper-writing-restart-handoff-20260731.md

当前权威状态：
- exact ISCF-BSCA-v1 已冻结为 paper-core candidate，不继续architecture search；
- Introduction v0.9-author-refinement 已暂时冻结，其中“优于horizon-specific models、核心组件有效、decoder可迁移”仍是需要正式实验兑现的provisional claims；
- 已完成的three-seed BSCA confirmation只证明其相对ISCF-EQUAL的小幅方向稳定收益，不等同于完整main result、ablation或transfer evidence；
- 当前允许existing-artifact inventory、source/protocol audit、实验矩阵设计和local prelaunch准备；
- local protocol patch、remote training和新的formal test均未授权。

请先完成paper-facing experiment consolidation，不要立即启动远程训练：
1. 逐checkpoint审计已有artifacts、validation selector、seed、checkpoint hash、test role与可复用性；
2. 建立逐table-cell manifest，标记reusable / retrain_required / source_patch_required / missing / excluded_with_reason；
3. 对齐并冻结四类baseline角色：horizon-specific standard、matched unified adaptation、native varied-horizon和modern native fixed-H；
4. 设计minimal sufficient而非盲目扩张的完整实验矩阵，覆盖：
   - Main I：一个unified ISCF-BSCA versus 多个horizon-specific systems；
   - Main II：matched unified benchmark；
   - 核心with/without ablations；
   - 两类backbone的decoder transfer；
   - efficiency与必要的mechanism diagnostics；
5. 严格遵守5 datasets × {96,192,336,720}、MSE/MAE、validation-only checkpoint selection与test-informed official-test边界；
6. 明确controls、seeds、resource estimate、GPU workload-aware scheduling、success/failure gates和rollback；
7. 先落地：
   - analysis/iscf_bsca_paper_experiment_consolidation_20260731/design_and_prelaunch_gate.md
   - configs/iscf_bsca_paper_experiment_protocol.json
8. 同步更新paper architecture、paper-mainline、research-roadmap与Stage C ledger，并完成最小诚实验证。

特别注意：
- 不要复跑contract匹配的已完成ISCF-BSCA-v1 confirmation；
- historical H720-only selector checkpoint不能直接进入matched paper table；
- external native baseline结果只作对应protocol角色，不冒充matched mechanism attribution；
- paper-core ablation和transfer默认end-to-end joint training，frozen replacement不能作方向级结论；
- 不得为了兑现Introduction结论选择性报告有利dataset/horizon/seed；
- 完成prelaunch gate后先向我汇报并请求local protocol patch、remote training和formal test的分级授权，不要自行启动。

请保留并忽略以下untracked目录，不要删除或提交：
- SRP-7C55/
- analysis/phase5_timealign_hss_h1c_capacity_preserving_gate_20260701_partial/
```
