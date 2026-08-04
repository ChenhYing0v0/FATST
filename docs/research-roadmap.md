# Research Roadmap

## ISCF-BSCA-MAIN-v1 H4L Wide HPO (2026-08-04)

Current cursor=`Step 6 wide search contract frozen / Step 7 local tooling pass / Step 8 remote resource smoke and training authorized / formal test pending separate gate`。H4K证明原117-trial search space的dataset-level与逐cell oracle上限均为30/56，因此H4L只针对ETTm2 0/8和Weather 2/8扩大范围，不复跑H4K，也不改变architecture、objective、scales或inference graph。

H4L为48个seed2021 profiles，ETTm2/Weather各24；对117个历史profiles做effective fingerprint audit后零重复。矩阵覆盖context/patch、capacity、decoder rank、learning rate、weight decay和layer norm边界，并从TimeAlign official scripts提取四组encoder parameter couplings作source prior，再与ISCF-BSCA decoder/optimizer组合。Trial budget扩展为60 epochs/patience12，checkpoint仍由four-H validation mean MSE选择；训练阶段test=0，一个dataset-level profile共同服务四H。Narrative gate=`pass_as_wide_hyperparameter_optimization_not_new_method`；effectiveness gate等待complete formal test，但该test尚未授权。Canonical report=`analysis/iscf_bsca_main_v1_hpo_20260731/h4l_wide_matrix_and_prelaunch.md`，machine contract=`configs/iscf_bsca_main_v1_hpo_wide_h4l.json`。Decision=`H4L_48_job_wide_matrix_frozen_remote_training_authorized_formal_test_false`。

## Section 3 Field-Style Alignment v0.4 (2026-08-03)

用户指出v0.3的rhetorical questions、Nature式短句与显式system-contract判断仍与
Introduction不一致。本轮检索并核对iTransformer、TimeMixer、TimeXer和
TimeMixer++的官方conference papers，将其Methods / model section的共同写法作为
领域风格标尺。Canonical audit=
`analysis/iscf_bsca_section3_style_calibration_20260803/style_audit.md`。

Section opening现只保留一个承上启下段，删除原P2 meta roadmap。3.1直接由
same-history / shared-target setting进入notation；3.2与3.4使用
`As shown in Figure ...`连接matched statistic与observation；3.3改为中性标题
`Accuracy under naive unified forecasting`；3.5以连续requirements paragraph引出
ISCF-BSCA。CHPC、CHPD/NCHPD、$\operatorname{UP}_H$、$R_{o,b,s}$、CFH、所有
evidence values与validation-only boundaries保持不变。Decision=
`section3_v0_4_field_style_alignment_pending_author_review`。

## Section 3 Concise Polish v0.3 (2026-08-03)

用户确认v0.2叙事链方向正确，但全文仍偏冗长。v0.3使用`nature-polishing`按
`algorithmic / methods+results / en / generic`路由精简
`docs/paper-drafts/iscf-bsca-problem-formulation-initial-draft.md`。Manuscript body
由2,308词降至1,576词，重点合并same-target与nested-trajectory重复解释、压缩
Figures 2--3 captions，并将future CFH protocol收敛到必要statistic和controls。

CHPC、CHPD/NCHPD、$\operatorname{UP}_H$、$R_{o,b,s}$和CFH均保留；Figure 2
仍只支持audited DLinear family缺少CHPC guarantee，Figure 3仍只支持
finite-capacity future-region heterogeneity。3.3没有把naive unified penalty写成
既成事实，3.5之前不引入ISCF-BSCA。Introduction v0.9、实验结果、远程状态与
授权边界未改。Decision=`section3_v0_3_concise_polish_pending_author_review`。

## ISCF-BSCA-MAIN-v1 H4K Targeted HPO (2026-08-03)

Current cursor=`Step 9 formal test complete / Step 10 effectiveness gate fail / rollback Step 6 HPO search-space design`。H4K保持frozen ISCF-BSCA architecture与objective，24个seed2021 jobs与96个standard-horizon test cells均已完整结束。

Narrative gate=`pass_as_targeted_hyperparameter_optimization_not_new_method`。24/24 test与checkpoint immutability通过；117-trial selector仅带来macro MSE/MAE 0.0199%/0.0286%改善，leading cells仍为15/28、15/28、30/56。ETTm2=0/8、Weather=2/8、H720=4/14，global/local gates全部失败。Failure attribution=`search_space_performance_shortfall`；automatic H4L、新训练、baseline与3-seed仍未授权。Canonical result=`analysis/iscf_bsca_main_v1_hpo_20260731/h4k_test_result_and_joint_hpo_decision.md`。Decision=`H4K_complete_continuous_improvement_no_new_leads_gate_fail_return_step6`。

## ISCF-BSCA-MAIN-v1 H4J Complete Test Decision (2026-08-03)

H4J 40/40 official-test artifacts和160/160 standard-horizon cells完整，checkpoint immutability 40/40 pass。H1--H4J 93 trials的frozen joint selector得到MSE 15/28、MAE 15/28、combined 30/56，相对H4J前23/56提高7 cells，但三项20/28、20/28和40/56 gates仍全部失败。共同7 datasets macro mean MSE/MAE分别改善0.234%/0.585%；相对TimeAlign Table 6 macro MSE/MAE分别低2.428%/0.520%，但结果仍是single-seed test-tuned published-context comparison。

Selector已达到所有93 trials的single-profile和逐cell diagnostic oracle上限30/56，因此failure attribution=`search_space_performance_shortfall`，不是selector、numeric或architecture failure。ETTm2=0/8、Weather=2/8、H720=4/14是下一轮主要缺口。Effectiveness gate=`performance_partial_pass_gate_fail`；rollback=Step 6 HPO matrix design。H4K未授权，必须先冻结minimal dataset-level search contract。Canonical report=`analysis/iscf_bsca_main_v1_hpo_20260731/h4j_test_result_and_joint_hpo_decision.md`。Decision=`H4J_complete_joint_HPO_material_partial_improvement_gate_fail_H4K_not_authorized`。

## ISCF-BSCA-MAIN-v1 H4J Joint-Objective Reset (2026-08-02)

Current cursor=`Step 6 contract frozen / Step 7 local tooling pass / Step 8 remote preflight pending`。H1/H2/H3A/H3B的53个test-tuned trials仍完整保留，但原MSE-only selected row相对frozen published per-cell targets只有MSE 14/28、MAE 9/28、combined 23/56；existing trials的dataset-level reselection upper bound为25/56，无法满足用户更正后的目标。

H4J不改变ISCF-BSCA architecture、objective、scales或inference graph。设计为40个seed2021 end-to-end jobs，其中ETTm2/Weather/Solar占28个。Trial checkpoint仍由four-H validation mean MSE选择；40/40 artifacts与checkpoint manifest完整后直接执行complete official test。Dataset selector固定为equal-weight joint MSE/MAE relative mean的1% guard后最大化leading cells；禁止per-H、per-metric、per-seed和per-cell profile selection。Success gate=MSE `>=20/28`、MAE `>=20/28`、combined `>=40/56`；Exchange因暂无同协议published target不进入56-cell denominator，但继续按joint within-search regret选一个shared profile。

Narrative gate=`pass_as_hyperparameter_optimization_not_new_method`；effectiveness gate=`pending_complete_H4J_test`。若未达目标，保留全部negative trials并另行冻结H4K；任何architecture redesign必须创建test-informed新candidate并回Step 4--6。Canonical report=`analysis/iscf_bsca_main_v1_hpo_20260731/joint_objective_reset_and_h4j_prelaunch.md`，machine contract=`configs/iscf_bsca_main_v1_hpo_joint_h4j.json`。Decision=`H4J_frozen_local_gate_pass_remote_resource_smoke_then_training_authorized`。

## Section 3 Narrative Refinement v0.2 (2026-07-31)

用户对v0.1的核心反馈不是公式错误，而是公式替代了叙事：reader从Related Work进入
Section 3后，尚未理解本节要回答什么、为什么回答以及证据如何导向method。v0.2已
按`what -> why -> formalization -> evidence -> implication -> boundary`重构全文，
文件仍为
`docs/paper-drafts/iscf-bsca-problem-formulation-initial-draft.md`，review=
`author_feedback_round1_integrated_pending_review`。

本轮revision decisions：

1. Section opening先提出两个问题：cross-horizon requests应保证何种coherence，
   unified decoder应如何组织future-domain sharing；
2. 3.1先给same-target semantic argument，再形式化input/output；删除
   $\Pi_{H_i}$ projection notation，直接写shared targets上的element-wise
   equality；
3. `future-step-indexed prediction function`在首次公式前解释，并明确是本文
   varied-horizon formulation天然满足CHPC；
4. accuracy disclaimer从3.1移至3.3，由`coherence != accuracy`自然引出
   $\operatorname{UP}_H$与D18 negative evidence boundary；
5. 3.2形成`question -> metric -> control -> Figure 2 -> implication`，3.4形成
   `decoder question -> sharing trade-off -> matched diagnostic -> Figure 3 ->
   out-of-sample boundary`；
6. manuscript移除neutral diagnostic的完整tensor derivation，只保留matched
   construction prose、$R_{o,b,s}$与formal CFH；详细contract继续由architecture
   与canonical evidence design承载；
7. 3.5分别标记task-derived、evidence-derived与method-level optimization
   requirements；Figure 3只直接支持future-region variation，sample/variable
   conditioning与BSCA trainability均留给后续ablation。

Introduction v0.9与所有evidence values未改。Figures 2--3仍是validation-only
illustrative evidence，CFH仍未formal established，P6 main/ablation/transfer
claims仍为provisional。Decision=
`section3_v0_2_narrative_refinement_pending_author_review`；本轮不扩张任何
implementation、remote training或formal test授权。

## Section 3 Initial Draft and Evidence Boundary Freeze (2026-07-31)

Section 3 manuscript-facing初稿已落地：
`docs/paper-drafts/iscf-bsca-problem-formulation-initial-draft.md`，
version=`v0.1-initial-draft`，review=`pending_author_review`。Introduction
`v0.9-author-refinement`保持不变。

本轮完成的写作决策为：

1. 3.1以$\mathbf X_o\in\mathbb R^{L\times C}$、supported horizon set
   $\mathcal H$和horizon-agnostic $g_\theta(\mathbf X_o,\tau,c)$定义task；
   CHPC是相同history/origin/preprocessing下的prefix projection identity；
2. 3.2定义origin-level CHPD与train-split scale-normalized NCHPD，Figure 2只证明
   audited DLinear horizon-specific systems不保证CHPC；
3. 3.3定义$\operatorname{UP}_H$但冻结negative evidence boundary：D18的
   horizon-specific/unified contrast小且受更大的measure-training差异混杂，
   因而不宣称naive unified forecasting存在稳定penalty；
4. 3.4以capacity-matched single-extent neutral family定义
   $R_{o,b,s}$、$E_{o,b,s}$与future formal CFH。Figure 3的winner sequence、
   crossings和8.112% headroom只作validation-selected descriptive oracle；
5. 3.5从one model、CHPC、multiple extents、target-level integration与stable
   joint learning导出design requirements，最后才引出ISCF-BSCA。

Problem、evidence、method与claim boundary继续分离：Figures 2--3均为
validation-only illustrative evidence；CFH尚未formal测量；Introduction P6的
unified superiority、component effectiveness与decoder portability继续等待完整
main/ablation/transfer tables。Method Figure 4仍为planned only。

Writing decision=
`section3_v0_1_complete_pending_author_review_evidence_bounded`。本轮没有执行或
授权new implementation、remote training与formal test；并行
ISCF-BSCA-MAIN-v1 H1 experiment cursor保持原记录。

## ISCF-BSCA Paper-Facing Experiment Prelaunch Freeze v2 (2026-07-31)

Canonical report为
`analysis/iscf_bsca_paper_experiment_consolidation_20260731/design_and_prelaunch_gate.md`，
machine-readable contract为
`configs/iscf_bsca_paper_experiment_protocol.json`。

Current cursor=`ISCF-BSCA-MAIN-v1 single-seed HPO and Main I 140-row published block complete；baseline execution authorization next`：

- exact `ISCF-BSCA-v1`与当前超参数只作原5数据集ablation anchor；不得直接进入
  Main I/II；
- `ISCF-BSCA-MAIN-v1`保持相同frozen architecture family，但必须在8 datasets
  上完成test-tuned HPO；每trial由validation选择checkpoint，每dataset由
  four-H mean official-test MSE选择一个profile服务四个H；
- Main I/II datasets=`ETTh1, ETTh2, ETTm1, ETTm2, Weather, ECL, Solar,
  Exchange`；ablation和transfer保留原5 datasets；
- HPO分H0 data/protocol parity、H1 anchor/resource smoke、H2 bounded coarse
  search、optional H3 top-2 stability、H4 selected-config freeze、H5 final
  training；当前primary seed=2021，additional seeds只在时间允许时按完整block
  扩展；TimeAlign encoder仅作source-audited search prior；
- Main I包含AMD、TimeMixer、DLinear、SimpleTM、iTransformer、PatchTST、
  TimePerceiver、SRSNet与TimeAlign；published-transcribed和official-native
  reproduction分层标记；
- published primary source为TimeAlign ICLR 2026 Table 6：覆盖TimeAlign、
  TimeMixer、DLinear、iTransformer、PatchTST的7个目标datasets，缺Exchange；
  AMD、SimpleTM、TimePerceiver、SRSNet及全部Exchange缺口走official
  reproduction；PDT只作secondary cross-check；TimeAlign Exchange seed2021
  script已按ETTh1 bootstrap本地实现但未运行；
- Main II保留DLinear/PatchTST matched unified、A6_FULL repo-native reference
  与ElasTST native varied-horizon context；
- seed2021 primary matrix不含HPO runs时为233 checkpoint slots、488 seed-horizon
  cells；23 primary-seed metric-evidence records可复用、210 new；另有30 existing
  extra-seed evidence保留；Main I包含140 published cells + 148 single-seed
  official-reproduction cells；
- historical H720-only selector、frozen replacement与external-native-as-matched
  均继续排除。

2026-08-02 post-HPO audit已将8个selected checkpoints/32 cells固定为Main I/II可复用，不再final retrain；TimeAlign Table 6的5 models × 7 datasets × 4 H=140 published rows完成双路径核验。共同7 datasets上ISCF相对TimeAlign aggregate MSE为`+2.199%`，但MAE为`-0.066%`，且只15/28 MSE cells、4/7 dataset means更优；因此当前定性为`competitive aggregate MSE / complete SOTA pending`。源表的5组Avg inconsistency与三种lookback描述必须披露。Main I下一最小remote block为TimeAlign-Exchange 4 jobs；Main II当前0 jobs可直接launch，须先完成DLinear/PatchTST Tier A source/protocol patch。Canonical report=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/post_hpo_main_i_published_audit_and_next_gate.md`。

下一rollback不是自动重启architecture search。若new-dataset protocol不一致回H0；
HPO不稳定回H1/H2并收窄已冻结search budget；冻结budget内test-tuned最优profile
未达到SOTA则完整报告并收窄claim，或另立candidate并重新冻结search contract；
matched/ablation/transfer失败按four-layer failure attribution回Step 4--6。

当前授权严格分层：

1. scoped TimeAlign Exchange local script=`completed_unlaunched`；
2. ISCF-BSCA-MAIN-v1 Tier A local protocol/source patch=`true`；
3. Tier B1 remote new-dataset/resource smoke=`true`；
4. Tier B2 bounded test-tuned HPO（含完整H2后的official-test profile ranking）
   =`true`；
5. Tier B3 selected-profile confirmation=`false`；
6. Tier C complete test-tuned reporting audit=`false`。

H1 16 jobs已全部完成，artifact、checkpoint selector和numeric health通过，
test jobs=0。五个旧dataset的H1 conservative checkpoints与既有seed2021
confirmation checkpoint hashes完全一致；复用既有test evidence表明五dataset
macro与TimeAlign published MSE仅差约+0.24%，但ETTm2仍有约+5.88%缺口，当前只
判定为`competitive_not_SOTA`。

上述launch gate已完成：H0 audit pass、6/6 new-dataset canary pass、16/16
resource smoke pass。Full H1已在commit `7361d9e`、GPUs 0/1/2上启动，
orchestrator PID=`545400`，output root=
`/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h1`。当前仍为
train/validation only，test jobs=0。H2现已冻结为24 additional jobs
（每dataset三个，H1+H2共40），config=
`configs/iscf_bsca_main_v1_hpo_h2.json`。下一gate为local contract、focused
commit/push、remote commit/GPU preflight和H2 resource canary；H2 24/24完成前
不得访问test。

H2 prelaunch已通过local/remote contract、9/9 new-dataset canary与24/24 full
resource smoke；随后24/24 full train/validation完成。Artifact、selector、numeric
health与checkpoint hash audit均通过，H1+H2共40 checkpoints，test仍为0。下一gate
为fail-closed 40-checkpoint official-test execution；任何partial/mixed artifact、
checkpoint mutation或160-cell不完整均阻断ranking。

用户于2026-08-01授权ECL和Solar的所有dataset-level test-tuned普通超参数调整，
包括扩展epochs/patience。首轮40-checkpoint test结果返回后，可以建立独立
test-informed H3 version，直接依据完整four-H test aggregate冻结下一批profiles；
validation只承担early stopping/checkpoint selection，不进行耗时的profile ranking。
仍禁止per-horizon、per-cell、per-seed或选择性报告。Canonical H2/prelaunch record=
`analysis/iscf_bsca_main_v1_hpo_20260731/h2_result_and_test_prelaunch.md`。

40-checkpoint test现已完整结束：40/40 checkpoints、160/160 cells、dense metrics、
provenance、invariants和checkpoint immutability全部pass。ECL selected mean MSE=
`0.151191`，通过TimeAlign `0.154` target；Solar=`0.196157`，相对`0.192`仍有
2.17% gap。H3A因此冻结为ECL一个budget-extension control与Solar八个test-informed
one-factor profiles，45 epochs/patience10。Canonical result/prelaunch=
`analysis/iscf_bsca_main_v1_hpo_20260731/h1_h2_test_result_and_h3a_prelaunch.md`；
config=`configs/iscf_bsca_main_v1_hpo_ecl_solar_h3a.json`。

H3A commit=`72e1f8f`，9/9 resource smoke通过；full train/validation随后9/9完成，
test=0。Artifact、finite metrics与checkpoint selector audit通过，9-row SHA256 manifest
已冻结；不做validation profile ranking，直接执行36-cell完整test。Launch record=
`analysis/iscf_bsca_main_v1_hpo_20260731/h3a_launch.md`，direct-test prelaunch=
`analysis/iscf_bsca_main_v1_hpo_20260731/h3a_training_result_and_test_prelaunch.md`。

## Paper-Experiments Parallel Handoff (2026-07-31)

新增experiment-workstream current entry：
`docs/stage-ledgers/stage-c-iscf-bsca-paper-experiments-restart-handoff-20260731.md`。
它与paper-writing handoff并行，前者负责paper-facing artifact inventory、baseline
consolidation、main/ablation/transfer/efficiency matrix与prelaunch gate，后者继续
Section 3 manuscript integration。

Experiment cursor=`E0 existing-artifact inventory and claim-to-table audit`。
当前仅授权read-only audit、design和local prelaunch文档准备；local protocol patch、
remote training与formal test未授权。必须先区分horizon-specific standard、
matched unified、native varied-horizon和modern native fixed-H四种baseline角色，
再冻结minimal sufficient matrix，不自动启动旧SC-MNB的65-run草案。

## Paper-Writing Restart Freeze (2026-07-31)

Introduction `v0.9-author-refinement`已由用户确认并暂时冻结为可用版本。
Paper-writing current entry为
`docs/stage-ledgers/stage-c-iscf-bsca-paper-writing-restart-handoff-20260731.md`；
旧D22 handoff仅作historical research lineage。

Active workstream=`Section 3 manuscript integration`。下一步依次完成task/CHPC
formulation、Figure 2 prefix-disagreement evidence、naive-unified evidence audit、
Figure 3 sharing-demand evidence与design requirements。Section 3冻结后再设计
Method Figure 4。当前不授权new method、remote training或formal test。

## Introduction v0.9 Author Refinement (2026-07-30)

- Figure 1a reference移至P2，功能限定为horizon-specific prefix disagreement；
- P3只承担horizon无关mapping与CHPC formalization；
- Figure 1 caption删除重复的conceptual/evidence句；
- P5用single-scope decoder显式对照ISCF；
- P6采用更强的预期性能、ablation与portability结论，但三类claim必须由后续完整
  tables兑现。

新增planned Method Figure 4，不在Introduction中插入第二张图。Figure 4将对比
single-scope forecasting与ISCF-BSCA，并展示scope-indexed field、
target-conditioned allocation和BSCA train-only path。当前P5不加入空figure
reference。Clean v0.9 draft与highlighted review copy均已建立。

## Introduction P2--P6 Structural Polish (2026-07-30)

Introduction v0.8已完成段落级优化：

- P2压缩horizon-specific system fragmentation与维护冗余；
- P3以horizon无关future-step-indexed mapping定义CHPC，并由Figure 1a说明
  overlap invariance；
- P4用uniform output mechanism自然引出latent-state sharing extent和
  future-region sharing-demand heterogeneity；
- P5先提出整合different sharing extents，再以independent history projections、
  scope-indexed field和target-conditioned allocation说明ISCF；
- P6压缩为problem、architecture、training三项贡献，并将实验归纳为
  horizon-specific comparison、component ablation与backbone transfer。

Main/transfer tables未完成前不写未经验证的`significant advantage`或portability
结论。下一步仍是冻结Section 3 definitions与Figures 2--3 captions。

## Introduction Compact Figure and Section 3 Evidence Placement (2026-07-30)

Introduction采用一张双面板constructed concept figure：

- panel a：三条horizon-specific curves在同一future step给出不同值；
- panel b：fine/intermediate/broad sharing的constructed risk curves在
  early/middle/late regions分别最低。

该图只说明概念，明确标注`not empirical data`，不展示dataset、metric、sample
或headroom。正文只声明Section 3将给出formal definitions与controlled real-data
evidence。原两张approved figures移至Section 3，暂按Figure 2（prefix
disagreement）与Figure 3（sharing-demand heterogeneity）组织。Concept Figure
1状态=`approved_for_manuscript_draft`，四种投稿格式已进入`paper-figures/`；
Introduction v0.7已修复段内硬换行。

## Section 3 Problem Evidence Full Search Result (2026-07-30)

五dataset validation matrix=`25/25 neutral + 20/20 DLinear`，test=false。最终
Figure 2选择ETTh2 maximum-disagreement origin805/channel0：shared-96 raw
differences相对H720为2.16--2.51，ETTh2 macro NCHPD五dataset最高。Figure 3选择
ETTm2 maximum-heterogeneity origin4177：五scales各赢2--3个regions，10/10 pairs
qualified crossing，mean winner margin=10.266%，descriptive headroom=8.112%。

两图已改为：

- integrated prefix trajectories + in-panel H720-relative mean differences +
  all-validation compact triangular NCHPD heatmap；
- region-best excess-risk heatmap + winner-colored realized-gain bars。

两图使用exact 183 mm output、muted semantic palettes与无figure-footer排版。
Prefix Figure 2为左右顶底对齐的two-panel layout；trajectory panel以thin solid
colors、sparse staggered marker shapes、subtle white strokes与lower-layer H720 reference
替代难辨认的dash-only encoding。
sharing heatmap不再以fixed s720逐列归一化，因而消除了s720自身相减造成的全白行；
fixed-s720 comparison仍由gain panel承担。SVG/PDF/PNG/TIFF与source data package
已生成；Nature QA=13 pass/1 warn/0 fail。
Historical selection decision=
`two_intro_figures_pass_illustrative_gate_etth2_prefix_ettm2_sharing`；其
Introduction placement已由上方布局取代。停止dataset/sample search；下一步转向
Section 3 definitions、captions与后续paper experiment consolidation。

## Introduction Evidence Full Visualization Search Step7A (2026-07-30)

用户授权补齐五datasets并从所有validation results中选择最清晰的prefix
disagreement与sharing-demand heterogeneity案例。Protocol=
`SC-UVHF-INTRO-EVIDENCE-FULL-SEARCH-v1`：

- prefix：每个dataset搜索所有`origin × channel`单元，以六个horizon pairs的
  mean-over-overlap disagreement选择maximum；
- sharing：每个origin按60-step regions、all-channel MSE形成`5 × 12`风险面；
  lexicographic selection依次最大化`supported_winner_count`、
  `distinct_winner_count`、winner entropy、qualified crossing pairs、
  winner margin与sample oracle headroom；
- 所有图显式披露maximum validation selection，只作illustrative existence；
  formal test与architecture effectiveness gate均为false。

新增31 runs补齐五dataset矩阵，已有14 runs复用。neutral pooling已vectorize为每个
scale最多两次batched pooling/LayerNorm；相对旧loop的forward与gradients五scale
等价，最大gap=`2.27e-13`。remote scheduler使用global three-GPU queue与longest
first ordering，消除固定GPU 0上的$s=1$ critical path。Local checker、Python
compile、JSON parse、bash syntax与full runner dry-run均通过。Decision=
`full_search_step7a_pass_remote_launch_next`。

首次launch暴露ETTh1/ETTm2未注册的prelaunch coverage fault；driver已及时停止，
该故障未进入训练结果。registry与checker修复后，commit `0808c80`的retry通过
五dataset construction、remote dry-run和三GPU preflight。retry前已完成
neutral=`13/25`、DLinear=`6/20`，remaining=`26`；首批三jobs健康且显存均低于
0.6 GiB。retry最终正常完成，current decision由上方result entry取代。

## Previous Introduction Figure Candidates (2026-07-29; provisional)

visualization-first search已得到两张论文候选图：

- Weather/DLinear：85% quantile H720-relative prefix-difference view + NCHPD
  heatmap；
- ETTm1/neutral matched scales：sharing-risk landscape + qualified crossover
  curves + region-wise scale contrast。

ETTm1的best fixed为s128，region 1偏好s1，多数中长regions偏好s128；
`s1_vs_s720`与`s8_vs_s720`均达到冻结0.5% bidirectional crossing margin。
它们作为已有artifacts进入上方五dataset full search，不再预先固定为最终选图。
两图只作illustrative problem evidence，不承担formal prevalence、method
effectiveness或architecture rejection。

## Introduction Visualization Candidate Search Extension (2026-07-29)

本轮目标明确为寻找具有说服力的illustrative figure，而不是以单次visualization
screen否定ISCF-BSCA architecture或论文problem logic。Prefix family已保留
H720-relative difference view候选。

下一步只执行ETTm1/seed2021的neutral five-scale sharing-only validation screen：
scales、training protocol、12×60 regions与frozen thresholds均不变；不重复
DLinear，不访问test。ETTh2/full matrix/formal test=false。Decision=
`continue_visualization_search_without_architecture_rejection`。

ETTm1 five-scale screen已在commit `465931f`上启动。GPU preflight与resource smoke
通过，s1/s8/s32首批jobs健康；current decision=
`ettm1_sharing_visualization_screen_running`。完成后只评估figure候选，不据此
修改或否定fixed ISCF-BSCA architecture。

## Introduction Evidence Visualization Pilot Remote Running (2026-07-29)

commit `9cc2d24e2359310dea2c6fc1764303a2da5d2c65`已推送并在
`529_Lab-3090`启动Weather/seed2021的9-run validation-only pilot。GPU preflight
显示GPU 0/1/2均为RTX 3090、各18 MiB used；CUDA resource smoke通过。启动后一次
有界检查确认neutral `s=1/8/32`已在三张GPU上运行，显存占用约0.9--1.0 GiB，
utilization为100%/71%/84%，未见即时错误。

Decision=`initial_weather_9run_validation_pilot_running`。当前cursor是等待远程完成，
之后同步9/9 validation artifacts与自动figure analysis；不高频polling，不访问
formal test，不自动启动fallback。若Weather可视化清晰则停止；若不清晰则先汇报，
再请求`ETTm1` fallback授权。详细启动证据见
`analysis/iscf_bsca_intro_evidence_visualization_pilot_20260729/remote_launch_record.md`。

## Introduction Evidence Visualization Pilot Step7A/7B (2026-07-29)

用户将两项Introduction evidence的近期目标收窄为visualization-first：

- initial dataset=`Weather`，seed=`2021`；
- prefix：DLinear H96/H192/H336/H720=`4 runs`；
- sharing：neutral s1/s8/s32/s128/s720=`5 runs`；
- total=`9 validation-only runs`；
- prefix example选择85% disagreement quantile，不取maximum/top-1%；
- Weather清晰即停止；fallback ETTm1/ETTh2未授权。

Neutral Step7A实现保持五scales exact matched parameters=`111312`，输出
`[B,720,C]`、candidate/pooled states=`[B,C,720,64]`，14/14 parameter groups
gradient pass；block sharing与endpoint difference contracts通过。两套figure
analyzers、DLinear `--skip-test` artifact path、9-job remote runner与config均已
完成，synthetic/local/dry-run gates通过。

Decision=`intro_evidence_visualization_step7b_pass_remote_launch_next`（已由上方
remote-running cursor取代）。当前授权
只覆盖initial 9 runs与validation analysis；formal design仍保留但deferred，
fallback/full matrix/formal test=false。

## Introduction Problem-Evidence Design v1 (2026-07-29; placement superseded)

Introduction证明实验已从旧的单一三联图拆成两项独立、可失败的protocol：

1. P2后为horizon-specific prefix disagreement。正式baseline family固定为
   DLinear、PatchTST、iTransformer，复用后续Main Results的五datasets ×
   四horizons × 三seeds=`180` native checkpoints；统计六个horizon pairs的
   train-scale normalized NCHPD。Figure 1包含pre-registered
   median-disagreement overlay与family heatmaps。它不证明accuracy更差，也不
   泛化到ElasTST等已有varied-horizon invariant methods。
2. P4后为future-region sharing-demand heterogeneity。Primary使用neutral
   raw-history single-scale decoder；所有scales共享exact parameterization与
   compute-heavy path，只改变parameter-free pooling/sharing extent。
   `S_diag={1,8,32,128,720}`与12个60-step regions均独立于最终ISCF boundaries；
   uniform full-domain MSE排除multi-prefix exposure confound。完整矩阵为
   75 end-to-end runs。

第二项的primary statistic不是same-test oracle，而是validation-selected region
schedule相对validation-selected fixed scale的official-test CFH。支持要求macro
CFH正向、3/5 datasets、2/3 seeds、稳定risk crossover与全部matched/numeric
controls同时通过。random grouping只在进一步claim temporal contiguity
specificity时追加，不作为basic existence gate。frozen replacement/A6 sensitivity
不得用于方向级拒绝。

Introduction Figure 1置于P2--P3之间；Figure 2紧跟P4且不提前展示ISCF/BSCA；
method architecture顺延为后续图。Decision=
`intro_problem_evidence_v1_design_frozen_pending_step7a_authorization`。当前
implementation、remote training与formal test均未授权；未来formal evidence
必须标记为`test_informed`，不作untouched holdout claim。

Placement amendment：上述真实数据figures已由当前Introduction移至Section 3；
当前Introduction Figure 1采用上方constructed two-panel concept layout。

## Introduction Round 1 Author Response (2026-07-29; visualization superseded)

`docs/paper-drafts/iscf-bsca-introduction-initial-draft.md`已更新为
`v0.2-round1`。当前叙事冻结方向为：

1. 大多数long-term forecasting工作仍采用horizon-specific protocol；
2. ElasTST及少量foundation models已经探索varied/flexible horizons，但该方向
   相对horizon-specific literature仍缺少充分、系统的发展；
3. 本文的独特研究链是明确task definition、分析future-domain内部的output-side
   sharing problem，并形成ISCF decoder与BSCA training response；
4. CHPC是varied-horizon forecasting的basic property，不是算法创新；
5. Introduction不展开comparison table或与native decoder主线无关的结构路线。

P4已加入finite-capacity bias--variance与pointwise-MSE Bayes boundary，但
problem-existence evidence仍pending。修订记录
`analysis/iscf_bsca_intro_round1_revision_20260729.md`冻结了neutral
capacity-matched end-to-end diagnostic family、region-wise risk crossover、
best-fixed versus region-oracle headroom、split角色与failure attribution。
当时建议的Introduction三联图
`horizon disagreement -> sharing-risk landscape -> ISCF-BSCA response`
已由上方evidence-v1的两个独立figures取代。

Decision=`intro_v02_round1_positioning_pass_problem_evidence_pending`。
当时的下一步cursor已完成并由上方Step7A authorization gate取代。
新implementation、remote training与formal test均未授权。

## ISCF-BSCA Introduction Initial Draft and Blind Review (2026-07-28)

Introduction P1--P5共识正文与v0.7 provisional P6已整合为clean manuscript
draft：`docs/paper-drafts/iscf-bsca-introduction-initial-draft.md`。独立blind
review只读取该六段正文，并用公开primary sources复核novelty；没有使用内部实验
历史或候选演化。

当前judgement=`major_revision / weak_reject_at_current_intro`，score=`4/10`。
主要原因不是主线不可读，而是贡献边界仍不足以抵抗最接近prior art：ElasTST已
研究varied-horizon output invariance，SRP/SRP++已提出step-invariant
representation bottleneck与step-specific adaptation，TimeMixer/N-HiTS覆盖
multi-scale future synthesis，MoLE覆盖adaptive expert weighting，Implicit
Forecaster覆盖output-side local/global motivation。ISCF仍需证明其scope-indexed
parameterization不是generic multi-head/MoE的重命名；P4需要neutral matched
problem evidence；BSCA只能claim ISCF-specific co-adaptation，不能claim generic
balancing novelty。

Decision=`intro_initial_draft_landed_major_revision_required`。下一rollback不是
method Step4或修改fixed model，而是paper consolidation：先做prior-art
positioning与problem-evidence/attribution design，再压缩P5术语并在main tables
完成后补headline results。P1--P5暂保留已冻结状态；P6、draft与review建议均为
provisional。new implementation、remote training、formal test=false。

## ISCF Framework and Introduction P5 v0.6 Consensus (2026-07-28)

ISCF暂时冻结为`Independent Scope-Conditioned Forecasting`，核心不再是multiple
fields加fusion，而是单一`scope-indexed forecast field`
$\mathcal F_\theta(\mathbf X,\tau,c,s)$。固定scope $s$对应该field的一个
`scope-conditioned slice`；各slices共享encoder、future-step representation与
future-step-specific synthesis vectors，只有scope-specific history projection
与latent-state sharing pattern独立。

scope的正式结构语义改为`future-step latent-state sharing scope`，其尺度称
`cross-step latent-state sharing extent`。中间state完整称
`history-conditioned, region-indexed latent state`，简称`scope-region latent
state`。原policy改为`target-conditioned scope allocation`
$\pi_\theta(s\mid\mathbf X,\tau,c)$，输出通过沿scope轴weighted contraction获得。
BSCA稳定scope slices与allocation之间的joint training，不claim generic
load-balancing novelty。

Introduction P5已按新框架重写并暂时冻结。P6 v0.5被新定义取代，下一步只重写
P6，不重新打开P1--P5。Decision=`intro_p5_v06_framework_consensus`；本轮不修改
implementation，不授权remote training或formal test。

## ISCF-BSCA Introduction P5--P6 v0.5 Draft (2026-07-28)

论文架构文档已形成Introduction方法概览与contributions讨论稿，状态为
`pending_user_confirmation`，不覆盖P1--P4冻结共识。

P5把Paragraph 4的problem层术语映射到method层：每个scope-coupled forecast
field都覆盖future domain，但具有独立history-to-mode map，并按一种
future-step coupling scope复用history-conditioned generation state；
step-specific synthesis保留target flexibility，forecast-target-wise fusion在
sample、variable与future-step层面组合fields。BSCA只作为ISCF-specific
train-only co-adaptation机制，作用边界是policy-mediated gradient allocation与
premature concentration，不claim forced specialization或generic balancing novelty。
模型继续采用horizon无关、future-step-indexed interface，不使用max-$T$-then-crop
的Introduction叙事。

P6暂定三项贡献：一是prefix-consistent unified multi-horizon formulation与可检验的
future-region sharing-demand hypothesis；二是ISCF完整output-side architecture
chain；三是BSCA及完整matched evidence。CHPC不单列method novelty，generic
multi-scale/multi-predictor/MoE原语不作为ISCF novelty。最终performance句须等待
horizon-specific、matched unified与modern baseline主表完整后再冻结。

Decision=`intro_p5_p6_v05_discussion_draft`。下一步先由用户逐段确认P5、P6，
再进入Related Work或Problem/Motivation具体实验冻结；本次没有new implementation、
remote training或formal test授权。

## ISCF-BSCA Paper Architecture and Introduction P1--P4 v0.4 (2026-07-24)

全文结构已落地到`docs/iscf-bsca-paper-architecture.md`。正文固定为六章：
Introduction、Related Work、Problem Formulation and Empirical Motivation、
ISCF-BSCA Method、Experiments、Conclusion；不设置独立Discussion。问题存在性证据在Method前使用已有baseline或
simple matched diagnostics建立，Experiments只评估最终模型、matched attribution、问题缓解、效率与transferability。

Introduction第1--3段冻结：$H$=`forecast horizon`，$\tau$=`future time step`，
$(\tau,c)$=`forecast target`。宏观模型称`unified multi-horizon forecaster`，不用`single-checkpoint`。
中文使用“horizon无关”，英文架构使用`horizon-agnostic`，性质使用`horizon-invariant`。

任务定义改为horizon无关、future-step-indexed field $g_\theta(\mathbf X,\tau,c)$；任意$H$-step forecast直接由
$\tau=1,\ldots,H$的predictions组成。CHPC来自同一future step不随requested horizon改变，而不是Introduction中的
max-$T$-then-crop叙事；ISCF以step-specific synthesis coefficients实现该接口，同时保留scope-level latent
sharing。不同horizons是nested outputs，不称为independent generation。

Introduction第4段冻结三层术语：问题=`future-region sharing-demand heterogeneity`，可检验表现=
`region-dependent sharing-scale preference`，方法=`future-step coupling scope`。sharing demand表示finite-capacity
decoder中一个history-conditioned generation state适宜被多宽范围的future steps共同复用。broad sharing可能降低
estimation variance但增加local-detail bias；fine-grained generation可能提高local flexibility但增加参数估计与优化
难度。不同future regions的multi-scale component importance只有通过这一bias--variance mechanism和matched evidence，
才能推出sharing-demand heterogeneity。

现有decoder不作global/independent二分：iTransformer与PatchTST是shared representation加step-specific rows；
DLinear是step-specific linear rows但仍共享输入分解和joint training；N-HiTS是预定义multi-scale trajectory
synthesis。future region仍定义为预测域内部连续future steps的集合，不是requested horizon；单个future step本身
没有coupling scope。

Problem Formulation的Evidence III使用simple baseline上的capacity-matched、end-to-end diagnostic predictors建立
region-wise risk crossing、best-scale变化与best-fixed-scale headroom；frozen probe和data-side multi-scale energy只作
secondary/descriptive evidence。若claim temporal contiguity，必须加入ordered versus random matched control。

Decision=`paper_architecture_v04_intro_p1_p4_consensus`。CHPC只作system contract，不单独claim算法创新；
不声称所有已有模型缺少CHPC、unified必然更弱、strict gradient conflict、canonical grouping必要或universal
conditional specialization。Introduction第5--6段与后续章节仍为provisional，下一步按用户顺序继续逐段讨论。
本次不改变frozen ISCF-BSCA-v1，不授权new training/test。

## ISCF-BSCA-v1 Three-seed Step9/10 Pass (2026-07-22)

10/10 confirmation trainings与single frozen formal test完整，合并seed2021后为60 test cells。BSCA相对same-seed EQUAL
MSE/MAE=`+0.3541%/+0.3073%`，41/60 MSE cells、3/3 seeds、4/5 datasets、4/4 horizons positive；ETTm2
mean=`-0.6506%`，满足冻结`>=-1%`边界但保留为limitation。15/15 initialization pairing、checkpoint nonmutation、
artifact protocol和internal health全部通过。

Decision=`passed_core_candidate_ready_for_paper_consolidation`。BSCA冻结为ISCF-native training contribution；不再调
lambda、按dataset/horizon rescue或叠加新router/loss。Post-hoc cluster bootstrap 95% interval跨0，故论文只claim
small directionally robust gain。下一步先完成ISCF architecture与BSCA objective的完整贡献链、claim boundary与ablation
设计，再进入modern baselines/main table；当前没有新training/test授权。

## ISCF-BSCA-v1 Three-seed Confirmation Prelaunch (2026-07-22)

用户授权按handoff继续seeds2022/2023 confirmation。10个new BSCA runs与FCC已有10个same-seed EQUAL controls构成
20-run confirmation matrix；与seed2021合并后审计60 test cells。无lambda/dataset/horizon tuning，无新loss/router。

冻结两级gate：direction robustness要求macro MSE/MAE positive、2/3 seeds、3/5 datasets、3/4 horizons；paper-core
promotion进一步要求macro MSE >=+0.3%、minimum dataset >-2%、ETTm2 >=-1%与health/nonmutation pass。

[Step 8 Launch] commit `72e3356`、three-GPU preflight与Weather resource smoke通过。10-run training于
2026-07-22 19:39:57 +08:00启动：seed2022=`GPU0/1`、seed2023=`GPU2`，首批三个jobs进入epoch1；formal test
保持`0/10`，仅在10/10 training artifacts完整后执行一次。Decision=
`confirmation_step8_training_active_formal_test_guarded`。
Local contract、10-job dry-run、10/10 formal guard和reference completeness均通过。下一步commit-pinned remote pull、
GPU/process audit与Weather smoke，通过后启动10 trainings；10/10前formal test硬禁用。

## ISCF-BSCA-v1 Step9/10 Result (2026-07-22)

Five trainings与single frozen formal test均5/5完整；candidate/EQUAL initialization hashes逐dataset全配对，checkpoint
nonmutation与protocol invariants通过。Official test MSE/MAE=`+0.3104%/+0.4902%`，15/20 cells、3/5 datasets、
3/4 horizons，刚好通过预注册gate；validation=`+0.6490%/+0.4492%`。

Mechanism health显示policy被稳定推向broad access：entropy `0.9983 vs 0.7913`、usage max `0.2042 vs 0.2528`；
pairwise arm L1仅小幅下降，oracle headroom仍32.56%。该结果支持balanced co-adaptation，不支持conditional-routing
specialization claim。ETTm2 test mean `-1.7375%`且H192/H336/H720 material negative，因此结论保持single-seed partial。

Decision=`performance_partial_pass_pending_confirmation_seed`。下一步只可在新授权与新冻结matrix后运行BSCA seeds
2022/2023，并复用对应EQUAL references；不允许lambda/dataset/horizon tuning。若跨seed positive再升paper core；否则
回Step4收窄claim，不否定fixed ISCF architecture。

## ISCF-BSCA-v1 Step4–7A and Frozen Test Gate (2026-07-22)

用户明确停止将 broad-anchor sufficiency 作为独立诊断门槛，授权直接形成 outcome-first method candidate。UPA-D2由
`ISCF-BSCA-v1`取代：ISCF-v0 architecture/inference unchanged；EQUAL objective上增加target/H-free uniform policy KL，
以25% ramp到0.1约束train-time scope-gradient allocation。

Primary-source audit确认generic load balancing/entropy regularization已有强prior且可能损害specialization。因此narrative
只在ISCF-specific完整链上conditional pass，EQUAL为same-architecture no-anchor attribution control。Step7A objective、
five-run runner、5/5-before-test hard guard、analyzer与code explanation已实现；local tensor/gradient contract待prelaunch
record汇总。Formal matrix冻结为5 datasets × H96/H192/H336/H720 × MSE/MAE，confirmation seeds=false。

Decision=`bsca_step4_6_conditional_pass_step7a_prelaunch`。下一步完成最小验证、commit/push、remote GPU/process preflight
和one-job Weather resource smoke；通过后启动5 runs，5/5完整后才允许一次formal test。

## ISCF PSA-D1 Result and UPA-D2 Gate (2026-07-22)

D1最终20/20 runs、80/80 validation cells与5/5 SHA-nonmutation diagnostics完整。New EQUAL逐checkpoint、metrics、
fused/arms/policy完全复现historical EQUAL，MSE/MAE gain=0，排除run drift。ARMERR/SHUFFLED相对new EQUAL仍为
`+0.6577%/+0.6557%` MSE，均17/20、5/5 datasets、4/4 horizons。

Decision=`joint_training_route_regularization_supported_as_carrier_clue`。结合D0 post-hoc negative，收益定位到
train-time arm-policy co-adaptation；但generic balancing与target semantics未归因，不升method/test。下一最小diagnostic
UPA-D2冻结information-free uniform-target KL，与controls共享weight/schedule；implementation/remote/test未授权。

## ISCF PSA-D1 v0.1 Diagnostic Repair (2026-07-22)

ETTh1已完成training与standard validation metrics，但diagnostic evaluator因缺少future-bin config在probe前失败。该结果
标记`diagnostic_protocol_fault_predecision`，没有research decision/test access。v0.1只增加evaluator contracts与dedicated
SHA-nonmutation validation replay runner；training matrix、checkpoints和gates不变。

当前其余training继续，结束前不remote pull。5/5 checkpoints后pull repair commit、GPU preflight、补做5 diagnostics，再
一次性运行full analyzer。

## ISCF PSA-D1 Step8 Remote Running (2026-07-22)

commit `f5275a4`、three-GPU idle preflight与Weather resource smoke通过。Smoke route weight/loss=0、five scope gradients
nonzero、initialization hash与three references一致。five-run validation已于`16:00:40+08:00`启动，初始0/5，
Weather/ETTm1/ETTh1 epoch 1 active，ETTh2/ETTm2 queued。

Decision=`psa_d1_five_run_validation_training_active_formal_test_disabled`。5/5后统一运行冻结H2/H3 analyzer；不读取
partial结果，不remote pull或改gates。formal test/confirmation/method promotion false。

## ISCF PSA-D1 Step7A and Prelaunch (2026-07-22)

用户明确授权PSA-D1 Step7A + five-run validation training。D1只新增contemporaneous `equal_skill` control，不改ISCF
architecture/loss。config/checker/runner/analyzer与code explanation已完成；source semantic diff=none、route loss/weight=0、
five-scope random-tensor gradients nonzero、dry-run=5 jobs，H2/H3 synthetic decision branches通过。

Decision=`psa_d1_step7a_pass_proceed_commit_remote_preflight`。下一步commit-pinned remote pull、GPU preflight与
Weather resource smoke，通过后启动five validation runs。formal test、confirmation seeds、method promotion false。

## ISCF PSA-D0 Result and Rollback (2026-07-22)

15-run LODO frozen diagnostic完整。Convex-uniform macro L1/MSE=`-0.2431%/-0.1218%`，1/5 datasets、2/15
runs joint-positive；4/5 folds虽选择nonzero alpha，但ETTh1/ETTm2/Weather held-out反转。scope-marginal与temperature
controls也分别为`-0.2570%/-0.1477%`、`-0.2615%/-0.2378%`。

Decision=`frozen_inference_shrinkage_not_supported`。H1 post-hoc policy overfit关闭，不做alpha/dataset/position
rescue；generic entropy/temperature route不升candidate。failure=`frozen_probe_negative_joint_training_unresolved`，
不能拒绝训练期co-adaptation。下一最小attribution control `SC-ISCF-PSA-D1`已冻结为five-dataset seed2021
contemporaneous EQUAL retrain，以区分H2 co-adaptation与H3 run drift；implementation、remote training与formal test
尚未授权，active method=none。

## ISCF Post-RSCC Step2/4 and PSA-D0 (2026-07-22)

ARMERR与SHUFFLED共同超过EQUAL约`+0.656%` validation MSE且彼此只差`0.0020%`；seed2021 probes进一步显示
两者fused relative L1=`0.00138--0.00462`、policy mean L1=`0.00254--0.00830`，均收敛到entropy约
`0.986--0.998`的near-uniform policy。正确coalition binding没有解释公共收益；exact route保持closed。

由于EQUAL是historical而非contemporaneous retrain，当前必须区分post-hoc policy overfit、joint-training
co-adaptation与run drift。Decision=`policy_shrinkage_problem_unresolved_proceed_d0_diagnostic_only`：冻结
`SC-ISCF-PSA-D0`，只复用15个existing validation replays，以fixed grid、147/109 source-aligned split和LODO
global-alpha selection审计frozen policy frontier。uniform endpoint、scope-marginal prior与temperature为controls。
forecast training、official test与method implementation均false；negative不得拒绝joint-training方向。

## ISCF-RSCC Step9 Decision (2026-07-22)

20/20 effective runs、80/80 validation cells与所有protocol/init/non-test checks通过。RSCC相对EQUAL
MSE/MAE=`+0.5189%/+0.3972%`，5/5 datasets与4/4 horizons通过primary validation gate；但相对
EQUAL-ARMERR/SHUFFLED分别为`-0.1414%/-0.1394%` MSE，两个matched controls都失败。

ARMERR和SHUFFLED相对EQUAL均约`+0.656%`且彼此只差`0.0020%` MSE。RSCC headroom保持正、gradients/usage
健康，但policy-credit Spearman从EQUAL `0.2052`降到`0.1539`。Decision=
`rscc_v1_control_attribution_fail_close_exact_route`；failure=`capacity_control_explains`。关闭exact SCC/RSCC，
不做formal test/seeds/lambda/router rescue；ISCF fixed base保留，回Step2/4重新做problem/narrative gate。

## ISCF-RSCC Step8 Remote Running (2026-07-22)

Weather RSCC/SHUFFLED resource smoke通过：相同initialization hash、相同EQUAL skill loss、nonzero route loss、five
scope gradients finite/nonzero，无OOM或numeric pathology。commit `020eea3`的15-run validation matrix已于
`14:12:34+08:00`在GPU0/1/2启动，首批三个Weather arms active；formal test=false。

Decision=`rscc_step8_validation_training_active_formal_test_disabled`。等待15/15与60/60 cells完整后一次性Step9；
不读取partial favorable cells，不修改config/gates，不做seed/lambda/router rescue。

## ISCF-RSCC Step7A (2026-07-22)

RSCC exact hybrid实现并通过contract tests：reliability loss逐值等于EQUAL，coalition/shuffled只改变route calibration；
15-job dry-run与existing PCC 36/36 regression通过。下一步Weather RSCC/SHUFFLED resource smoke；通过后才启动15 runs。

## ISCF-SCC Step9 Failure and RSCC Rollback (2026-07-22)

SCC-v0 validation相对EQUAL为`-3.1750%/-1.7742%` MSE/MAE，且未超过FUSED/ARMERR/SHUFFLED。all numeric和
gradient health通过，但coalition headroom由`+18.08%`反转为`-14.93%`。failure=`intervention_point_wrong`：删除
equal-skill破坏arm reliability。v0关闭，不做seed/lambda rescue。

Step5冻结唯一successor RSCC-v1：EQUAL reliability + exact coalition KL；15 new-run matrix匹配EQUAL-ARMERR与
RSCC-SHUFFLED。当前只授权Step7A实现，remote/test false。

## ISCF-SCC Step8 Remote Running (2026-07-22)

resource smoke通过，20-run matched validation已从commit `91e466a`在GPU0/1/2启动；首次状态为0/20 complete，前三个
Weather jobs处于epoch 1。下一步等待全matrix完成后统一Step9分析，不读取partial favorable cells；formal test仍false。

## ISCF-SCC Step7A and Step7B Gate (2026-07-22)

exact SCC/shuffled objective已实现；credit full detach、uniform fallback、dedicated shuffle RNG与five-scope gradient logging均通过
contract tests。20-run validation config与dry-run通过。Decision=`step7a_pass_step7b_remote_validation_authorized`；下一步
Weather SCC/SHUFFLED resource smoke，通过后启动20 runs。test仍false。

## ISCF-SCC D0B Pass and Step5–6 (2026-07-22)

修正为147/109 source-sample-aligned split后，D0B仍通过全部gate：median predicted L1 gain=`1.3727%`，15/15
positive，14/15 shuffle-binding，vs standalone median=`+0.5143` point、13/15 positive，5/5 datasets三seed全正。

SCC-v0 narrative gate现`pass_to_step7a_matched_validation_only`。exact objective为harmonic fused L1 + ramp-to-.1的
coalition KL，credit stop-gradient、all-negative uniform fallback、不保留individual arm loss；inference graph不变。下一步只
授权Step7A实现和contract tests；remote validation/formal test仍false。

## ISCF-SCC D0 Result and D0B (2026-07-22)

D0完成15-run frozen validation replay。median target-visible coalition headroom=`17.9766%`，15/15 nondegenerate且
15/15超过scope-label shuffle；但fixed-label topology只在ETTh1/ETTh2跨seed稳定，machine decision=`unresolved`。
按预注册rollback不进入Step7，也不把当前frozen diagnostic用于方向级拒绝。

D0B冻结60/40 blocked-row low-capacity ridge probe，输入只含inference-available arms/policy/position，比较coalition、
standalone与16个horizon-marginal shuffles。下一步commit/push后在existing validation NPZ上offline执行；无forecast
training/test access。active method仍none。

## ISCF-SCC D0 Validation Replay Prelaunch (2026-07-22)

15个historical ISCF NPZ缺少exact per-coordinate direct policy，只有bin-level usage；因此无法从one fused equation唯一恢复
five policy weights。按Step2–6预注册fallback，冻结same checkpoints的15-run validation-only replay，补齐
`probe_direct_policy [256,720,5]`。runner只调用frozen checkpoint evaluator，输出到repo-external root，并检查checkpoint
SHA256 nonmutation；training/test入口均false。

local preflight通过，three-GPU state均18 MiB、0% utilization。Decision=
`d0_validation_replay_prelaunch_pass_remote_forward_authorized`。下一步commit/push、remote fast-forward并执行15个
validation forwards；随后运行D0 analyzer。active method仍none。

## ISCF Post-FRSC Step2–6 Innovation Portfolio (2026-07-22)

用户将研究范围扩大为：固定ISCF architecture base，但允许探索与其原生耦合的loss、training和architecture extension；目标是
形成连贯paper story、补充创新边界并提升official-test性能。现有15-run function evidence显示arms已有稳定function diversity与
median `8.5813%` oracle headroom，但fusion仅9/15超过best fixed arm。代码审计确认`equal_skill`实际为
`fused loss + uniform individual arm target loss`，没有提供coalition-specific role signal。

候选portfolio把`SC-ISCF-SCC-v0 — Scope Coalition Credit`列为primary diagnostic-gated route；fused-only/skill annealing
只作objective control，CPSI successor deferred，SPS/FRSC exact routes保持closed。SCC利用ISCF dense fusion闭式计算
leave-one-scope-out risk，train-only校准existing direct policy；inference architecture、requested-H input与latency不变。
最新primary-source audit确认generic expert loss、orthogonality/diversity、structural anchor、frequency experts、Shapley和
counterfactual routing均已有直接prior，所以claim只允许位于完整ISCF-specific chain。

Decision=`scc_problem_diagnostic_proposed_active_method_none`；narrative gate=`conditional_pass_to_d0_only`。下一步只复用
existing 15 checkpoints/probes完成D0 coalition-credit problem audit；implementation、remote training、formal test和modern
baselines均false。详细记录：
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_frsc_step26_20260722/step2_6_innovation_portfolio_and_scc_gate.md`。

## ISCF-FRSC Step4–6 Decision (2026-07-22)

exact SPS-v0完成20/20 validation但相对identity MSE `-2.3123%`，failure attribution为hard capacity restriction；不否定
ISCF。BSC frozen affine readout无positive cell。FRSC以$Q_s=P_s+(1-\alpha)(I-P_s)$保留full rank并对scope外方向做
nonzero gradient conditioning。D1.1 frozen canonical在alpha .55达到MSE `+0.7997%`、5/5 datasets、4/4 horizons，random
为`-8.9750%`；best global为`+0.8677%`，所以scope attribution仍未成立，但risk envelope足够competitive以进入新候选
Step4–6。candidate=`SC-ISCF-FRSC-v0`，narrative conditional pass；下一步Step7A。新remote training、formal test、loss/router
和requested-H conditioning均为false。详细记录：
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_frsc_step46_20260722/step4_6_design_and_remote_gate.md`。

## Current Cursor

| Field | Content |
| --- | --- |
| `stage` | `StageC-UVHF` |
| `current_step` | paper consolidation；Introduction frozen；Section 3 integration |
| `active_question` | 如何把CHPC与sharing-demand两项问题证据组织成严谨的Problem Formulation and Empirical Motivation |
| `active_candidates` | exact `ISCF-BSCA-v1` frozen paper core；Figures 1--3 approved；Method Figure 4 planned |
| `future_validation_suite` | ETTh1, ETTh2, ETTm1, ETTm2, Weather |
| `active_protocol` | `docs/stage-ledgers/stage-c-iscf-bsca-paper-writing-restart-handoff-20260731.md` |
| `restart_handoff` | `docs/stage-ledgers/stage-c-iscf-bsca-paper-writing-restart-handoff-20260731.md` |
| `method_implementation` | none authorized；remote training/formal test false |
| `rollback_point` | Section 3 evidence不足则收窄claim；不得通过重启closed method search修补叙事 |

## ISCF-FRSC Step9 Validation Decision

20/20 new runs与25/25 effective audits完整，无numeric pathology或test access。candidate vs identity MSE/MAE=
`-1.2745%/-0.4184%`，primary guards全部失败；same-alpha scope vs global为`+0.7215%`，但best-global-a045把优势压至
`+0.0703%`，canonical vs random仅`+0.1781%`且MAE负向。internal activity健康，但oracle-headroom preservation
guard失败。

Decision=`frsc_v0_validation_continuation_not_supported_rollback_step4`。这是development stop，不是formal-test direction
rejection。exact FRSC-v0关闭，ISCF architecture prior保留；不做seed/alpha/per-cell/loss/router rescue。下一步回到Step4提出
新的scope utilization mechanism，并重新通过narrative/design gate。

## ISCF-FRSC Step7B Prelaunch Decision

matrix冻结为`frsc_scope_a055/frsc_global_a055/frsc_global_a045/frsc_random_a055` × five datasets × seed2021，
共20个new runs；5个历史`SPS identity` checkpoints只作frozen reference，形成25-run effective audit。primary performance、
best-global、same-alpha global、random binding和internal health五层角色分离，validation不能替代formal test。

local prelaunch `37/37`通过。用户于`2026-07-22`明确授权推进remote training。Decision=
`frsc_step8_remote_validation_authorized_formal_test_disabled`；下一步commit/push、remote fast-forward、GPU/process preflight、
two-arm resource smoke并启动20-run validation。test、confirmation seeds和modern baselines仍false。

remote commit=`9069e87`，GPU/process preflight与Weather candidate/random resource smoke通过。正式matrix于
`2026-07-22T10:41:20+08:00`启动，runner PID=`3559159`，初始`0/20`且三个Weather jobs进入epoch 1。Decision=
`frsc_step8_training_active_formal_test_disabled`；完成后先做validation audit，不自动访问test。

## ISCF-SPS Step4–7A Decision

用户将ISCF multi-scope architecture固定为design prior。SAC negative不删除，但不再作为停止研究的条件；其新的解释是
“当前实现没有充分利用scope”。代码审计定位到具体缺口：五个scopes虽有independent history maps和不同group states，
却共同使用逐target unrestricted `identity_synthesis/nonlinear_synthesis [T,K]`，因此coarse scope仍可生成generalist
full forecast，scope extent不会持续约束output/gradient。

`SC-ISCF-SPS-v0`在raw arms进入既有direct fusion前加入scope-native local-DCT projector。rank规则为
$r_s=\min(s,\max(1,\mathrm{round}(Ks/720)))$；$P_s=C_sC_s^\top$同时定义forecast subspace与该scope map收到的
backward error subspace。candidate新增0 trainable parameters，保持single MSE/equal-skill objective，不增加router、
requested-H或auxiliary specialization loss。

Step4–6 narrative gate为`conditional_pass`。NHITS/N-BEATS已覆盖hierarchical interpolation/additive components，
TimeMixer/FreqMoE已覆盖multi-scale predictors/frequency experts，MoE specialization objectives也有直接prior；所以claim
限定为`future-output coupling scopes -> scope-native synthesis/gradient subspaces -> target-wise full-domain composition`
及matched attribution，不claim DCT、interpolation或expert diversity primitive。

Step7A全部通过：paired parent/identity/candidate/global/random parameter hash一致；identity-parent max gap
`8.34e-7`；prefix gap`0`；orthonormal/idempotence errors `3.22e-15/1.53e-16`；five gradients非零；production
model/CLI通过。Decision=`conditional_pass_as_scope_utilization_architecture_step7a_complete`。下一步冻结candidate/
identity/global/random的validation-first Step7B matrix；remote training与formal test均false。

## ISCF-SPS Step7B Prelaunch Decision

20-run matrix完整覆盖四个new-training arms和五个frozen natural profiles。scope candidate相对identity是primary validation
effectiveness；global projection判断收益是否只是generic smoothing；random partition只判断canonical scope binding归因，不允许
方向级拒绝ISCF。每个run除H96/192/336/720 MSE/MAE外，还需保存raw/projected/removed arms、direct policy、arm diversity、
oracle headroom、future-bin winners与projection retention。

local gate `19/19`通过。首轮prelaunch发现配置错误地把projector的$K$写成256；production实际按设计使用dataset-matched
mode rank 106/109/116。仅修正显式rank/degrees contracts，未改forward。runner normal launch在authorization=false时exit 3，
test split也在dry-run前硬拒绝。

Decision=`step7b_prelaunch_pass_wait_remote_authorization`。下一步只有在明确授权后才commit-pinned remote pull、执行
`nvidia-smi`、dual resource smoke并启动20-run validation matrix。validation结果返回前不设计formal test；test、confirmation
seeds与modern baselines仍false。

## ISCF-SPS Step8 Remote Authorization

用户于`2026-07-22`明确授权“启动20-run validation matrix”。authorization只开放seed2021的20个frozen trainings；
`formal_test_access_authorized=false`保持不变。启动顺序固定为local commit/push、remote fast-forward、three-GPU preflight、
Weather scope/identity resource smoke、正式matrix。20/20完成后先分析validation，不自动执行test。

Decision=`step8_remote_validation_authorized_formal_test_disabled`。

remote commit=`48afd12`，three-GPU preflight与two-arm resource smoke已通过。20-run training于
`2026-07-22T00:17:31+08:00`启动，runner PID=`2787170`，初始`validation=0/20`，首批三个Weather jobs均进入
epoch 1。Decision=`step8_training_active_formal_test_disabled`；完成后先做validation audit，不自动访问test。

## ISCF-v0 SAC Step9/10 Decision

formal matrix完整：25/25 new tests、60/60 effective runs、240/240 standard rows、15/15 internal health与25/25
checkpoint nonmutation。Q1-WIDE gate通过：MSE/MAE=`+0.8496%/+0.5996%`，5/5 datasets、4/4 horizons、
2/3 seeds。RANDOM gate失败：`-0.1990%/-0.4347%`，1/5 datasets、0/4 horizons、1/3 seeds。

ISCF相对A6_FULL仍有`+1.3584%/+0.9144%`，所以它是strong performance carrier；但canonical contiguous/nested
grouping没有超过same-parameter random grouping，exact收益只能归因到generic independent branches。该剩余claim受现有
multi-branch/independent-expert prior覆盖，不能单独paperize。

Decision=`temporal_scope_structure_not_supported_generic_independent_branches_explain`。关闭ISCF exact
paperization，active method清空，modern baselines保持false；rollback Step2/4，不做任何test-informed rescue。

## ISCF-v0 Post-CPSI Step4/5 Decision

exact CPSI-v1关闭后，没有继续搜索interaction operator。新的problem boundary把ISCF定义为future-output coupling-scope
factorization：五个independent maps改变的是future coordinates在nonlinear synthesis前的latent sharing extent，而不是
history sampling scale、requested horizon或expert task ID。

现有ISCF vs A6_FULL three-seed MSE/MAE=`+1.3584%/+0.9144%`，5/5 datasets、4/4 horizons、3/3 seeds；
D1.1 15/15超过direction/random-init controls，private response median=`0.7197`。但这些都是test-informed carrier/problem
evidence；ISCF最初作为SIFF control，不能post-hoc直接promote。2024--2026 primary sources也已覆盖generic multi-scale
predictors、multi-branch complementarity、expert gating与forecasting sub-task agents。

Decision=`conditional_pass_as_output_coupling_scope_architecture_pending_sac`。SAC冻结两项primary controls：

1. `Q1-WIDE`：相同五scope/synthesis/policy/objective，active-param gap最大`0.4646%`，只用一个wide shared mode map；
2. `RANDOM-PARTITION`：与ISCF相同readout/rank/params，只随机化48/144/360 scopes的future grouping。

新增training仅为Q1-WIDE seeds2022/2023的10 runs与RANDOM三seed的15 runs。candidate不改、loss/router不改；remote
training和single formal test仍需新授权。两项test gate都过才进入modern baselines；任一失败即回portfolio decision。

## ISCF-v0 SAC Step7B Prelaunch Decision

SAC local execution chain已完成且`18/18`通过。25个new jobs精确覆盖Q1-WIDE seeds2022/2023与
RANDOM-PARTITION seeds2021/2022/2023；35个historical references由two source config/run-audit SHA256 contracts
冻结，合计60 effective runs和240 standard-horizon rows。validation只选checkpoint；formal test必须等待25/25
training并在单独process中执行。

five-profile static contracts确认canonical/random parameter count、readout/Encoder initialization与global RNG
post-state一致；endpoint scopes相同、中间scopes不同。Q1 max active-parameter gap=`0.464638%`。runner syntax、dry-run、
analyzer smoke、remote scanner fallback和unauthorized-launch rejection均通过。

Decision=`step7b_prelaunch_pass_waiting_remote_authorization`。当前不含training/validation/test evidence，active method仍
none。下一步仅在显式授权后执行commit-pinned remote pull、`nvidia-smi`、Weather-RANDOM与ETTm2-Q1 resource
smokes；通过后启动25-run matrix。formal test、modern baselines、router、second loss仍不自动授权。

## ISCF-v0 SAC Step8 Training Authorization

用户于`2026-07-21`明确授权“继续推进SAC remote training”。config现只开放25个frozen new trainings；
`formal_test_access_authorized=false`保持不变。training继续使用four-horizon validation mean MSE选择checkpoint，
不得因validation排序删除arm或改变gate。

Decision=`step8_remote_training_authorized_formal_test_pending`。下一步先commit/push、remote fast-forward、三卡
preflight与Weather-RANDOM/ETTm2-Q1 resource smoke；smoke通过才launch。25/25完成后不自动执行test。

remote现已fast-forward到commit `78cbcf4`；三卡preflight空闲，两项resource smoke finite/no-OOM。25-run
training于`18:58:40+08:00`在GPU0/1/2启动，supervisor PID=`2383292`，首批三个Weather jobs active，
training/test=`0/25,0/25`。Decision=`step8_training_active_formal_test_not_authorized`；训练期间冻结repo/config/gates，
完成后等待test授权。

## ISCF-v0 SAC Step8 Validation Artifact Audit

training于`20:24:32+08:00`完成，new training/checkpoints/validation=`25/25`，formal test=`0/25`。联合35个
historical references后，60/60 run audits、240/240 validation rows和15/15 internal-health pairs完整；无numeric或
protocol pathology。analyzer新增`--validation-only`，不会调用official-test decision；同时修复A6不使用PCSD却被
无意义partition字段误判的问题，ISCF canonical/random checks未放宽。

validation observation：ISCF over Q1-WIDE MSE/MAE=`+1.0704%/+0.7538%`；canonical over RANDOM为
`-0.1823%/-0.3075%`。前者是independent maps的positive lead，后者是contiguous/nested partition的negative lead；
两者都不能替代test gate。

Decision=`formal_test_ready_pending_user_authorization`。下一步只能在独立授权后开放一次冻结formal test；在此之前
active method仍none，不进入modern baselines，不按validation结果做rank/partition/seed/loss/router调整。

用户已于`2026-07-21`独立授权SAC formal test。config切换为`authorized_prelaunch`与
`formal_test_access_authorized=true`；授权只覆盖现有25个new checkpoints的一次完整test，不允许retraining、mutation或
per-cell tuning。Decision=`step9_formal_test_authorized`。下一步commit-pinned remote launch；official-test结果返回前
active method仍none。

首次formal launch在test loader创建前因missing `diagnostic_protocol.future_bins`停止，test access artifacts仍0/25、
checkpoint未变。归因为exact protocol preflight gap。当前rollback到Step7B runtime repair：补入固定8-bin contract、
runner静态断言，并先做validation-split real-checkpoint smoke；不回到method design或training。

## ISCF-v1-CPSI Step9/10 Decision

25 new runs与single formal test完整。CPSI vs ISCF-v0 test MSE/MAE=`-2.2128%/-1.6987%`，1/5 datasets，
4/20 MSE cells；vs A6_FULL=`-0.7775%/-1.0606%`。validation同向negative，四horizon全部负，internal health
25/25正常，因此触发material effectiveness fail而非mild/inconclusive或numeric failure。

CPSI优于SELF/COMMON，但落后LINEAR/POST。LINEAR vs ISCF仅`+0.0217%/+0.0472%`，属于tie且function class
可吸收；POST vs ISCF MSE/MAE=`-0.4882%/-0.6362%`，也不promote。

Decision=`cpsi_v1_exact_performance_fail_return_step4_5`。active method清空，ISCF-v0继续作为carrier；下一步先做
source-informed Step4/5 redesign，不授权新implementation、training、test、confirmation、router或second loss。

## ISCF-v1-CPSI Step7B Prelaunch Decision

frozen matrix为25 new trainings + 10 hashed historical references；35 effective runs覆盖H96/192/336/720的140 MSE
与140 MAE cells。runner将validation-only training与formal test拆开，并要求25/25 training artifacts齐全才开放一次
test。SELF/LINEAR/COMMON/POST始终是intermediate diagnostics，不因validation排序删除。

初始18/18后，remote smoke发现缺少`rg`会导致negated scanner false pass；加入`grep` fallback并扩展为19/19。
profile/config/auth、control governance、reference hashes、five constructors、paired parent hash、runner syntax/dry-run、
scanner与analyzer smoke均通过。evaluator已落盘六类CPSI internal RMS；analyzer分别裁决effectiveness、
matched attribution、health与failure cause。

Decision=`step7b_prelaunch_pass_step8_authorized`。下一步commit-pinned remote pull、GPU preflight、dual resource smoke后
启动25-run seed2021 matrix；25/25后才执行single formal test。confirmation、router、second loss保持false。

Step8已在commit `5d2330e`启动。三张3090 preflight空闲；修复scanner fallback后Weather-CPSI与ETTm2-POST
resource smokes通过。初始状态training/test=`0/25,0/25`，formal-test mode为0；25/25前继续禁止test。

## ISCF-v1-CPSI Step7A Implementation Decision

five production modes已接入`TimeAlign` coupling path与active CLI。81/81 local cases覆盖zero-init exact parent morph、
scope equivariance、private-absent zero message、two-stage gradient opening、真实model full/prefix forward、five CLI和five
profile parameter formulas。五arms base initialization hash相同，morph gap均为0；未发现NaN、permanent zero path或shape/
parameter mismatch。

Decision=`step7a_local_pass_step7b_prelaunch_next`。active method状态提升为
`implementation_ready_effectiveness_pending`，但这不是performance pass。下一步只做Step7B runner/analyzer、historical
reference contract与resource prelaunch；remote/test execution仍false。

## ISCF-v1-CPSI Step6 Design Decision

用户将SELF/LINEAR/COMMON/POST-SYNTH定位为intermediate diagnostics，并要求不因轻微负向在official test前关闭
mechanism。Step6将这一要求转化为可审计协议：validation只选checkpoint和监测health，全部protocol-valid arms都进入
冻结test；controls在test上决定attribution，而不决定是否允许test access。

SELF、LINEAR、COMMON已实现formula-level exact `3Lr` matching；COMMON使用
$W_o[\mathrm{GELU}(W_a\mu)\odot\mathrm{GELU}(W_b\mu)]$ broadcast，避免dummy parameters。POST-SYNTH直接在
`[B,C,S,720]` forecasts上作同构interaction，`r_post=round(Lr/720)`；module gap小于1.95%，total-model gap小于
0.041%，因此可作诚实placement diagnostic。global $r=32$在test前冻结。

formal matrix为25 new trainings，加ISCF-v0/A6_FULL 10 historical references，经hash/contract审计后形成35 effective
runs与140 test MSE/MAE cells。CPSI vs ISCF macro MSE `[-0.5%,+0.3%)`定义为inconclusive而非direction failure；
initial support要求`>=+0.3%`、3/5 datasets、10/20 cells及MAE guard。controls的`±0.3%`是attribution tie band，
不能用轻微差异claim necessity。

Decision=`step6_pass_step7a_local_authorized`。下一步只实现production modes、five arms与local checker，并同步
code explanation；remote/test execution在Step7B前保持false。

## ISCF Step5 Common–Private Interaction Decision

Step5首先证明：若$M_s=W_sh+b_s$，任意fixed linear scope mixing
$\widetilde M_s=\sum_jA_{sj}M_j$均等价于新的独立affine map。因此Cross-Stitch matrix、linear peer mean、
linear common/private decomposition和fixed graph diffusion不扩展ISCF-v0 mode function class，只可作为optimization
controls。plain peer MLP也被降为generic control，因为其收益可能来自另一组affine projection与额外depth。

working candidate `ISCF-v1-CPSI`只在`[B,C,S,D,K]` pre-synthesis modes上计算
$\mu=\operatorname{mean}_sX_s$、$\delta_s=X_s-\mu$，并用
$W_o[\operatorname{GELU}(W_c\mu)\odot\operatorname{GELU}(W_p\delta_s)]$生成mode interaction。shared无bias
maps使operator permutation-equivariant，common/private任一路缺失时message为零；`W_o=0`精确包含ISCF-v0。
以global theory rank 32估算只增加`0.5983%–2.2399%` active parameters，但rank仍需Step6冻结。

Bayes boundary不变：candidate不增加information，只可能提供finite-capacity inductive bias。DMSC v5、Deep Sets、Set
Transformer、Cross-Stitch、MoLE与TimeExpert使generic “multi-scale/expert interaction” claim不可用；论文边界必须绑定
future-output coupling scopes、Step4 controlled relation evidence、linear impossibility与matched attribution。

Decision=`step5_theory_pass_step6_control_design_next`，narrative=
`conditional_pass_to_step6_as_task_coupled_common_private_interaction`。Step6必须冻结exact-parameter SELF/LINEAR/COMMON
和诚实POST-SYNTH placement controls；若placement/capacity matching无法成立，则实现前关闭或降低claim。当前
active_method=none，不实现、不训练、不访问formal test，不新增router/loss。

## ISCF Step4 Response-Relation Decision

Step4将上一轮common residual降为shared-target-confounded clue，并以validation-label-free local response重新审计。
D1 primary的relation、random-init和noncollapse gates通过，但16-direction topology仅2/5；64-direction validity check恢复
5/5，归因`diagnostic_estimator_variance`。新冻结D1.1改用disjoint rows offset64、新seed与64 directions，最终15/15
超过direction-null和matched random-init，common/private median=`0.2803/0.7197`，4/5 datasets topology稳定。

Decision=`scope_response_relation_confirmed_for_step5_theory`。存在问题被收紧为：independent future-output scopes已经学习
pre-synthesis response dependence，但ISCF-v0只在完整forecast后late fusion。Narrative=
`conditional_pass_to_step5_as_single_pre_synthesis_architecture_problem`。

这不支持two-factor output compression、ordered scale、universal fixed graph或router。Step5只允许研究作用于
`[B,C,S,D,K]` modes、permutation-equivariant、zero-interaction包含ISCF-v0的最小operator；Deep Sets、Set Transformer、
Cross-Stitch、MoLE和multiscale coordination是mandatory boundaries。generic primitive不足以通过method narrative gate。

## ISCF-v0 Carrier Freeze and Function Audit

用户于`2026-07-21`授权把FCC的`SIFF_INDEPENDENT_EQUAL`以新identity `ISCF-v0`固定为后续research carrier。
该动作不改写SIFF-v2的ordered-attribution failure，也不把control role自动升级为paper method。exact contract由
`configs/stage_c_iscf_v0_carrier.json`冻结：五个independent scopes、direct policy、equal-skill、dataset-wise
ranks、natural profiles和three-seed checkpoint protocol保持不变。

由existing 60-cell × MSE/MAE table派生，ISCF-v0相对A6_FULL为`+1.3584%/+0.9144%`，MSE在
5/5 datasets、4/4 horizons、3/3 seeds正向。该结果确立strong carrier角色，但仍是test-informed post-hoc
comparison，不是独立预注册method claim。

低成本function audit只复用15个existing diagnostic NPZ，没有training、dataset loader或new test access。四项gate：

1. aligned low dimension：`0/15`，fail；median EV2 `0.6281`低于shift-null p95 `0.7223`；
2. common/private：15/15 common高于null，median private `0.0680`，pass但dataset heterogeneity显著；
3. complementarity：median oracle headroom `8.5813%`、unique best scopes `3`，pass；
4. topology：4/5 datasets cross-seed stable，pass；但canonical order rho median仅`0.2121`。

Decision=`function_relation_unresolved_requires_narrow_step4_audit`。generic shared-private、low-rank task relation、
multi-scale mixing与adaptive experts已有Cross-Stitch/Factorial MTL/TimeMixer/Pathformer/Moirai-MoE/M2FMoE等强prior。
下一步必须先完成non-ordered common/scope-specific relation的source-informed Step4 gate；不实现method，不新增
router/loss，不按test选择scope set。

## Post-D21 Unconstrained Reset

用户与研究审计共同确认：过去的`exact projectivity + requested-H禁用 + full-T prefix crop`组合显著压缩了真正的
horizon-adaptive自由度。后续这些contract不再是强制设计约束，只作为可能机制或controls，由理论与实验选择。

但pointwise MSE下，同一fixed past与future coordinate的Bayes conditional mean并不因requested horizon改变；所以
简单加入H embedding没有自动的统计必要性。新的Step2将自由度来源分为finite-capacity tradeoff、target-coordinate
information access、nonseparable/decision risk、future context、compute/resolution与probabilistic joint target。
当前task保持deterministic MSE/MAE，优先审计前两项。

A6-LBF经重新评估后保留为strong carrier/control，不作为standalone paper core：其性能证据可用，但basis、
projectivity与harmonic measure的独立novelty不足，且现代varied-horizon baseline comparison仍不完整。

`SC-D22-HFA` D22-A/B现已完成。D18 H1..720 full-test curves显示：SPEC96 own-H为`+1.2748%`且5/5
datasets正向，但SPEC192/SPEC336分别为`-0.1386%/-0.6385%`；三个specialists在standard-horizon向量上均
0/5 dataset Pareto-dominate A6_MEASURE。A6_MEASURE相对A6_FULL在五个lead-time bins全部5/5正向。
因此decision=`finite_capacity_frontier_not_supported`，H96只保留为局部optimization clue。

D22-A/B不能直接回答target-coordinate raw-history information access。D14 dual-carrier three-seed headroom使
D22-C具有条件合理性。D22-C static/prelaunch现已通过：neutral/raw-history为primary，六臂复用完全相同参数集与
初始化；global/pooled/order-shuffled/target-shuffled/generic为controls，A6 sensitivity尚未授权。local synthetic
execution与machine decision pipeline通过，seed2021 five-dataset完整validation/test problem gate已获授权，等待
commit/push和3090 GPU preflight。只有D22-C通过，才允许source-informed设计
`lead-time-conditioned evidence operator`；第二contribution只能来自首个E2E operator暴露的真实训练瓶颈，
不得预先指定loss/router。

首次v1 remote launch在training-only log中发现RevIN-normalized loss对near-zero variance rows产生$10^3$量级
隐式放大；在任何dataset/test artifact完成前终止。v1.1只把loss改为RevIN重建后的dataset-standardized MSE，
其余冻结合同不变，并从新目录/新checkpoints重跑。该修正属于
`optimization_or_numeric_pathology`，不构成problem result。

D22-C v1.1正式decision=`target_coordinate_information_access_supported`。ordered相对最关键generic control为
test MSE `+2.5228%`、MAE `+1.6484%`，15/20 cells、4/5 datasets、4/4 horizons；validation MSE为
`+2.5410%`，parameter gap为0。相对其余四controls均约`+13.7%–17.5%`且20/20 cells正向。Weather对generic
为`-1.0900%`，完整保留为heterogeneity evidence。

由于CATS/TimePerceiver/MQTransformer/TQNet已覆盖query-to-history retrieval，D22-C arm不能升级method。
Step4-6现冻结`SC-D23-FCMI`：将query context精确拆为trajectory-wide main与zero-mean coordinate interaction，
分别变换后合成；generic与standard query decoder均为exact contained cases。matched dual-branch control用于排除
multi-branch capacity解释。narrative gate conditional pass，只授权local Step7A，不授权remote/test或第二loss。

Step7A production gate现为11/11 pass。memory/query/context/main/interaction/output shapes正确；
$\operatorname{mean}_t\Delta_t$最大`1.82e-7`，FCMI与standard-dual initial morph最大差`6.33e-8`；
generic control不读取interaction，四条关键gradient均finite/nonzero，order shuffle保持value marginal并改变
value-position binding。五个profiles内dual controls参数严格相等。FCMI相对A6 active parameters少83%–95%，
故Step7B必须冻结dense capacity-matched control后才能讨论remote matrix。

Step7B prelaunch现为21/21 pass。dense control用zero-init low-rank temporal residual匹配A6 active parameters；
五个profiles采用rank `234/250/234/241/247`（Weather/ETTm1/ETTh2/ETTh1/ETTm2），实际参数gap仅
`0.0914%–0.1321%`，initial standard-dual function gap不超过`8.94e-8`，两阶段gradient与非零residual均通过。
formal matrix冻结8 arms × 5 datasets × seed2021 = 40 runs；全部arms形成160个official-test cells和
160个validation cells。target-shuffle原拟validation-only，但因其参与方向级attribution，在任何test access前
改为formal control。validation只作four-horizon checkpoint selection与健康诊断，official test才承担
effectiveness与matched attribution。runner在authorization=false时固定exit 3。
decision=`step7b_prelaunch_pass_waiting_remote_test_authorization`；该节点当时remote/test为false。

用户2026-07-20以“按计划继续推进工作”独立授权冻结的seed2021 40-run/160-cell formal matrix。该授权只覆盖
remote training与一次official-test audit；confirmation seeds、paper-method promotion、matrix/profile/gate修改
仍为false。Step8必须先commit/push、remote pull、`nvidia-smi`以及Weather-FCMI和ETTm2-DENSE resource smoke。

Step8已于`2026-07-20T17:57:10+08:00`从commit `4ff439c`在GPU0/1/2启动。preflight显示三卡空闲，
Weather-FCMI与ETTm2-DENSE两项2-batch smoke finite/pass；首批Weather FCMI/DENSE/A6进入训练。
40/40完整前不得pull、改matrix或启动confirmation。launch provenance见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/d23_step8_remote/remote_launch_record.md`。

Step9/10现已完成。FCMI vs A6 test MSE为`-21.7343%`、0/20；validation也为`-16.1434%`、0/20。
decomposition/generic/target controls通过，但order与capacity失败；五dataset internal health全过。DENSE vs
STANDARD_DUAL为`+15.4825%`、19/20，DENSE vs A6仅`-0.3284%`，说明effective
capacity/function-class解释主要差距。validation-fit dense/FCMI、dense-plus-interaction与A6-plus-interaction
三项conditional diagnostics均test反转。A6/DENSE validation-fit blend也从validation `+0.5127%`反转test
`-0.1707%`；固定等权虽test `+1.4680%`，validation却为`-0.4009%`，不能据此选择allocation或router。

机器decision=`fails_a6_internal_valid`；研究decision=
`fcmi_v1_failed_capacity_control_explains_return_step2_3`。FCMI-v1关闭且不补rescue；target-coordinate
problem evidence保留。Implicit Forecaster、BasisFormer、S2TX与attention attribution prior又阻止把
“dense trajectory main + coordinate interaction”本身写成新贡献，故direct Step4 successor不通过，
回同一task的Step2/3。

Step2/3 phase/time-warp probe已关闭：A6/DENSE的一阶derivative相对affine control仅约`+0.03%`，且curvature与
shifted-derivative controls不弱；PULSE、PhaseFormer与prediction-delay work又构成直接prior。SC-D24-CTB因此不做
phase router，而审计另一个更基础的问题：frozen strong carrier是否存在raw-history可识别的48-step coarse future
deformation。协议只访问validation，按forecast origin使用first/middle/last thirds作fit/purge/evaluate，并以
sorted-history和target-shuffled排除marginal/capacity解释。即使通过也只返回Step4，不授权method、training或test。

D24-v1 10/10返回后发现regularization semantics错误：$X^\top X+\lambda I$没有按数万fit rows归一化，导致
conditional maps severe extrapolation。该结果按`design_fault_suspected`处理，不能拒绝problem。v1.1仅修正为
$X^\top X+n\lambda I$，冻结normalized $\lambda=\{0.01,0.1,1\}$，其余matrix与gates不变。

D24-v1.1 10/10 protocol valid且official test access为0。ordered vs marginal为
`-8.5950%/-8.6168%`（A6/DENSE），vs sorted为`-9.4741%/-8.8197%`，vs target-shuffled为
`-14.1002%/-13.4974%`；所有primary horizons均0/4正向。strong shrinkage也未恢复absolute transfer。
Decision=`close_exact_coarse_deformation_probe_return_step2_4_consolidation`。不做D24 feature/bin/lambda/
nonlinear/seed rescue；broader nonlinear direction不被方向拒绝，但没有candidate authorization。

Post-D24 consolidation现完成。现有链条通过scientific problem-boundary gate，但未通过method-paper narrative
gate：它区分了pure requested-H、future-coordinate evidence access与trajectory capacity，却没有给出同时满足
三者的正向paper-core method。故不启动D25，也不把negative/control evidence包装成contributions。

下一步为`SC-MNB Step1-3`：冻结ElasTST、CATS、TimePerceiver、SRSNet及A6_FULL/A6_MEASURE的modern
native-baseline reproduction protocol。ElasTST承担single-weight varied-horizon角色；CATS与TimePerceiver承担
future/target-query角色；SRSNet承担modern selective-patch performance角色。per-H fixed models、single-weight
models与foundation/pretrained models必须分表。当前只允许source/protocol/prelaunch audit，execution、remote
training与official test仍false。若A6广泛落后，后续不得继续在A6 interface上堆叠method。

SC-MNB Step1-3 source audit现完成。official commits固定为ElasTST
`d49f7e41c2db7ac3208816225885b6e3f61c0fb3`、CATS
`58854fc759d608ce400f378be83f4513960e505d`、TimePerceiver
`7e30cc07b51c709f408409fd60a34c81ae8990be`与SRSNet
`6ee35d498f48eefecf84530b362b137de38e6592`。CATS/TimePerceiver training loop的per-epoch test access、
CATS ETTm2-H96 typo、SRSNet file-level license trace/metric equivalence与ElasTST 10-batch semantics均是
launch blockers。
planned matrix为65 external runs/80 cells，但local protocol patch、remote training与official test继续false。

2026-07-21用户将paper strategy重置为SIFF-first：`SC1-SIFF-v2-EQ-ATTR-v1`保留为immutable performance-near
parent，其历史failure不变；SC-MNB不再是active performance gate，只作为source/control inventory。新candidate
`SC1-SIFF-v3-TSAF-v1`以Target-Scale Allocation Field替换history-conditioned generic router：future-coordinate
与ordered log-scale共同产生sample-shared allocation，SIFF arms本身仍依赖history。该设计不输入requested H，
不增加第二loss，也不恢复CCSF、D17-D21或parameter rescue。

Step4-6 narrative/design gate现为conditional pass。最新primary-source audit将claim收紧到完整链
`request invariance -> target-specific scale demand -> shared scale-indexed output-coupling field -> target-scale
allocation`，不claim首次multi-scale、future query、MoE、learned basis或decoder-side forecasting。Step7A现已
production-local 26/26通过，覆盖history/request invariance、scale semantics、参数公式、gradient与TimeAlign
constructor。Step7B现冻结9 effective arms/45 runs/180 test cells，其中4个historical arms共20 runs经checkpoint
hash复核后复用，5个new arms共25 runs必须from-scratch joint training。旧direct-policy independent没有复用；
target-only independent ranks重新按TSAF active parameters匹配，最大gap 0.3619%。prelaunch为15/15 cases、10/10
categories，25/25 CLI、5/5 two-step gradients、paired initialization、runner refusal与analyzer synthetic smoke均
通过。prelaunch时只读remote preflight显示3 GPUs idle、20/20 reference hashes一致；尚未pull、resource-batch smoke、
training或test，且当时需等待独立remote/test authorization。confirmation和SC-MNB execution均false。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v3_tsaf_step7b_prelaunch_report.md`。

2026-07-21用户已独立授权冻结的25-run seed2021 remote training，以及25/25 training完整后的一次45-run/180-cell
formal test。generic evaluator现识别`effective_arms`，config补齐equal-skill training contract、coupling scales与
future bins；runner将training与`FORMAL_TEST_ONLY`严格分离。授权记录时等待commit/push、remote pull与两项resource smoke；
confirmation、paper promotion和post-hoc matrix/gate修改仍false。

commit `6cef063`已remote pull，两项resource smoke finite/no-OOM；25-run training于10:17:06在GPU0/1/2启动。
首批Weather candidate/categorical/permuted进入epoch 1。该launch snapshot为training 0/25、formal test 0/25；训练期间不再pull，
25/25 completeness前不得启动formal-test mode。

Step9/10现已完成。25/25 new training、25/25 new formal test与45/45 effective audit完整；45个checkpoint hashes
unique，逐dataset encoder initialization一致，checkpoint nonmutation 25/25通过。TSAF相对A6_MEASURE test
MSE/MAE为`-1.2854%/-1.3146%`，相对SIFF-v2 parent为`-1.0422%/-0.9183%`，两项均0/4 horizon
wins，paper-facing effectiveness fail。

matched attribution同样全部失败：ordered field vs categorical `-1.0191%`，ordered scale vs permuted
`-0.0796%`，target coordinate vs global `-0.0405%`，shared field vs independent `-1.2785%`。internal health
全过说明实现路径活跃且不是numeric pathology，但不能覆盖negative effectiveness。validation中TSAF相对parent
`+0.7700%`，test反转为`-1.0422%`，禁止换selector或按test重选epoch。

independent target-only相对parent的MSE `+0.2383%`只保留为single-seed weak lead：低于0.3% primary threshold，
且其预注册角色是control，不能post-hoc晋升。Decision=
`close_tsaf_v1_shared_field_design_keep_siff_v2_immutable_parent`。TSAF-v1不补seed/rank/width/readout/loss rescue；
SIFF-v2继续immutable，当前没有active successor method，回SIFF-first Step2/4。

post-TSAF Step2现用四个existing E2E arms完成field-family × policy-information的$2\times2$ audit。
`independent target-only`同时把Q2 ordered field改为Q5 independent fields、把direct policy改为static-target，
并在ETTh2/ETTm1/Weather把rank从116改为115，因此其`+0.2383%`不是单因素effect。全20-cell test interaction
MSE/MAE为`+0.5265%/+0.4246%`，但严格同rank的ETTh1+ETTm2子集为`-0.3097%/-0.1175%`；validation
same-rank近零，Weather split reversal。latest primary-source audit还显示independent experts、query-specific
selection、multi-scale gating与shared-plus-residual gating均已有强覆盖。Decision=
`independent_target_only_weak_lead_not_supported_for_step4`。

该决定只关闭weak control的post-hoc promotion，不关闭SIFF-v2。下一节点是immutable SIFF-v2 final paper-claim
consolidation：允许陈述其相对A6_FULL、PCSD_EQUAL、constant、permuted与Q1-wide的正证据；不得陈述ordered
strictly superior to independent、target-only allocation成立或已超过A6_MEASURE。claim gate完成前不实现successor、
不执行SC-MNB performance matrix、不启动remote training，也不预设第二loss/router。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_post_tsaf_independent_factorial_audit.md`。

SIFF-v2 final paper-claim Step4-6现完成。narrative gate为
`conditional_pass_as_single_architecture_contribution`：唯一method contribution收紧为“把future-output coupling
extent作为ordered log-scale coordinate，以共享history-conditioned components生成nested full-domain forecast
operators并target-wise融合”。不claim首次multi-scale、MoE、future query或requested-H conditioning；equal-skill
只作共同训练contract，不包装为第二loss contribution。

该narrative pass不覆盖effectiveness blockers。用户于2026-07-21指定FCC以`A6_FULL`替代`A6_MEASURE`，现固定
`SIFF_EQUAL/A6_FULL/SIFF_INDEPENDENT_EQUAL`三臂，在五datasets补seeds2022/2023共30 new runs，复用
seed2021形成45 effective runs/180 test cells。`SIFF vs A6_FULL`是architecture与objective共同改变的
method-package comparison；ordered-field attribution只来自same-objective independent control。历史
`A6_MEASURE` negative保留，但不进入FCC metrics或gate。

两项SIFF comparisons都沿用MSE `+0.3%`、MAE为正、3/5 datasets、3/4 horizons与至少2/3 seed macro为正的
gates。只有同时通过才进入modern baselines；任一失败即停止SIFF paper-core rescue，不回rank、loss、router或
readout tuning。local prelaunch现25/25通过，30/30 jobs与15/15 historical references均通过审计；remote
training及30/30 training后的single formal test已获授权但尚未启动。Decision=
`step7b_prelaunch_pass_proceed_commit_remote_preflight`。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v2_fcc_v1_prelaunch/prelaunch_report.md`。

Step8已从commit `87bea35`启动：remote fast-forward、三张3090 preflight及Weather-SIFF/ETTm2-independent
resource smokes均通过。30-run training于`2026-07-21T12:54:37+08:00`开始，首批三个Weather jobs active；
formal test为0/30，runner在30/30 training前拒绝test mode。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v2_fcc_v1_prelaunch/remote_launch_record.md`。

FCC Step9/10现完整结束。SIFF相对A6_FULL的test MSE/MAE为`+1.2497%/+0.7549%`，5/5 datasets、4/4
horizons、3/3 seeds正向，package effectiveness通过；相对independent control却为`-0.1272%/-0.1733%`，
validation也为`-0.3224%/-0.5015%`。45/45 protocol、unique hashes、paired initialization、checkpoint
nonmutation及internal health均通过，因此不是design/numeric fault。Decision=
`performance_pass_attribution_blocked_stop_fcc_promotion`，failure attribution=`capacity_control_explains`。
SIFF-v2不晋升paper core，不补任何rescue，不执行modern baselines/formal ablations；当前回paper portfolio decision，
`active_method=none`。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v2_fcc_v1/step9_10_result_and_portfolio_decision.md`。

用户2026-07-20决定暂不承担task pivot成本。该scope决定把上一版“D22-C有效失败即停止整个
deterministic-MSE search”改为：关闭exact D22-C v1并回joint Step2/3，在同一task边界寻找不同的falsifiable
problem；不允许D22-C seed/width/readout/representation rescue，也不恢复D17-D21。

本节以下的post-CCSF、D17-D21与SIFF记录均为chronological history；新会话不得从其中恢复已关闭候选或重新施加
已被顶部unconstrained reset撤销的hard constraints。

## post-CCSF Step 2/4 Reset

exact projectivity给出一个直接约束：对任意$H\leq K$，若$F_H(x)=P_HF_K(x)$，则整个family必可写为
$F_H(x)=P_HF_T(x)$。因此requested horizon不能改变共享prefix；“exact-projective且$H$-adaptive prediction”
不再作为论文问题。ElasTST已覆盖varied-horizon invariance与horizon reweighting，ProNet与Implicit Forecaster又
覆盖output dependency和global wave decoding，普通multi-scale/future-dependency表述不足以形成Contribution 1。

prefix-safe future-context D17-v1已完成。正式validation→test结果为causal vs pointwise `-3.0356%`、causal vs
shuffled `-2.3616%`；pointwise本身相对parent `-28.7314%`，并有>100%局部退化。故exact post-hoc protocol关闭，
future-context保持direction unresolved但不进入Step4。

当前转向`SC-D18-SPC` problem diagnostic：使用相同full-domain A6 architecture，分别以H96/H192/H336 prefix loss
训练horizon-specific oracle controls，并与强A6_MEASURE比较own-H test performance。D18只回答exact projectivity
是否存在accuracy cost；separate models不是贡献。machine-readable protocol与prelaunch已`11/11`通过，
15个shape/projectivity/gradient checks全部通过；现只授权15个new training runs与完整test-informed problem audit，
不授权method implementation。Contribution 2继续停在Step2。

详见`analysis/stage_c_post_ccsf_step24_reset_20260719/d17_result_and_failure_attribution.md`与
`analysis/stage_c_post_ccsf_step24_reset_20260719/soft_projectivity_step2_problem_audit.md`。

Step3与代码契约见
`analysis/stage_c_post_ccsf_step24_reset_20260719/d18_prelaunch/step3_prelaunch_report.md`。

remote preflight与Weather SPEC96 resource smoke通过后，D18于`2026-07-19T14:31:07+08:00`在GPU0/1/2启动。
运行中保持commit`c843178`、config/gates与25-unit matrix不变；完成后先执行完整problem gate，不查看或挑选局部
有利cells。

## D18 Step 9–10 Result And Step 2 Rollback

D18现已25/25 artifact units完整。specialists相对`A6_MEASURE`仅`+0.1659%` MSE、`7/15` cells、
`2/5` datasets与`1/3` horizons为正；七项预注册categories仅通过prediction deformation与protocol/numeric两项。
相反，`A6_MEASURE`相对`A6_FULL`为`+1.7980%`且15/15 cells正向，说明表面specialization收益主要由统一
measure training解释，而不是strict projectivity的accuracy cost。

validation也只为`+0.4205%`、7/15 cells，且H336为`-0.7301%`；15个specialists没有best epoch卡训练预算。
因此失败不是test-only reversal、arms collapse或明显optimization pathology。decision=
`measure_training_explains_close_soft_architecture_route_return_step2`。

soft projectivity、consistency-$\lambda$ sweep与requested-H feature全部关闭。下一步先以最新Implicit Forecaster作为
`SC-D19-IFC control_only`完成source/code/theory audit：比较A6 learned basis与implicit frequency/amplitude/phase
trajectory generation是否还有真实headroom。IF本身已有NeurIPS 2025直接prior，不是本项目method。harmonic
measure weighting也继续只作mandatory control。

D19 Step4/5现已完成。official implementation确认所有pred_len都先用`spectrum_size=720`经iFFT生成full
trajectory，再crop loss prefix；因此synthesis天然满足当前projectivity contract，但upstream仍按horizon分别训练。
IF相对A6同时引入nonlinear polar synthesis与raw input spectrum skip，所以Step6必须冻结`A6_MEASURE`、
`IF_MEASURE`、`IF_NOSKIP_MEASURE`与`DIRECT_NONLINEAR_MATCHED_MEASURE`四arm，不能只做IF-vs-A6。
Step6现已冻结five datasets、seed2021、four-horizon validation selector、15个新训练run + 5个复用A6，以及
IF/direct逐profile参数差不超过0.1%的matched control；静态gate为9/9。Decision=
`step6_pass_step7a_local_only`；下一步只实现production heads与shape/projectivity/gradient/parameter/CLI
local gate，remote/test/paper method继续false。

Step7A code-theory audit发现v1错误继承upstream 96-point lookback，而A6 natural真实`seq_len=720`。由于尚未
发生D19 training/test，v1被审计性保留并由v1.1修复：Encoder、IF skip与matched direct均读取同一720-point
history，history rFFT为361 bins，direct重新匹配后parameter gap为0.0036%–0.0098%。

v1.1 production implementation现已114/114通过：15 CLI、60 shape/projectivity、12 parameter、
10 gradient、paired initialization、numeric、source-reference与model wiring全部pass；maximum prefix gap=0。
Step7B进一步以31/31冻结15个new runs、5个复用A6、80个official-test cells、四层analyzer、checkpoint
non-mutation与D19 amplitude/phase diagnostics。Decision=`step7b_prelaunch_pass_step8_launch_next`；
只授权seed2021一次冻结的control audit，confirmation与paper-method promotion仍false。

Step8已于`2026-07-19T18:52:43+08:00`以commit `da011c8`在GPU0/1/2后台启动。Weather IF与ETTm2
matched-direct双resource smoke均finite、无OOM；首批Weather IF、Weather IF-no-skip与ETTm1 IF已进入训练。
Step9现已20/20 units、80/80 test cells完整。IF相对A6为`-3.6117%` MSE、`-3.6519%` MAE、
3/20 MSE cells；相对matched direct为`-0.8075%` MSE；只有history-spectrum skip相对no-skip为
`+1.6191%` MSE、16/20 cells。internal health全部通过，故不是collapse或numeric pathology。

Validation上IF相对A6也为`-10.9950%`、0/20 cells，排除test-only reversal；但D19 total params为A6的
7.94×–10.29×，且12/15 new arms在epoch1达到best checkpoint，故failure attribution收紧为
`readout_or_head_design_wrong + readout-scale/optimization mismatch suspected`，不能方向级否定所有structured
decoder。D19 control关闭，不补seeds、不做width/LR sweep；回Step2/4评估compact generation与direct history
statistic是否能形成新的paper problem。

详见`analysis/stage_c_post_ccsf_step24_reset_20260719/d18_step9/d18_step9_four_layer_diagnostic.md`与
`analysis/stage_c_post_ccsf_step24_reset_20260719/d19_step9/d19_step9_deep_audit.md`。

## Post-D19 Contribution 1 Step 2/4 Viability Audit

external primary-source refresh确认FITS、FBM、Implicit Forecaster、PhaseFormer、BasisFormer、FlowState、N-HiTS与
TimePerceiver已经覆盖compact frequency interpolation、time-frequency basis、amplitude/phase synthesis、phase
routing、functional basis与target-coordinate decoding等邻近链条；ICML 2024 linear-model analysis进一步限制了
“Fourier coordinates本身构成新function class”的说法。因此smaller IF与history-phase-continued atoms均未通过
method narrative gate，不能用D19 negative后的width/LR rescue启动新candidate。

D19只留下一个尚未验证的问题：720-history spectrum在IF内相对no-skip为正，能否transfer到strong A6 coefficient
operator，并超过同维fixed random orthogonal history projection。下一步冻结为`SC-D20-CST diagnostic_only`：
`A6_MEASURE_RETRAIN/A6_CST_SPEC/A6_CST_RANDOM`共同from-scratch E2E训练，保持full-T synthesis、prefix crop、
five datasets、four-horizon validation selector与official-test完整矩阵。SPEC必须同时通过vs A6 transfer与vs RANDOM
specificity，才能只返回Step4设计native non-residual operator；generic concat head不得因结果正向而升级为paper method。

Decision=`step2_4_complete_d20_diagnostic_step6_next`。当时只授权Step6 exact dimension/normalization/bin/init/matrix
freeze，不授权implementation、remote、test access或Contribution claim。详见
`analysis/stage_c_post_ccsf_step24_reset_20260719/post_d19_step24_compact_statistic_viability_audit.md`与
`Papers/post-d19-compact-statistic-decoder-audit.md`。

D20 Step6现已冻结：normalized history经fixed orthonormal projection得到`[B,C,64]` summary；SPEC使用
32组non-DC real Fourier cos/sin modes，RANDOM使用seed20260719 Gaussian QR subspace。两者以相同
`Linear(R+64,256)`进入A6 learned-basis coefficient operator，并通过zero-init summary columns与paired
Encoder/basis/base-head保持三臂初始function完全相同。所有arms仍先生成full 720 trajectory，再做prefix crop。

matrix固定为`A6_MEASURE_RETRAIN/A6_CST_SPEC/A6_CST_RANDOM` × five datasets × seed2021，共15个from-scratch
runs与60个official-test cells；SPEC必须分别对A6与RANDOM通过相同`0.3% + 11/20 + 3/5 datasets + 3/4
horizons + nonnegative MAE` gates。static checker为14/14，projection orthogonality误差不超过`3.763e-15`、
initial/prefix gap为0、summary path gradient与deformation均非零。

Decision=`step6_pass_step7a_local_only`。下一步只实现production buffers/readout/paired initialization/CLI与local
shape-projectivity-gradient-hash tests；remote、official test、confirmation与paper-method仍false。详见
`analysis/stage_c_post_ccsf_step24_reset_20260719/d20_step6/step6_diagnostic_design.md`。

D20 Step7A production gate现为`9/9`：15个CLI cases、15个constructors、60个shape/prefix cases与10个
summary-gradient cases全部通过，initial output gap与prefix gap均为0；SPEC/RANDOM各增加16,384参数，且paired
Encoder/basis/base-head hash逐dataset一致。Step7B prelaunch进一步为`10/10`，冻结15个from-scratch runs、60个
official-test cells、validation four-horizon mean-MSE checkpoint selector、checkpoint non-mutation与四层analyzer。

Decision=`step7b_prelaunch_pass_step8_authorized`。当前允许seed2021一次完整remote/test diagnostic；confirmation与
paper-method promotion继续false。即使结果双门槛正向，也只能返回Step4设计native non-residual operator。

Step8已于`2026-07-20T11:55:45+08:00`以commit`9573cd7`在GPU0/1/2后台启动。Weather-SPEC与ETTm2-RANDOM
resource smoke均finite且无OOM；首批Weather-SPEC、Weather-RANDOM与ETTm1-SPEC已进入epoch 1。当前不值守，
完成后进入Step9完整15-unit/60-cell四层审计。

## D20 Step 9–10 Result And Step 2/4 Rollback

D20现已15/15 runs、60/60 official-test cells完整。SPEC相对A6为`-0.7614%` MSE、`-0.5831%` MAE，
8/20 MSE cells、2/5 datasets与0/4 horizons；transfer gate明确失败。SPEC相对RANDOM为`+0.1412%` MSE、
14/20 cells、4/5 datasets、3/4 horizons，属于weak directional specificity但低于冻结0.3% gate。

11项internal health全部通过；SPEC prediction contribution为RANDOM的1.78–4.44×，不是collapse。validation上SPEC
相对A6为`+0.5755%`，到test发生sign reversal；SPEC-vs-RANDOM也从`+0.7288%`衰减到`+0.1412%`。因此formal
decision保留`compact_spectrum_transfer_failed`，但failure attribution收紧为
`validation_test_mismatch + intervention_point_wrong`，不得方向级解释为history spectrum无价值。

exact D20关闭，不补seeds、不做scalar gate/width/LR sweep；Contribution 1回Step2/4。最新external refresh表明
generic spectral robustness已被Frequency Matters、DropoutTS、Fremen与AEA覆盖。只保留future-distance
predictive-support作为provisional problem family，先执行D20-D1 within-model contribution direction/scale oracle，
不授权新training或method。

D20-D1现已完成：SPEC contribution相对其co-adapted base为`+26.8928%`、39/40 bins有益、median oracle
alpha=`1.2649`；RANDOM也为`+9.0422%`、35/40。scalar shrink/normalization rescue被否定；新增path的重要性主要
反映joint responsibility relocation，而非对A6的独立增量。D1也没有支持future-distance envelope，因为SPEC path
几乎所有bins内部有益。Decision=`d1_complete_scalar_fix_rejected_coadaptation_explains_return_step2_3`。

下一步不实现operator，而是先设计跨机制、validation-fit→official-test的problem diagnostic，检验support是否
past-identifiable且split-stable；失败则关闭该provisional family。

[D21-EVS Step2/3] 宽泛`future-distance support`经external source audit收紧为Evidence-Validity Surface：route
relative risk必须包含past × future-region interaction，而不是region-only或sample-only主效应。D14-A两carrier三seed
crossing/oracle作为headroom；D20/D1不作正向existence evidence。validation拟合centered log-risk，official test只
评估transfer；global/region/history-global/additive/permuted/oracle controls全部冻结。Step7A descriptor为192维，
synthetic policy recovery与static invariants通过。decision=`d21_evs_step7a_pass_remote_checkpoint_evaluation_next`；
只授权读取seed2021 frozen D14 checkpoints，不授权new forecasting training、method或confirmation。

[D21-EVS Step9/10] 100/100 exports与完整official-test gate完成。Oracle相对region仍有neutral `+7.6399%`、A6
`+10.4053%` headroom，但最强neutral HGB interaction相对additive仅`+0.0347%`，低于冻结`0.1%`；A6为
`-0.0069%`。validation chronological forward HGB曾为`+0.3092%/+0.4406%`，到test缩小或反转，证明paper定义所需
split stability不成立。Permuted controls表明past signal存在，但additive sample+region解释大部分realized gain。
Decision=`close_exact_evs_problem_split_stability_failed_return_step2`；不补seed、不做representation/readout rescue，
Contributions 1/2共同回Step2。

## SIFF_EQUAL Attribution Step 6 Freeze

`SIFF_EQUAL` 已从“性能正向但归因阻塞”推进为 `SC1-SIFF-v2-EQ-ATTR-v1`：

1. 10-arm matrix同时包含`A6_FULL/A6_MEASURE`、`PCSD_MEASURE/PCSD_EQUAL`、
   `SIFF_MEASURE/SIFF_EQUAL`和四个EQUAL-context SIFF controls；
2. Phase A固定为5 datasets × 10 arms × seed2021，共50 runs与200个standard-horizon test cells；
3. `SIFF_EQUAL`必须逐项超过A6_FULL、A6_MEASURE、PCSD_EQUAL、constant、permuted、Q1-wide与independent；
4. 统一结论分为paper-facing effectiveness、matched mechanism attribution、internal mechanism health与failure
   attribution四层，内部健康度不得挽救negative performance gate；
5. Step6 16/16与Step7A 13/13 categories已通过；下一步只做Step7B prelaunch，remote/test仍未授权。

若`SIFF_MEASURE`未超过`PCSD_MEASURE`但EQUAL comparison通过，claim只能收紧为equal-skill-trained scale
field。只有Phase A七项hard comparisons全部通过，才允许seeds2022/2023 confirmation。

详见`analysis/stage_c_siff_equal_attribution_step6_20260718/step6_attribution_protocol.md`。

Step7A现已完成50-job CLI wiring、35个unique model constructors、10条objective gradient paths、matched
parameter accounting与scale-component intervention artifact。新增component统计固定policy并逐个移除
scale-field component，保存raw-scale `full - ablated`；它是non-additive diagnostic，不改变production forward。
checkpoint evaluator与四层analyzer smoke均通过，remote runner在authorization=false时固定拒绝launch。

Decision=`step7a_local_pass / step7b_prelaunch_next`。详见
`analysis/stage_c_siff_equal_attribution_step7a_20260718/step7a_implementation_gate_report.md`。

Step7B prelaunch进一步以9/9 categories冻结正式授权：50 runs/200 test cells、four-horizon validation
checkpoint、official test primary gate、single formal access与confirmation hold。3090三张GPU预检均约15 MiB
used、无训练进程；远程历史dirty CSV与本次路径不重叠并必须保留。下一步为pull、dry-run、resource smoke与后台
launch。详见`analysis/stage_c_siff_equal_attribution_step7b_prelaunch_20260718/prelaunch_report.md`。

Step8已于2026-07-18 11:12:03在3090 GPU0/1/2启动。remote dry-run与Weather-SIFF_EQUAL resource smoke先行
通过，首批Weather的A6_FULL/A6_MEASURE/PCSD_MEASURE正常进入epoch 3/3/1。当前不高频值守；50/50完成后同步
完整test metrics、invariants与component artifacts，再执行四层Step9。详见
`analysis/stage_c_siff_equal_attribution_step8_remote_20260718/remote_launch_record.md`。

## SIFF_EQUAL Attribution Step 9 Result

Phase A已50/50完成，200/200 standard-horizon test cells、50/50 invariants与paired initialization均通过。
七项hard comparisons中：

1. SIFF_EQUAL超过A6_FULL `+1.6436%`与PCSD_EQUAL `+0.5906%`，但低于A6_MEASURE `-0.2366%`；
2. ordered超过constant `+0.9393%`、permuted `+0.3959%`、Q1-wide `+1.1619%`，但相对independent仅
   `+0.2580%`，未达冻结`0.3%`；
3. internal health 7/7，通过finite、projectivity、oracle、diversity、entropy与component-use gates。

结论不是mechanism未执行，而是simple objective与independent-scope controls解释了paper-facing gain。exact v1关闭，
confirmation保持false，回Step4重新研究conditional headroom到learned fusion的转化问题。详见
`analysis/stage_c_siff_equal_attribution_step9_20260718/step9_four_layer_diagnostic.md`。

## Post-Step9 Candidate Freeze And Step4 Improvement Audit

用户决定把`SC1-SIFF-v2-EQ-ATTR-v1`保留为本阶段最接近论文级performance的候选。该决定只改变portfolio
status，不推翻Step9 attribution failure：v1固定为`frozen_performance_near_candidate / performance_partial_pass`，
其source commit、config/profile hash与五个checkpoint hashes记录于
`configs/stage_c_siff_equal_attribution_v1_candidate_freeze.json`。

复用现有artifacts的diagnostic表明，SIFF_EQUAL policy best-arm match为29.24%、skill alignment为0.0277，
policy-weighted expected arm loss相对uniform仅+0.0762%；two-fold static convex fusion却比learned fusion高
2.2112%且8/10 dataset-fold为正。bounded affine相对convex只多0.1203%，因此首要问题是router看不到/学不到
relative arm competence，而不是softmax convex hull太窄。

决策：保留v1作为当前candidate与mandatory parent reference；研究正式进入Step4 source-informed redesign，
优先将`arm-contrast-aware policy + synchronous competence calibration`送入Step5 theory feasibility，并同时审计
`A6_MEASURE anchor containment`。generic deeper MLP、top-k、entropy loss、直接扩Q/rank/scales不进入下一步。
详见`analysis/stage_c_siff_candidate_step4_source_audit_20260718/source_informed_improvement_audit.md`。

## CCSF Step 5 Theory Feasibility

代码审计确认旧PCC已实现same-forward detached arm-error route supervision，因此Step4中“同步calibration”不能作为
未测试创新。CCSF核心收紧为target-free arm-contrast information path；calibration只作为co-designed weak
supervision，并必须设置loss-only control。

预冻结contrast diagnostic使用5 datasets × 2 row folds。contrast相对coordinate-only expected arm MSE
`+1.8348%`、10/10 folds正；相对shuffled `+1.7085%`、10/10；best-arm accuracy相对existing policy提高
15.31 percentage points。5/5 gates通过，支持contrast identifiability，但仍是test-derived offline evidence。

Step5冻结的provisional operator保留v1 logits，并用scope-shared scorer读取dimensionless pointwise/scope-group
contrast；correction为零时包含v1。full T computation + prefix crop给出strict projectivity。旧PCC teacher除以
cross-arm std，会放大near-equal arms的噪声；provisional relative-regret teacher改按mean error归一化，并用
`1-normalized entropy`降低ambiguous supervision。temperature grid只作geometry evidence，禁止从test选择。

显式A6 anchor branch因capacity/ensemble confound退出method，A6_MEASURE保留为mandatory external baseline。
Decision=`conditional_theory_pass_to_step6`；Step6必须冻结v1、loss-only、architecture-only、full、shuffled/zero、
generic capacity、independent与A6 controls，当前implementation/remote均false。详见
`analysis/stage_c_siff_ccsf_step5_theory_20260718/step5_theory_feasibility.md`。

## CCSF Step 6 Narrative And Control Gate

候选版本冻结为`SC1-SIFF-v2-CCSF-v1-preimplementation`，并通过5/5 static gates。method保留v1 logits，使用
scope-shared六维target-free contrast descriptor产生零初始化logit correction；完整T=720计算后才crop，requested
horizon与benchmark bins不进入model。新增scorer为2,881参数，ordered/independent CCSF在五dataset上的总参数gap
均低于0.5%。

归因矩阵以`SIFF-v1/CCSF × EQUAL/RELCAL` 2×2为核心，另含A6_MEASURE、old standardized teacher、
zero-contrast same-capacity、permuted-contrast与matched-independent controls，共10 arms。temperature只允许从
`{0.05,0.1,0.25}`通过五dataset共同validation macro score选择，不允许per-dataset或test选择；该15-run pilot尚未
授权。正式Phase A冻结为50 runs/200 cells，confirmation为100 runs/400 cells，但remote/test/confirmation均false。

Decision=`step6_pass_step7a_local_only`。下一步只实现production forward/objective/control adapters、prefix/gradient/
parameter tests与remote refusal gate；10项hard comparisons全部通过前不得形成joint claim。详见
`analysis/stage_c_siff_ccsf_step6_20260718/step6_narrative_control_gate.md`。

## CCSF Step 7A Local Implementation

production path已实现`arms [B,C,S,T] -> contrast [B,C,T,S,6] -> shared 43-64-1 correction -> v1 logits +
correction -> projective fusion`。新增参数严格2,881；ordered/independent总参数gap最大0.3833%。相同seed的v1与
三个ordered CCSF controls具有相同base hash和initial forecast，gap为0。true/zero/permuted descriptors及nonzero
intervention、contrast-to-arm gradient均通过，四readout的prefix gap均为0。

relative/standardized calibration objectives、50-job adapters、30 constructors、10 gradient paths、two-step correction
optimization与diagnostic tensor contract全部就绪。local gate=18/18。`tau=0.1`只作synthetic smoke；正式shared
temperature没有选择。remote template在当前authorization下exit 3，即使手工改authorization也因Step7B未冻结而
exit 4。

Decision=`step7a_local_pass_step7b_next`；下一步先设计Step7B validation pilot/prelaunch boundary，不得直接启动
pilot、remote或test。详见
`analysis/stage_c_siff_ccsf_step7a_20260718/step7a_implementation_gate_report.md`。

## CCSF Step 7B Temperature-Pilot Prelaunch

Step7B把temperature selection与formal effectiveness严格拆开。pilot固定为唯一`ccsf_relcal` arm、5 datasets、
`{0.05,0.1,0.25}`和seed2021，共15 runs/60 validation cells。checkpoint仍由H96/H192/H336/H720 validation MSE
平均选择；shared temperature则由完整5×4 validation macro MSE选择，并列取更大temperature。禁止per-dataset、
per-horizon或test-informed selection。

本地prelaunch gate=14/14；runner dry-run=15 jobs，synthetic tie选择器通过。pilot weights/checkpoints不会进入formal
comparison；实际pilot选定temperature后，必须生成新的formal candidate version并重新审核50-run Phase A的runner、
evaluator、internal artifacts与test metadata。当前只授权validation pilot remote，formal Phase A、official test与
confirmation仍为false。Decision=`step7b_temperature_pilot_prelaunch_pass`。详见
`analysis/stage_c_siff_ccsf_step7b_prelaunch_20260718/prelaunch_report.md`。

## CCSF Step 8 Validation Pilot Launch

commit`06d0ffc`已同步至3090远端；dry-run 15/15与Weather/tau0.1 resource smoke通过。完整pilot于
2026-07-18 15:31:11启动，GPU0/1/2并行执行三个Weather temperatures，driver PID=`654232`。输出固定在
`/home/yingch/exp_outputs/r-2026-fatst/stage_c_siff_ccsf_temperature_pilot_v1`。

当前decision=`step8_validation_temperature_pilot_running`。不值守、不访问test；完成后先做15-run/60-cell
completeness与selection artifact audit，再冻结formal candidate identity。详见
`analysis/stage_c_siff_ccsf_temperature_pilot_step8_remote_20260718/remote_launch_record.md`。

### Runtime failure correction

首次driver于15:41退出，但completion audit为0/15。三个Weather temperatures均从epoch1开始NaN；direction rejection
无效。定向复现证明旧group RMS在exact zero contrast处forward finite、backward有7200个NaN gradients。加入
`contrast_epsilon`后zero-contrast gate与三个temperature的三步optimization共3/3 categories通过；Step7B重新检查
15/15 categories通过。resource smoke收紧为三train batches，retry写入独立external root。

Decision=`runtime_repair_local_pass_remote_smoke_next`；formal Phase A/test/confirmation仍为false。详见
`analysis/stage_c_siff_ccsf_runtime_repair_20260718/runtime_failure_and_repair_report.md`。

三batch真实Weather smoke随后通过：3-step train loss 1.785318、validation MSE 1.221736，checkpoint/metrics存在且
无nan/inf。commit`7045c80`的retry1于15:54:21在GPU0/1/2启动15-run matrix。当前decision=
`step8_repaired_validation_pilot_running`；用户通知完成前不轮询。详见
`analysis/stage_c_siff_ccsf_temperature_pilot_retry1_step8_remote_20260718/remote_relaunch_record.md`。

## CCSF Pilot Result And Formal Candidate Freeze

retry1已完成15/15 runs与60/60 validation cells，9/9 protocol/result categories通过。tau0.25的macro MSE为
0.568165，相对tau0.1与tau0.05分别+0.1415%/+0.2991%；cell/dataset/horizon wins为17/20、4/5、4/4。
ETTm1中长horizon偏好更低tau，但per-dataset selection被预注册规则禁止。

formal candidate固定为`SC1-SIFF-v2-CCSF-v1-tau25`，所有10 arms从头训练，不复用pilot checkpoint。当前
decision=`freeze_tau025_formal_candidate_prelaunch_next`；下一步实现50-run official-test Phase-A runner/evaluator/
internal artifacts/four-layer analyzer并通过新的prelaunch gate，remote/test仍未授权。详见
`analysis/stage_c_siff_ccsf_temperature_pilot_retry1_result_20260718/pilot_result_and_candidate_freeze.md`。

## CCSF tau0.25 Formal Phase-A Prelaunch

formal Step7B已实现并通过15/15 categories。matrix固定为10 arms × 5 datasets × seed2021 = 50 runs与
200 official-test cells；训练只以validation四horizon mean MSE选择checkpoint，之后单次test evaluator读取完整
H96/192/336/720 scorecard并验证checkpoint hash不变。

新增internal artifacts保存final/base policy、base/correction logits与六维contrast descriptor；four-layer analyzer
同时执行Step6的10项hard comparisons、architecture-objective interaction、internal health与failure attribution。
所有50条CLI、contract hashes、runtime repair、三batch smoke合同与test authorization均通过。

Decision=`formal_phase_a_prelaunch_pass_remote_launch_next`。Phase A/test已授权，confirmation seeds仍为false；
下一步必须commit/push、remote pull、`nvidia-smi`与Weather CCSF_RELCAL三batch smoke后再启动正式矩阵。详见
`analysis/stage_c_siff_ccsf_v1_tau25_phase_a_20260718/formal_phase_a_prelaunch_report.md`。

Step8已于`2026-07-18T17:27:08+08:00`从commit`604e1b8`在GPU0/1/2启动。remote 15/15 gate与Weather
CCSF_RELCAL三batch smoke先行通过，首批三个Weather jobs均已进入训练。运行期间不值守、不改协议；50/50返回后
先审计200/200 test cells、实际test dates、checkpoint non-mutation与CCSF internal artifacts，再执行four-layer
Step9。详见`analysis/stage_c_siff_ccsf_v1_tau25_phase_a_20260718/remote_launch_record.md`。

## CCSF Formal Result And D2–D4 Closure

formal Phase-A已完成50/50 runs与200/200 test cells。full CCSF相对A6_MEASURE为`-0.8567%`，相对SIFF-v1
EQUAL为`-0.6159%`；architecture、objective、zero-contrast与ordered-field specificity均失败。internal arms未
collapse且oracle为5.30%–12.50%，但learned allocation相对uniform五dataset全部为负。

D2证明region aggregation改善individual-arm competence identifiability，却没有满足相对pointwise的mixture-margin
gate；D3证明convex-mixture cross terms相对best-arm只贡献约`1.34%–1.38%`，不足以支持covariance-aware主线；
D4证明sharpening与hard routing不能修复soft policy。故exact CCSF与其region/covariance/temperature修补全部关闭，
confirmation不启动。

Decision=`close_contrast_policy_family_return_step2_4`。SIFF-v2-EQ-ATTR-v1继续冻结为performance-near parent，
但当前无active child method、无training authorization。下一步必须从fixed-past unified multi-horizon generation的
decoder contract重新做Step2/4 external-source与历史failure-boundary审计，不得转入router auxiliary sweep。完整
报告见`analysis/stage_c_siff_ccsf_v1_tau25_phase_a_20260718/step9_four_layer_and_redesign_audit.md`。

## Fair Re-audit Step 9–10 Result

70/70 runs与280/280 test cells通过protocol。结论发生三层分化：

1. PCSD_DIRECT相对A6为`-0.8562%`，exact direct candidate继续关闭；内部诊断仍保留representation与
   credit-starvation线索；
2. PCC相对prior仅`+0.0806%`，且在SIFF上相对EQUAL为`-0.2663%`，exact PCC关闭；
3. SIFF_EQUAL相对PCSD_EQUAL为`+0.5906%`并通过预注册gate；相对A6为`+1.6436%` MSE、
   `+0.9084%` MAE，是当前最佳carrier；
4. SIFF+PCC相对A6虽为`+1.3812%`并通过performance gate，但PCC降低SIFF_EQUAL性能，不能形成双贡献归因。

旧best-H720下SIFF_EQUAL vs PCSD_EQUAL四-horizon validation为`-2.3897%`，本次four-H checkpoint下变为
validation `+0.1469%`、test `+0.5906%`。旧SIFF failure包含明确checkpoint假失败。但SIFF在PRIOR/PCC下不稳，
且未超过independent control的macro gate，因此只能标记`partial_pass_attribution_blocked`。

下一步不是直接补seed，而是回Step6冻结EQUAL-context controls与`A6_MEASURE_ONLY`。详见
`analysis/stage_c_fair_reaudit_v1_20260717/step9_10_conclusion.md`。

补充internal mechanism audit确认：PCSD_DIRECT arms的loss CV为`36.11%`，同一run最佳固定arm仍比fused差
`18.05%`；PCSD_PCC/SIFF_PCC的row-bin oracle headroom达到`16.17%/18.17%`，但policy entropy为
`0.965/0.975`，说明PCC增加了未利用的conditional headroom而非形成有效routing。SIFF_EQUAL则同时保持
`+1.6436%` test gain、`6.39%` oracle headroom与非零arm差异。MCCA没有进入本次14-arm矩阵，其状态修正为
`historical_validation_negative / fair_test_not_reaudited / inactive`。

## Test-Primary Fair Historical Re-audit

用户决定暂停CTD，先按最新规则公平评估PCSD、PCC与SIFF。规则现明确：validation只选择checkpoint与支持
debug/diagnostic；正式机制pass/fail统一看official test的H96/H192/H336/H720。历史best-H720 checkpoints不直接
复用，A6与所有candidate/control均from-scratch重训，并由四horizon validation MSE平均选择checkpoint。

冻结矩阵包含14 arms × 5 datasets × seed2021，共70 runs与280 test cells。它同时覆盖A6→PCSD、PCC相对
equal/prior controls、SIFF在equal/prior/PCC下相对PCSD，以及constant/permuted/Q1-wide/independent controls。
Step7A通过70 CLI、40 model construction、prefix identity、profile hash与paired Encoder initialization共9类
gate；Step8于2026-07-18 00:52完成70/70，Step9/10结论见上节。

由于test已成为统一benchmark decision surface，所有后续candidate均标记`test_informed`；项目不再声称test是
untouched holdout。完整协议见
`analysis/stage_c_fair_reaudit_v1_20260717/preregistered_protocol.md`。

## SIFF/MCCA Step 9–10 Result And Measure Rollback

55/55 new validation runs与25/25 matched references通过protocol audit。SIFF architecture main effect为
`-1.5015%`、2/5 datasets；MCCA相对same-mass PCC为`-0.0250%`、2/5；joint相对A6为
`-0.5621%`、4/5。exact SIFF-v1/MCCA-v1均未通过Phase-A，confirmation、Phase B与test保持false。

controls给出两个局部正信号：ordered SIFF相对permuted为`+1.1177%`、5/5；MCCA transport相对pointwise为
`+0.4736%`、4/5，capability marginal相对uniform OT为`+0.1182%`、5/5。但ordered未超过Q1-wide/
independent macro gate，PCSD MCCA相对PCC为0/5，因此这些结果不能挽救exact methods。

failure attribution发现SIFF不是全面fit失败：ETTm2 SIFF+MCCA相对PCSD+MCCA在H1为`-669.49%`，H720却为
`+0.6013%`。dense all-prefix AUC等价于target weight
$w_t=T^{-1}\sum_{H=t}^{T}H^{-1}$。code audit随后确认PCSD/SIFF coupling arms的fused training loss已经使用
exact harmonic target measure（L1），所以不能归因于flat training；未决问题是L1-vs-MSE与H720 checkpoint。

当时decision：停止exact SIFF-v1/MCCA-v1 development；因>100%局部pathology，不作scale-coordinate方向级否决，
MCCA只保留transport/marginal ingredients。按后续test-primary治理，该段不是formal test rejection：SIFF已公平
复评并修正为partial pass，MCCA尚未公平复评且保持inactive。回滚Step4执行`SC-D16` external-first measure audit。

source audit现已完成：NeurIPS 2024 ElasTST直接覆盖harmonic horizon reweighting与weighted checkpoint；
Loss Shaping/QDF进一步覆盖future-step weighting。standalone PHMA narrative fail，新增HR arms也因现有training
已exact harmonic而冗余。只保留`SC-D16-CTD diagnostic_only`：下一步Step5/6冻结ETTm2上的
PCSD/SIFF/constant/Q1四条per-epoch trajectories；implementation/remote/test仍false。

详见
`analysis/stage_c_post_pcc_step7b_seed2021_20260717/step9_10_result_and_failure_attribution.md`。
source audit见
`analysis/stage_c_post_pcc_step4_measure_audit_20260717/source_informed_measure_audit.md`。

## Paper-Facing Four-Horizon Reevaluation

项目规则现固定为validation development与official test main/ablation均使用H96/H192/H336/H720；dense curve/AUC
默认只作mechanism diagnostic与unified-horizon补充证据。旧Step7B artifacts回溯重算后，SIFF architecture为
`-2.3509%`、8/20 cells、2/5 datasets；MCCA为`-0.1357%`、7/20、1/5；joint vs A6为`-1.3325%`、
14/20、4/5。三项均fail，旧decision不反转。

SIFF architecture按horizon为`-6.3186/-2.6027/-1.0522/+0.5698%`，对应H96/192/336/720。说明问题不是
dense H1指标单独制造，而是明确的short/mid-to-long tradeoff。该audit为validation-only retrospective screen，
继承best-H720 checkpoint且未重选epoch；test=false。详见
`analysis/stage_c_post_pcc_standard_horizon_reevaluation_20260717/standard_horizon_reevaluation_report.md`。

## SC-D16-CTD Step 5/6 Design

diagnostic已冻结为ETTm2 pathology-locus四条trajectory：
`PCSD_EQUAL/SIFF_EQUAL/SIFF_CONSTANT_EQUAL/Q1_WIDE_EQUAL`。四arms共享seed2021、equal-skill exact
harmonic-L1 objective、batch32与learning rate $10^{-4}$，固定运行20 epochs而不由H720 early stopping截断；
每epoch一次full validation forward生成paper-facing H96/H192/H336/H720与dense H1..720 MSE/MAE。

同一trajectory离线选择`best_standard_mse`、`best_h720_mse`、`best_dense_mse_auc`与
`best_dense_mae_auc`。只有SIFF best-standard同时移除H1>100% pathology、在四标准horizon超过
PCSD/constant/Q1且不以long-bin >1%退化换取，才允许five-dataset validation confirmation。若所有checkpoint
rules下H1 ratio仍>2，回Step2关闭scale-field方向。

decision=`diagnostic_design_pass_step7a_local_only`；下一步实现per-epoch evaluator与selected-state retention，
remote/test仍false。详见
`analysis/stage_c_d16_ctd_step56_20260717/step56_diagnostic_design.md`。

[Pause] 2026-07-17用户明确暂停CTD。上述design与rollback边界保留，但Step7A、remote与test均不继续；当前cursor
已切换到PCSD/PCC/SIFF公平重评估。

## Post-D11 Joint Mainline Reset

[Decision] D11只否定short/long directional gradient conflict，未否定D6 locality-coherence crossing、RGNB
geometry或Contribution 1 slot。不能从D11直接跳到Contribution 2；旧MIPR也因problem evidence与prior-art边界
不足而关闭。

历史post-D11主线曾把unified multi-horizon forecasting定义为同一future function在nested prefix-risk family下的
rank-limited逼近：

1. `PRISM`用$W_\mu$-isometric、prefix-localized forecast frame替换A6 unconstrained basis，保留free
   coefficient head与domain-only crop；
2. `CAPE`用train-only cross-fitted predictions估计conditional-mean covariance，使frame优先保留history可预测
   的future energy，而不是raw-label noise；
3. 两者形成`prefix localization on/off × predictable/raw covariance`的`2x2` factorial；
4. D12只进入Step 2-3 diagnostic；不实现method、不读test、不改Encoder。

D12 risk-aligned v2最终只支持1/5 datasets：CAPE关闭，PRISM joint route未进入D12-B，整条forecast-frame
mainline按预注册规则关闭并回滚Step 2。上述内容只作为历史设计，不是当前论文claim。

## Deferred Post-D12 Forecast-Revision Surface Idea

[Strong Evidence] D3-D8只稳定保留future-support geometry、short/local与long/global crossing及A6 free
operator capacity；D9-D12依次关闭history-scale mapping、future-component conflict与predictable-frame
allocation。A6 flatten是bijective reshape，但`PD -> 256`一次global compaction使所有future targets共享
同一coefficient state；patch-direct target access仍未被验证。

[External Boundary] MQ-RNN与Forking-Sequences已覆盖multi-FCD forecast grid；N-BEATS-S、On Forecast
Stability与Forecast AC已覆盖generic revision/stability optimization；forecast rationality literature已给出
conditional-mean revision moment。因此grid、stability penalty、martingale theory与continuous target query
均不能单独claim创新。

[Historical Provisional Mainline] 把基本对象定义为
$F(o,\tau)=E[Y_\tau\mid\mathcal F_o]$：row是multi-horizon forecast，column是same-target revision path。

1. `SC1-NIFRO`：causal patch memory产生`Delta[B,C,P,T]`，沿origin axis prefix scan得到forecast surface；
   $H$只裁剪latest row；linear control必须contain A6 readout。
2. `SC2-IARL`：不压小所有revision，而约束
   $E[e_{new}\Delta]=0$与$E[e_{old}^2-e_{new}^2]=E[\Delta^2]$，使revision energy由accuracy gain解释。
3. 两项status均为`proposed_step2_3`；它们共享surface object，但尚无internal practical headroom evidence。

[Decision] 用户确认该问题适合作为下一篇独立SCI核心。完整idea已转移到根目录`New-idea.md`；D13 protocol
保留为未来restart artifact，当前状态`deferred_next_paper`，不再执行。

## Fixed-Past Mainline Reset: Adaptive Output Coupling

[Accepted Critique] ordered patch memory只描述Encoder–Decoder information interface；即使full patch memory有用，
该问题也同样适用于single-horizon forecasting，不能成为multi-horizon论文主线。旧`CADMO/CPGA`因此标记
`rejected_by_narrative_scope`，原patch-memory D14降为`D14-P auxiliary_interface_probe`且当前不执行。

[Core Problem] Direct、AR、MIMO、DIRMO与future-query decoder的核心差异之一，是future targets共享
predictive function的scope。经典方法通常固定一个strategy或在模型外选择block size；unified multi-horizon
model若只统一输出长度，却固定一种output coupling，仍未统一forecasting strategy。

[Theory Boundary] deterministic separable MSE的Bayes predictor可逐target写成conditional mean，显式future
dependency不是population-risk必要条件。新主线只研究finite-sample/finite-capacity下parameter sharing引起的
bias–variance–flexibility trade-off。

[Internal Evidence]

1. A6是global low-rank/MIMO-like endpoint；
2. D6在disjoint validation上出现short/local `+1.1964%`、long/local `-1.2675%`与12/15 crossing，但该证据
   属于basis support，只有间接意义；
3. D8/JAPO要求新operator contain A6 global function，而不是整体替换；
4. B13/PMFO未支持current recurrent transition，首版不采用AR output feedback；
5. history-conditioned coupling choice尚无直接证据，D9-D10/JAPO形成高风险先验。

[External Boundary] Direct/MIMO/DIRMO与2025 Stratify已覆盖fixed strategy与output-size continuum；CATS、
MQTransformer、TimePerceiver覆盖future/target queries；Implicit Forecaster覆盖global wave decoding；MQF2覆盖
probabilistic future dependency；dynamic ensemble、meta-learning与TimeRouter覆盖expert/model routing。因此任一
primitive都不能单独claim创新。

[Historical Provisional Mainline, Superseded By PCSD-CF Reset]

1. `SC1-PCSD`：Projective Coupling-Spectrum Decoder在同一fixed future domain内表示point、multiple block与
   global sharing scopes；policy依赖history与target coordinate，不读取requested $H$；global arm contain A6；
2. `SC2-CCRL`：Cross-fitted Coupling-Regret Learning用train-only OOF losses为sample × target-region coupling
   policy提供counterfactual supervision；generic cross-validation/regret/routing不计创新；
3. novelty只允许落在完整
   `fixed past -> exact-prefix decoder -> point-to-global coupling spectrum -> counterfactual coupling policy -> no
   external strategy search`链条。

该版本中的CCRL已因two-stage teacher/student inconsistency在Step7A前退出paper core；当前active mainline为
`PCSD-CF direct control first`，见后文`CCRL Retirement And PCSD-CF Reset`。

[Returned Gate] D14-A0在neutral PCA64 carrier上完成5 datasets × 3 folds。carrier skill 4/5且numeric/split
invariants pass，但stable crossing 0/5、sample × bin oracle仅0.0586%、canonical-vs-random -0.1427%。exact
PCA64 + linear RRR evidence失败。

[Failure Correction] A0匹配factor params但未匹配rank-manifold effective DoF，且five-scale full-risk spread
最多0.04036%，没有形成足够function-level contrast。故方向级拒绝无效，归因为
`intervention_point_wrong + capacity_control_incomplete`。

[A1 Design Gate] A1不再调A0 rank，而以E2E grouped nonlinear head改变point/block/global hidden-bank sharing。
所有scales均经GELU正负对构造证明包含full-affine map；80个parameter/partition/affine与20个forward/gradient
local cases通过，最大parameter gap 0.1646%。neutral raw-history carrier是primary direction gate；A6-natural只作
paper-carrier sensitivity。由于A6 architecture/profile围绕global basis decoder形成，A6-negative不能拒绝scale。

[Completed Execution Order] neutral seed2021 -> neutral gate -> A6-natural sensitivity -> seeds2022/2023
dual-carrier confirmation。该D14-A串行协议已完成；D14-B后来在Step7A前被consistency audit取消。

[A1 Neutral Returned] 40/40 complete；function separation、carrier skill、crossing均5/5；oracle macro 7.6753%；
canonical-vs-random 0.8945%且5/5正。sequential row re-evaluation修复了official validation shuffle造成的artifact
alignment fault，未重训checkpoint。neutral只授权A6 sensitivity；single-seed不能直接形成paper claim。

[A1 A6 Returned] 45/45 complete；A6同样5/5 crossing，strict oracle 9.1504%，sample-over-bin 8.5429%，
contiguity 0.6661%且5/5正。neutral strict/sample分别6.9978%/6.7555%，dual-carrier problem evidence一致。
但train-selected/validation-best GroupedMLP相对A6-LBF H720 macro为-2.9435%/-1.6855%，所以fixed grouped head
不是method candidate。该single-seed结果当时只授权seeds2022/2023 confirmation，最终判定见下一段。

[A1 Three-Seed Confirmation] 新增170/170 runs完成，three-seed dual-carrier gate均pass。neutral/A6均为5/5
stable crossing；strict oracle为7.1107%/9.1259%，sample-over-bin为6.7948%/8.5990%。contiguity均为4/5 stable
datasets，故只能claim broad default而非universal law。A6 train-selected/validation-best GroupedMLP相对LBF仍为
-2.6886%/-1.4879%，所以problem confirmed但method仍未ready。该时点只授权D14-B Step4-6
source/theory/narrative audit；最终设计判定见下一段。

[D14-B1 Step4-6] 2026-07-16 external audit确认TimeFuse已覆盖sample-level adaptive fusion，TimeRouter已覆盖
oracle-best labels、context/CV/forecast features与nonlinear router，AME-TS已覆盖structural-prior KL。CCRL novelty
风险上调为high。理论上OOF squared-error differences可识别conditional relative risk，但expert-risk不是mixture
MSE；故CCRL收紧为`actual fused forecast loss + auxiliary cross-fitted centered-risk`。冻结两个gate：B-P检验
history+target predictability，B-C要求hybrid相对matched direct fusion、hard-oracle与in-sample controls有独立增量。
只授权Step7A local implementation；remote/method/test仍false。完整设计见
`analysis/stage_c_d14b_crossfit_regret_20260716/d14b_step46_source_theory_design_audit.md`。

[CCRL Retirement And PCSD-CF Reset] 后续training-consistency audit确认D14-B1需要独立fold × scale teachers、
只覆盖部分training samples的OOF labels，再监督architecture不同且持续更新的joint PCSD arms；因此存在
teacher-student mismatch、stale target和非最终图工程成本。CCRL在Step7A前取消并降为
`diagnostic_only_not_scheduled`。研究返回PCSD Step4-6，提出`PCSD-CF`：一个shared history-to-future mode
field经scope pooling产生全部point/block/global states，使用direct synthesis而非A6 residual，且以构造性映射
exact contain A6。external audit将DeepONet coordinate synthesis、PoU local operator mixture、Soft MoE与
TimeFuse direct fusion列为mandatory boundaries。narrative gate只对local implementation conditional pass；完整
报告见`analysis/stage_c_pcsd_native_reset_20260716/pcsd_cf_step46_source_theory_design_audit.md`。

[Frozen Boundary] neutral raw-history carrier是primary；A6 sensitivity也从头E2E joint training，但其negative只表示
carrier interface/profile不确认。最终paper effectiveness仍须matched E2E，不能用frozen replacement gap通过或拒绝。

## Completed Foundation

### SC0 natural carrier

[Decision] dataset 可有自然结构偏好，但不得为每个新机制重新精调。使用 validation-only 两阶段小 grid
一次性冻结：Weather=P12/D64/ff128、ETTm1=P24/D32/ff64、ETTh2=P12/D64/ff128、
ETTh1=P24/D64/ff128、ETTm2=P48/D64/ff128。params 差异只报告，不参与选择。新增ETTh1/ETTm2的14-run
validation-only extension与3-seed stability gate已通过；five-dataset contract已冻结。

### Natural baseline test reference

[Fact] 2026-07-13 完成 3 datasets × 3 seeds × 8 horizons，72/72 test metrics；checkpoint/profile 均在
test 前冻结，`selection_used_test=false`。该 reference 只用于后续对比，不允许反向修改 protocol。

[Risk] ETTh2 H48 test MSE CV=`5.30%`，后续必须报告三 seed；这与训练期 validation best-vs-last
`31.63%-44.95%` 恶化不是同一统计。

### Research reset and archive

[Decision] StageB 不再是 active cursor。旧 scripts、local candidates、configs 与 protocol/code docs 已移入
archive；`analysis/` 作为不可变 evidence store 保留。活动入口只保留 natural A6 carrier、baseline test 与
PMFO/PIR diagnostic。

## Step 1: Prior-Art Audit

已确认的 novelty pressure：

- ElasTST：horizon-invariant placeholders 与 horizon reweighting；
- TimePerceiver：target timestamp queries；
- FlowState：functional basis + dynamic horizon/resolution；
- Implicit Forecaster：implicit future waves；
- TransDF/QDF：label decorrelation与task covariance weighting。

[Decision] explicit horizon conditioning、continuous coordinate query、simple functional basis、simple harmonic
step weighting 都不能单独成为 paper core。wavelet/refinement/neural-operator专项审计已在2026-07-13
Step 4-6完成，并进一步排除了generic hierarchical interpolation与learnable lifting claim。

## Step 2-3: Completed Problem Diagnostics

[Decision] D1-v2已完成：PMFO structure与frozen ordered-memory gate均3/3；PIR aggregate gate 3/3。
SC1通过problem gate；SC2以measure-conditional形式通过。以下内容转为已完成problem record。

### SC1-PMFO

问题：A6 已按`basis[:H]`直接计算H步输出，但只提供single dense rank-256 future subspace。是否存在稳定
的nested coarse-to-fine future structure，A6 `memory: [B,C,P,D]`是否保留该信息，以及新的operator能否在
不读取horizon ID的前提下提供refinement/local-support computation？

Gate：至少2/3 datasets、3 seeds支持evaluation-space future deviation与baseline residual的stable increment
structure；frozen A6必须优于zero-deviation baseline，且patch shuffle/collapse必须产生至少1%的SSE恶化。
Linear probe只作辅助量，negative R2之间的差值不得形成pass。
learned basis geometry用于区分“容量足够但缺层次”与“subspace本身不足”。若失败，rollback Step 2；
不得用同步更换Encoder与decoder掩盖归因。

### SC2-PIR

问题：deployment horizon measure 的变化是否产生跨 dataset 的非平凡 gradient/risk差异，并且 nested
increments 是否提供 raw step reweighting之外的解释量？

Gate：至少 2/3 datasets 显示稳定 gradient direction变化；projected risk必须超越 ElasTST-style harmonic
weights 的必然结果。若失败，关闭 PIR；horizon measure 只保留为 protocol/evaluation定义。

## Step 4-6: Completed Design Gate

2026-07-13已完成：

1. external primary-source matrix表明arbitrary horizon、functional basis、hierarchical interpolation、
   learned lifting与raw harmonic weighting均不能单独成文；
2. SC1收紧为`PMFO-RCT`：future interval tree的detail位于父尺度正交补，H只做domain pruning；
3. mixed-radix `(90,30,10,5,1)` orthogonality/refinement/prefix invariants均在`1.33e-15`内通过；
4. SC2收紧为`MIPR`：$\widetilde W_\mu=\sum_lQ_lW_\mu Q_l$，是L2 measure-induced
   block-diagonal surrogate，不是exact raw risk；
5. 预注册dense/no-transition/no-conservation与raw/random-projector controls；
6. SC1/SC2均标记`narrative_ready`，但SC2实现必须等待SC1 operator contract。

## Step 7-10: PMFO-RCT v1 Result

1. Step 7A已完成：90/90 shape-prefix cases及refinement/conservation/locality gate通过，不训练；
2. Step 7B完成ETTm1+ETTh2+Weather、seed2021的15-run matched-control screen；
3. PMFO-RCT相对A6 macro `-1.0955%`且三dataset均退化，effectiveness gate失败；
4. conservation相对no-conservation macro `+2.3393%`，保留；recursive transition相对no-transition仅
   `+0.0486%`，v1 claim撤回；
5. decision=`rollback_step4`；SC2-MIPR与joint factorial暂停，不得建立在失败operator上。

禁止在最小 gate 前加入 Encoder innovation、MoE、router、auxiliary reconstruction 或 per-horizon tuning。

## Step 4 Redesign Audit: Completed

2026-07-13 source-informed redesign audit完成：

1. A6 effective operator为$W=BA\in\mathbb R^{720\times768}$；覆盖rank-256 affine family至少需要
   `316,112`维，而PMFO v1 readout只有`212,010` parameters。因此同为256维latent不构成functional
   containment；
2. A6 operator在fixed block90/30 boundaries上的jump ratio仅`0.989-1.009`，没有跨dataset regime-change
   证据；block90 rank16 capture又从ETTh2 `0.4595`到Weather `0.8025`，不支持统一激进local rank；
3. PMFO 8 root nodes的history-patch profile cosine为`0.936-0.994`、entropy为`0.976-0.995`；nodes学习了
   不同signed projection，但没有清晰history-region specialization；
4. PRISM与LeapTS进一步占据generic multiresolution tree与adaptive scale scheduling；nested basis、lifting、
   Net2Net/Network Morphism只可作为数学工具；global low-rank + hierarchy residual也已有Asymmetric MMF压力；
5. 新候选暂定`SC1-FPMO`：future-domain function-preserving multiresolution operator morphism。它只通过
   Step 4 source-level conditional gate，未通过Step 5 theory或Step 6 design gate。

详细统计定义、source matrix与failure attribution见active protocol。PMFO-RCT v1继续关闭，不做调参复活。

## Step 5 FPMO Theory Feasibility: Partial Pass

1. 采用任意正整数$T$可构造的balanced unbalanced-Haar interval basis；orthogonality、perfect
   reconstruction、A6 morphism与native prefix restriction在9个$T$、53个$(T,H)$ cases通过，max gap
   `5.329e-14`；
2. shared-latent `FPMO-M0`与A6 function class完全相同，只能作exact morph control；
3. independent-scale `FPMO-DS`可逐depth factorize A6 effective map，因此exact containment成立；
4. T720的group sizes/rank caps均为
   `[1,1,2,4,8,16,32,64,128,256,208]`，sum=720，所以DS class等价full affine；
5. exact containment、independent scale maps与总latent budget 256不能同时成立；这是Step5 no-go boundary；
6. native restriction成立，但全部scale latents仍可能对任意$H$执行，故撤销“比A6更快”claim；
7. decision=`partial_pass_step6_design_only`；M0与direct-atom DA降为controls，DS尚未narrative-ready。

## Step 6 FPMO Narrative / Control Gate: Rejected

1. T720下每个group满足$k_l=n_l$，所以$D_lA_l$可表示任意block map；linear DS与DA拥有完全相同的
   full-affine function class；
2. 该等价对任意orthogonal coordinates与任意row grouping成立，当前factorization没有scale-specific
   function constraint；
3. deep linear/matrix factorization prior art支持“factorization可能改变implicit optimization bias”，
   但这不构成新的future-scale operator；当前Adam + L1 joint training也不满足直接移植现有定理的条件；
4. 加入per-scale nonlinearity会成为新候选：automatic exact A6 containment、matched dense/random controls
   与prior-art boundary都需重新审计，不能作为DS的implementation detail；
5. DS可少写出inactive atom coefficients，但dense scale factors仍需先构造全部720维scale latents，故
   prefix algebra不产生独立efficiency claim；
6. decision=`rejected_by_narrative_gate`；M0/DA/DS-L只作controls，rollback Step 2/3，MIPR继续held。

## SC1-D4 Structured-Basis Audit: Completed And Rolled Back

1. 315/315 frozen-memory fits完成，test未使用，PCA只由fit targets构造，315 fits均finite；
2. D3 signal复现：H720 balanced相对random orthogonal `+2.7181%`，5/5 datasets通过；
3. locality成立：balanced相对permuted interval八horizon macro `+1.6324%`，8/8 horizons为正；
4. exact midpoint balancing不特异：相对random interval tree仅`+0.2742%`，未过0.5% gate；
5. standard structured bases解释accuracy：balanced相对DCT-II/PCA-fit分别`-0.8609%/-1.5050%`；
6. decision=`standard_structured_basis_explains_gain_return_step2`。fixed balanced basis可作generation component，
   但不能以独特accuracy claim单独成为Contribution 1。

## SC1-PLGO Step 5 Theory Feasibility: Partial Pass

1. 构造Restricted-Global Nested Basis：root保持global DCT subspace，balanced intervals递归生成children
   scaling union相对parent的orthogonal local details；
2. direct restricted-DCT QR暴露最高`3.110e17` condition number；stable local Chebyshev chart保持同span并将
   最大condition降至`1.784e3`；
3. 12个$(T,r_g)$、101个selected prefixes与3,731个all-$H$ bounds通过，max algebraic gap
   `2.141e-13`；
4. square `PLGO-ONB-M0`可exact morph A6，但只是isometric reparameterization，无新function；
5. naive global/local union虽有frame bounds$[1,2]$，却有$r_g$维coefficient kernel；
6. T720、$r_g=16$ independent-group rank caps sum=720且等价full affine，capacity control解释收益；
7. native support pruning成立，但H1需102个active atoms，generator-level speedup未证明，效率claim撤回；
8. decision=`partial_pass_step6_design_only`；RGNB只冻结为mathematical scaffold，method/training仍false。

## PCSD-CF Step 7A Local Gate: Passed

1. `PCSDCouplingFieldReadout`已接入A6-natural active readout，真实forward为
   `memory [B,C,P,D] -> z [B,C,R] -> modes [B,C,4,256] -> arms [B,C,5,720] -> policy -> [B,H,C]`；
2. five profiles × 13 horizons的65个direct prefix cases与5个真实model integration cases全部exact crop，max gap `0`；
3. arbitrary-A6 mapping在$R=768/1536/3072$的float32 maximum output/arm gap为
   `3.815e-6/2.384e-6`，float64为`3.109e-15/5.329e-15`；
4. scope Jacobian-sharing classes严格为`720/15/5/2/1`；canonical/random minimum arm NRMSE为
   修正fan-in初始化后为`0.131493/0.023079`，equal-zero initial policy gap为`0`；
5. five-profile module与ETTh2真实Encoder-PCSD E2E two-step gradients finite/active；canonical/random trainable
   parameter values与shapes相同，只改变fixed partition buffers；
6. coupling-field core参数为A6 decoder的`3.0291-3.6184x`，含policy为`3.1006-3.7224x`，FLOP静态估算为
   `7.97-13.93x`，故Step7B必须保留dense capacity control与remote resource smoke；
7. decision=`step7a_local_pass_step7b_design_only_next`。该结果只通过implementation/theory contract，不是
   effectiveness evidence；remote、SC2、test均false。

## PCSD-CF Step 7B Step9/10 Result

seed2021 validation-only screen于17:26:52正常结束，60/60 protocol/artifacts与paired initialization通过。DIRECT
相对A6 macro -1.5833%、0/5，method gate失败；相对dense matched +2.3492%、5/5，相对random +0.4499%、3/5，
capacity/random explanations排除。25/25 DIRECT same-run scope arms相对独立fixed E2E training退化，median
89.95%，failure attribution=`design_fault_suspected_joint_credit_starvation`。SC1不跑confirmation，回Step4保留
training-aware representation question；SC2-PCC完成Step2-5 source/theory audit，15/15 local cases通过，下一步只做
Step6 control matrix、optimization与rollback design。Contribution-2 implementation、test与confirmation seeds保持false。

## PCSD-CF Milestone Test Audit: Exact V1 Rejected

2026-07-16冻结的12 arms × 5 datasets official test audit完成60/60，checkpoint hash与no-retraining invariants全过。
DIRECT相对A6为-1.3994%、1/5；相对equal/static/dense/random的macro gain均为负。validation上的dense advantage
发生test reversal，但A6 primary gate在两split均失败，exact PCSD-CF-v1因此在Step10关闭。

same-run oracle test headroom为+2.0197%、3/5，且25/25 DIRECT arms仍under-trained，median 90.6647%。按预注册
decision map归为`test_fail_with_arm_headroom`：PCC可进入test-informed Step6 design，但不得据此宣布training机制成立，
也不得按test dataset/horizon调参。下一rollback point是PCC是否超越measure-only、equal-skill、capability-only、
route-only及generic balancing controls。

## SC2-PCC Step 5 Theory Feasibility

plain fused arm/router与PCC附加credit的四个gradient identities均以float64 autograd验证，最大误差
`5.20e-18`；dense-prefix measure identity误差`4.44e-16`，full-domain prefix crop gap为`0`。history × target
crossed synthetic policy达到capability KL `1.50e-11`与argmax accuracy `1.0`。这只证明output-level skill floor、
router credit与projective measure可同时成立；shared-parameter gradient cancellation、moving target、arm
homogenization与真实capability predictability仍未解决。decision=`conditional_pass_step6_design_only`；下一步必须
冻结`MEASURE_ONLY/EQUAL_SKILL/CAPABILITY_SKILL_ONLY/ROUTE_ONLY/full PCC`等controls后才可讨论Step7A。

## SC2-PCC Step 6 Source-Informed Redesign

fresh external search发现time-series Expert Loss Integration已直接训练expert losses，ICLR 2026 graph MoE也已使用
negative per-expert loss teacher、gate KL与uniform warm-up。因此原pointwise PCC-v0不能承担Contribution 2，降为
closest-prior control。研究显式回滚Step4/5后提出test-informed `PCC-v1-TI`：先计算全部$H=1..720$的scope
prefix risks，再用harmonic prefix-target incidence把capability输运为不含requested-H的target-coordinate credit。

exact nested-risk/transport identity误差`0`，19/19 local design cases通过；全局固定continuous schedule与9 new arms ×
5 datasets validation matrix已冻结。narrative gate仅conditional pass：full transport必须超过A6、plain、pointwise v0
与pointwise prior composition，并显著恢复25个arm pairs；否则按generic control、readout ceiling、shared-gradient或
numeric pathology分别回Step4/5。decision=`step6_pass_step7a_local_authorized`，remote/test false。

## SC2-PCC-v1-TI Step 7A Local Implementation

`layers/PCC.py`已实现nine frozen modes、dense-prefix measure、pointwise/prefix capability、harmonic transport与continuous
schedule；TimeAlign只增加显式training-details path，默认三元组、parameter count与inference output不变。real PCSD batch
中的raw-scale arm fusion gap为`8.88e-16`，arbitrary prefix gap为`0`，五个scope auxiliary gradients均非零。

35/35 gates通过：vectorized/direct loop最大差`2.22e-16`、transport identity gap`0`、nine decompositions 9/9、
adapter optimizer step finite且只访问train/val。该结果只建立implementation correctness，不建立effectiveness。
decision=`step7a_pass_prelaunch_audit_next`；下一步仅做45-run runner/analyzer/resource audit，remote/test仍false。

## SC2-PCC-v1-TI Step 7B Prelaunch

45-run Phase-A matrix已按dataset-major slow-first顺序固定：nine objective modes × Weather/ETTm1/ETTm2/ETTh1/
ETTh2。45个production CLI contracts、frozen hashes、endpoint-mode initialization pairing、validation-only authorization、
shared-gradient evaluator与analyzer synthetic smoke均通过；prelaunch categories为8/8。

新runs只训练PCSD DIRECT架构的nine objectives；A6/plain DIRECT/dense/five fixed scopes复用冻结seed2021 references，
不重训。analyzer同时执行performance、pointwise-prior specificity、25-pair arm recovery、pairwise NRMSE retention、policy
collapse与best-val shared-gradient diagnostics。decision=`step7b_prelaunch_pass_remote_seed2021_authorized`；只授权45-run
validation Phase A，test、confirmation seeds与conditional Phase B继续false。

2026-07-17 remote dry-run和GPU0 resource smoke通过，commit `282b96c`已在GPU 0/1/2后台启动45-run
matrix。单次startup audit确认三个Weather jobs进入training、显存占用正常、runner与workers存活。当前进入11-step
Step8，停止长期值守；45/45返回后再进入Step9/10。launch provenance见
`analysis/stage_c_sc2_pcc_step7b_prelaunch_20260717/remote_launch_record.md`。

## SC2-PCC-v1-TI Step9/10 Result And Step4 Rollback

45/45 PCC runs与15/15 references通过本地复算。full PCC相对A6 macro `+0.9627%`、3/5，相对plain
`+2.4927%`、5/5，25/25 arm pairs改善且median relative reduction `98.01%`。但相对closest prior composed
仅`+0.1050%`，低于`0.2%`门槛；five-dataset pairwise NRMSE retention仅`20.57%-41.13%`，低于50%。

formal decision=`generic_or_pointwise_control_explains_return_step4`。`EQUAL_SKILL`已经解释full PCC相对A6 gain的
88.90%，说明arm recovery主要通过same-label homogenization完成；harmonic transport未形成独立performance/horizon
signature。exact v1不进入Phase B、seeds或test。

Step4 external-first audit进一步确认generic expert loss、structural routing prior、heterogeneous experts、orthogonality/
variance diversity与balanced OT assignment均已有直接prior art。provisional next pair为：(1) coupling scale作为internal
coordinate生成scope-conditioned history modes；(2) projective target-measure rows与scope-skill-budget columns约束的
competitive credit。两者只进入Step5 proof，不实现、不remote。详见
`analysis/stage_c_sc2_pcc_step7b_seed2021_20260717/step9_10_result_and_failure_attribution.md`与
`analysis/stage_c_post_pcc_step4_redesign_20260717/source_informed_redesign_audit.md`。

## SIFF/MCCA Step5 Theory Feasibility

10/10 float64 cases通过：SIFF的Q1 containment gap `3.5527e-15`、prefix gap `0`，current constant-coordinate
scope gap `0`而SIFF witness gap `1.0`；MCCA row/column marginal gaps分别`6.25e-17/1.11e-16`，crossed
best-scope mass相对uniform增加`0.6667`，dominant-arm case minimum scope mass `0.2`，skill/router gradients均finite。

该pass只证明algebraic feasibility。generic wider head、heterogeneous experts与BASE/SSR-style OT仍可能解释完整primitive，
production rank/parameter/marginal也未冻结。decision=`step5_theory_pass_step6_source_design_next`；下一步Step6，不授权
implementation、remote或test。详见
`analysis/stage_c_post_pcc_step5_theory_20260717/step5_theory_feasibility.md`。

## SIFF/MCCA Step6 Source-Informed Design

external-first primary-source audit确认：DirMO/Stratify已覆盖固定block-size strategy；CViT/conditioned neural fields/
HyperDeepONet覆盖coordinate-conditioned operators；BASE/Expert Choice/Selective Sinkhorn覆盖balanced assignment与
expert capacity。因此SIFF不能claim coordinate field primitive，MCCA不能claim OT/Sinkhorn或anti-starvation primitive。

production SIFF-v1固定$Q=2,D=4,K=256$：`hidden [B,C,R] -> component modes [B,C,Q,D,K] -> log-scale
basis [S,Q] -> scale-indexed modes [B,C,S,D,K] -> existing scope pooling/shared synthesis`。Q1 exact containment与
same-parameter constant control成立；Q1-wide和independent-scope integer-rank controls在五profiles上的最大parameter
gap为`0.3893%`。params只作attribution，不作profile/candidate选择。

MCCA-v1把batch-channel-target rows的projective mass与scope columns做log-domain I-projection。column marginal被定义为
current PCC在同一progress下给予每个scope的**完全相同总skill mass**，所以方法差异只剩credit放置位置：PCC逐target
均匀撒floor，MCCA在global coverage下竞争分配。float64/float32 marginal gap为`3.86e-10/1.04e-7`，same-mass gap
`5.55e-17/2.98e-8`，22/22 design cases通过。

Phase A冻结`PCSD/SIFF × EQUAL/PCC/MCCA`的$2\times3$ factorial，加SIFF constant/permuted、Q1-wide、independent、
dense matched与pointwise/uniform-OT controls。narrative gate=`conditional_pass`，只授权Step7A local implementation；
remote、confirmation和test均false。详见
`analysis/stage_c_post_pcc_step6_design_20260717/step6_source_method_control_design.md`。

## SIFF/MCCA Step7A Implementation and Step7B Prelaunch

production `SIFFCouplingFieldReadout`、MCCA objective与dense measure-only control已落地。Step7A 36/36通过：
Q1/A6 containment gap `0`，constant collapse `3.55e-15`，float32 MCCA marginal gap `4.47e-8`，same-mass
PCC gap最大`2.78e-17`，arm/policy gradients均非零。该证据只完成implementation/numeric gate，不等于method
effectiveness。

Step7B prelaunch 8/8通过；冻结11 new arms × five datasets = 55 runs，复用未改变的`PCSD_EQUAL/PCSD_PCC`
及A6/PCSD/dense references。seed2021 validation-only remote现已授权；test、confirmation与conditional Phase B仍为
false。remote resource smoke已通过，55-run matrix于`2026-07-17T14:59:22+08:00`从commit `7a9e5c7`启动；
artifacts返回后进入Step9 attribution，不允许依据partial runs改设计。
详见`analysis/stage_c_post_pcc_step7a_local_20260717/step7a_implementation_gate_report.md`和
`analysis/stage_c_post_pcc_step7b_prelaunch_20260717/prelaunch_report.md`。

## SC1-JAPO Step 7A: Production Gate Passed, Step 8 Authorized

1. `memory [B,C,P,D] -> h [B,C,PD]`是可逆reshape，不是pooling；D8失败不能归因于flatten本身；
2. 真正边界是`alpha_j = psi(d_j)^T A h`：自由A6 temporal table被descriptor-generated fixed separable
   operator替代；
3. 直接atom-to-patch cross-attention缺少future-support/history-patch canonical alignment，且B14与OFormer/GNOT/
   BasisFormer/TimePerceiver共同阻断该shortcut；
4. geometry-only linear expert mixture可代数吸收到一个更宽PAF；固定总rank时无新function class，扩rank时由
   capacity control解释，因此不推进；
5. 唯一保留候选为`SC1-JAPO`：free RGNB expert maps生成coefficients，joint gate同时读取history context与atom
   geometry；requested H只选择active atoms；
6. 令所有experts表示同一A6-equivalent RGNB map时，任意convex gate仍精确复现A6；4个$T$ cases最大误差
   `1.137e-13`，无dense bypass containment通过；
7. generic nonlinear decoder、MoE、geometry gating与step-specific representation均已有先例；novelty只允许落在
   joint history-atom operator、RGNB projectivity与multi-horizon domain-only contract的完整组合；
8. 22个prefix cases的shared coefficient/output最大误差`1.172e-13`；requested H只改变active set；
9. scalar construction得到$f(h)=h\tanh(h)$，affine second difference=`1.523188`，证明joint gate严格超出
   fixed affine PAF；geometry-only mixture仍以`8.882e-15`误差collapse为fixed operator；
10. exact containment不是initialization recipe：identical experts使router gradient严格为0，首版必须independent
    from-scratch initialization；
11. uniform/history-only/atom-only/PERM/RANDOM same-bank controls冻结；params差异不用于选择；
12. Step6冻结两个independent full-rank experts（$E=2,K=256$）与factorized multiplicative router（$G=32$）；
13. basis init使用$\sqrt{E/K}$恢复uniform-mixture initial variance；router output std=`0.01`，五profiles entropy
    min=`0.999855`、usage=`0.4980–0.5020`，所有joint gradients nonzero；
14. seven arms固定为A6/JOINT/UNIFORM/HISTORY/ATOM/PERM/RANDOM；所有JAPO arms paired expert bank；
15. seed2021先做35-run validation-only screen；严重失败早停，模糊结果只补seed2022，pass后补seed2023；
16. Step6 decision=`SC1-JAPO narrative_ready_step7a_local_implementation_only`；当时只授权本地编码；
17. production `JAPOReadout`实现六个same-bank modes，requested $H$只选择active atoms；
18. 210/210 prefix与35/35 gradient cases通过；最大gap分别为`4.768e-7`与patch rewrite `5.722e-6`；
19. 七arms Encoder hashes paired，六JAPO arms expert-bank hashes paired且within-bank experts独立；
20. runner/analyzer dry-run固定35 jobs、validation-only、full-H720 L1与best-val；
21. decision=`step7a_pass_remote_screen_authorized`；只授权seed2021 Step8，test/SC2继续held。
22. commit `90e4164`在3090 GPUs0/1/2完成seed2021 35-run matrix；output root固定为repo-external路径；
23. 35/35 artifacts、protocol、from-scratch paired initialization、prefix与patch invariants均通过，无numeric pathology；
24. JOINT vs A6 dense MSE macro=`-1.3754%`、0/5 positive；vs same-bank median macro=`-0.0780%`、2/5；
25. immediate-fail=false、provisional-pass=false，冻结decision=`seed2021_inconclusive_run_seed2022_only`；
26. 五个JOINT router normalized entropy均不低于`0.993263`，提示under-specialization，但单seed不足以区分
    optimization variance与exact design weakness，不能据此拒绝理论方向或临时改loss；
27. two-seed gate固定为先对每个dataset/arm求seed2021/2022 metric mean，再原样执行Step6 provisional threshold；
28. commit `3d37440`于`2026-07-15T11:37:11+08:00`在GPUs0/1/2启动并完成seed2022 unchanged matrix；
29. 70/70 audit通过；two-seed JOINT vs A6=`-1.2435%`、0/5，vs same-bank median=`-0.1175%`、1/5；
30. canonical geometry相对PERM/RANDOM仍为`+0.2229%/+0.1259%`，但JOINT不及UNIFORM/HISTORY/ATOM，
    `capacity_control_explains=true`；
31. 两seed router entropy均接近1，under-specialization复现；这支持exact head/intervention weakness，不构成
    projective conditional operator方向级否定；
32. decision=`two_seed_mean_fail_stop_and_attribute`：JAPO exact v1关闭，seed2023/test/SC2停止，回Step4
    source-informed redesign audit。
33. 2026-07-15系统复盘把正证据收紧为RGNB geometry、exact projectivity、local-support crossing与A6自由算子；
    把fixed tree、shared separable PAF和weak expert mixing关闭为exact designs，而非方向级否定；
34. 下一步为`SC1-D9 History-Support Operator Evidence Audit`：从A6 learned operator验证history-scale ×
    future-support coupling是否超越scale permutation/random controls。该实验预注册为`diagnostic_only`，通过也只
    授权Step4-5候选设计，失败则回Step2/3。
35. D9-A完成15/15 exact audits，Parseval max gap=`7.5381e-16`；macro rho=`0.173810`，positive
    datasets=`2/5`，permutation/random-basis gates=`1/5`与`0/5`，故primary hypothesis失败；
36. global-root与details之间存在15/15正向binary contrast，但它是post-hoc observation且details内部不单调，
    不能挽救D9。D9-B取消，回Step2/3设计D10 raw history–future scale identifiability。
37. D10 Step2/3 protocol已冻结：history DCT与future RGNB使用相同七组sizes，但所有cells进一步固定为16→16；
    binary 2×2与detail-only 6×6 monotone gates分离，paired history/future permutations阻断coordinate/capacity解释；
38. D10使用chronological train fit、20% temporal gap、train holdout与official validation；不读取test，不训练
    forecast model。当前只授权diagnostic implementation与remote evidence。
39. D10 artifacts/invariants完成：binary effect/direction/control=`2/5,0/5,2/5`；detail-monotone
    effect/control=`4/5,4/5`但best-count=`0/5`、mapping permutation=`2/5`；
40. decision=`raw_aligned_scale_not_supported_rollback_step2`。partial off-diagonal signal缺少跨dataset统一mapping，
    不得事后升级adaptive router；D9+D10共同关闭history-scale aligned routing，下一步审计future-component问题。
41. D11 external audit确认Time-o1已覆盖transformed label alignment与task-overload，FreDF/DBLoss覆盖
    frequency/component losses；generic component loss不能成为本项目创新边界；
42. D11 exact identity冻结为`sum_g J^T P_g v = J^T v`，直接分解output gradient而非错误地相加prefix
    component energies；MSE primary、L1 replication；
43. strict directional conflict必须negative dot；low positive cosine与norm ratio分别归为heterogeneity和magnitude
    imbalance。RGNB必须超过DCT/3 random controls才支持future-support-specific problem；remote前method/SC2/test false。
44. D11 accepted v2完成15 checkpoints：strict directional conflict=`0/5 datasets`，support-specific component
    gate=`2/5`，generic responsibility redistribution=`3/5`，magnitude=`2/5`；all invariants pass；
45. 所有validation MSE total paths/batches均为positive dot，same-component跨short/long negative fraction也为0；
    因此SC1 conflict-aware decoder问题为`hypothesis_false`，不是architecture或hyperparameter failure；
46. short measure对RGNB groups 5/6严格zero-gradient，long shares分别约`0.064107/0.020441`；该现象收紧为
    projective supervision coverage observation，只授权Contribution 2 Step1-3 prior-art/equivalence audit；
47. Time-o1、Loss Shaping Constraints与generic task weighting/sampling形成强overlap压力。未经Step1-3证明完整
    `measure -> inclusion probability -> unbiased/controlled risk -> non-equivalence -> falsifiable benefit`链条，
    不实现coverage normalization、MIPR、PCGrad或joint factorial。
48. post-D11 external audit确认：完整T720 label可用时raw horizon-measure risk可一次精确计算，generic
    importance sampling不构成必要机制；MIPR删除cross-scale terms但D11没有支持删除必要性，正式retired；
49. 新主线回到joint Step2-3：PRISM从nested prefix family推导risk-localized frame，CAPE以train-only
    cross-fitted predictions估计predictable covariance；两者先过D12，不直接实现method或读取test。
50. D12-v1暴露uniform normalized risk mismatch；v2以$s_x^2$对齐raw MSE并复用相同pilots，所有invariants
    通过但只1/5 datasets支持。CAPE与joint PRISM route关闭，D12-B取消；回滚Step2并重新开放两个slots。

## SC1-PLGO Step 6 Design Gate: Conditional Pass, D7 Required

1. `PLGO-PAF`的atomwise tensor contract在$T=16/96/720/721$共33个prefix cases通过，max gap
   `4.547e-13`；$H$不进入descriptor/generator，rank上界仍为256；
2. generic branch-trunk、nonlinear query decoder、HyperNetwork、basis coefficient attention、timestamp query与
   functional basis decoder已有直接先例；overlap用于收紧component claim，不自动否决task-specific组合；
3. internal B11 basis-conditioned field被no-basis/constant-slot controls解释，B14 retrieval demand只有1/6
   settings、0/3 datasets通过；新PAF不得复活atom-specific history retrieval；
4. narrowed PAF只读取shared flattened memory，并以RGNB descriptors生成free temporal table的受限替代；
5. compact width256参数仅为A6 readout的0.696-0.880，可能capacity-restricted；near-budget width694约为
   0.9996-0.9998，却可能memorize descriptors而失去geometry attribution；
6. decision=`conditional_narrative_pass_d7_required`。PAF保留为provisional contribution candidate；D7通过并
   返回Step6冻结method contract前不进入Step7。

## SC1-D6 Confirmation And Step 4 Outcome

D6在未使用的validation batches8-15完成225/225：b144相对global DCT short `+1.1964%`、long
`-1.2675%`，12/15 primary units crossing，short-positive/long-negative分别覆盖4/5与5/5 datasets。
problem gate通过。external primary-source audit确认basis generation、wavelet coefficients、multiscale
interpolation与dynamic target length均已有先例；provisional `SC1-PLGO`只以projective local-global co-synthesis
进入Step5。balanced interval保留为local support scaffold，不claim exact midpoint novelty。

## SC1-D2 Core3 Precheck: Partial

1. 99/99 head-only runs完成，test/freeze/validation/basis/Parseval invariants通过；
2. full affine相对rank256 macro `-0.5661%`，不支持rank expansion是统一瓶颈；
3. strongest dense nonlinear相对full affine macro `-6.4492%`；ETTh2虽fit/inner-holdout更低，official
   validation恶化约19%-24%，属于temporal generalization failure而非未优化；
4. true scale相对strongest dense macro `+4.0358%`被ETTh2 dense overfit放大，且只2/3 datasets为正；
5. true interval basis相对random basis macro `+2.3137%`，3/3 datasets、9/9 seeds为正；
6. true depth grouping相对同basis random grouping macro `-0.2212%`，仅Weather稳定为正；
7. 初版combined random median会隐藏第6项，已在formal5前拆成random-group与random-basis两个mandatory gates；
8. decision=`partial_core3_basis_geometry_signal_only`；不进入Step4，先完成两套profile calibration与formal5。

并行control prerequisite：按validation-only natural grid校准ETTh1与ETTm2 profile。未来broad screen固定为
五dataset全arms seed2021；通过后对五dataset全部decisive arms运行seeds2021/2022/2023。增加dataset降低
cross-dataset偶然性，multi-seed才降低training stochasticity。协议见
`docs/experiments/stage-c-five-dataset-validation-policy.md`。

## SC1-D2 Formal5: Closed

1. five-dataset profiles已冻结；formal5完成165/165 fits，test/freeze/validation/basis/Parseval invariants pass；
2. full affine相对rank256 macro `+0.6780%`，只3/5 datasets达到2/3 seeds为正；rank不是统一瓶颈；
3. strongest dense相对full affine macro `-6.4715%`，ETTh1/ETTh2存在fit/holdout改善但official validation恶化的
   temporal generalization gap；
4. true scale相对strongest dense `+4.5202%`，但该值被上述dense gap放大，不能单独支持scale机制；
5. true basis相对random basis `+3.0635%`，5/5 datasets、15/15 seeds为正；
6. true grouping相对same-basis random grouping仅`+0.0947%`，只有2/5 datasets通过方向一致性，平均只击败
   `1.53/3` controls；
7. exact hypothesis=`hypothesis_false`；否定边界仅为final frozen-memory head上的balanced-depth independent
   nonlinear grouping；
8. decision=`scale_alignment_not_supported_reformulate_step2`；basis main effect因缺失factorial cell仍未识别。

完整解释见`analysis/stage_c_sc1_d2_formal5_20260714/research_interpretation.md`。

Step 7B证据见`analysis/stage_c_step7b_pmfo_rct_20260713/step7b_screening_report.md`。

## Historical Boundary

reset 前完整路线保存在 `docs/archive/pre-stage-c-reset-20260713/`。历史实验结果位于 `analysis/`，只有在
active ledger明确引用其 failure attribution 时才可用于新决策。
