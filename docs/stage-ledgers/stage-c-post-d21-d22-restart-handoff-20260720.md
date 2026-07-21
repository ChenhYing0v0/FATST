# Stage C Post-D21 / D22 Restart Handoff

## 0. 使用方式

本文件是新会话的唯一首读入口。它只保存当前有效状态、约束、证据边界和下一动作；详细历史仍由
`paper-mainline`、`research-roadmap`、Stage C ledger与`analysis/`承担。

新会话必须按以下顺序读取：

1. `AGENTS.md`；
2. 本 handoff；
3. `analysis/stage_c_post_d21_unconstrained_reset_20260720/step2_problem_and_a6_viability_audit.md`；
4. `analysis/stage_c_post_d21_unconstrained_reset_20260720/d22_ab_bayes_frontier_audit.md`；
5. `analysis/stage_c_post_d21_unconstrained_reset_20260720/d22c_result_and_step4_handoff.md`；
6. `analysis/stage_c_post_d21_unconstrained_reset_20260720/d23_fcmi_step46_design_audit.md`；
7. `analysis/stage_c_post_d21_unconstrained_reset_20260720/d23_fcmi_step7a_implementation_audit.md`；
8. `analysis/stage_c_post_d21_unconstrained_reset_20260720/d23_step7b_prelaunch/prelaunch_report.md`；
9. `analysis/stage_c_post_d21_unconstrained_reset_20260720/d23_step8_remote/d23_step9_10_result_and_rollback.md`；
10. `analysis/stage_c_post_d21_unconstrained_reset_20260720/d24_ctb_step23_design_audit.md`；
11. `analysis/stage_c_post_d21_unconstrained_reset_20260720/d24_ctb_result_and_rollback.md`；
12. `analysis/stage_c_post_d21_unconstrained_reset_20260720/post_d24_paper_story_and_modern_baseline_gap_audit.md`；
13. `analysis/stage_c_post_d21_unconstrained_reset_20260720/sc_mnb_step13_source_and_protocol_audit.md`；
14. `analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v2_reactivation_and_tsaf_step46_audit.md`；
15. `analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v3_tsaf_step7a_implementation_audit.md`；
16. `docs/code-explanation/stage-c-siff-v3-tsaf-step7a.md`；
17. `analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v3_tsaf_step7b_prelaunch_report.md`；
18. `docs/code-explanation/stage-c-siff-v3-tsaf-step7b.md`；
19. `analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v3_tsaf_step8_remote_authorization_and_launch.md`；
20. `docs/code-explanation/stage-c-siff-v3-tsaf-step8.md`；
21. `analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v3_tsaf_step9_10_result_and_rollback.md`；
22. `docs/stage-ledgers/stage-c-unified-forecasting-redesign.md`；
23. `docs/paper-mainline.md`；
24. `docs/research-roadmap.md`。

若上述文件与更旧的聊天、archive或历史段落冲突，以本文件和三份主线文档顶部的最新cursor为准。

## 1. 当前权威状态

| Field | Content |
| --- | --- |
| `project` | R_2026_FATST |
| `stage` | StageC-UVHF |
| `handoff_date` | 2026-07-21 |
| `source_parent_commit` | `d647874`（Step7B开始前parent） |
| `current_step` | SC1-SIFF-v3-TSAF Step9/10 complete；exact v1 closed；return Step2/4 |
| `active_problem` | independent target-only weak signal是否能形成新的SIFF-first paper problem？ |
| `active_method` | none；SIFF-v2 immutable paperization parent |
| `method_training_authorized` | completed seed2021 matrix only；new candidate/confirmation false |
| `remote_training_authorized` | no new run；completed one formal test on 2026-07-21 |
| `next_action` | consolidate paper problem/narrative；do not promote control post-hoc |
| `conditional_next` | new Step2/4 and narrative/design gate before any implementation or remote run |
| `rollback` | TSAF-v1 closed，Step2/4；SIFF-v2不变；SC-MNB execution false |

当前工作树存在两个与本次handoff无关的untracked目录，必须原样保留，不得在新会话中清理、归档或提交：

- `SRP-7C55/`；
- `analysis/phase5_timealign_hss_h1c_capacity_preserving_gate_20260701_partial/`。

## 2. 本轮正式转向

### 2.1 已撤销为硬约束

以下项目不再是新architecture必须满足的先验条件：

- exact projectivity；
- requested horizon禁止进入模型；
- 必须生成full-$T=720$ trajectory再prefix crop；
- 必须兼容A6 decoder/interface；
- 必须预先形成`decoder + training loss`两个contributions。

它们仍可作为设计选择或matched controls，但必须由问题、理论和实验决定。

### 2.2 仍然有效的理论护栏

在fixed past、pointwise separable MSE且requested horizon不携带额外信息时，同一future coordinate的Bayes最优
conditional mean不依赖requested horizon。故“允许输入H”不等于“H-adaptation有统计必要性”。

新自由度必须明确来自以下之一：

1. finite-capacity/optimization tradeoff；
2. target-coordinate-specific history evidence access；
3. nonseparable/decision risk；
4. different known-future context；
5. compute/resolution contract；
6. probabilistic joint target。

当前任务仍以deterministic MSE/MAE为primary，D22优先审计1和2；其余四类属于显式task pivot。

## 3. A6、SIFF与历史路线的当前角色

| Item | Current Role | Boundary |
| --- | --- | --- |
| `A6-LBF-natural-baseline` | strong carrier / mandatory control / possible component | 不足以standalone承载高水平论文；新设计无需强制兼容 |
| `A6_MEASURE` | strong training control | harmonic horizon measure受ElasTST直接prior覆盖，不能单独claim |
| `SIFF-v2-EQ-ATTR-v1` | immutable performance-near parent | internal 7/7，但未超过A6_MEASURE且independent control阻塞归因；不修改exact v2 |
| `SIFF-v3-TSAF-v1` | closed exact candidate | effectiveness/attribution fail，internal health pass；不补confirmation/rescue |
| `D14 crossing/oracle` | historical clue | oracle不等于past-identifiable benefit；D21已证明interaction split-unstable |
| `D17-D21` | closed evidence | 不做representation/readout/seed rescue |
| `CTD` | paused by user | 新会话不得自动恢复 |
| `NIFRO/IARL/New-idea.md` | deferred next-paper idea | 不占当前paper slot |

## 4. 必须保留的关键结果

1. A6-LBF相对TimeAlign lineage有真实性能收益，但modern varied-horizon baseline comparison仍不完整；
2. `A6_MEASURE > A6_FULL`约`+1.8762%` MSE，20/20 standard cells正向；
3. D18 specialists相对A6_MEASURE仅`+0.1659%`、7/15 cells，finite-capacity horizon frontier证据弱；
4. SIFF_EQUAL相对A6_FULL `+1.6436%`，但相对A6_MEASURE `-0.2366%`；
5. D21 oracle headroom为`7.64%/10.41%`，但interaction相对additive仅`+0.0347%/-0.0069%`，且validation→test不稳定；
6. 因此当前没有active paper-core method，也没有已经成立的two-contribution chain。
7. D22-A/B dense复核后，SPEC96 own-H为`+1.2748%`、5/5 datasets，但SPEC192/SPEC336为负，
   0/15 arm-dataset Pareto dominance；decision=`finite_capacity_frontier_not_supported`；
8. A6_MEASURE相对A6_FULL在五个lead-time bins全部5/5正向；H96只保留局部optimization clue；
9. D22-C static/prelaunch已通过；仅冻结的neutral/raw-history diagnostic remote/test获授权，paper method仍未授权。
10. D22-C v1.1已完整通过problem gate；FCMI Step7A production-local 11/11通过，尚未训练；
11. FCMI Step7B prelaunch 21/21通过；40 runs、160 official-test cells、160 validation cells和dense
    capacity control已冻结；
12. 用户2026-07-20已独立授权上述seed2021 remote/test matrix；confirmation与paper method仍未授权；
13. commit `4ff439c`已通过三卡preflight与两项resource smoke；40-run matrix完整结束；
14. FCMI vs A6为`-21.7343%`、0/20；DENSE vs STANDARD_DUAL为`+15.4825%`、19/20，
    DENSE vs A6仅`-0.3284%`；FCMI-v1关闭并回Step2/3。
15. phase/time-warp probe specificity仅约`+0.03%`且direct prior充分；phase router关闭。D24-CTB冻结为
    validation-only raw-history conditional coarse-deformation diagnostic。
16. D24-v1 10/10 protocol valid但ridge penalty未按fit rows归一化，severe extrapolation标记
    `design_fault_suspected`；v1.1只修正normalized ridge semantics后重跑。
17. D24-v1.1 10/10、840 metrics、720 comparisons完整，test access=0；ordered相对marginal/sorted/shuffled
    全部macro negative且0/4 horizons；exact probe关闭并回Step2/4 consolidation。
18. Post-D24 consolidation确认完整证据链scientifically coherent，但method-paper narrative incomplete；
    当前没有正向paper-core method，modern native-baseline gap成为blocking gate。
19. `SC-MNB` P0固定ElasTST、CATS、TimePerceiver、SRSNet与A6_FULL/A6_MEASURE；当前只允许Step1-3
    source/protocol audit，implementation、remote training与official test均false。
20. 四个official commits与65-run/80-cell roles已冻结；CATS/TimePerceiver per-epoch test access、CATS
    ETTm2-H96 typo、SRSNet file-level license trace/metric equivalence与ElasTST 10-batch semantics阻塞
    prelaunch。
21. 用户2026-07-21明确选择SIFF-first论文落地；恢复SIFF program但不改写v2历史failure，SC-MNB降为supporting
    inventory且execution false。
22. TSAF Step4-6冻结：保留SIFF arms，以future-coordinate × ordered-log-scale field产生sample-shared
    allocation；不输入requested H/history hidden，不增加第二loss。
23. TSAF narrative gate conditional pass；Step7A production-local 26/26通过。
24. TSAF Step7B prelaunch为15/15 cases、10/10 categories；formal matrix固定45 effective runs/180 cells，
    其中20个historical references复用、25个new arms × datasets必须joint from-scratch。
25. 20/20 remote reference checkpoint hashes与frozen audit一致；旧direct independent不得复用为target-only control。
26. target-only independent matched ranks为ETTh1/ETTh2/ETTm1/ETTm2/Weather=`109/115/115/106/115`，最大
    active-parameter gap 0.3619%。
27. 只读remote preflight显示3 GPUs idle；当时未pull、未resource-batch smoke、未训练、未test。
28. 用户2026-07-21授权冻结的seed2021 25-run training与一次完整formal test；confirmation仍false。
29. evaluator已补齐`effective_arms` matrix support，runner分离training与formal-test mode；下一步先commit-pinned
    remote pull与两项resource smoke，25/25 training前不得访问test。
30. commit `6cef063`已pull；Weather-TSAF与ETTm2-independent resource smoke finite/no-OOM。
31. 25-run training于`2026-07-21T10:17:06+08:00`在GPU0/1/2启动；initial training/test=0/25，formal-test mode
    未启动。训练期间不得pull或修改config/gates。
32. TSAF Phase A已25/25 new training、25/25 new formal test和45/45 effective audit完整；formal-test commit
    `4cc96f21e23c159e37757c66ec2e5c68358c5718`，checkpoint nonmutation 25/25通过。
33. TSAF相对A6_MEASURE test MSE/MAE为`-1.2854%/-1.3146%`，相对SIFF-v2 parent为
    `-1.0422%/-0.9183%`，两项均0/4 horizon wins，paper-facing effectiveness fail。
34. ordered-field、ordered-scale、target-coordinate、shared-field comparisons的MSE分别为
    `-1.0191%/-0.0796%/-0.0405%/-1.2785%`；matched attribution全fail。internal health全过不能覆盖结果。
35. validation中TSAF相对parent为`+0.7700%`，test反转。independent target-only相对parent的`+0.2383%`
    只作single-seed weak lead，低于0.3% primary threshold，不得post-hoc promotion。
36. Decision=`close_tsaf_v1_shared_field_design_keep_siff_v2_immutable_parent`。当前无active successor method，
    no confirmation/rescue，回SIFF-first Step2/4；SC-MNB execution和Contribution 2继续false。

## 5. D22-HFA 的执行顺序

### D22-A：Bayes/task boundary（completed）

- 写清requested H是纯request还是带来新information/risk/context/compute contract；
- 审计放宽约束后的primary-source prior；
- 禁止从“自由度更大”直接推出method necessity。

### D22-B：existing-artifact finite-capacity frontier（completed）

优先复用D18 specialists、A6_MEASURE、dense-horizon与checkpoint artifacts，回答：

- specialists是否形成稳定own-H/Pareto advantage；
- shared model是否在特定lead-time segment系统性让步；
- tradeoff是否超过measure control、seed和split波动。

本步骤没有训练新模型、没有访问新test选择candidate、没有恢复CTD。完整结果见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/d22_ab_bayes_frontier_audit.md`。

### D22-C：completed problem supported

问题限定为future coordinate与history token/patch的joint access是否稳定超越：

- global compressed state；
- pooled memory；
- order-shuffled memory；
- target-shuffled query；
- matched-capacity generic control。

ordered patch memory只作诊断载体，不作论文主语。neutral/raw-history为primary，A6为sensitivity。frozen
component replacement只能形成conditional evidence，不能方向级拒绝E2E method。

六臂neutral/raw-history runner与machine aggregator已实现；所有arms共享完全相同trainable parameter set、
seed2021 initialization、optimizer、window selection与validation selector。local synthetic smoke已完成六臂
forward/backward、checkpoint、validation/test和decision artifacts；parameter gap为0。

首次v1 remote launch在training-only阶段发现RevIN-normalized loss被near-zero history variance放大；在任何
dataset/official-test artifact完成前终止。v1.1只改为RevIN重建后的dataset-standardized MSE，architecture、
arms、seed、selector与gates不变；必须使用新output/checkpoints完整重跑。

five-dataset diagnostic现已完整执行并同步。D22-C raw arm不升级method；A6 sensitivity不再需要启动。
Contribution 2仍未授权。

### D22-C result与D23 handoff

D22-C v1.1 complete decision=`target_coordinate_information_access_supported`。ordered相对generic为test MSE
`+2.5228%`、MAE `+1.6484%`、15/20 cells、4/5 datasets、4/4 horizons；validation/test同号，parameter gap 0。
Weather 4/4 horizons负向，阻止universal claim。

query/cross-attention primitive已由CATS、TimePerceiver、MQTransformer与TQNet覆盖。新candidate
`SC-D23-FCMI`只在完整chain上claim：把query-retrieved context精确分为trajectory main与zero-mean coordinate
interaction，generic和standard query均为contained cases，并用matched dual-branch control排除capacity解释。
Step4-6 conditional pass；Step7A现已11/11通过，remote/test/paper promotion仍false。FCMI相对A6 active
parameters少83%–95%，后续formal design必须加入dense capacity-matched control。

Step7B现已21/21通过。`DENSE_DUAL_MATCHED`使用profile-specific low-rank temporal residual，使五个profiles
与A6 active parameters差距仅`0.0914%–0.1321%`，同时zero-init保持standard-dual initial function；
coefficient与basis的分阶段gradient及active residual均通过。formal matrix固定8 arms × 5 datasets ×
seed2021 = 40 runs：全部8 arms进入160个official-test cells和160个validation cells。
`TARGET_SHUFFLED_QUERY`原拟validation-only，但因参与方向级attribution，在任何test access前修正为formal
control。dense arm是capacity attribution control，不是第二个method或Contribution 2。

Step9/10已完成：40/40 protocol valid，FCMI vs A6 test MSE/MAE为`-21.7343%/-10.9242%`、0/20。
decomposition/generic/target controls通过，但order与capacity失败；internal health 5/5。DENSE几乎恢复A6，
并相对STANDARD_DUAL为`+15.4825%`。三种FCMI validation-fit conditional blends全部test反转；
A6/DENSE validation-fit blend同样test反转，固定等权只有test-only正信号，不能授权allocation/router。
decision=`fcmi_v1_failed_capacity_control_explains_return_step2_3`。

## 6. 研究与实验治理

- 外部调研默认广泛web search并优先primary sources；Zotero只作seed；
- paper-facing/formal mechanism evaluation使用official test的`{96,192,336,720}`完整矩阵；
- validation只用于checkpoint、普通超参数、debug与解释性diagnostic；
- official test已是`test_informed benchmark decision surface`，不得声称untouched；
- final confirmation应增加冻结后未参与设计的新datasets；
- paper-core比较默认matched end-to-end joint training；frozen replacement只作`diagnostic_only`；
- 一个problem diagnostic失败后必须明确failure attribution，不得直接堆叠下一机制；
- 远端实验前必须commit/push、`nvidia-smi`检查并使用`529_Lab-3090`的`moe`环境；
- 远端训练输出默认写入`/home/yingch/exp_outputs/r-2026-fatst`。

当前five-dataset profiles保持：ETTh1、ETTh2、ETTm1、ETTm2、Weather。dataset之间允许不同自然profile；
同一dataset的机制比较必须共享profile，params差异不参与profile选择。

## 7. 当前执行定义

1. 读取SIFF-v2 freeze/Step9、CCSF closure、Post-D24 consolidation与TSAF Step9/10 result；
2. 保持SIFF-v2 immutable，不改写其A6_MEASURE/independent failures；
3. `SC1-SIFF-v3-TSAF-v1`已关闭；不得补seed、rank、width、readout、loss或selector rescue；
4. 当前没有new official test、remote training、confirmation或method implementation授权；
5. SC-MNB保留为supporting source/control inventory，不执行65-run matrix；
6. 下一步回Step2/4形成新的problem/narrative/design gate；independent control不得post-hoc晋升；
7. 不恢复CCSF、D17-D21、H embedding、region/covariance/temperature或Contribution 2预设。

## 8. 禁止无损重启时发生的漂移

- 不把A6重新写成已经成立的论文主体；
- 不把SIFF内部健康度写成paper-core pass；
- 不把SIFF-v2历史failure改写为pass，也不直接修改其immutable identity；
- 不把TSAF称为有效或仍active；它已在完整formal test上失败并关闭exact v1；
- 不把independent target-only的`+0.2383%` weak signal直接改名为method或补confirmation；
- 不恢复EVS、CCSF、PCC、PCSD、JAPO、D19或D20的参数/seed rescue；
- 不因为放宽约束就直接实现explicit H embedding；
- 不把ordered patch memory升格为论文主线；
- 不在Contribution 1 problem gate之前设计Contribution 2；
- 不导入`R_2026_FSA`代码、配置或artifact，除非用户明确批准具体来源与范围。

## 9. Restart Prompt

本文件末尾Prompt与交付给用户的Prompt应保持一致；新会话以工作区当前文件为准，不依赖旧聊天记忆。

```text
请在 /Users/river/PaperResearch/Project/R_2026_FATST 中继续 Stage C 研究。

首先严格阅读并遵守仓库 AGENTS.md，然后按顺序阅读：
1. docs/stage-ledgers/stage-c-post-d21-d22-restart-handoff-20260720.md
2. analysis/stage_c_post_d21_unconstrained_reset_20260720/step2_problem_and_a6_viability_audit.md
3. analysis/stage_c_post_d21_unconstrained_reset_20260720/d22_ab_bayes_frontier_audit.md
4. analysis/stage_c_post_d21_unconstrained_reset_20260720/d22c_result_and_step4_handoff.md
5. analysis/stage_c_post_d21_unconstrained_reset_20260720/d23_fcmi_step46_design_audit.md
6. analysis/stage_c_post_d21_unconstrained_reset_20260720/d23_fcmi_step7a_implementation_audit.md
7. analysis/stage_c_post_d21_unconstrained_reset_20260720/d23_step7b_prelaunch/prelaunch_report.md
8. analysis/stage_c_post_d21_unconstrained_reset_20260720/d23_step8_remote/d23_step9_10_result_and_rollback.md
9. analysis/stage_c_post_d21_unconstrained_reset_20260720/d24_ctb_step23_design_audit.md
10. analysis/stage_c_post_d21_unconstrained_reset_20260720/d24_ctb_result_and_rollback.md
11. analysis/stage_c_post_d21_unconstrained_reset_20260720/post_d24_paper_story_and_modern_baseline_gap_audit.md
12. analysis/stage_c_post_d21_unconstrained_reset_20260720/sc_mnb_step13_source_and_protocol_audit.md
13. analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v2_reactivation_and_tsaf_step46_audit.md
14. analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v3_tsaf_step7a_implementation_audit.md
15. docs/code-explanation/stage-c-siff-v3-tsaf-step7a.md
16. analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v3_tsaf_step9_10_result_and_rollback.md
17. docs/stage-ledgers/stage-c-unified-forecasting-redesign.md
18. docs/paper-mainline.md
19. docs/research-roadmap.md

当前权威状态是：D22-C v1.1 problem gate仍为`target_coordinate_information_access_supported`；SC-D23-FCMI 40-run/160-cell Step9/10已完成，decision=`fcmi_v1_failed_capacity_control_explains_return_step2_3`。FCMI相对A6 test MSE为`-21.7343%`、0/20；DENSE相对STANDARD_DUAL为`+15.4825%`、19/20且相对A6仅`-0.3284%`。exact projectivity、requested-H禁用、full-T prefix crop、A6 interface compatibility以及预设decoder+loss双贡献均不再是硬约束；但在fixed past、pointwise MSE且H不携带额外信息时，同一future coordinate的Bayes conditional mean不依赖requested horizon，因此不得直接实现H embedding/router。

A6-LBF仅是strong carrier/control/possible component，不足以standalone承载论文；`SC1-SIFF-v2-EQ-ATTR-v1`是immutable performance-near parent，其相对A6_FULL `+1.6436%`、相对A6_MEASURE `-0.2366%`、相对independent `+0.2580%`与internal 7/7均必须完整保留。D17-D21、CCSF与D24 exact routes均关闭；CTD paused。

已确认D22-C ordered相对generic为test MSE `+2.5228%`、MAE `+1.6484%`、15/20 cells、4/5 datasets、4/4 horizons；其余四controls均20/20正向。Weather相对generic 4/4负向，必须保留generic fallback并禁止universal claim。

Step7A已验证zero-mean interaction、standard-query exact morph、main/interaction/query/output gradients、dual parameter matching与35个production CLI cases。Step7B又冻结`DENSE_DUAL_MATCHED`、8 arms × 5 datasets × seed2021的40-run matrix、160个official-test cells、160个validation cells、四层gates与failure rollback。dense control相对A6 active parameter gap为`0.0914%–0.1321%`，只作capacity attribution，不是method或第二项contribution。CATS/TimePerceiver等已覆盖query-to-history primitive；FCMI只在main–interaction decomposition与generic/standard containment的完整chain上作provisional claim。

Step9/10中decomposition、generic与target controls通过，但order和capacity失败；internal health 5/5，negative不是numeric pathology。validation-fit dense/FCMI、dense-plus-interaction和A6-plus-interaction diagnostics全部test反转，且只属于frozen cross-model conditional evidence。A6/DENSE allocation同样没有split-stable、validation-identifiable正证据。direct dense+FCMI successor未过Step4 narrative gate。

SC-D24-CTB-v1.1已完成：10/10、840 metrics、720 comparisons，official test access为0。ordered history相对marginal约`-8.6%`、相对sorted约`-9%`、相对target-shuffled约`-14%`，所有primary horizons均0/4正向。exact coarse deformation probe关闭，不做feature/bin/lambda/nonlinear/seed rescue。

用户2026-07-21明确选择SIFF-first paperization。`SC1-SIFF-v3-TSAF-v1`的25/25 new training、25/25 formal test与45/45 effective audit现已完成。TSAF相对A6_MEASURE test MSE/MAE为`-1.2854%/-1.3146%`，相对SIFF-v2 parent为`-1.0422%/-0.9183%`；ordered-field、ordered-scale、target-coordinate与shared-field attribution均fail。internal health全过只说明路径活跃，不改变negative effectiveness。validation中相对parent的`+0.7700%`在test反转。

Decision=`close_tsaf_v1_shared_field_design_keep_siff_v2_immutable_parent`。TSAF-v1关闭，不补seed/rank/width/readout/loss rescue；SIFF-v2继续immutable paperization parent。independent target-only相对parent的`+0.2383%`仅为低于primary threshold的single-seed control weak lead，不得post-hoc晋升。当前无active successor method，回SIFF-first Step2/4；new implementation、remote/test、confirmation、Contribution 2和SC-MNB execution均未授权。

完成后同步更新analysis report、docs/paper-mainline.md、docs/research-roadmap.md和Stage C ledger，执行最小诚实验证，并按AGENTS.md提交、推送。请从专业时序预测研究员角度进行审计，不要为了凑两个contributions而预先设计第二个loss/router。
```
