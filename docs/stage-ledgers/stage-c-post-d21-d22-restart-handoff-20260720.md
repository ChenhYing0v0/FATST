# Stage C Post-D21 / D22 Restart Handoff

> **Superseded current-entry notice（2026-07-31）**
>
> 本文件仅保留为D22 research lineage。当前paper-writing新会话必须先读
> `docs/stage-ledgers/stage-c-iscf-bsca-paper-writing-restart-handoff-20260731.md`，
> 不得把本文件中的旧active method、authorization或next action恢复为当前状态。

## 0. 使用方式

本文件曾是D22阶段新会话的首读入口，现仅用于追溯当时的状态、约束、证据边界和
下一动作；详细历史仍由`paper-mainline`、`research-roadmap`、Stage C ledger与
`analysis/`承担。

新会话必须按以下顺序读取：

在本 handoff 后，先读最新
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_frsc_step26_20260722/rscc_step9_result_and_rollback.md`，
再按下列历史证据顺序继续。

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
22. `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_function_audit_20260721/result_and_step4_handoff.md`；
23. `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_step4_scope_relation_20260721/step4_result_and_step5_handoff.md`；
24. `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_step5_common_private_interaction_20260721/step5_theory_and_narrative_gate.md`；
25. `configs/stage_c_iscf_v1_cpsi_step5.json`；
26. `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v1_cpsi_step6_design_20260721/step6_control_design_and_test_policy.md`；
27. `configs/stage_c_iscf_v1_cpsi_step6.json`；
28. `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v1_cpsi_step7a_20260721/step7a_implementation_audit.md`；
29. `docs/code-explanation/stage-c-iscf-v1-cpsi-step7a.md`；
30. `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v1_cpsi_step7b_prelaunch_20260721/prelaunch_report.md`；
31. `docs/code-explanation/stage-c-iscf-v1-cpsi-step7b.md`；
32. `configs/stage_c_iscf_v1_cpsi_step7b.json`；
33. `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v1_cpsi_step8_remote_20260721/remote_launch_record.md`；
34. `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v1_cpsi_step9_10_20260721/step9_10_result_and_rollback.md`；
35. `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_post_cpsi_step45_20260721/step4_5_scope_independence_narrative_gate.md`；
36. `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_sac_step7b_prelaunch_20260721/prelaunch_report.md`；
37. `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_sac_step8_remote_20260721/remote_authorization_and_launch.md`；
38. `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_sac_step8_remote_20260721/validation_artifact_audit_and_test_handoff.md`；
39. `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_sac_step9_10_20260721/step9_10_result_and_rollback.md`；
40. `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_sps_step46_20260721/step4_6_design_and_step7a_audit.md`；
41. `docs/code-explanation/iscf-sps-v0.md`；
42. `configs/stage_c_iscf_sps_v0.json`；
43. `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_sps_step7b_prelaunch_20260721/prelaunch_report.md`；
44. `configs/stage_c_iscf_sps_step7b.json`；
45. `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_sps_step8_remote_20260722/remote_authorization_and_launch.md`；
46. `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_sps_step9_validation_20260722/step9_result_and_bsc_step4_handoff.md`；
47. `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_frsc_step46_20260722/step4_6_design_and_remote_gate.md`；
48. `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_frsc_step7b_prelaunch_20260722/prelaunch_report.md`；
49. `docs/code-explanation/iscf-frsc-step7b.md`；
50. `configs/stage_c_iscf_frsc_step7b.json`；
51. `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_frsc_step8_remote_20260722/remote_authorization_and_launch.md`；
52. `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_frsc_step9_validation_20260722/step9_validation_result_and_rollback.md`；
53. `docs/code-explanation/stage-c-iscf-v0-sac-step7b.md`；
54. `configs/stage_c_iscf_v0_scope_attribution_confirmation.json`；
55. `configs/stage_c_iscf_v0_carrier.json`；
56. `configs/stage_c_iscf_v0_scope_response_d1_1_confirmation.json`；
57. `docs/stage-ledgers/stage-c-unified-forecasting-redesign.md`；
58. `docs/paper-mainline.md`；
59. `docs/research-roadmap.md`。

若上述文件与更旧的聊天、archive或历史段落冲突，以本文件和三份主线文档顶部的最新cursor为准。

## 1. 当前权威状态

| Field | Content |
| --- | --- |
| `project` | R_2026_FATST |
| `stage` | StageC-UVHF |
| `handoff_date` | 2026-07-22 |
| `source_parent_commit` | `a3022e9`（PSA-D1 v0.1 analyzer complete） |
| `current_step` | PSA-D1 complete；return Step4 UPA-D2 diagnostic gate |
| `active_problem` | information-free uniform train-time anchor是否足以解释control co-adaptation gain |
| `active_method` | none；UPA-D2 design-only；ISCF-v0 fixed base；closed routes unchanged |
| `method_training_authorized` | false |
| `remote_training_authorized` | false；UPA-D2 implementation/training/test未授权 |
| `next_action` | request authorization for UPA-D2 Step7A + five validation runs |
| `conditional_next` | D2 positive只确认uniform-anchor problem；仍须new Step4 narrative gate |
| `rollback` | H1/H3 closed；H2 carrier clue supported；generic balancing不可直接升method |

当前工作树存在两个与本次handoff无关的untracked目录，必须原样保留，不得在新会话中清理、归档或提交：

- `SRP-7C55/`；
- `analysis/phase5_timealign_hss_h1c_capacity_preserving_gate_20260701_partial/`。

### 1.1 ISCF-SPS current decision

用户已把ISCF multi-scope architecture固定为本轮design prior。SAC negative不得删除或改写为positive；它现在作为
“current scope geometry underutilized”的motivation。active candidate=`SC-ISCF-SPS-v0`在五个raw arms进入既有direct
fusion前加入scope-native orthonormal local-DCT projectors，使$P_s$同时限定forward forecast与backward error
subspace。candidate新增0 trainable parameters，不增加loss/router/requested-H。

Step7A local contract已通过：identity-parent gap `8.34e-7`、prefix gap `0`、projector errors不超过
`3.22e-15`、five scope gradients nonzero、canonical/random/global outputs可辨、production model/CLI通过。该结果不含
training/validation/test evidence。下一步只允许冻结Step7B validation-first matrix；不得直接remote launch或访问formal
test。

Step7B现已冻结scope/identity/global/random × five datasets × seed2021的20-run validation matrix。evaluator保存
raw/projected/removed arms、direct policy、diversity/oracle/bin metrics；local gate `19/19`通过。runner在authorization=false
时exit 3并硬拒绝test split。在2026-07-22授权前不得remote launch；该边界随后已由下述explicit authorization更新。

用户已于`2026-07-22`授权remote validation。commit `48afd12`完成fast-forward，three-GPU preflight和Weather
scope/identity smoke通过；20-run matrix于`00:17:31+08:00`启动，runner PID=`2787170`，初始`0/20`。formal test仍false。

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
| `ISCF-v0` | fixed architecture prior / strong carrier | SAC exact canonical claim fail；FRSC exact continuation fail；仍按用户要求保留architecture，不直接promote paper method |
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
37. 用户2026-07-21明确把FCC `SIFF_INDEPENDENT_EQUAL`以`ISCF-v0`新identity固定为后续research carrier；
    code identity、ranks、profiles、direct policy、equal-skill和three-seed checkpoints保持exact frozen。
38. existing complete test table派生ISCF vs A6_FULL MSE/MAE=`+1.3584%/+0.9144%`，5/5 datasets、
    4/4 horizons、3/3 seeds正向；该结果是post-hoc carrier evidence，不是method promotion。
39. 15-run function audit仅复用existing NPZ：common/private、complementarity、topology通过，aligned low-dimension
    `0/15`失败；decision=`function_relation_unresolved_requires_narrow_step4_audit`。当前不实现method、不训练、
    不访问new test，下一步为non-ordered common/scope-specific relation的Step4 source/narrative gate。
40. Step4指出residual common受shared target difficulty混淆，改用validation-label-free hidden directional response；
    D1 primary在16 directions下relation/random/noncollapse pass，但topology仅2/5。
41. 预先声明的64-direction estimator validity check把topology恢复到5/5，故primary topology failure归为
    `diagnostic_estimator_variance`，不得方向级拒绝或post-hoc promotion。
42. 新冻结D1.1使用disjoint validation rows offset64、新seed、64 directions、128 null与8 random-init controls：
    15/15超过两类controls，common/private=`0.2803/0.7197`，4/5 topology稳定；无targets/training/test。
43. Decision=`scope_response_relation_confirmed_for_step5_theory`；narrative conditional pass为single pre-synthesis
    architecture problem。active_method仍none；下一步只做`[B,C,S,D,K]` non-ordered operator、morphism与matched
    controls的Step5 theory/design，generic set mixing、router、second loss与remote/test均未授权。
44. Step5证明fixed linear scope mixing可被ISCF各scope的独立affine `mode_weight/mode_bias`精确吸收；Cross-Stitch
    matrix、linear common/private和fixed graph不形成新function class。plain peer MLP也因extra projection/depth
    attribution不净而降为generic control。
45. working candidate `ISCF-v1-CPSI`以scope mean与zero-sum deviation的multiplicative product更新pre-synthesis
    modes；scope-equivariant、无requested H/order/target，`W_o=0`精确包含ISCF-v0；只允许finite-capacity解释。
46. Decision=`step5_theory_pass_step6_control_design_next`；full method gate仍未通过。Step6必须解决exact SELF/
    LINEAR/COMMON及诚实POST-SYNTH control；implementation、training、test、router、second loss均false。
47. 用户要求四个controls作为intermediate diagnostics，不能因轻微负向在test前放弃mechanism。Step6因此冻结
    validation只作selector/health，全部protocol-valid arms进入同一次official-test audit。
48. SELF/LINEAR/COMMON均exact `3Lr`；POST直接作用于forecast arms，derived rank使total-model parameter gap
    小于0.041%。r=32、zero-init morph与paired initialization已冻结。
49. formal matrix为25 new trainings + 10 historical references = 35 effective runs、140 MSE和140 MAE cells。
    test inconclusive band为CPSI vs ISCF `[-0.5%,+0.3%)`；轻微负向不方向级拒绝。
50. Decision=`step6_pass_step7a_local_authorized`；当前只授权local implementation，remote/test须等待Step7A/7B。
51. CPSI/SELF/LINEAR/COMMON/POST五个production readouts已实现；parent initialization后再创建interaction
    matrices，保持base RNG path与ISCF exact paired。
52. Step7A local checker在`r2026-fsa`中81/81通过：five model/CLI、zero morph、equivariance、private-absent
    zero message、two-stage gradients与profile parameters均pass；没有training/validation/test。
53. Decision=`step7a_local_pass_step7b_prelaunch_next`；active method为implementation-ready/effectiveness pending，
    下一步只做Step7B，remote/test继续false。
54. Step7B冻结25 new trainings + ISCF/A6_FULL 10 historical references；35 effective runs形成140 MSE与140 MAE
    official-test cells，validation只作selector/health。
55. prelaunch初始18/18；remote resource smoke暴露无`rg`时negated scanner false pass，训练本身finite但verdict无效。
    加入`grep` fallback后machine gate为19/19；必须在正式训练前重跑smoke。
56. SELF/LINEAR/COMMON/POST保持intermediate diagnostics；runner在25/25 training前硬拒绝test，轻微validation
    negative不能淘汰任何arm。
57. Decision=`step7b_prelaunch_pass_step8_authorized`；用户授权seed2021 remote与训练完成后的single formal test；
    confirmation、router、second loss仍false。
58. commit `5d2330e`已remote pull；三卡preflight空闲，修复后的Weather-CPSI/ETTm2-POST resource smokes通过。
59. 25-run training于`17:09:43+08:00`启动，initial training/test=`0/25,0/25`；25/25前test hard-blocked。
60. training与formal test均25/25完成；checkpoint nonmutation、protocol与health通过。
61. CPSI vs ISCF test MSE/MAE=`-2.2128%/-1.6987%`，1/5 datasets；vs A6_FULL亦负，触发material fail。
62. LINEAR是最强control但vs ISCF仅`+0.0217%/+0.0472%` tie且function-class可吸收；不promote。
63. Decision=`cpsi_v1_exact_performance_fail_return_step4_5`；active method none，confirmation/rescue false。
64. post-CPSI Step4/5把ISCF-v0收紧为output-coupling-scope conditional candidate；SAC冻结Q1-WIDE与
    RANDOM-PARTITION两项promotion-blocking controls。
65. SAC Step7B新增10 Q1 + 15 RANDOM jobs，并hash复用35 historical references，形成60 effective runs；
    local prelaunch `18/18`通过。
66. canonical/random在five profiles上parameter/init/Encoder/RNG exact paired，endpoint scopes相同、中间scopes不同；
    Q1 max active-param gap=`0.464638%`。Decision=`step7b_prelaunch_pass_waiting_remote_authorization`。
67. 用户授权SAC training-only；commit `78cbcf4`已remote fast-forward，三卡preflight空闲。
68. Weather-RANDOM seed2021与ETTm2-Q1 seed2023 resource smokes finite/no-OOM。
69. 25-run training于`2026-07-21T18:58:40+08:00`启动，PID=`2383292`，initial training/test=`0/25,0/25`；
    formal test仍false，完成后必须停下等待授权。
70. training于`20:24:32+08:00`完成：25/25 checkpoints、25/25 validation、0/25 test；联合references后
    60/60 runs、240/240 rows与health 15/15通过。
71. validation中ISCF over Q1-WIDE MSE/MAE=`+1.0704%/+0.7538%`；canonical over RANDOM=
    `-0.1823%/-0.3075%`。后者是negative lead但不能作validation-level rejection。
72. Decision=`formal_test_ready_pending_user_authorization`；active method仍none，formal test access仍false。
73. 用户于`2026-07-21`独立回复“授权SAC formal test”；config status=`authorized_prelaunch`、formal test true，
    只开放现有25 checkpoints的一次完整test，禁止retraining/mutation/tuning。
74. 首次formal launch因missing diagnostic bins在test loader前停止；test=0/25、checkpoint unchanged。exact repair只补
    8-bin contract与runner assertion，val smoke后按原matrix重启。
75. repair commit `6bbc3fc`的validation real-checkpoint smoke通过；formal test于`21:04:16`重启并于`21:07:02`
    完成，25/25 new tests、60/60 effective audits、240/240 standard rows与25/25 nonmutation完整。
76. Q1-WIDE gate通过：MSE/MAE=`+0.8496%/+0.5996%`；RANDOM gate失败：
    `-0.1990%/-0.4347%`，仅1/5 datasets、0/4 horizons、1/3 seeds。
77. Decision=`temporal_scope_structure_not_supported_generic_independent_branches_explain`；ISCF降为carrier-only，
    modern baselines false，rollback Step2/4。
78. SPS-v0 hard projection相对identity validation MSE=`-2.3123%`，但scope相对global=`+0.9041%`；hard capacity
    restriction关闭并触发full-rank FRSC设计。
79. FRSC frozen D1.1 scope-a055相对identity=`+0.7997%`、5/5 datasets、4/4 horizons；best global=`+0.8677%`，
    因而只支持conditional Step4–6 gate。
80. FRSC Step7A与Step7B prelaunch通过；20-run validation matrix于commit `9069e87`启动，formal test始终false。
81. 20/20 new runs、25/25 effective audits、100/100 validation rows完整，无numeric pathology；candidate vs identity
    MSE/MAE=`-1.2745%/-0.4184%`，0/4 horizon wins。
82. same-alpha scope vs global MSE=`+0.7215%`，但vs best-global-a045仅`+0.0703%`；canonical vs random仅
    `+0.1781%`且MAE负向。Decision=`frsc_v0_validation_continuation_not_supported_rollback_step4`。
83. exact FRSC-v0关闭且不进formal test；ISCF-v0按用户要求保留为fixed architecture prior/carrier，active method none，
    下一步Step4 source-informed scope-utilization redesign。
84. 用户于2026-07-22扩大范围：ISCF core固定，但允许基于ISCF探索loss、training与architecture extension；旧SAC/
    CPSI/SPS/FRSC negatives保留为diagnosis/control，不再形成“禁止探索objective”的局部边界。
85. existing evidence将缺口定位为coalition credit：ISCF vs A6_FULL `+1.3584%` MSE，oracle headroom median
    `8.5813%`，fusion仅9/15超过best fixed；`equal_skill`是fused + uniform individual target loss。
86. latest source audit确认expert-specific loss、orthogonality/diversity、structural prior、frequency experts、Shapley与
    counterfactual routing均有直接prior；generic loss/router不具独立novelty。
87. working route=`SC-ISCF-SCC-v0`，用ISCF dense fusion的closed-form leave-one-scope-out risk校准existing policy，
    inference不变；当前只`conditional_pass_to_d0_only`，active method仍none。
88. 下一步复用15个existing ISCF probes做D0，必须超过uniform/standalone-error/shuffled controls并确认credit
    cross-seed stability；D0前不实现、不remote train、不访问new formal test。
89. historical NPZ key audit发现只有bin-level policy usage，缺少exact `probe_direct_policy [256,720,5]`；one fused
    equation不能唯一反演five weights，故禁止approximate substitution。
90. 预注册fallback现冻结same 15 checkpoints的validation-only replay；runner只做forward，source checkpoint前后
    SHA256必须一致，new training/test均false。
91. D0 config/analyzer/runner通过local smokes；GPU0/1/2均18 MiB、0% utilization。Decision=
    `d0_validation_replay_prelaunch_pass_remote_forward_authorized`；下一步先commit/push，再remote fast-forward执行。
92. D0 15/15 validation replay完成；median coalition oracle headroom=`17.9766%`，nondegeneracy与shuffle specificity
    均15/15通过，但fixed-label seed topology仅2/5 datasets稳定。Decision=
    `coalition_credit_unresolved_requires_validation_diagnostic_redesign`；不得进入Step7。
93. D0B冻结target-free information-access probe：60/40 blocked rows、fixed ridge、16 horizon-marginal shuffles与
    standalone-credit matched probe。只读取existing validation NPZ；forecast training/method implementation/test均false。
94. nominal 153/103 split因切开multivariate channel group被标记`diagnostic_protocol_fault_predecision`；修正为147/109
    后重跑15 cells，所有gate仍通过：median gain=`1.3727%`，15/15 positive，14/15 shuffle binding，vs standalone
    median=`+0.5143` point且13/15 positive。
95. Decision=`coalition_credit_information_access_supported_return_step5_6`。SCC-v0 exact objective与五臂matched
    attribution已冻结，narrative gate=`pass_to_step7a_matched_validation_only`；Step7A true，remote/test false。
96. Step7A实现exact SCC/shuffled modes、dedicated RNG与five-scope gradient logging；target-visible credit full detach，
    inference unchanged。SCC checker、existing PCC 36/36 regression和20-job dry-run通过。
97. Decision=`step7a_pass_step7b_remote_validation_authorized`；先Weather SCC/SHUFFLED resource smoke，通过后才启动
    20-run validation。formal test/modern baselines保持false。
98. Weather SCC/SHUFFLED resource smoke通过，five scope gradients全部nonzero。commit=`91e466a`，GPU0/1/2均18 MiB、
    0% preflight后启动20-run matrix；首次status=0/20，前三个Weather jobs已进入epoch 1。formal test=false。
99. SCC-v0 25/25 runs、100/100 validation cells完整；vs EQUAL MSE/MAE=`-3.1750%/-1.7742%`，且输给
    FUSED/ARMERR/SHUFFLED。numeric/gradient健康，但median coalition headroom从`+18.0775%`变`-14.9326%`。
100. Decision=`scc_v0_failed_return_step5_reliability_preserving_design`，failure=`intervention_point_wrong`。v0关闭，
    不做seed/lambda rescue。唯一允许RSCC-v1保留EQUAL reliability并附加coalition KL；Step7A true，remote/test false。
101. RSCC Step7A实现EQUAL+coalition与EQUAL+shuffled modes；skill loss与EQUAL逐值相等，existing PCC 36/36与
     15-job dry-run通过。Decision=`rscc_step7a_pass_resource_smoke_authorized`；正式15 runs conditional，test false。
102. RSCC Step9 control attribution fail后，function-level audit确认ARMERR/SHUFFLED learned functions近似，且都把
     policy推到near-uniform；exact coalition route保持closed。
103. latest primary-source audit确认generic forecast-combination shrinkage、stacking weight complexity、TS MoE balancing、
     diversity-aware weights与forecastability routing均已有直接prior；不得把entropy/uniform/temperature loss直接包装为创新。
104. EQUAL为historical而非contemporaneous retrain，H1 inference policy overfit、H2 training co-adaptation与H3 run drift
     仍混杂。`SC-ISCF-PSA-D0`只检验H1，frozen negative不得作direction rejection。
105. PSA-D0冻结15 existing replays、fixed grids、147/109 source-aligned split、LODO global selection和three control
     families。existing artifact analysis=true；new training、test、checkpoint mutation与method implementation=false。
106. PSA-D0 15/15完整：convex-uniform L1/MSE=`-0.2431%/-0.1218%`，1/5 datasets、2/15 runs；marginal与
     temperature controls也macro negative。Decision=`frozen_inference_shrinkage_not_supported`。
107. source-fit nonzero alpha在held-out datasets反转，禁止alpha/dataset/position rescue。该结果只关闭H1 post-hoc
     shrinkage；frozen fairness要求H2 joint-training仍保持unresolved。
108. 下一最小attribution control是five-dataset seed2021 contemporaneous EQUAL retrain，用于区分H2与H3 run drift。
     该D1 design已冻结但launch未获remote authorization；active method none，formal test=false。
109. D1 control design现已冻结，包含recovery ratio、dataset/horizon gates与no-test boundary；implementation、resource
     smoke、five-run training和method promotion全部false，等待明确授权。
110. 用户明确授权D1 Step7A + five-run validation。config/checker/runner/analyzer与code explanation已完成；local
     contracts、5-job dry-run、run-drift/co-adaptation synthetic decisions均通过。
111. Decision=`psa_d1_step7a_pass_proceed_commit_remote_preflight`。remote launch仍须commit-pinned pull、GPU preflight与
     Weather smoke；formal test、confirmation seeds、method promotion false。
112. commit=`f5275a4`已pull；GPU0/1/2 idle preflight与Weather smoke通过，route=0、5/5 gradients nonzero、init hash
     匹配historical/ARMERR/SHUFFLED。
113. five-run validation于`16:00:40+08:00`启动，PID=`3975446`，initial 0/5；Weather/ETTm1/ETTh1 epoch 1 active。
     5/5前不得partial selection，formal test仍false。
114. ETTh1 training/metrics完成后diagnostic evaluator在probe前因missing future bins失败；这是predecision protocol fault，
     没有H2/H3 result或test access。
115. v0.1只增加evaluator contracts与checkpoint-nonmutation validation replay；不改training/checkpoints/gates。等待
     all training process结束后才pull repair并补5 diagnostics。
116. D1最终20/20 effective runs、80/80 cells与5/5 diagnostics完整；new/historical EQUAL checkpoints、metrics、
     fused/arms/policy逐值相同，run drift排除。
117. ARMERR/SHUFFLED vs new EQUAL MSE=`+0.6577%/+0.6557%`，5/5 datasets、4/4 horizons。Decision=
     `joint_training_route_regularization_supported_as_carrier_clue`；只作carrier clue，test=false。
118. 下一唯一diagnostic UPA-D2使用information-free uniform-target KL匹配control schedule/weight，区分broad anchor与
     target variation。Design true；implementation/remote/test false。
119. 用户随后明确降低“broad anchor是否足以解释收益”类独立诊断优先级，并授权`ISCF-BSCA-v1` Step4–7A、
     five-run seed2021 training及5/5完成后的一次frozen formal test；UPA-D2被该method candidate取代。
120. BSCA保持ISCF-v0 architecture/inference/EQUAL skill loss，只加入target/H-free uniform policy KL；25% ramp到0.1。
     Generic balancing不是novelty claim，完整贡献链为policy-mediated scope-gradient allocation与balanced co-adaptation。
121. Formal gate冻结为vs EQUAL test MSE >=+0.3%、MAE >0、3/5 datasets、3/4 horizons、20/20 cells与health/nonmutation
     pass。confirmation seeds、per-cell tuning和lambda search保持false。
122. BSCA seed2021 five trainings与five formal tests全部完成；candidate/EQUAL initialization hashes全配对，test
     checkpoint SHA nonmutation，20/20 cells与internal health完整。
123. Official test MSE/MAE=`+0.3104%/+0.4902%`，15/20 cells、3/5 datasets、3/4 horizons，刚好通过冻结gate；
     validation=`+0.6490%/+0.4492%`。
124. Policy entropy `0.9983 vs 0.7913`、usage max `0.2042 vs 0.2528`，pairwise arm L1 `0.1165 vs 0.1219`，
     oracle headroom `32.56% vs 33.01%`。支持balanced co-adaptation，不支持strong conditional specialization。
125. ETTm2 test mean `-1.7375%`且H192/H336/H720 material negative。Decision=
     `performance_partial_pass_pending_confirmation_seed`；confirmation seeds2022/2023仍未授权，禁止test-informed tuning。
126. 用户随后回复“继续按计划推进实验”，授权seeds2022/2023 confirmation的10个BSCA trainings与10/10完成后的
     single frozen formal-test matrix；FCC的10个same-seed EQUAL artifacts全部复用。
127. Confirmation不改objective/lambda/profile/rank/selector。Three-seed direction gate冻结为macro MSE/MAE positive、
     2/3 seeds、3/5 datasets、3/4 horizons；paper-core gate额外要求MSE >=+0.3%、minimum dataset >-2%、ETTm2 >=-1%。
128. Local config/runner/analyzer/checker、10-job dry-run、10/10-before-test guard与reference completeness通过。
     Decision=`confirmation_step7b_prelaunch_pass_remote_resource_smoke_next`。
129. commit `72e3356`已remote fast-forward；GPU0/1/2 preflight均`18 MiB`、Weather resource smoke通过。10-run
     confirmation于`2026-07-22T19:39:57+08:00`启动，首批Weather2022、ETTm1-2022、Weather2023进入epoch1。
     Decision=`confirmation_step8_training_active_formal_test_guarded`；formal test `0/10`，10/10前禁止执行。
130. 10/10 training与single frozen formal test完成，合并seed2021得到60/60 cells；15/15 candidate runs的artifacts、
     paired initialization、checkpoint nonmutation与internal health通过。
131. Three-seed MSE/MAE=`+0.3541%/+0.3073%`，3/3 seeds、4/5 datasets、4/4 horizons positive；ETTm2=
     `-0.6506%`。Decision=`passed_core_candidate_ready_for_paper_consolidation`。
132. BSCA claim限定为ISCF-specific balanced co-adaptation；post-hoc cluster bootstrap跨0，禁止universal/large-gain claim。
     下一步paper consolidation先于modern baselines；当前新training/test=false。

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

1. 首读FRSC Step9 validation result与本handoff顶部cursor；
2. exact FRSC-v0在validation continuation gate关闭，不进入formal test；
3. ISCF-v0 code、direct policy、equal-skill objective、ranks与profiles保持fixed architecture prior；
4. same-alpha scope topology positive保留为conditional clue，best-global/random/identity failures必须同时保留；
5. active paper-core method为none，rollback Step4 source-informed scope-utilization redesign；
6. 新candidate必须重新完成problem/narrative/design gate，不得把FRSC结果用作per-dataset/alpha tuning；
7. 不运行modern baselines，不做FRSC seed/alpha/loss/router/requested-H rescue；
8. SC-MNB保留为supporting source/control inventory；不恢复CCSF、D17-D21或Contribution 2预设。

## 8. 禁止无损重启时发生的漂移

- 不把A6重新写成已经成立的论文主体；
- 不把SIFF内部健康度写成paper-core pass；
- 不把SIFF-v2历史failure改写为pass，也不直接修改其immutable identity；
- 不把TSAF称为有效或仍active；它已在完整formal test上失败并关闭exact v1；
- 不把independent target-only的`+0.2383%` weak signal直接改名为method或补confirmation；
- 不把CPSI的Step5 theory pass写成method/design/effectiveness pass；exact v1已material fail；
- 不省略CPSI-SELF、CPSI-LINEAR、CPSI-COMMON或POST-SYNTH placement attribution；
- 不把CPSI失败反向写成“independence理论上最优”；
- 不把ISCF现有test-informed control结果直接写成paper-core pass；
- 不在Q1-WIDE与RANDOM-PARTITION attribution前进入modern-baseline performance matrix；
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
2. analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_frsc_step26_20260722/rscc_step9_result_and_rollback.md
3. analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_frsc_step26_20260722/rscc_step8_remote_launch.md
4. analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_frsc_step26_20260722/scc_step9_result_and_rscc_step5_6_design.md
5. analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_frsc_step26_20260722/step2_6_innovation_portfolio_and_scc_gate.md
6. docs/stage-ledgers/stage-c-unified-forecasting-redesign.md
7. docs/paper-mainline.md
8. docs/research-roadmap.md

当前权威状态是：ISCF-v0保持fixed architecture base/carrier。SCC-v0已在完整20-run validation上失败：相对EQUAL MSE=`-3.1750%`，且coalition headroom从`+18.08%`反转为`-14.93%`；failure=`intervention_point_wrong`，不做seed/lambda rescue。

已有证据显示ISCF vs A6_FULL test MSE/MAE=`+1.3584%/+0.9144%`，oracle headroom median=`8.5813%`，但fusion只在9/15 runs超过best fixed arm；代码确认`equal_skill`实际为fused loss + uniform individual arm target loss，没有coalition-specific role signal。

RSCC-v1已完成20/20 effective runs与80/80 validation cells。相对EQUAL MSE/MAE=`+0.5189%/+0.3972%`，5/5 datasets与4/4 horizons通过primary gate；但相对EQUAL-ARMERR/SHUFFLED分别为`-0.1414%/-0.1394%` MSE，且policy-credit Spearman从`0.2052`降到`0.1539`。

Decision=`rscc_v1_control_attribution_fail_close_exact_route`，failure=`capacity_control_explains`。SCC/RSCC exact coalition route关闭，不做formal test、seed、lambda、epsilon、fallback或router rescue。ISCF-v0保持fixed base/carrier，当前active method=none；下一步回Step2/4做source-informed problem/narrative audit，新gate前不实现method、不remote train、不运行modern baselines。

完成后同步更新analysis report、docs/paper-mainline.md、docs/research-roadmap.md和Stage C ledger，执行最小诚实验证，并按AGENTS.md提交、推送。请从专业时序预测研究员角度进行审计，不要为了凑两个contributions而预先设计第二个loss/router。
```
