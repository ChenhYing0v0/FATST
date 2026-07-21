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
35. `configs/stage_c_iscf_v0_carrier.json`；
36. `configs/stage_c_iscf_v0_scope_response_d1_1_confirmation.json`；
37. `docs/stage-ledgers/stage-c-unified-forecasting-redesign.md`；
38. `docs/paper-mainline.md`；
39. `docs/research-roadmap.md`。

若上述文件与更旧的聊天、archive或历史段落冲突，以本文件和三份主线文档顶部的最新cursor为准。

## 1. 当前权威状态

| Field | Content |
| --- | --- |
| `project` | R_2026_FATST |
| `stage` | StageC-UVHF |
| `handoff_date` | 2026-07-21 |
| `source_parent_commit` | `b0b9d7c`（Step7B worktree parent） |
| `current_step` | CPSI Step9/10 material fail；rollback Step4/5 |
| `active_problem` | scope relation evidence需要哪类non-absorbable operator，而非CPSI elementwise product？ |
| `active_method` | none；ISCF-v0 strong carrier retained |
| `method_training_authorized` | false；formal matrix complete；confirmation false |
| `remote_training_authorized` | false pending new Step4-6 gate |
| `next_action` | source-informed Step4/5 problem/function-class redesign；do not implement yet |
| `conditional_next` | new narrative/theory/control gate pass -> only then Step6/7A |
| `rollback` | exact CPSI/SELF/COMMON closed；LINEAR control only；POST not promoted；router/loss blocked |

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
| `ISCF-v0` | frozen strong research carrier；Step4 problem pass | D1.1 relation confirmed；paper method pending；不继承ordered SIFF claim |
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

1. 读取ISCF-v0 carrier、Step4 relation confirmation与Step5 CPSI theory gate；
2. 保持ISCF-v0 frozen carrier，不改写SIFF-v2 ordered-attribution failure；
3. Step6已通过test-first design gate；active_method仍为none，Step7A local implementation已授权；
4. SELF/LINEAR/COMMON/POST均为intermediate diagnostics且进入formal test，不作validation淘汰门；
5. 当前没有remote training、official test execution或confirmation授权，须先完成Step7A/7B；
6. SC-MNB保留为supporting source/control inventory，不执行65-run matrix；
7. 不恢复CCSF、D17-D21、H embedding、router、second loss或Contribution 2预设。

## 8. 禁止无损重启时发生的漂移

- 不把A6重新写成已经成立的论文主体；
- 不把SIFF内部健康度写成paper-core pass；
- 不把SIFF-v2历史failure改写为pass，也不直接修改其immutable identity；
- 不把TSAF称为有效或仍active；它已在完整formal test上失败并关闭exact v1；
- 不把independent target-only的`+0.2383%` weak signal直接改名为method或补confirmation；
- 不把CPSI的Step5 theory pass写成method/design/effectiveness pass；
- 不省略CPSI-SELF、CPSI-LINEAR、CPSI-COMMON或POST-SYNTH placement attribution；
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
2. analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_function_audit_20260721/result_and_step4_handoff.md
3. analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_step4_scope_relation_20260721/step4_result_and_step5_handoff.md
4. analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_step5_common_private_interaction_20260721/step5_theory_and_narrative_gate.md
5. configs/stage_c_iscf_v1_cpsi_step5.json
6. analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v1_cpsi_step6_design_20260721/step6_control_design_and_test_policy.md
7. configs/stage_c_iscf_v1_cpsi_step6.json
8. analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v1_cpsi_step7a_20260721/step7a_implementation_audit.md
9. docs/code-explanation/stage-c-iscf-v1-cpsi-step7a.md
10. docs/stage-ledgers/stage-c-unified-forecasting-redesign.md
11. docs/paper-mainline.md
12. docs/research-roadmap.md

当前权威状态是：用户已将FCC matched independent-scope model冻结为`ISCF-v0` carrier。existing complete test table中，ISCF-v0相对A6_FULL MSE/MAE=`+1.3584%/+0.9144%`，5/5 datasets、4/4 horizons、3/3 seeds正向；这是test-informed carrier evidence，不是新method promotion。SIFF-v2 ordered attribution仍失败且immutable，TSAF/CCSF/D17-D21均关闭，CTD paused。

Step4 D1.1只读disjoint validation histories，不读取targets/test、不训练：15/15 runs超过direction-null与architecture-identical random-init p95，common/private response median=`0.2803/0.7197`，4/5 datasets topology稳定。Decision=`scope_response_relation_confirmed_for_step5_theory`，只支持“pre-synthesis response dependence / late-only fusion”问题，不支持ordered scale、low-rank graph或router。

Step5现已完成。数学审计证明任意fixed linear scope mixing均可被ISCF独立affine mode maps精确吸收，因此Cross-Stitch matrix、linear common/private和fixed graph不形成新function class；plain peer MLP也因generic extra projection/depth归因不净而未进入working method。

working candidate为`ISCF-v1-CPSI`：对`[B,C,S,D,K]` modes计算scope mean与zero-sum deviation，用shared、无bias的common/private bottlenecks及elementwise product生成pre-synthesis native mode interaction。它permutation-equivariant，不读取requested H/order/target；`W_o=0`精确包含ISCF-v0；common或private任一路缺失时message为零。它不改变Bayes information set，只允许finite-capacity inductive-bias解释。

最新external primary-source audit确认Deep Sets/Set Transformer/Cross-Stitch/MoLE/DMSC v5/TimeExpert已覆盖generic set、shared-private、expert mixture及multi-scale coordination primitives。因此允许的paper boundary只能是`future-output coupling scopes -> controlled common/private response evidence -> linear reparameterization boundary -> pre-synthesis multiplicative interaction -> matched attribution`完整链，不能claim首次multi-scale/expert interaction。

用户要求SELF/LINEAR/COMMON/POST作为intermediate diagnostics，不因轻微负向在test前关闭mechanism。Step6现已完成：前三者exact `3Lr` matching；POST直接作用于forecast arms，derived rank使total-model parameter gap小于0.041%；global `r=32`冻结。validation只作selector/health，全部protocol-valid arms进入一次完整official-test audit。

formal matrix为25 new trainings，加ISCF-v0/A6_FULL 10 historical references，共35 effective runs、140 MSE与140 MAE cells。CPSI vs ISCF macro MSE `[-0.5%,+0.3%)`定义为inconclusive，不方向级拒绝；initial support要求`>=+0.3%`、3/5 datasets、10/20 cells与MAE guard。controls使用`±0.3%` attribution tie band。

Step7A production implementation现已完成。五个readout modes在parent ISCF初始化后创建interaction matrices；local checker在conda `r2026-fsa`中81/81通过。parent/output/arms/policy morph gap均为0，scope equivariance、private-absent zero message、five real `TimeAlign.Model` forwards、CLI、profile parameter formulas与two-stage gradient opening全部通过。没有training、validation comparison或test access。

Decision=`step7a_local_pass_step7b_prelaunch_next`。`ISCF-v1-CPSI`现为implementation-ready/effectiveness-pending active method。下一步只允许构建并审核25-run manifest、training/test separation、historical ISCF/A6 hashes、internal diagnostics与frozen analyzer。remote training与formal test execution仍须等待Step7B、commit-pinned pull、GPU preflight和25/25 training完成；confirmation、router、second loss均未授权。

完成后同步更新analysis report、docs/paper-mainline.md、docs/research-roadmap.md和Stage C ledger，执行最小诚实验证，并按AGENTS.md提交、推送。请从专业时序预测研究员角度进行审计，不要为了凑两个contributions而预先设计第二个loss/router。
```
