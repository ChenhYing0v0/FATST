# Paper Mainline

## Current Position

| Field | Content |
| --- | --- |
| `paper_target` | 高水平 SCI 期刊时间序列预测论文 |
| `working_title` | TBD；provisional architecture base=`ISCF` |
| `current_stage` | `StageC-UVHF` active；StageB 已归档 |
| `current_11_step` | ISCF-BSCA-v1 Step9/10 complete；single-seed performance partial pass |
| `source_evidence` | A6-LBF-r256 historical/source-faithful performance |
| `mechanism_control` | same-run end-to-end A6；frozen A6仅作reference/conditional diagnostic |
| `test_reference` | 3 datasets × 3 seeds × 8 horizons，72/72 complete |
| `future_validation_suite` | ETTh1/ETTh2/ETTm1/ETTm2/Weather；five natural profiles frozen |
| `active_ledger` | `docs/stage-ledgers/stage-c-unified-forecasting-redesign.md` |
| `restart_handoff` | `docs/stage-ledgers/stage-c-post-d21-d22-restart-handoff-20260720.md` |
| `paper_core_status` | active candidate=`ISCF-BSCA-v1`；`performance_partial_pass_pending_confirmation_seed` |

[ISCF-BSCA-v1 Confirmation Prelaunch, 2026-07-22] 用户回复“继续按计划推进实验”，授权上一handoff中冻结的
seeds2022/2023 confirmation。Matrix为10个new BSCA trainings、10个reused same-seed EQUAL controls；合并seed2021后
形成three-seed 60-cell official-test surface。Objective、lambda=0.1、25% ramp、profiles、ranks、selector与gates均不调。

Direction gate：macro MSE/MAE >0、2/3 seeds、3/5 datasets、3/4 horizons。Paper-core gate额外要求macro MSE
>=+0.3%、minimum dataset >-2%、ETTm2 >=-1%与full health/nonmutation。Local 10-job dry-run、reference 10/10、
test authorization与10/10-before-test guard通过。Decision=`confirmation_step7b_prelaunch_pass_remote_resource_smoke_next`。

[ISCF-BSCA-v1 Step9/10, 2026-07-22] 5/5 trainings、5/5 frozen formal tests、20/20 standard-horizon cells
完整；candidate与EQUAL五个datasets的all initialization hashes exact paired，test checkpoint SHA before/after不变。
BSCA vs EQUAL test MSE/MAE=`+0.3104%/+0.4902%`，15/20 cells、3/5 datasets、3/4 horizons，全部冻结
performance gates刚好通过。Validation为`+0.6490%/+0.4492%`。

内部机制与设计一致：policy entropy `0.9983 vs 0.7913`、marginal max usage `0.2042 vs 0.2528`；arms未
collapse，但pairwise L1从0.1219降至0.1165，oracle headroom `32.56% vs 33.01%`。因此支持的是balanced
scope co-adaptation/gradient access，不是更强conditional specialization。Weather/ETTm1/ETTh1 positive；ETTh2 tie，
ETTm2 test `-1.7375%`且存在validation/test reversal，必须完整报告。

Decision=`performance_partial_pass_pending_confirmation_seed`。当前只完成seed2021，不能升级passed core或声称robustness；
confirmation seeds 2022/2023 尚未授权。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_bsca_v1_step9_10_20260722/step9_10_result_and_confirmation_handoff.md`。

[ISCF-BSCA-v1 Step4–7A, 2026-07-22] 用户将低成本诊断优先级改为 outcome-first paper route，并明确授权
Step4–7A、five-run seed2021 training 与 5/5 完成后的一次 frozen formal test。UPA-D2 diagnostic 被
`ISCF-BSCA-v1 (Balanced Scope Co-Adaptation)`取代，不再单独执行。

BSCA 保留 ISCF-v0 exact architecture、EQUAL arm-skill loss 与 inference graph，只加入 train-only、target/H-free
`KL(uniform || policy)`；weight 在前25% progress从0 ramp至0.1。Narrative novelty限定为
`dense temporal-scope outputs -> policy-mediated fused-gradient allocation -> balanced co-adaptation`完整链，不声称
generic load balancing 首创。Matched no-mechanism control固定为ISCF-EQUAL。

Frozen formal gate：相对EQUAL test macro MSE `>=+0.3%`、MAE `>0`、至少3/5 datasets与3/4 horizons MSE positive，
20/20 cells完整且checkpoint/non-numeric/internal health通过。confirmation seeds仍未授权。Decision=
`bsca_step4_6_conditional_pass_step7a_prelaunch`。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_bsca_v1_step46_20260722/step4_6_narrative_design_and_test_gate.md`。

[ISCF PSA-D1 Result, 2026-07-22] 5/5 new、20/20 effective runs与80/80 validation cells完整，5/5 diagnostic
replays通过checkpoint SHA nonmutation，official test access=0。Contemporaneous EQUAL与historical EQUAL的checkpoint、
metrics、fused、arms、policy在5/5 datasets逐值完全相同；MSE/MAE=`0/0`，H3 run drift关闭。

ARMERR/SHUFFLED相对new EQUAL MSE/MAE分别=`+0.6577%/+0.4476%`与`+0.6557%/+0.4544%`，
均17/20 cells、5/5 datasets、4/4 horizons positive。Decision=
`joint_training_route_regularization_supported_as_carrier_clue`：结合D0 negative，公共收益属于training-time arm-policy
co-adaptation，而非post-hoc smoothing；但target semantics与generic balancing novelty仍未建立。

下一唯一diagnostic=`SC-ISCF-UPA-D2`：information-free uniform policy anchor，检验broad train-time anchor是否足以复现
controls。Design已冻结，implementation/remote/test false。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_rscc_step24_20260722/psa_d1_result_and_step4_rollback.md`。

[ISCF PSA-D1 v0.1 Protocol Repair, 2026-07-22] ETTh1 training/validation metrics完成后，diagnostic evaluator因config
缺少`diagnostic_protocol.future_bins`在probe forward前报`KeyError`。Decision=
`diagnostic_protocol_fault_predecision_repair_frozen_training_continues`；没有H2/H3 result或test access。

v0.1只补evaluator training contract、eight future bins与validation-replay authorization；training matrix/arguments/
checkpoints/gates不变。其余training继续，结束前不remote pull；之后只对5 checkpoints做SHA-nonmutation validation
replay。详见`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_rscc_step24_20260722/psa_d1_v01_diagnostic_protocol_repair.md`。

[ISCF PSA-D1 Step8 Launch, 2026-07-22] commit=`f5275a4`完成remote fast-forward；GPU0/1/2 preflight均18 MiB、
0%，无compute process。Weather smoke确认EQUAL route weight/loss=0、five scope gradients nonzero，且initialization
hash与historical EQUAL/ARMERR/SHUFFLED完全相同。

5-run validation matrix于`16:00:40+08:00`启动，PID=`3975446`；Weather/ETTm1/ETTh1分别在GPU0/1/2进入epoch 1，
ETTh2/ETTm2 queued。Decision=`psa_d1_five_run_validation_training_active_formal_test_disabled`；5/5前不读取partial
metrics，formal test=false。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_rscc_step24_20260722/psa_d1_step8_remote_launch.md`。

[ISCF PSA-D1 Step7A, 2026-07-22] 用户明确授权Step7A与five-run validation training。one-arm config、existing-runner
wrapper、source/objective checker与20-run/80-cell H2/H3 analyzer已实现。Local contracts确认training/evaluator相对RSCC
launch commit无semantic diff，EQUAL route loss/weight严格为0，five scope gradients可达；dry-run=5 jobs，analyzer的
run-drift/co-adaptation synthetic branches均通过。

Decision=`psa_d1_step7a_pass_proceed_commit_remote_preflight`。下一步commit/push后做remote GPU preflight与Weather
resource smoke；通过后才启动5 runs。formal test、confirmation seeds与method promotion保持false。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_rscc_step24_20260722/psa_d1_step7a_prelaunch.md`。

[ISCF PSA-D0 Result, 2026-07-22] 15/15 existing EQUAL validation replays的LODO frozen-policy diagnostic完整。
Convex-uniform shrinkage macro L1/MSE=`-0.2431%/-0.1218%`，仅1/5 datasets、2/15 runs joint-positive；
selected alpha为`[.3,0,.2,.5,.75]`，4/5 nonzero却在ETTh1/ETTm2/Weather held-out反转。scope-marginal与
temperature controls同样macro negative。

Decision=`frozen_inference_shrinkage_not_supported`：关闭post-hoc uniform/temperature rescue，H1
`inference_weight_overfit`不受支持。failure=`frozen_probe_negative_joint_training_unresolved`；该结果不能拒绝
joint-training co-adaptation。下一识别节点`SC-ISCF-PSA-D1`已冻结为five-dataset seed2021 contemporaneous EQUAL
control，用于区分H2 co-adaptation与H3 run drift；implementation/remote training/test尚未授权。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_rscc_step24_20260722/psa_d0_result_and_rollback.md`
与`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_rscc_step24_20260722/psa_d1_contemporaneous_equal_control_design.md`。

[ISCF Post-RSCC Step2/4 / PSA-D0, 2026-07-22] function-level audit确认ARMERR与SHUFFLED不仅validation
MSE彼此只差`0.0020%`，其seed2021 fused relative L1仅`0.00138--0.00462`、policy mean L1仅
`0.00254--0.00830`，且最终policy entropy均约`0.986--0.998`。二者共同实现near-uniform policy，却不共享正确
scope-credit binding；这是finite policy flexibility的problem clue，而非新mechanism结论。

关键confound是EQUAL为historical reference，并未在本轮contemporaneously retrain。现冻结`SC-ISCF-PSA-D0`：复用
15个EQUAL validation replays，以LODO选择global convex-shrinkage alpha，在source-sample-aligned 147/109 split上
区分`inference_weight_overfit`、joint-training co-adaptation与run drift。该probe不训练、不访问test；generic entropy、
temperature、uniform KL与forecast-combination shrinkage已有强prior，不能直接作为Contribution 2。Decision=
`policy_shrinkage_problem_unresolved_proceed_d0_diagnostic_only`。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_rscc_step24_20260722/step2_4_policy_shrinkage_problem_audit.md`。

[ISCF-RSCC Step9 Result, 2026-07-22] 15/15 new、20/20 effective runs与80/80 validation cells完整，
protocol/init/invariants/non-test audit全部通过。RSCC vs EQUAL MSE/MAE=`+0.5189%/+0.3972%`，15/20 cells、
5/5 datasets、4/4 horizons，validation primary gate通过；但RSCC分别输给EQUAL-ARMERR与SHUFFLED
`-0.1414%/-0.1394%` MSE，均只3/20 cells、1/5 datasets、1/4 horizons。

ARMERR与SHUFFLED彼此MSE仅差`+0.0020%`，且都比EQUAL约`+0.656%`；RSCC policy-credit Spearman
`0.1539`也低于EQUAL `0.2052`。headroom保持`+18.2940%`且gradient/usage健康，故不是numeric或reliability
failure，而是no-binding matched controls解释收益。Decision=`rscc_v1_control_attribution_fail_close_exact_route`，
failure=`capacity_control_explains`。不开formal test，不做seed/lambda/router rescue；ISCF fixed base保留，回Step2/4。
详见`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_frsc_step26_20260722/rscc_step9_result_and_rollback.md`。

[ISCF-RSCC Step8 Launch, 2026-07-22] Weather RSCC/SHUFFLED resource smoke通过：两臂共享initialization
hash，`skill_loss=0.7632371485`逐值相等，route loss非零，five scope gradients均finite/nonzero，且无
Traceback/OOM/NaN/Inf。commit=`020eea3`已remote fast-forward；GPU0/1/2 preflight均18 MiB、0%。

冻结的15-run validation matrix于`14:12:34+08:00`启动，首批Weather RSCC/SHUFFLED/EQUAL-ARMERR分别位于
GPU0/1/2；output=`/home/yingch/exp_outputs/r-2026-fatst/stage_c_iscf_rscc_v1_step7b`，runner PID=
`3836251`。Decision=`rscc_step8_validation_training_active_formal_test_disabled`；只在15/15完整后统一Step9，
不得读取partial favorable cells。formal test、confirmation seeds与modern baselines仍false。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_frsc_step26_20260722/rscc_step8_remote_launch.md`。

[ISCF-RSCC Step7A, 2026-07-22] reliability-preserving coalition/shuffled modes实现完成；RSCC skill loss与EQUAL
逐值相等，coalition credit/route boundary沿用v0。15-job config与runner、RSCC checker、existing PCC 36/36 regression通过。
Decision=`rscc_step7a_pass_resource_smoke_authorized`；只先允许Weather smoke，正式15 runs conditional，test=false。

[ISCF-SCC Step9 / RSCC Rollback, 2026-07-22] 25/25 runs与100/100 validation cells完整。SCC-v0 vs EQUAL
MSE/MAE=`-3.1750%/-1.7742%`，并分别以MSE `-0.0150%/-0.1663%/-0.0428%`输给FUSED/ARMERR/
SHUFFLED。numeric/gradient健康，但median coalition headroom从EQUAL的`+18.0775%`反转为SCC的`-14.9326%`。
Decision=`intervention_point_wrong`：关闭v0，禁止seed/lambda rescue。

唯一允许successor=`SC-ISCF-RSCC-v1`：保留EQUAL reliability loss，只附加exact coalition policy KL，并冻结
EQUAL/ARMERR/SHUFFLED controls。narrative gate仅通过到Step7A；remote/test false。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_frsc_step26_20260722/scc_step9_result_and_rscc_step5_6_design.md`。

[ISCF-SCC Step8 Launch, 2026-07-22] Weather SCC/SHUFFLED resource smoke通过，five scope gradients均nonzero；
source commit=`91e466a`。GPU0/1/2 preflight均18 MiB、0%，20-run seed2021 matched validation matrix已启动，output=
`/home/yingch/exp_outputs/r-2026-fatst/stage_c_iscf_scc_v0_step7b`。current=`running`；partial result不得选择，
formal test=false。详见`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_frsc_step26_20260722/scc_step8_remote_launch.md`。

[ISCF-SCC Step7A, 2026-07-22] exact coalition/shuffled objective、dedicated RNG、per-scope gradient logging、20-run
validation config与runner完成。target-visible credit path完整detach，route KL只校准policy；model forward/inference未变。
SCC checker、existing PCC 36/36 regression、20-job dry-run与syntax/compile checks通过。Decision=
`step7a_pass_step7b_remote_validation_authorized`；先resource smoke再启动20 runs，formal test仍false。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_frsc_step26_20260722/step7a_scc_implementation_and_remote_gate.md`。

[ISCF-SCC D0B / Step5–6, 2026-07-22] corrected 147/109 source-sample-aligned held-out probe完整通过：median
target-free L1 gain=`1.3727%`，15/15 positive，14/15超过shuffle三指标p95，vs standalone median=`+0.5143`
percentage point且13/15 positive。Decision=`coalition_credit_information_access_supported_return_step5_6`。

SCC-v0现冻结为ISCF-native train-only coalition policy calibration；inference graph不变。narrative gate=
`pass_to_step7a_matched_validation_only`，只授权exact loss和contract tests；20-run validation training须在Step7A及launch
record通过后授权，formal test仍false。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_frsc_step26_20260722/d0b_result_and_step5_6_scc_design.md`。

[ISCF-SCC D0 Result / D0B, 2026-07-22] 15-run frozen validation replay完成。D0的headroom、nondegeneracy、
standalone-distinction OR gate与shuffle specificity通过，但fixed-label cross-seed topology仅2/5 datasets稳定，因此
decision=`coalition_credit_unresolved_requires_validation_diagnostic_redesign`，不得进入Step7。该失败不等于
`hypothesis_false`：dynamic coordinate credit未必要求independent seeds的固定scope label同构。

D0B现已冻结为held-out information-access diagnostic：只用arms/policy/position拟合低容量ridge，验证credit是否存在
target-free可预测分量，并与horizon-marginal shuffle及standalone-credit probe匹配比较。active method仍none；forecast
training、method implementation和formal test均false。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_frsc_step26_20260722/d0_result_and_d0b_information_access_plan.md`。

[ISCF-SCC D0 Prelaunch, 2026-07-22] historical 15-run NPZ audit确认已有arms/fused/targets与bin-level policy usage，
但缺少closed-form leave-one-scope-out所需的exact `probe_direct_policy [256,720,5]`。policy反演欠定，禁止用
least-squares或bin averages替代。按预注册fallback，现冻结same 15 checkpoints的validation-only replay；只做forward并
保存exact policy，source checkpoint执行SHA256 nonmutation。

D0 analyzer与runner已完成local preflight：`py_compile`、synthetic analyzer/evaluator smokes、JSON parse、`bash -n`和
diff checks通过。GPU0/1/2 preflight均18 MiB、0% utilization。Decision=
`d0_validation_replay_prelaunch_pass_remote_forward_authorized`；remote replay需先commit/push/fast-forward，
new training、formal test、method implementation仍false。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_frsc_step26_20260722/d0_prelaunch_and_validation_replay.md`。

[ISCF Post-FRSC Step2–6 Portfolio, 2026-07-22] 用户扩大研究范围：ISCF core保持固定，但允许围绕它探索
loss、training和architecture coupling，以补充连贯创新并提升official-test性能。已有证据把首要缺口定位为
coalition credit assignment，而不是缺少arm diversity：ISCF vs A6_FULL MSE/MAE=`+1.3584%/+0.9144%`，
oracle headroom median=`8.5813%`，但fusion只在9/15 runs超过best fixed arm；代码同时确认`equal_skill`以
uniform individual L1 target loss把所有arms拉向同一conditional-median target。

primary working route=`SC-ISCF-SCC-v0 — Scope Coalition Credit`。它保留ISCF inference architecture，用dense
fusion的closed-form leave-one-scope-out risk构造train-only coalition credit，校准既有direct policy，并以fused-only
替代uniform individual arm supervision。TIGER、Shapley-MoE、Expert Loss Integration、AME-TS、MoHETS与最新
specialization objectives已覆盖counterfactual routing/expert loss/structural prior/diversity primitives，故novelty只能位于
`future-output coupling scopes -> exact coalition risk -> matched E2E coordination -> unified-horizon gain`完整链。

Decision=`scc_problem_diagnostic_proposed_active_method_none`。narrative gate只
`conditional_pass_to_d0_only`；下一步复用15个existing ISCF artifacts审计coalition credit是否stable、nondegenerate且
不同于standalone arm error。D0前不实现、不remote train、不访问new formal test。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_frsc_step26_20260722/step2_6_innovation_portfolio_and_scc_gate.md`。

[FRSC Step9 Validation Result, 2026-07-22] 20/20 new runs、25/25 effective audits、100/100 validation rows完整，
20/20 invariants通过且无numeric pathology/test access。scope-a055相对identity MSE/MAE=`-1.2745%/-0.4184%`，
7/20与10/20 cells，2/5 datasets、0/4 horizons，primary continuation gate失败。candidate相对same-alpha global为
`+0.7215%` MSE、19/20、5/5、4/4，说明scope topology有条件作用；但相对best-global-a045仅`+0.0703%`，
低于`+0.1%` gate，且相对random仅`+0.1781%` MSE、MAE `-0.0330%`。

Decision=`frsc_v0_validation_continuation_not_supported_rollback_step4`。不开formal test，不做seed/alpha/loss/router rescue；
关闭exact FRSC-v0但保留ISCF architecture prior。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_frsc_step9_validation_20260722/step9_validation_result_and_rollback.md`。

[FRSC Step8 Launch, 2026-07-22] remote已fast-forward到`9069e87`；GPU0/1/2 preflight均18 MiB、0% utilization、
无compute process。Weather candidate/random resource smokes确认alpha=.55、minimum eigenvalue=.45、full-rank且无
Traceback/OOM/NaN/Inf。20-run validation matrix于`10:41:20+08:00`启动，runner PID=`3559159`；初始
`validation=0/20`，三个Weather jobs均进入epoch 1。预计约1.5–2小时。formal test仍false。

Decision=`frsc_step8_training_active_formal_test_disabled`。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_frsc_step8_remote_20260722/remote_authorization_and_launch.md`。

[FRSC Step7B Prelaunch, 2026-07-22] 4个new arms × five datasets × seed2021的20-run validation matrix已冻结；
复用5个历史identity checkpoints后，完整analysis surface为25 runs/100 standard-horizon rows。candidate必须同时超过
identity、same-alpha global、best-tuned global和random-binding controls。local gate `37/37`通过，包含full-rank、paired
initialization、runner/test boundary和analyzer contracts。用户已授权推进remote training；下一动作是commit-pinned remote
pull、GPU/process audit、Weather candidate/random resource smoke和正式launch。formal test、confirmation seeds、modern
baselines、new loss/router/requested-H保持false。

Decision=`frsc_step8_remote_validation_authorized_formal_test_disabled`。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_frsc_step7b_prelaunch_20260722/prelaunch_report.md`。

[FRSC Step7A, 2026-07-22] 新readout=`iscf-full-rank-scope-conditioning`已本地实现。alpha=0与ISCF parent exact
gap为0；candidate/identity/global/random parameter hash一致；alpha .55 minimum eigenvalue=.45；five scope gradients全部
finite/nonzero；production model/CLI/full-prefix contracts通过。Step7A只建立code-theory consistency，不含method effectiveness。
下一步已由Step7B prelaunch与用户remote authorization更新。

[SPS Step9 and FRSC Step4–6, 2026-07-22] SPS 20/20 validation matrix完整且无numeric pathology，但scope-canonical相对
identity MSE/MAE=`-2.3123%/-1.0937%`；其相对global `+0.9041%/+0.8461%`说明local geometry有条件价值，exact
failure归因于hard capacity restriction。BSC frozen readout 20/20 MSE cells负向，关闭exact diagnostic。FRSC把hard projector改为
invertible $Q_s=P_s+(1-\alpha)(I-P_s)$；frozen D1.1 canonical在alpha .55为`+0.7997%` MSE、5/5 datasets、
4/4 horizons，random为`-8.9750%`，但best global为`+0.8677%`。因此FRSC仅通过conditional narrative/design gate，
E2E必须超过identity、random与best-tuned global；Step7A、remote training与formal test均尚未授权。

[ISCF-SPS Step8 Remote Authorization, 2026-07-22] 用户明确授权并启动冻结20-run validation matrix。授权只覆盖
scope/identity/global/random × five datasets × seed2021的from-scratch training；formal test、confirmation seeds和modern
baselines不包含在内。下一动作是commit-pinned remote pull、GPU preflight、two-arm resource smoke，通过后启动正式matrix。

Decision=`step8_remote_validation_authorized_formal_test_disabled`。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_sps_step8_remote_20260722/remote_authorization_and_launch.md`。

[ISCF-SPS Step8 Launch, 2026-07-22] remote已fast-forward到`48afd12`；GPU0/1/2 preflight均为18 MiB、0%
utilization且无compute process。Weather scope/identity resource smokes finite且无OOM。20-run matrix于
`00:17:31+08:00`启动，runner PID=`2787170`；首批Weather scope/identity/global jobs进入epoch 1，初始
`validation=0/20`。formal test仍false。

Decision=`step8_training_active_formal_test_disabled`。training期间冻结repo/config/gates，完成后先做validation audit。

[ISCF-SPS Step7B Prelaunch, 2026-07-21] validation-first matrix已冻结为scope/identity/global/random四arms ×
ETTh1/ETTh2/ETTm1/ETTm2/Weather × seed2021，共20个from-scratch matched runs、80个standard-horizon validation cells。
所有arms共享natural profiles、dataset-matched rank、direct policy、equal-skill objective与four-horizon checkpoint selector。

evaluator现显式保存raw/projected/removed arm tensors、direct policy、arm predictions、oracle与bin metrics；analyzer预注册
effectiveness、global smoothing attribution、random binding attribution和specialization health。random只是attribution control，
不得单独方向级拒绝ISCF。local prelaunch `19/19`通过，runner在未授权时exit 3且硬拒绝test split。

Decision=`step7b_prelaunch_pass_wait_remote_authorization`。目前没有新training/validation/test evidence；remote training、
formal test、confirmation seeds和modern baselines仍false。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_sps_step7b_prelaunch_20260721/prelaunch_report.md`。

[ISCF-SPS Step4–7A, 2026-07-21] 用户将ISCF multi-scope architecture固定为本项目的design prior，并要求从
arm specialization/utilization继续研究。SAC negative完整保留，但角色从stop decision改为design diagnosis：ISCF超过
Q1-WIDE说明independent maps有效，而canonical不超过random、fusion仅9/15超过best fixed arm，说明现有shared
unrestricted target synthesis没有充分兑现scope geometry。

新candidate=`SC-ISCF-SPS-v0 — Scope-Projected Synthesis`。它保留五个independent maps、scope groups、direct
policy、single objective和full-T crop，在每个raw arm进入fusion前施加scope-native orthonormal local-DCT projector。
projector不仅约束forward resolution，也以$P_s^\top=P_s$过滤该scope map收到的error gradient；不增加trainable
parameters、second loss、router或requested-H input。identity/global/random controls已冻结，NHITS/N-BEATS/TimeMixer/
FreqMoE及2025–2026 expert-specialization literature限定了claim只能位于完整task-specific chain。

Step7A local contract通过：identity-parent max gap=`8.34e-7`，prefix gap=`0`，basis orthonormal/projector
idempotence errors=`3.22e-15/1.53e-16`，five scope gradient norms全部finite/nonzero，canonical/random/global
outputs可辨，production model `[1,720,2]`与CLI contract通过。Decision=
`conditional_pass_as_scope_utilization_architecture_step7a_complete`。下一步只冻结validation-first Step7B matrix；remote
training、formal test与modern baselines仍未授权。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_sps_step46_20260721/step4_6_design_and_step7a_audit.md`。

[ISCF-v0 SAC Step9/10 Result, 2026-07-21] 25/25 new formal tests与60/60 effective audits完整，240/240
standard-horizon rows、25/25 checkpoint nonmutation和15/15 internal-health checks通过。首次launch的missing-bin
preflight gap已通过8-bin repair与validation real-checkpoint smoke解决，正式test access count为1。

ISCF-v0 over Q1-WIDE MSE/MAE=`+0.8496%/+0.5996%`，5/5 datasets、4/4 horizons、2/3 seeds，primary
gate通过；independent maps的收益不是shared-width capacity。canonical over RANDOM-PARTITION为
`-0.1990%/-0.4347%`，仅1/5 datasets、0/4 horizons、1/3 seeds，所有primary guards失败。ISCF相对A6_FULL
仍为`+1.3584%/+0.9144%`，说明performance carrier强，但不能建立temporal-scope mechanism claim。

Decision=`temporal_scope_structure_not_supported_generic_independent_branches_explain`。ISCF-v0降为strong
carrier/control，exact paperization route关闭；不做rank、seed、partition、loss、router或requested-H rescue，modern
baselines不启动。rollback到Step2/4 contribution-boundary consolidation。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_sac_step9_10_20260721/step9_10_result_and_rollback.md`。

[ISCF-v0 SAC Step9 Formal-test Authorization, 2026-07-21] 用户独立回复“授权SAC formal test”。授权只覆盖
SC1-ISCF-v0-SAC-v1冻结的25个new checkpoints的一次official-test audit；candidate、controls、ranks、partition seed、
objective、checkpoint rule、metrics与gates不变，禁止重训与checkpoint mutation。

Decision=`step9_formal_test_authorized`。下一步commit/push、remote fast-forward、checkpoint/hash与GPU preflight后，
只运行`FORMAL_TEST_ONLY=1`；完成后联合35个historical references执行60-run/240-cell Step9/10 analyzer。

首次launch在任何test loader创建前因SAC config缺失evaluator所需的`diagnostic_protocol.future_bins`而停止；
training/test仍为`25/25,0/25`，checkpoint未变。该事件标记为`exact_protocol_preflight_gap`。repair仅补入8个
diagnostic bins并增加runner静态边界断言；不改变candidate、forecast、metrics或gates。validation-split真实checkpoint
smoke通过前不得重新launch。

[ISCF-v0 SAC Step8 Validation Audit, 2026-07-21] frozen new training已`25/25`完成，remote checkpoints与
validation artifacts均`25/25`，formal-test artifacts为`0/25`。联合35个hashed historical references后，validation
matrix为60/60 runs、240/240 standard-horizon rows；protocol audit 60/60与internal health 15/15通过，无
Traceback、OOM、NaN或Inf。

validation-only observation中，ISCF over Q1-WIDE MSE/MAE为`+1.0704%/+0.7538%`，4/5 datasets、4/4
horizons、3/3 seeds MSE正向；canonical over RANDOM-PARTITION为`-0.1823%/-0.3075%`，仅2/5 datasets、
1/4 horizons、1/3 seeds MSE正向。后者是必须保留的negative lead，但validation不允许通过或拒绝机制，尤其项目已有
validation→test reversal先例。

Decision=`formal_test_ready_pending_user_authorization`。这只是artifact/protocol readiness，不是paper-core或
attribution pass。formal test仍false；获得独立授权后才可更新config并运行一次冻结25-run `FORMAL_TEST_ONLY=1`。
详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_sac_step8_remote_20260721/validation_artifact_audit_and_test_handoff.md`。

[ISCF-v0 SAC Step8 Authorization, 2026-07-21] 用户明确授权继续SAC remote training。授权范围严格限定为
Step7B冻结的25 new runs；formal test仍false，25/25 training完成后必须停在validation artifacts并等待独立授权。
下一动作是commit-pinned remote pull、三卡`nvidia-smi`与Weather-RANDOM/ETTm2-Q1 dual resource smoke；smoke
finite/no-OOM后才启动正式matrix。candidate、gates、partition seed、rank、loss与policy均不变。

Decision=`step8_remote_training_authorized_formal_test_pending`。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_sac_step8_remote_20260721/remote_authorization_and_launch.md`。

[ISCF-v0 SAC Step8 Launch, 2026-07-21] commit `78cbcf4`已remote fast-forward；GPU0/1/2 preflight均为
18 MiB、0% utilization且无compute process。Weather-RANDOM seed2021与ETTm2-Q1 seed2023 resource smokes
finite/no-OOM。25-run training于`18:58:40+08:00`启动，supervisor PID=`2383292`；首批三个Weather jobs进入
epoch 1，初始training/test=`0/25,0/25`。formal-test execution mode为0且config authorization=false。

Decision=`step8_training_active_formal_test_not_authorized`。训练期间不pull、不改matrix/gates；25/25后停止并等待test授权。

[ISCF-v0 SAC Step7B, 2026-07-21] candidate code未修改。SAC runner、three-source analyzer与frozen
manifest已实现，local prelaunch `18/18`通过。新矩阵为Q1-WIDE seeds2022/2023的10 runs与
RANDOM-PARTITION three seeds的15 runs；另以两份SHA256-frozen source audits复用ISCF-v0/A6_FULL各15 runs
及Q1 seed2021五runs，形成60 effective runs、240 standard-horizon metric rows。

canonical/random在five profiles上的active parameters、PCSD/Encoder initialization和post-construction RNG均匹配；
scope1/720 partitions相同、48/144/360不同，输出finite且非恒等。Q1 signed active-param gaps复核一致，最大绝对值
`0.464638%`。runner将validation-only training和formal test分离，25/25 training前拒绝test，且当前未授权normal
launch会以exit code 3终止。

Decision=`step7b_prelaunch_pass_waiting_remote_authorization`。ISCF仍未promote；remote training、formal test、modern
baselines、router与second loss均false。获得明确授权后才做commit-pinned pull、GPU preflight、dual resource smoke与
25-run training。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_sac_step7b_prelaunch_20260721/prelaunch_report.md`。

[ISCF-v0 Post-CPSI Step4/5 Gate, 2026-07-21] CPSI的material failure没有被反向包装成“independence最优”。研究改为审计
ISCF-v0本身能否成为单一architecture contribution：五个scope不是input resolutions或requested horizons，而是规定
哪些future coordinates在nonlinear synthesis前共享latent state；每个scope拥有独立`history -> mode` map，但仍共享
Encoder、target synthesis与final policy。

现有three-seed carrier evidence相对A6_FULL MSE/MAE为`+1.3584%/+0.9144%`，5/5 datasets、4/4 horizons、
3/3 seeds正向；D1.1确认scope-specific responses非随机冗余；ordered SIFF与CPSI interaction均未超过ISCF。然而
ISCF最初是control且结果已test-informed，仍不能直接promote。最新TimeMixer、FreqMoE、MAFS、HMformer与M²FMoE
已覆盖multi-scale predictors、independent experts、sub-task agents和multi-branch fusion，故generic“多分支/多尺度”
不是贡献。

Decision=`conditional_pass_as_output_coupling_scope_architecture_pending_sac`。下一步Scope Attribution Confirmation只回答
两项阻塞归因：ISCF是否超过active-param gap不超过`0.4646%`的near-matched `Q1-WIDE` shared map，以及是否超过same-parameter
`RANDOM-PARTITION`，后者破坏中间scopes的temporal contiguity/nesting。两项都通过才进入modern baselines；任一失败，
ISCF降为strong carrier/control，不做rank、seed、partition、loss或router rescue。candidate code不变；25个new
control trainings与formal test仍未授权，但Step7B local prelaunch已通过。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_post_cpsi_step45_20260721/step4_5_scope_independence_narrative_gate.md`与
`configs/stage_c_iscf_v0_scope_attribution_confirmation.json`。

[ISCF-v1-CPSI Step9/10, 2026-07-21] 25/25 trainings、25/25 formal tests与10 historical references全部
protocol valid。CPSI相对ISCF-v0 test MSE/MAE为`-2.2128%/-1.6987%`，仅4/20与5/20 cells、1/5 datasets
正向；H96/H192/H336/H720 MSE均负，ETTh2平均退化`5.0538%`。相对A6_FULL亦为
`-0.7775%/-1.0606%`。这同时满足预注册material-negative macro/dataset与single-dataset条款。

internal health 25/25 pass，CPSI message/latent/output norms均finite/nonzero，故不是dead path或numeric pathology。
CPSI虽优于SELF与COMMON，却分别落后LINEAR/POST `2.2586%/1.7093%` MSE。最强LINEAR相对ISCF仅
`+0.0217%/+0.0472%` MSE/MAE，落在tie band且理论上可被independent affine scope maps吸收，只能作optimization
control，不能升级为贡献。

Decision=`cpsi_v1_exact_performance_fail_return_step4_5`。关闭exact CPSI-v1、confirmation与rescue；保留ISCF-v0
strong carrier。更广义scope interaction未被方向级否定，但任何新operator必须重新通过Step4/5 function-class与narrative
gate，不得增加router/second loss掩盖失败。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v1_cpsi_step9_10_20260721/step9_10_result_and_rollback.md`。

[ISCF-v1-CPSI Step7B, 2026-07-21] frozen config将CPSI/SELF/LINEAR/COMMON/POST五arms × five datasets
固定为25个new trainings，并以checkpoint hash复用ISCF-v0/A6_FULL 10个historical references，构成35 effective
runs、140 MSE cells与140 MAE cells。validation只作four-horizon checkpoint selector与health；runner在25/25
training artifacts完成前硬拒绝formal test，且test前后核验checkpoint nonmutation。

machine prelaunch初始18/18；remote smoke发现remote image无`rg`会使negated log scanner false pass，故在正式训练前
加入`grep` fallback并扩展为19/19。matrix/auth/governance、10 references、five model constructors、paired parent hash、
runner syntax/dry-run、scanner与analyzer smoke全过。evaluator新增common/private/left/right/latent/message RMS和trained output norm，
用于区分dead path、capacity explanation与mechanism attribution。四controls保持intermediate diagnostics：轻微负向不会
阻断test；只有完整test MSE/MAE后才按Step6 severity决定exact v1。

Decision=`step7b_prelaunch_pass_step8_authorized`。用户已授权seed2021的25-run training及全部完成后的单次formal
test；confirmation仍false。下一步为commit/push、remote pull、GPU preflight和Weather-CPSI/ETTm2-POST resource
smokes。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v1_cpsi_step7b_prelaunch_20260721/prelaunch_report.md`。

[ISCF-v1-CPSI Step8 Launch, 2026-07-21] commit `5d2330e`已remote pull；三张3090 preflight为空闲。
修复remote无`rg`导致的scanner false-pass风险后，Weather-CPSI与ETTm2-POST resource smokes均finite/no-OOM。
25-run matrix于`17:09:43+08:00`启动；training于18:03:54完成，formal test于18:10:55完成，均25/25。

[ISCF-v1-CPSI Step7A, 2026-07-21] production implementation新增五个readout modes，在parent ISCF初始化后创建
interaction matrices，保持paired base RNG path。local checker在conda `r2026-fsa`中81/81通过：readout 50/50、
equivariance 5/5、semantics 1/1、two-stage gradient 10/10、真实`TimeAlign.Model` 5/5、CLI 5/5、profile
parameters 5/5。五arms的parent/output/arm/policy morph gap均为0，model parent hash一致；首次backward只打开
zero-init output projection，一次synthetic update后两组input gradients和message全部finite/nonzero。

Decision=`step7a_local_pass_step7b_prelaunch_next`。该结果只排除hard implementation pathology，不含validation/test
MSE/MAE。下一步实现25-run manifest、training/test separation、historical ISCF/A6 hash audit、internal diagnostics和冻结
analyzer；remote/test仍false。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v1_cpsi_step7a_20260721/step7a_implementation_audit.md`与
`docs/code-explanation/stage-c-iscf-v1-cpsi-step7a.md`。

[ISCF-v1-CPSI Step6 Design, 2026-07-21] 用户要求四个controls作为intermediate diagnostics，不因轻微负向
提前关闭机制，并走到official-test MSE/MAE后判断effectiveness。Step6据此将validation角色限定为checkpoint与health：
所有protocol-valid arms无论validation排序都进入同一次冻结test audit；controls只决定claim attribution，不决定test access。

四个controls现已完成公平设计。SELF、LINEAR、COMMON与CPSI完全同为`3Lr`参数、无bias、`W_o=0` exact
ISCF morph；COMMON用two common projections的product而非dormant padding。POST-SYNTH直接作用于
`[B,C,S,T]` forecasts，不使用任意fixed projection；derived `r_post=round(Lr/T)`使added-module gap小于
1.95%、total-model gap小于0.041%。global pre-synthesis rank冻结为`r=32`。

formal matrix预注册为five new arms × five datasets × seed2021=`25` new trainings，另复用并审核ISCF-v0与
A6_FULL的10个historical references；全部35 effective runs形成140 MSE与140 MAE test cells。轻微test结果进入
`test_inconclusive`：CPSI vs ISCF macro MSE在`[-0.5%,+0.3%)`不方向级拒绝。initial support要求至少`+0.3%`、
3/5 datasets、10/20 cells且MAE不低于`-0.3%`；material negative要求macro不高于`-0.5%`且4/5 datasets为负，
或单dataset退化达到5%。这些阈值在implementation/test前冻结。

Decision=`step6_pass_step7a_local_authorized`。当前只授权local implementation与shape/morphism/gradient/parameter/
CLI checks；remote/test须等待Step7A与Step7B prelaunch、commit-pinned pull及GPU preflight。confirmation、router和
second loss仍为false。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v1_cpsi_step6_design_20260721/step6_control_design_and_test_policy.md`。

[ISCF Step5 Theory Gate, 2026-07-21] 代码与数学审计确认：对独立affine scope modes做任何fixed linear mixing，
均可精确吸收到新的`mode_weight/mode_bias`，所以Cross-Stitch式`5×5` matrix、linear common/private sharing与
fixed graph diffusion不形成新function class。plain peer-mean MLP也因“额外affine projection + generic depth”归因不净而
未进入working method。

Step5保留工作候选`ISCF-v1-CPSI`：在`[B,C,S,D,K]` modes进入`_scope_forecast`前，计算scope mean
$\mu$与zero-sum deviation $\delta_s$，再以
$W_o[\operatorname{GELU}(W_c\mu)\odot\operatorname{GELU}(W_p\delta_s)]$生成native mode interaction。
该path在scope/metadata共同置换时equivariant；`W_o=0`精确包含ISCF-v0；无bias product保证common或private任一
缺失时interaction为零。它不改变Bayes information set，只允许解释为finite-capacity inductive bias。

最新primary-source audit将claim进一步收紧：Deep Sets/Set Transformer已覆盖generic set interaction，Cross-Stitch覆盖
linear shared/private activation mixing，MoLE覆盖forecast expert output mixture，DMSC v5已直接覆盖multi-scale gated
coordination与adaptive routing，TimeExpert也把expert interaction放在output之前。故允许的贡献只能是
`future-output coupling scopes -> controlled common/private response evidence -> linear reparameterization boundary ->
pre-synthesis multiplicative interaction -> matched attribution`完整链，不能claim multi-scale coordination或expert mixing。
Decision=`step5_theory_pass_step6_control_design_next`；Step6必须解决exact-parameter SELF/LINEAR/COMMON与诚实的
POST-SYNTH placement controls。implementation、remote、formal test、router与second loss仍为false。
详见`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_step5_common_private_interaction_20260721/step5_theory_and_narrative_gate.md`。

[ISCF-v0 Step4 Response Relation, 2026-07-21] test residual common被确认含shared-target confound，故不再作为
existence gate。新的label-free `ISCF-SRA-D1/D1.1`在frozen validation histories上测量五个scope对共同hidden
perturbation的central response。primary 16-direction topology仅2/5稳定；预先声明的64-direction validity check恢复
5/5，说明原估计存在Monte Carlo design fault。随后D1.1使用disjoint validation rows与新seed确认：15/15 runs同时
超过independent-direction null和architecture-identical random-init p95，median common/private response=
`0.2803/0.7197`，pair distance=`1.3440`，4/5 datasets topology跨seed稳定。Decision=
`scope_response_relation_confirmed_for_step5_theory`。

Step4 narrative仅以single architecture problem conditional pass：ISCF scopes学得pre-synthesis response dependence，但
当前只在完整arm forecasts后做late scalar fusion。下一步研究non-ordered scope-set interaction；Deep Sets/Set Transformer、
Cross-Stitch、MoLE与multiscale forecasting mixing均是mandatory prior/control，generic set mixing本身不可claim。
active_method仍为none；不实现、不训练、不访问formal test，不新增router/second loss。
详见`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_step4_scope_relation_20260721/step4_result_and_step5_handoff.md`。

[ISCF-v0 Freeze and Function Audit, 2026-07-21] 用户明确将FCC的matched independent-scope arm固定为
`ISCF-v0 — Independent-Scope Coupling Field`后续研究基底。code仍使用
`siff-independent-scope-control`以保持15个checkpoints exact-compatible；$Q=5$、`I_5` scale basis、direct
policy、equal-skill objective、dataset-wise ranks、natural profiles与four-horizon selector全部冻结。由既有完整FCC
test table派生的ISCF-v0 vs A6_FULL MSE/MAE为`+1.3584%/+0.9144%`，5/5 datasets、4/4 horizons、
3/3 seeds正向；该比较为test-informed post-hoc carrier evidence，不是新method gate。

existing NPZ function audit现完成：common/private residual、scope complementarity与cross-seed topology三项通过，
aligned low-dimensional relation为`0/15`失败。median common/private energy=`0.9320/0.0680`，oracle headroom
`8.5813%`，4/5 datasets topology稳定；但EV2 `0.6281`低于shift-null p95 `0.7223`，scale-order rho仅
`0.2121`，fusion只在9/15 runs超过best fixed arm。Decision=
`function_relation_unresolved_requires_narrow_step4_audit`。这阻止立即实现low-rank relation matrix、恢复ordered
SIFF或新增router/loss；下一节点仅为non-ordered common/scope-specific relation的Step4 source/narrative audit。
详见`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_function_audit_20260721/result_and_step4_handoff.md`。

[Constraint Reset, 2026-07-20] 后续不再把exact projectivity、requested horizon禁用、A6 interface compatibility或
full-$T$ prefix crop当作新方法的先验硬约束。它们可以成为合理设计或matched controls，但必须由problem、理论与
实验选择。与此同时，放宽约束不等于自动支持H-conditioned shared-prefix：在separable pointwise MSE下，同一fixed
past与future coordinate的Bayes conditional mean不依赖requested horizon。新自由度必须明确来自finite-capacity
tradeoff、target-coordinate evidence access、nonseparable risk、future context、compute/resolution contract或
probabilistic joint target。

[A6 Role Audit] A6-LBF保留为strong carrier、mandatory control与possible component，不直接升级为standalone
paper core。其性能收益真实，但learned basis、projectivity与harmonic horizon measure分别存在N-BEATS/N-HiTS、
FlowState/Implicit Forecaster与ElasTST等强prior，当前也缺少相对最新target-query/varied-horizon方法的完整优势。
本轮`SC-D22-HFA`先审计finite-capacity horizon frontier，再按证据决定是否设计小型target-coordinate
information-access diagnostic；Contribution 2不预先指定。

[D22-A/B Decision] Bayes/task audit确认：在同一fixed past、pointwise MSE且requested H不改变information set、
distribution或utility时，同一future coordinate的Bayes conditional mean不依赖H。D18 H1..720 official-test curves
进一步给出`finite_capacity_frontier_not_supported`：SPEC96 own-H虽为`+1.2748%`且5/5 datasets正向，但
SPEC192/SPEC336分别为`-0.1386%/-0.6385%`；三个specialists在`{96,192,336,720}`上均0/5 dataset
Pareto-dominate A6_MEASURE。A6_MEASURE相对A6_FULL在五个lead-time bins全部5/5正向，measure control解释
主要稳定收益。

H96只保留为局部optimization clue，不授权soft projectivity、H embedding、router或seeds。D22-A/B不能直接回答
target coordinate是否需要对raw history作specific evidence access；D14 dual-carrier three-seed headroom使一次
D22-C小诊断具有条件合理性。D22-C现已完成static/prelaunch：neutral/raw-history六臂共用完全相同module、
trainable parameters、seed initialization与selector，只改变global/pooled/ordered/order-shuffled/
target-shuffled/generic retrieval contract；local synthetic smoke与aggregator均通过。ordered patch memory仍不是
论文主语，A6 sensitivity、paper method和Contribution 2均未授权。用户已授权冻结seed2021 five-dataset完整
problem gate；remote/test只在commit/push与GPU preflight后启动。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/d22c_prelaunch_design_audit.md`。

[D22-C v1 Numeric Correction] 首次remote launch在training-only阶段发现Weather/ETTm2的RevIN-normalized loss被
near-zero within-window variance放大到$10^3$量级；在任何dataset/test artifact完成前终止。该run只说明
`optimization_or_numeric_pathology`，不作problem判断。v1.1保持architecture、arms、seed、selector与gates不变，
只将training loss移到与evaluation一致的dataset-standardized scale，并使用全新output/checkpoints。

[D22-C Result] v1.1完整five-dataset × six-arm problem gate通过。`ORDERED_TARGET_ACCESS`相对
global/pooled/order-shuffled/target-shuffled的test MSE gain分别为`+17.2910%/+17.5308%/+17.0826%/
+13.7449%`，均20/20 cells、5/5 datasets正向；相对最强`GENERIC_MATCHED`为MSE `+2.5228%`、MAE
`+1.6484%`、15/20 cells、4/5 datasets、4/4 horizons，validation/test macro为`+2.5410%/+2.5228%`。
parameter gap为0，ordered attention与prediction均未collapse。decision=
`target_coordinate_information_access_supported`。

该结果只通过problem/matched-attribution gate，不是paper-facing method effectiveness。Weather相对generic在
4/4 horizons为负、dataset macro `-1.0900%`，因此不能claim universal target-query superiority。CATS、
TimePerceiver、MQTransformer与TQNet已直接覆盖future/temporal query读取history；raw cross-attention不能升级为
Contribution 1。

[SC-D23-FCMI] Step4-6提出`Future-Coordinate Main–Interaction operator`：对standard query context
$S_t$计算trajectory main $\bar S$与zero-mean interaction $\Delta_t=S_t-\bar S$，再以
$W_{\rm main}\bar S+W_{\rm int}\Delta_t$作native forecast state。$W_{\rm int}=0$精确包含generic case，
$W_{\rm main}=W_{\rm int}$精确包含standard query decoder；没有H embedding、router或第二loss。Step7A production
implementation现为11/11 pass：zero-mean residual最大`1.82e-7`，standard morph最大差`6.33e-8`，
main/interaction/query/output gradients均finite/nonzero，dual controls在五个natural profiles参数严格相等。
FCMI相对A6 active parameters少约83%–95%，故未来formal matrix必须增加dense capacity-matched control。
Step7B design/prelaunch现为21/21 pass。`DENSE_DUAL_MATCHED`以profile-specific low-rank temporal residual把
active parameters匹配到A6的`0.2%`以内，并在zero-init下保持standard-dual initial function；五个profiles的
实际gap为`0.0914%–0.1321%`，coefficient与basis分阶段gradient均finite/nonzero。formal matrix冻结为
8 arms × 5 datasets × seed2021 = 40 runs，全部arms进入160个official-test cells和160个validation cells。
`TARGET_SHUFFLED_QUERY`在任何test access前由validation-only修正为formal control，因为它参与方向级
attribution，不能由validation pass/reject。effectiveness、standard/generic decomposition、
order、dense capacity与initialization attribution均已预注册，validation仍只选checkpoint。

Step7B本身只得到`step7b_prelaunch_pass_waiting_remote_test_authorization`，没有训练结果。用户随后于
2026-07-20以“按计划继续推进工作”独立授权冻结的seed2021 40-run/160-cell remote/test matrix；
confirmation seeds与method promotion仍为false。下一步先做commit-pinned remote pull、GPU preflight和两项
resource smoke。dense control不是method component或第二项contribution。

[SC-D23-FCMI Step8] commit `4ff439c`已完成remote pull、三张RTX 3090 preflight以及Weather-FCMI/
ETTm2-DENSE resource smoke；smoke finite且无OOM。40-run matrix于`2026-07-20T17:57:10+08:00`在
GPU0/1/2启动，首批Weather FCMI/DENSE/A6均进入训练。运行期间不改config/gates，不启动confirmation；
完整40/40返回后才执行冻结four-layer analyzer。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/d23_step8_remote/remote_launch_record.md`。

[SC-D23-FCMI Result] 40/40 runs、160/160 validation与160/160 official-test cells完整，protocol/internal
health全部通过，但FCMI相对A6 test MSE/MAE为`-21.7343%/-10.9242%`、0/20 cells。FCMI相对
standard-dual、generic-dual与target-shuffle分别为`+1.3409%/+6.0060%/+9.2071%`，说明
main–interaction与coordinate semantics在弱query family内有效；但order control为`-0.4536%`且
validation/test反转。

capacity control给出决定性解释：DENSE相对STANDARD_DUAL test MSE为`+15.4825%`、19/20，DENSE相对A6
仅`-0.3284%`。三种FCMI validation-fit frozen complementarity diagnostics全部在test反转；A6/DENSE
validation-fit blend也从validation `+0.5127%`反转test `-0.1707%`，而固定等权仅出现test-only正信号。因此
decision=`fcmi_v1_failed_capacity_control_explains_return_step2_3`：关闭FCMI-v1，不补seed/width/readout，
也不把dense main直接升级successor；D22-C problem evidence保留，当前回deterministic-MSE fixed-past
Step2/3。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/d23_step8_remote/d23_step9_10_result_and_rollback.md`。

[SC-D24-CTB] Step2/3现冻结一个validation-only problem diagnostic：在D23的A6与DENSE冻结checkpoint上，
检验ordered raw history能否超越global、channel/marginal、recent、order-destroyed sorted-history与
target-shuffled controls，识别48-step coarse future deformation。first-third fit、middle-third purge、
last-third evaluate；official test、training、method、router与第二loss均false。256-row phase probe仅给出约
`+0.03%` derivative-specific gain且被curvature/shift controls解释，故phase router路线关闭。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/d24_ctb_step23_design_audit.md`。

D24-v1的10/10 inference与checkpoint invariants通过，但发现ridge penalty未按fit row count归一化，
使$\lambda$在数万rows下近似无效并造成severe chronological extrapolation。v1标记
`design_fault_suspected`，不得用于problem rejection。v1.1只改为$X^\top X+n\lambda I$并冻结normalized
$\lambda=\{0.01,0.1,1\}$；其余validation-only contract不变。

[SC-D24-CTB Result] v1.1 normalized-ridge 10/10、840 metric rows、720 comparison cells完整，official test
access为0。ordered history相对marginal在A6/DENSE上为`-8.5950%/-8.6168%`，相对sorted为
`-9.4741%/-8.8197%`，相对target-shuffled为`-14.1002%/-13.4974%`；所有primary horizon aggregates
均0/4正向。即使$\lambda=1$，ordered correction相对原forecast仍约`-15%`。exact coarse deformation
hypothesis关闭，不做feature/bin/lambda/nonlinear rescue；broader direction只记为
`unresolved_but_unsupported`。当前回Step2/4 consolidation，不启动D25。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/d24_ctb_result_and_rollback.md`。

[Post-D24 Consolidation] `Bayes boundary -> finite frontier negative -> target access positive -> capacity
explanation`形成scientifically coherent problem boundary，但没有形成完整method-paper narrative。pure requested-H
不提供新Bayes information；future coordinate仍可能需要不同history computation；D23进一步表明target access与
strong trajectory function class是两个必须独立控制的维度。当前最强主张是design/control principle，不是
paper-core method。

modern native-baseline gap现为blocking：P0必须包括ElasTST、CATS、TimePerceiver、SRSNet及A6_FULL/
A6_MEASURE。外部baseline必须在official repositories按native contract复现，并把single-weight varied-horizon、
per-H fixed-model与foundation/pretrained结果分表。当前只进入`SC-MNB Step1-3` protocol设计；
implementation、remote training与official test仍false。若A6不具modern competitiveness，后续不得继续围绕其
interface堆叠architecture。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/post_d24_paper_story_and_modern_baseline_gap_audit.md`。

[SC-MNB Source Audit] 四个P0 official sources已在repo外只读审计并固定commit。ElasTST是唯一
single-weight P0 baseline；CATS、TimePerceiver与SRSNet均为per-H独立训练。source audit还发现CATS/
TimePerceiver每epoch读取test loss、CATS ETTm2-H96 dataset identifier typo、SRSNet file-level license/
metric-equivalence待证和
ElasTST `limit_train_batches=10`待确认。decision=
`source_set_frozen_protocol_repairs_required_before_prelaunch`；当前local patch、remote training和official test
仍false。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/sc_mnb_step13_source_and_protocol_audit.md`。

[SIFF Paperization Reset, 2026-07-21] 用户明确选择`SC1-SIFF-v2-EQ-ATTR-v1`作为当前论文落地的最接近路径。
该版本继续immutable，历史Step9仍是`performance_partial_pass_attribution_blocked`：相对A6_FULL
`+1.6436%`、相对A6_MEASURE `-0.2366%`、相对independent `+0.2580%`。恢复的是SIFF research program，
不是把失败改写为pass或授权seed/width/rank/objective rescue。SC-MNB降为supporting prior/control inventory，
65-run baseline execution继续false。

新的provisional successor为`SC1-SIFF-v3-TSAF-v1`。它保留scale-indexed full-domain arm generator，以
future-coordinate × ordered-log-scale `Target-Scale Allocation Field`替代缺乏稳定证据的history-conditioned
generic router。allocation不读取requested H、history hidden或future labels；history dependence保留在每个SIFF
arm内。equal-skill继续作为单一training contract，不包装为第二项loss contribution。Step4-6 narrative/design gate
为conditional pass。Step7A production-local现26/26通过：allocation对history/sample/channel严格不变，
target/scale语义、full-domain crop、参数公式、五条gradient与真实TimeAlign constructor均通过。

[TSAF Step7B] prelaunch现15/15 cases、10/10 categories通过。formal matrix固定为9 effective arms × 5 datasets =
45 runs/180 official-test cells；其中A6_FULL、A6_MEASURE、PCSD_EQUAL与immutable parent共20个历史end-to-end
references经remote checkpoint SHA256 20/20复核后复用，5个new arms共25 runs必须from-scratch joint training。
旧history-conditioned independent reference没有复用；新的independent target-only ranks重新按TSAF active parameters
匹配为ETTh1/ETTh2/ETTm1/ETTm2/Weather=`109/115/115/106/115`，最大relative gap 0.3619%。25/25 CLI、
5/5 two-step policy gradients、逐dataset encoder/TSAF initialization pairing、runner refusal与four-layer analyzer
synthetic smoke均通过。validation只选checkpoint；runner不执行test。在该prelaunch节点
`remote training=false / official test=false / confirmation=false`，因此该节点不是performance pass。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v3_tsaf_step7b_prelaunch_report.md`。

[TSAF Step8 Authorization] 用户于2026-07-21授权冻结的25-run seed2021 remote training，并授权在25/25 training
完整后对45-run/180-cell effective matrix执行一次formal test。confirmation seeds、paper-core promotion与matrix/gate
修改仍false。training runner与formal-test evaluator已分离，test mode要求checkpoint SHA256 nonmutation；授权记录时尚未
remote pull、resource smoke、training或test。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v3_tsaf_step8_remote_authorization_and_launch.md`。

[TSAF Step8 Launch] commit `6cef063`已remote fast-forward；Weather-TSAF与ETTm2-independent两项2-batch resource
smoke finite且无OOM。25-run training于`2026-07-21T10:17:06+08:00`在GPU0/1/2启动，首批三个Weather jobs进入
epoch 1，初始显存约1.6–2.0 GiB。formal test保持0/25，只有training 25/25完整后才单独启动。

[TSAF Step9/10 Result] 25/25 new training、25/25 new formal test与45/45 effective-run audit均完整；
formal-test commit为`4cc96f21e23c159e37757c66ec2e5c68358c5718`，45个checkpoint SHA256 unique，逐dataset
encoder initialization完全matched且test nonmutation 25/25通过。TSAF相对`A6_MEASURE` test MSE/MAE为
`-1.2854%/-1.3146%`，相对SIFF-v2 parent为`-1.0422%/-0.9183%`，两项均0/4 horizon wins，
paper-facing effectiveness fail。

四个matched questions也全部fail：ordered-field vs categorical `-1.0191%`，ordered-scale vs permuted
`-0.0796%`，target-coordinate vs global `-0.0405%`，shared-field vs independent `-1.2785%`。all-finite、
arm diversity、nonconstant target-scale surface、order sensitivity、allocation entropy、scale-component contribution
与request invariance全部通过，只证明路径活跃，不能覆盖negative effectiveness。validation中TSAF相对parent曾为
`+0.7700%`，formal test反转为`-1.0422%`，所以不得换selector或按test重选epoch。

capacity-matched independent target-only相对parent有MSE `+0.2383%`、MAE `+0.0898%`的single-seed weak signal，
但低于0.3% primary threshold，且预注册角色是control；不得post-hoc改名或直接补confirmation。Decision=
`close_tsaf_v1_shared_field_design_keep_siff_v2_immutable_parent`：关闭TSAF-v1 exact shared field，不做
seed/rank/width/readout/loss rescue；SIFF-v2 identity不变，当前无active successor method，回SIFF-first Step2/4。
详见`analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v3_tsaf_step9_10_result_and_rollback.md`。

[Post-TSAF 2x2 Problem Audit] `independent target-only`不是“移除SIFF history router”的单因素arm：它同时把
Q2 ordered field改为Q5 independent fields、把direct policy改为static-target，并在ETTh2/ETTm1/Weather改变
rank 116到115。四个existing E2E arms的factorial audit显示，全20-cell test interaction MSE/MAE虽为
`+0.5265%/+0.4246%`，严格same-rank ETTh1+ETTm2子集却为`-0.3097%/-0.1175%`；validation same-rank
近零，Weather发生split reversal。因此`+0.2383%` weak gain不能归因于target-only、independent field或二者
stable interaction。Decision=`independent_target_only_weak_lead_not_supported_for_step4`；不补seed/rank/router
rescue，不创建successor。

SIFF-v2的paper claim同步收紧：保留其相对A6_FULL、PCSD_EQUAL、constant、permuted与Q1-wide的正证据，
但不claim ordered field严格优于independent、history-conditioned policy不可替代、target-only allocation已成立，
或SIFF-v2已超过A6_MEASURE。下一节点先完成immutable SIFF-v2 final paper-claim consolidation；在该narrative
gate前不执行modern-baseline performance matrix、不实现新method、不启动remote training，也不预设第二loss/router。
详见`analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_post_tsaf_independent_factorial_audit.md`。

[SIFF-v2 Final Paper-Claim Gate] 原样SIFF-v2以单一architecture contribution获得conditional narrative pass：把
future-output coupling extent作为ordered log-scale coordinate，以共享history-conditioned components生成nested
full-domain forecast operators并target-wise融合。论文不claim首次multi-scale、MoE、future query或H conditioning；
direct policy不作为novel router，`equal_skill`只作共同训练contract，不包装为第二method contribution。

这不是paper-core effectiveness pass。用户于2026-07-21指定FCC以`A6_FULL`替代`A6_MEASURE`，因此现冻结
`SIFF_EQUAL/A6_FULL/SIFF_INDEPENDENT_EQUAL × 5 datasets × seeds2022/2023`共30 new runs，复用seed2021
形成three-seed evidence。`SIFF_EQUAL vs A6_FULL`只回答完整method package effectiveness，因为architecture与
objective同时变化；ordered-field attribution只由same-objective independent control承担。`A6_MEASURE`完全退出
FCC matrix、metrics与machine gate，但其历史negative evidence保留在limitations中。

两项primary comparisons均须通过MSE `+0.3%`、MAE为正、3/5 datasets、3/4 horizons与至少2/3 seed macro
为正；任一失败即停止SIFF paper-core rescue，不回rank/loss/router/readout tuning。local prelaunch现25/25通过，
30/30 jobs与15/15 historical references完整、unique且initialization paired。用户已授权冻结matrix的remote
training，并授权30/30 training完整后执行一次formal test；下一步为commit-pinned remote pull、GPU preflight与
两项resource smoke。Decision=`step7b_prelaunch_pass_proceed_commit_remote_preflight`。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v2_fcc_v1_prelaunch/prelaunch_report.md`。

[SIFF-v2 FCC Step8 Launch] commit `87bea35`已remote fast-forward；GPU0/1/2 preflight均约24.1 GiB free，
Weather-SIFF seed2022与ETTm2-independent seed2023 two-batch smokes finite且无OOM。30-run training于
`2026-07-21T12:54:37+08:00`在三张RTX 3090启动，首批Weather的SIFF/A6_FULL/independent均已进入训练。
formal test仍为0/30，只在30/30 training完整后执行。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v2_fcc_v1_prelaunch/remote_launch_record.md`。

[SIFF-v2 FCC Step9/10 Result] 30/30 new training、30/30 formal test与45/45 effective-run audit完整；180/180
official-test cells、45 unique checkpoint hashes、paired initialization与test nonmutation全部通过。SIFF相对
`A6_FULL`的test MSE/MAE为`+1.2497%/+0.7549%`，5/5 datasets、4/4 horizons、3/3 seeds正向，确认完整
method package相对用户指定source carrier稳定提升。

但是ordered-field attribution失败：SIFF相对same-objective capacity-matched independent control的test MSE/MAE为
`-0.1272%/-0.1733%`，仅2/5与1/5 dataset wins、2/4与0/4 horizon wins、1/3 seed wins；validation也为
`-0.3224%/-0.5015%`，不是split reversal或单seed异常。internal health六项均通过，只说明路径活跃，不能覆盖
negative attribution。Decision=`performance_pass_attribution_blocked_stop_fcc_promotion`，failure attribution=
`capacity_control_explains`。SIFF-v2不晋升paper core，不补seed/rank/width/readout/router/loss，不启动modern
baseline matrix或formal ablations；当前回paper portfolio decision，`active_method=none`。A6_MEASURE历史negative
继续保留。详见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v2_fcc_v1/step9_10_result_and_portfolio_decision.md`。

[Scope Decision, 2026-07-20] 用户明确要求当前项目暂不转出`deterministic-MSE fixed-past architecture search`。
因此D22-C有效失败只关闭exact v1并回joint Step2/3，不再自动停止整个search；但不得用seed、width、readout或
representation rescue重开D22-C，也不得恢复D17-D21或预设第二loss/router。

[Supersession Notice] 本文件后续D17-D21、projectivity与旧Contribution slots段落属于历史证据，不得覆盖顶部
`Constraint Reset`与restart handoff。旧文中“requested-H关闭”“full-T必须保留”等决定只关闭当时exact candidate，
不再是新问题的全局硬约束。

[Evaluation Rule] official test split现固定为所有正式机制评估、paper-core effectiveness与Step9-10决策的
primary gate；validation只负责checkpoint selection、普通超参数选择、debug与解释性diagnostic，不能判定机制
pass/fail。默认checkpoint score为validation H96/H192/H336/H720 MSE平均。test已成为
`test_informed benchmark decision surface`，不再声称untouched；禁止按dataset/horizon/cell反向调参。

[Historical Fair Audit] 70/70公平重评估中，`SIFF_EQUAL`相对A6为`+1.6436%` MSE、`+0.9084%` MAE，
17/20 MSE cells、4/5 datasets、4/4 horizons，当时是已测试arms中的最佳performance carrier。`SIFF+PCC`虽相对A6
`+1.3812%`，但PCC相对SIFF_EQUAL为`-0.2663%`，因此不能解释为双贡献joint成功。PCSD_DIRECT
`-0.8562%`，PCC specificity与SIFF objective-robustness均fail。

[Attribution Freeze] `SC1-SIFF-v2-EQ-ATTR-v1`已冻结10-arm EQUAL-context matrix。主效果比较为
`SIFF_EQUAL`分别对A6_FULL、A6_MEASURE与PCSD_EQUAL；机制specificity比较为其分别对
constant/permuted/Q1-wide/independent EQUAL controls。七项comparison必须逐项通过，内部oracle、arm
difference、policy entropy与component-use只解释机制健康度，不能替代paper-facing effectiveness或matched
attribution。Step6为16/16，Step7A为13/13，Step7B prelaunch为9/9；seed2021的完整50-run official-test
Phase A现已完成。结果为main effectiveness 2/3、EQUAL-context attribution 3/4、internal health 7/7：
`SIFF_EQUAL`未超过`A6_MEASURE`（MSE `-0.2366%`、MAE `-0.3961%`），且对independent control仅
`+0.2580%`，低于冻结`0.3%` gate。Step9 paper-core gate仍失败，seeds2022/2023 confirmation保持false；但用户在
Step9后将该exact artifact保留为`frozen_performance_near_candidate`，作为当前最强候选与下一轮redesign parent。

## Research Thesis

论文研究对象仍固定为`fixed-past unified multi-horizon generation`，但旧coupling-adaptive thesis已在CCSF
Step9–10后撤回为historical hypothesis，不能继续作为当前论文主张。

[Fact] 若要求对任意$H\leq K$满足$F_H(x)=P_HF_K(x)$，则必有
$F_H(x)=P_HF_T(x)$。requested horizon因此不能改变共享prefix。当前论文不再同时声称“shared-prefix严格不变”
与“prediction随requested horizon自适应”。

prefix-safe future-context D17没有通过problem gate：causal相对pointwise `-3.0356%`、相对shuffled
`-2.3616%`，pointwise correction相对parent亦为`-28.7314%`。该结果带有validation→test transfer pathology，
只关闭frozen post-hoc correction，不方向级否定E2E operator；但没有正向证据时不允许进入Step4实现。

D18现已关闭该问题。25/25 artifact units与15/15 own-H official-test cells完整：specialists相对
`A6_MEASURE`仅`+0.1659%` MSE、7/15 cells、2/5 datasets与1/3 horizons为正，七项gates只通过2项；
`A6_MEASURE`相对`A6_FULL`却为`+1.7980%`且15/15 cells全正。因此表面specialization收益由measure training
解释，未证明exact projectivity存在稳定accuracy cost。

[Decision] controlled soft projectivity、consistency penalty sweep与requested-H feature关闭，不补D18 seeds。
Contribution 1/2均回Step2。当前新的问题收紧为：

> 在不把requested horizon作为semantic input、保持single full-$T$ trajectory与prefix crop的条件下，
> trajectory-level structured decoder是否仍能超过strong A6_MEASURE learned-basis generator？

下一步`SC-D19-IFC`只把Implicit Forecaster作为source-informed control，比较frequency/amplitude/phase wave
generation与A6 learned basis，并加入no-skip和matched MLP controls。IF已有NeurIPS 2025直接prior，不能作为本项目
Contribution 1；D19只判断decoder forecasting phase是否还有真实headroom。Step4/5 code audit已确认upstream
本身固定生成720 points再crop，因此projectivity可直接保持；但IF同时增加nonlinear synthesis与raw input
spectrum skip，不能用单一IF-vs-A6结果判断wave mechanism。

Step6已将`A6_MEASURE/IF_MEASURE/IF_NOSKIP_MEASURE/DIRECT_NONLINEAR_MATCHED_MEASURE`四arm、
five datasets、seed2021、four-horizon validation selector与80个official-test cells完整冻结，静态gate 9/9。
IF与matched-direct逐profile参数差小于0.1%，用于分离polar/frequency structure与generic nonlinear/capacity。
Step7A发现v1误把upstream 96-point lookback作为本地skip contract，而A6 natural实际读取720 points；v1在任何
training/test前被v1.1替代。v1.1让Encoder、IF skip和matched direct读取相同720-point normalized history，
并重新匹配参数。

v1.1 Step7A现已114/114通过，maximum prefix gap为0，paired Encoder与IF/no-skip decoder hashes、required
gradients、numeric与production CLI均通过。Step7B prelaunch又以31/31通过，冻结15个new runs、5个复用A6、
80个official-test cells与四层诊断。seed2021一次remote/test control audit已授权；confirmation seeds与
paper-method promotion仍未授权。

D19 Step9现已完成：IF相对A6为`-3.6117%` MSE、`-3.6519%` MAE，仅3/20 MSE cells；相对
parameter-matched direct为`-0.8075%` MSE。IF相对no-skip则为`+1.6191%` MSE、16/20 cells，说明720-point
history-spectrum shortcut提供了真实信息，但polar/frequency synthesis没有得到matched-control支持。全部internal
health gates通过，validation上IF对A6也为`-10.9950%`、0/20，因此不是collapse或test-only reversal。

[Decision] D19 exact control关闭，不补confirmation seeds，也不做width/LR engineering sweep。由于D19 heads为
A6参数量的7.94×–10.29×且12/15 new arms在epoch1达到best checkpoint，失败归因为当前
`readout_or_head_design_wrong`并怀疑readout-scale/optimization mismatch；不能据此方向级否定所有trajectory-
structured decoder。Contribution 1回Step2/4，history-spectrum skip只作为后续compact method的设计证据，
不能单独充当multi-horizon Contribution。

[Post-D19 Step2/4 Boundary] external primary-source refresh进一步确认：FITS、FBM、Implicit Forecaster、
PhaseFormer、BasisFormer、FlowState、N-HiTS与TimePerceiver已分别覆盖compact spectrum interpolation、
time-frequency basis、amplitude/phase synthesis、compact phase routing、functional basis与target-coordinate
decoding。可逆linear transform后接unconstrained linear head也不产生独立function class。因此smaller IF与
history-phase-continued atoms未通过method narrative gate；“直接接入spectrum”也不能成为Contribution 1。

当前只保留`SC-D20-CST diagnostic_only`：检验D19的history-spectrum skip收益能否transfer到A6 full-trajectory
coefficient operator，并超过same-dimensional fixed random orthogonal history projection。三臂必须共同from-scratch
E2E训练且共享A6 Encoder、learned basis、measure objective、full-T generation与prefix crop。只有SPEC同时超过
A6与RANDOM并通过internal health，才允许回Step4设计native non-residual coefficient operator；D20本身和generic
concat head均禁止结果后升级为paper method。当时只授权Step6，implementation/remote/test均false。

[D20 Step6 Freeze] exact diagnostic使用`A6_MEASURE_RETRAIN/A6_CST_SPEC/A6_CST_RANDOM`三臂。SPEC与RANDOM
分别以fixed 64-dimensional real low-frequency Fourier subspace和same-dimensional Gaussian-QR orthogonal
subspace读取normalized 720-history；两者共享`Linear(R+64,256)`、learned temporal basis与full-T prefix-crop
contract。summary columns zero-init且base block/Encoder/basis paired，使三臂初始prediction完全一致，但summary
weight首步gradient非零。

five datasets × seed2021形成15个from-scratch runs/60 test cells，不复用A6 checkpoint。SPEC必须同时对A6
transfer和对RANDOM specificity通过对称gates；只超过A6仍由generic history access/capacity解释。Step6 static
gate为14/14，decision=`step6_pass_step7a_local_only`。当前只授权local implementation；remote/test/confirmation/
paper method仍false。

[D20 Step7A/7B] production实现保持A6 base coefficient path，并加入独立zero-init summary-to-coefficient path；SPEC与
RANDOM只在fixed projection geometry上不同。local gate为9/9，initial/prefix gap均为0；prelaunch为10/10，
15-run/60-cell matrix、validation four-horizon checkpoint selector、formal-test授权、checkpoint non-mutation与四层
analyzer均已冻结。当前decision=`step7b_prelaunch_pass_step8_authorized`；该实验仍是test-informed
problem diagnostic，不是Contribution 1 method。

[D20 Step8] commit`9573cd7`的15-run/60-cell matrix已于`2026-07-20T11:55:45+08:00`在三张3090后台启动；
双resource smoke与首批三任务启动检查通过。当前不值守；结果返回前不改变candidate、matrix、checkpoint或gate。

[D20 Step9–10] 15/15 runs与60/60 test cells完整。SPEC-vs-A6为`-0.7614%` MSE、8/20 cells、2/5
datasets、0/4 horizons，transfer fail；SPEC-vs-RANDOM为`+0.1412%`、14/20、4/5、3/4，只有弱directional
specificity且未达0.3% gate。11项internal health全过，validation SPEC-vs-A6 `+0.5755%`到test反转，因此只关闭
q64 additive coefficient injection；failure attribution为`validation_test_mismatch + intervention_point_wrong`，不能
方向级否定history-spectrum。

[Post-D20 Boundary] scalar gate/normalization rescue与generic spectral robustness均不进入method。当前仅保留
`future-distance predictive support`为provisional problem：同一history evidence对不同future coordinates的有效性
可能不同，但尚未证明past-identifiable或split-stable。先执行D20-D1 contribution oracle；Contribution 2继续Step2。

[D20-D1 Result] SPEC contribution相对其co-adapted base为`+26.8928%`、39/40 bins有益，median oracle alpha
`1.2649`；RANDOM也为`+9.0422%`、35/40。完整两臂却都差于A6，证明within-model path importance不能当作
incremental mechanism evidence；joint model发生了non-identifiable responsibility relocation。scalar shrink rescue
关闭，future-distance support也未被D1支持，继续停在problem-unverified Step2/3。

[D21 Problem Redefinition] 外部primary-source审计后，宽泛的`future-distance support`已被替换为
`Evidence-Validity Surface (EVS)`：在一次full-$T$、prefix-projective生成中，internal forecast-construction
route的conditional relative risk是否是past pattern $x$与future coordinate $\tau$的non-separable函数。该定义要求
past × future-region interaction同时超过global fixed、region-only、sample-only、additive sample+region与permuted
history controls。若region-only解释收益，问题退化为静态segmented decoder；若sample-only解释，问题退化为
generic adaptive fusion，均不能支撑当前multi-horizon主线。

D14-A dual-carrier、three-seed crossing与sample-over-bin oracle是problem headroom；D20/D1只限定co-adaptation
边界。D21 seed2021使用五个独立训练的D14 canonical scope arms，在validation拟合past-only centered log-risk，
official test只评估transfer。TimeFuse、TimeRouter、Synapse、TimeMixer、MQTransformer、TimePerceiver与ElasTST
分别构成external model fusion、routing、multiscale predictor、target-query与projectivity mandatory boundaries。
当前problem narrative/design gate与Step7A local implementation已通过；100个frozen-checkpoint val/test evaluations
已于commit `53661b1`在三张3090启动。new forecasting model training、paper method与confirmation seeds仍未授权。

[Provisional Two-Slot Logic] 只有D21 problem gate通过后，两个slots才可进入Step4：Contribution 1负责在single
projective decoder内表示past-by-coordinate route validity；Contribution 2负责在同一end-to-end forward graph中给
这些routes分配与最终fused forecast一致的credit。offline oracle teacher、stale cross-fit labels与requested-H
conditioning均被排除。当前这只是可证伪的论文蓝图，不是method claim。

[D21 Returned Decision] 100/100 exports完整。D14 oracle opportunity在official test仍为neutral `+7.6399%`、A6
`+10.4053%`，但neutral HGB interaction相对mandatory additive control只有`+0.0347%`，低于冻结`0.1%`；A6为
`-0.0069%`。validation chronological forward interaction曾有`+0.3092%/+0.4406%`，到test缩小或反转。因此
exact EVS的split-stability与material non-separability没有成立；generic sample/additive effects解释了大部分可学习
收益。`SC-D21-EVS`关闭，不补seeds、不做representation rescue。上述Provisional Two-Slot Logic作为失败的
historical hypothesis保留，不能再写成active paper thesis；两个slots共同回Step2。

forecast-revision surface已转移到根目录`New-idea.md`，状态`deferred_next_paper`；它不再是当前论文问题。

## Contribution Slots

[Decision] `CADMO/CPGA`因问题范围只落在history-interface而退出active slots；它们不是实验方向级失败，而是
`rejected_by_narrative_scope`。ordered patch memory只保留为`D14-P auxiliary_interface_probe`，不决定论文主线。

两个active slots曾由`PCSD/CCRL` provisional占用；2026-07-16 training-consistency audit已将CCRL降为
`diagnostic_only_not_scheduled`，第二slot重新开放。D14-A0的neutral linear RRR gate已返回：stable crossing
0/5、sample × bin oracle macro 0.0586%、canonical-vs-random -0.1427%。但其factor params并不等于
rank-manifold effective DoF，且scale risks最多只相差0.04036%，所以方向级拒绝无效。A1随后以matched grouped
nonlinear heads强制不同sharing topology，且全部scales包含full-affine map；neutral raw-history作为primary gate，
A6-natural作为from-scratch E2E sensitivity carrier。

neutral seed2021现已返回：40/40 complete，function separation/carrier skill/crossing均5/5，sample × bin oracle
macro 7.6753%，canonical-vs-random 0.8945%且5/5正。train-only fixed scale跨datasets落在48/360/720，说明一个
固定scope没有统一支配。该结果是problem evidence，不是PCSD performance，随后已由A6与multi-seed gate复核。

A6-natural也已返回5/5 crossing。three-seed confirmation进一步确认：neutral/A6均为5/5 stable crossing；扣除
validation-best fixed scale后的strict oracle为7.1107%/9.1259%，再扣除每个future bin固定scale后的
sample-specific headroom仍为6.7948%/8.5990%。因此adaptive coupling problem已从single-seed clue升级为
dual-carrier、three-seed direct evidence。contiguity仅在两carrier各4/5 datasets稳定，不能claim universal
temporal grouping law。另一方面，GroupedMLP相对A6-LBF H720仍落后2.6886%，所以D14-A确认的是研究问题，
不是Contribution 1 method performance。

### Retained Contribution 1 Candidate: SIFF_EQUAL v2

`SIFF_EQUAL`在PCSD coupling field上用ordered continuous scale coordinate生成共享history modes，并由
equal-skill objective缓解same-run arms的credit starvation。Step9确认equal-skill确实把SIFF arm loss CV从
`111.85%`降到`3.45%`，并形成`6.39%` oracle headroom；该训练机制按设计工作，但不是足够的paper-facing
Contribution 1。

新冻结的`SC1-SIFF-v2-EQ-ATTR-v1`不再只问“SIFF是否比PCSD更准”，而是要求完整链条同时成立：

1. 相对A6_FULL、A6_MEASURE和PCSD_EQUAL具有paper-facing effectiveness；
2. 相对constant、permuted、Q1-wide和independent EQUAL controls具有architecture specificity；
3. arms、policy与ordered component内部路径健康且不collapse；
4. 失败时按hypothesis/intervention/readout/optimization/capacity归因，不能用oracle headroom挽救negative gate。

50/50 runs与200/200 test cells通过protocol。`SIFF_EQUAL`分别超过A6_FULL、PCSD_EQUAL、constant、permuted与
Q1-wide controls，但没有超过A6_MEASURE，且对independent control的`+0.2580%`未达冻结margin。内部路径7/7
健康不能挽救negative effectiveness/attribution gate。实验结论仍为
`performance_partial_pass_attribution_blocked`；portfolio status更新为`frozen_performance_near_candidate`，
confirmation未授权。完整不可变清单见
`configs/stage_c_siff_equal_attribution_v1_candidate_freeze.json`。

[Retained Evidence] ordered scale information相对constant/permuted/Q1-wide有稳定局部价值，equal-skill能修复
SIFF-specific arm starvation，D14 coupling-crossing problem仍成立。下一轮不能继续微调当前SIFF；必须先解释为何
simple A6_MEASURE已经取得主要收益，以及如何把multi-arm conditional headroom转成超过A6_MEASURE和independent
scope的fused forecast。现有artifact进一步显示：SIFF_EQUAL policy的best-arm match仅29.24%、skill alignment
0.0277；two-fold static convex fusion比learned fusion高2.2112%，而bounded affine相对convex仅多0.1203%。
因此首要瓶颈定位为fusion calibration/information set，而非convex geometry；当前正式进入以v1为parent的Step4
source-informed redesign，优先评估arm-contrast-aware policy与synchronous competence calibration。详见
`analysis/stage_c_siff_candidate_step4_source_audit_20260718/source_informed_improvement_audit.md`。

Step5现已完成关键修正与理论审计。旧PCC已覆盖same-forward detached-error route supervision，因此calibration loss
本身不再claim novelty；CCSF的核心变量收紧为`policy读取target-free arm contrast`。现有probe的two-fold
diagnostic 5/5 gates通过：contrast相对coordinate-only expected arm MSE allocation为`+1.8348%`且10/10 folds
为正，相对shuffled contrast为`+1.7085%`且10/10，best-arm accuracy由existing policy的28.83%升至44.14%。

provisional CCSF在v1 logits上加入scope-shared contrast-conditioned correction，令correction为零时严格退化到v1；
完整T domain计算后再crop，projectivity保持。training只保留为co-designed weak supervision：使用relative regret而非
旧PCC的cross-arm std standardization，并以teacher entropy downweight ambiguous cells。显式A6_MEASURE anchor branch
因capacity/ensemble confound退出method，A6_MEASURE继续作为mandatory baseline。status=
`conditional_theory_pass_to_step6 / implementation_false`。详见
`analysis/stage_c_siff_ccsf_step5_theory_20260718/step5_theory_feasibility.md`。

Step6已把候选冻结为`SC1-SIFF-v2-CCSF-v1-preimplementation`。Contribution 1是projective coupling scopes上的
target-free arm-contrast-conditioned fusion；Contribution 2是与该information path共同设计的confidence-weighted
relative competence calibration，而不是泛称首个error-supervised routing。核心2×2为
`SIFF-v1/CCSF × EQUAL/RELCAL`；另加入A6_MEASURE、standardized teacher、zero-contrast same-capacity、permuted
contrast与parameter-matched independent-field controls。完整Phase A为10 arms × 5 datasets × seed2021 = 50 runs/
200 official-test cells，但本Step只通过5/5 static gates并授权Step7A local implementation；validation temperature
pilot、remote、formal test与confirmation仍为false。两项paper claim只有在10项hard comparisons与内部健康层同时
通过后才成立；否则按architecture/objective/field/numeric层分别回滚。详见
`analysis/stage_c_siff_ccsf_step6_20260718/step6_narrative_control_gate.md`。

Step7A production implementation现已通过18/18 local categories。CCSF不改Encoder/SIFF arms，而是在
`arms [B,C,S,T]`后构造`contrast [B,C,T,S,6]`，以共享`43 -> 64 -> 1` scorer修正v1 logits；新增2,881
parameters。相同seed下三个ordered CCSF controls与v1的base hash一致、initial output gap为0；四类CCSF readout
在5个prefix上的projectivity gap均为0。relative teacher满足stop-gradient、scale invariance与tie-confidence contract，
zero/permuted controls的tensor/gradient semantics及两步optimization path均通过。50-job adapters、30 constructors、
10 objective gradient paths与diagnostic tensors已就绪，但所有证据均为synthetic construction，不包含dataset
training。status=`step7a_local_pass_step7b_next`；15-run temperature pilot、remote、test、confirmation仍为false。
详见`analysis/stage_c_siff_ccsf_step7a_20260718/step7a_implementation_gate_report.md`。

Step7B现已冻结validation-only shared-temperature pilot并通过14/14 prelaunch categories。只训练
`ccsf_relcal`，使用5 datasets × 3 temperatures × seed2021 = 15 runs/60 validation cells；每个run由四horizon
validation MSE平均选择checkpoint，再以5×4 macro validation MSE选择一个全局共享temperature。并列时按预注册
规则选择更大值，不允许per-dataset/test选择。pilot checkpoint不复用；选定temperature后才创建formal candidate并
重新训练完整10-arm matrix。当前只授权pilot remote，formal Phase A、official test与confirmation仍为false。
详见`analysis/stage_c_siff_ccsf_step7b_prelaunch_20260718/prelaunch_report.md`。

Step8 validation pilot已从commit`06d0ffc`在3张RTX 3090上启动。首批三个Weather temperatures分别占用
GPU0/1/2；resource smoke、15-job dry-run与no-test guard均通过。运行期间不值守、不改变协议；完成后必须先验证
15/15 runs和60/60 validation cells，再选择唯一shared temperature。该pilot不是effectiveness gate，formal Phase A、
official test与confirmation仍为false。启动记录见
`analysis/stage_c_siff_ccsf_temperature_pilot_step8_remote_20260718/remote_launch_record.md`。

[Runtime Correction] 首次driver虽退出，但实际为0/15：三个Weather temperatures均在首次parameter update后进入
NaN，未产生checkpoint/metrics/selection，test未访问。原因定位为zero contrast时
`sqrt(mean(group_contrast²))`的0点反向不定义；这属于`optimization_or_numeric_pathology`，不能拒绝CCSF方向。
加入同一`1e-6` epsilon后，identical-arm NaN gradients由7200降为0，三temperature × 三AdamW steps共9/9 finite；
prelaunch现为15/15。下一步只执行三batch remote smoke与同协议retry，不改变temperature/dataset/selection。
详见`analysis/stage_c_siff_ccsf_runtime_repair_20260718/runtime_failure_and_repair_report.md`。

repair commit`7045c80`的真实Weather三batch smoke已通过，checkpoint与metrics finite。retry1已于15:54:21在
GPU0/1/2重启首批三个Weather temperatures，独立external root避免与0/15失败attempt混合。当前不值守、不访问
test；完成后先做完整性与shared-temperature selection audit。启动记录见
`analysis/stage_c_siff_ccsf_temperature_pilot_retry1_step8_remote_20260718/remote_relaunch_record.md`。

retry1结果现已15/15 runs、60/60 validation cells并通过9/9 audit。shared selection固定tau0.25：macro validation
MSE/MAE=`0.568165/0.453679`，相对tau0.1 MSE +0.1415%、相对tau0.05 +0.2991%；17/20 cells、4/5 datasets、
4/4 horizons最优，ETTm1中长horizon是明确例外。该margin只用于hyperparameter selection，不是mechanism evidence。
formal candidate现冻结为`SC1-SIFF-v2-CCSF-v1-tau25`，pilot weights不复用。下一步只实现50-run正式Phase-A
prelaunch tooling，remote/test仍为false。详见
`analysis/stage_c_siff_ccsf_temperature_pilot_retry1_result_20260718/pilot_result_and_candidate_freeze.md`。

formal Phase-A Step7B prelaunch现已15/15通过。新runner固定10 arms × 5 datasets × seed2021 = 50个from-scratch
runs与200个official-test标准cells；每个run仍只用validation四horizon mean MSE选checkpoint，test evaluator验证
授权并禁止checkpoint mutation。CCSF内部artifact新增final/base policy、base/correction logits与contrast descriptor，
four-layer analyzer将10项hard comparisons、interaction、internal health和failure attribution分开。Phase A/test现已
授权，confirmation仍为false；下一步是commit/push、3090 resource smoke与正式启动。详见
`analysis/stage_c_siff_ccsf_v1_tau25_phase_a_20260718/formal_phase_a_prelaunch_report.md`。

commit`604e1b8`已在3090完成remote prelaunch复核与Weather/CCSF_RELCAL三batch resource smoke，正式50-run
Phase A于`2026-07-18T17:27:08+08:00`使用GPU0/1/2启动。首批为Weather的A6_MEASURE、SIFF-v1 EQUAL与
SIFF-v1 RELCAL。当前只记录running状态，不预判effectiveness；完成后执行完整four-layer Step9，confirmation仍
未授权。启动记录见`analysis/stage_c_siff_ccsf_v1_tau25_phase_a_20260718/remote_launch_record.md`。

formal artifacts现已50/50 runs、200/200 official-test cells完整。CCSF_RELCAL相对A6_MEASURE为
`-0.8567%` MSE/`-0.7251%` MAE，相对SIFF-v1 EQUAL为`-0.6159%/-0.3262%`；architecture-only、
objective-only、true-vs-zero contrast与ordered-vs-independent均为负。唯一正向hard control是true contrast
相对permuted `+0.3568%`，只能说明semantic sensitivity，不能说明net utility。confirmation因此取消。

随后D2-D4完成failure attribution：region aggregation能把contrast的expected-arm specificity稳定提高到
`+1.29%–+1.87%`，但相对pointwise的额外mixture utility不足；simplex mixture相对best single arm只多
`+1.34%–+1.38%`，不支持covariance为主矛盾；global sharpening/hard routing在scope-native widths全部失败。
因此`contrast descriptor + competence teacher + readout temperature`整条exact route关闭。SIFF-v2-EQ-ATTR-v1
只作为性能接近发表水平但归因阻塞的immutable parent，当前论文没有可声称的两项core contributions，正式回到
Step2/4重构。详见
`analysis/stage_c_siff_ccsf_v1_tau25_phase_a_20260718/step9_four_layer_and_redesign_audit.md`。

### Historical Contribution 1 Parent: PCSD-CF

`PCSD-CF`（Projective Coupling-Spectrum Decoder with a Coupling Field）在固定future domain上表示多个
output-sharing scopes：

$$
\mathcal S=\{1,b_1,b_2,\ldots,T\},\qquad
\hat y_\tau=\sum_{s\in\mathcal S}\alpha_s(X,\tau)\hat y_\tau^{(s)}.
$$

$s=1$接近Direct/independent query，$s=T$包含A6 global MIMO-like arm，中间$s$是parallel block scopes。
它不是五个完整models：同一history state先生成一个shared mode field，future-coordinate descriptors再按scope
pooling成point/block/global predictive states，并共享target synthesis rows。requested $H$不进入operator或
policy，只执行$F_H=\mathcal R_HF_T$。

固定coordinate field的constant mode与zero-mean modes允许构造任意A6 mapping：令nonconstant mode maps及
nonlinear synthesis weights为零，global pooling即精确退化为`coeff [B,C,256] × basis [T,256]`。这是
function-class containment，不是warm-start或learned-capacity preservation。首个control只使用actual fused
forecast loss端到端训练，不加入risk、oracle、balance、diversity或counterfactual auxiliary loss。

[Novelty Boundary] Direct/MIMO/DIRMO/Stratify已覆盖固定strategy与block-size continuum；CATS、MQTransformer、
TimePerceiver已覆盖future/target queries；Implicit Forecaster已覆盖global wave decoding。因此PCSD只能claim完整
链条：`one projective parameter field -> scope pooling changes future-output state sharing -> simultaneous
point-to-global operators -> exact A6 subspace -> sample/target policy -> no requested-H semantics`。
DeepONet/PoU-MoE已覆盖coordinate synthesis与local operator mixture，因此这些primitive不计novelty。
Step7A production implementation现已通过全部9类local gates：five-profile shape/integration、65个dense/arbitrary
prefix cases、float32/float64 A6 containment、720/15/5/2/1 Jacobian-sharing topology、random-parameter arm
separation、partition-only parameter equality、module与真实Encoder-PCSD two-step gradients、static accounting及
protocol exclusion均通过。float32 containment maximum output gap为`3.815e-6`，float64为`3.109e-15`；
修正fan-in初始化后canonical/random minimum pairwise arm NRMSE为`0.131493/0.023079`。PCSD field core相对
A6 decoder参数为
`3.0291-3.6184x`，含policy为`3.1006-3.7224x`，所以dense capacity control仍是Step7B硬要求。
status=`pcsd_cf_step7a_local_pass / effectiveness_unready`。

Step7B production/prelaunch现已冻结：A6、exact-paired M0、five fixed scopes、equal、static-target、direct、random
partition与dense nonlinear matched共12 arms × 5 datasets。A6/M0相同seed的operator hash与初始输出gap均为`0`；
full PCSD arms共享完全相同的trainable initialization；dense control参数gap低于`0.1%`。primary metric在结果返回前
固定为validation dense-H1..720 MSE AUC，test=false。seed2021现已60/60返回：DIRECT相对A6为0/5、macro
`-1.5833%`，plain-training method gate失败；相对dense matched为5/5、`+2.3492%`，相对random为3/5、
`+0.4499%`。25/25 DIRECT same-run arms相对对应fixed E2E training退化，median `89.95%`，因此状态收紧为
`partial_representation_signal_training_blocked`；不启动confirmation seeds。

`SC-D15-T1` frozen official test audit已于2026-07-16完成60/60。DIRECT相对A6 test macro `-1.3994%`、1/5；
相对equal/static/dense/random分别为`-0.4984/-0.5304/-0.8942/-0.1164%`，method gate失败。validation上的
`DIRECT > dense` +2.3492%在test反转，但DIRECT相对A6在validation/test均整体失败。因此exact PCSD-CF-v1
status=`rejected_effectiveness_test`，不得作为paper claim或运行confirmation seeds。

test仍显示same-run oracle macro +2.0197%、3/5 datasets为正，且25/25 DIRECT arms低于对应independent fixed
training，median degradation 90.6647%。这保留了joint credit starvation线索，但不证明PCC有效；预注册decision为
`test_fail_with_arm_headroom`，只授权PCC进入`test_informed` Step6 design。

### Retired Core Candidate: CCRL

`CCRL`（Cross-fitted Coupling-Regret Learning）不期待ordinary mixture loss自动产生expert specialization。它在
train split内chronological cross-fit每个coupling arm，对held-out training sample与target region构造centered
relative risk：

$$
r^{cf}_{i,b,s}=L^{cf}_{i,b,s}-\frac1{|\mathcal S|}\sum_jL^{cf}_{i,b,j}.
$$

policy仅由inference可见history与target coordinate预测relative risk。Step4-6审计已纠正原soft-label解释：对MSE，
weighted expert risk只是mixture loss上界，不能把$\operatorname{softmax}(-r)$称为optimal fusion。因此候选training
principle固定为actual fused forecast loss + auxiliary cross-fitted risk distillation；matched direct-fusion是mandatory
control。

[Retirement Reason] FFORMA/TimeFuse已覆盖feature-based sample fusion；TimeRouter已覆盖oracle labels、context/CV/
forecast features与nonlinear routing。更关键的是，CCRL需要独立fold experts生成稀疏OOF labels，再监督共享
representation的最终PCSD；teacher/student architecture不一致，labels会随joint arms更新而stale，额外工程不属于
最终推理图。它技术上可作辅助loss，但不是统一的single-run end-to-end training principle。

[Decision] status=`diagnostic_only_not_scheduled`；D14-B1在Step7A前取消。旧source/theory/config保留为历史control，
不得继续实现或作为Contribution 2 claim。

### Contribution 2 Candidate: PCC-v1-TI

第二个contribution必须原生依赖PCSD的same-run arms，并直接服务最终fused forecast。D15-A现已证明generic
capacity/numeric不是主因，并发现25/25 DIRECT-run arms相对相同scope独立fixed E2E training退化，median
89.95%；learned policy虽非one-hot collapse，但future-bin usage variation仅L1 0.0051-0.0440。因此旧working
hypothesis `SC2-ICC`先收紧为`SC2-PCC-v0`（Projective Coupling Credit）。但Step6 external audit发现forecasting
Expert Loss Integration已覆盖direct expert loss，ICLR 2026 graph MoE也已覆盖negative per-expert loss teacher、
gate KL与uniform warm-up。故pointwise v0不再作为paper candidate，只保留mandatory control。

新candidate `SC2-PCC-v1-TI`从全部dense nested prefixes构造scope risk
$R_s(H)=H^{-1}\sum_{t\le H}e_s(t)$与capability $q_s(H)$，再按prefix-target incidence输运到natural target：

$$
c_s(t)=\frac{\sum_{H=t}^{T}q_s(H)/H}{\sum_{H=t}^{T}1/H}.
$$

这样router仍不接收requested $H$，但其target明确核算target $t$在全部包含它的forecast prefixes中的scope credit。
带floor的transport训练arms，无floor transport训练router；continuous schedule从equal skill平滑过渡到capability
credit，不冻结参数、不使用teacher checkpoint、EMA或second forward。19/19 Step5b/6 cases通过，exact transport
identity gap为`0`，crossed case相对pointwise target差`0.616407`。

[Novelty Boundary] generic expert loss、loss-teacher gate、warm-up、load balancing、router-expert coupling与gradient
surgery均不计novelty。完整claim只允许落在
`nested projective risks -> harmonic credit transport -> target-coordinate coupling arm/router co-training`；且必须在
同一matrix超过`POINTWISE_PCC_V0`与`POINTWISE_PRIOR_COMPOSED`才能保留。

Step7A现已完成：35/35 local gates通过，vectorized/direct loop最大差`2.22e-16`，raw-scale arm fusion
gap`8.88e-16`，默认inference output、parameter count与prefix projectivity不变；real PCSD batch五个scope均获非零
auxiliary output gradient。status=`step7a_pass_prelaunch_audit_next`；这些结果只证明implementation correctness，
不证明performance；该节点只曾授权Step7B prelaunch audit。

Step7B最终45/45完成。full PCC相对A6为`+0.9627%`、3/5，相对plain为`+2.4927%`、5/5，25/25 arms
恢复；但相对closest prior composition仅`+0.1050%`，且pairwise arm-output diversity只保留plain的
`20.57%-41.13%`。`EQUAL_SKILL`已解释full PCC相对A6 gain的88.90%。因此exact PCC-v1-TI status=
`validation_screen_failed_exact_design`：不进入Phase B、confirmation或test，回Step4。

### Historical Validation-Negative Pair: SIFF-v1 / MCCA-v1

原PCSD-CF/PCC pair只保留为problem evidence：前者暴露arm starvation，后者证明direct supervision可恢复skill，
同时暴露same-label homogenization。Step6已把新pair冻结为：

1. `SC1-SIFF-v1`：固定$Q=2$的Scale-Indexed Forecast Field，用continuous log-scale basis从同一history生成
   `modes [B,C,S,D,K]`，再经真实point/block/global pooling生成完整$T=720$ function；
2. `SC2-MCCA-v1`：在all-prefix projective target measure下，把与PCC**完全相同的per-scope total skill mass**从
   per-target uniform floor改为competitive target assignment。

两者都不输入requested $H$，也不使用teacher、second forward或two-stage labels。conditioned neural fields、
HyperDeepONet、DirMO/Stratify、BASE、Expert Choice与Sinkhorn已覆盖底层primitives，因此允许的paper claim只能是
task-specific complete chain，不能把scale coordinate或balanced OT单独写成创新。

Step6 22/22通过：SIFF Q1/constant containment成立；five-profile Q1-wide/independent matched controls最大parameter
gap `0.3893%`；MCCA float32 marginal gap `1.04e-7`，与PCC same column mass gap `2.98e-8`。实验固定为
`PCSD/SIFF × EQUAL/PCC/MCCA`的$2\times3$ factorial和七个归因controls。status=
`conditional_narrative_pass / effectiveness_pending`。Step7A production gate随后36/36通过：Q1/A6 exact gap `0`，
constant collapse `3.55e-15`，float32 MCCA marginal gap `4.47e-8`，same-mass PCC gap最大`2.78e-17`。Step7B
prelaunch 8/8通过并冻结55个new runs；最终55/55 new与25/25 references均有效，test=false。

Step9/10 formal result：

- SIFF architecture main effect `-1.5015%`、2/5 datasets；
- MCCA over same-mass PCC `-0.0250%`、2/5；
- SIFF+MCCA over A6 `-0.5621%`、4/5；
- ordered over permuted `+1.1177%`、5/5，但ordered未超过Q1-wide/independent macro gate；
- PCSD MCCA over PCC为`-0.1092%`、0/5；transport over pointwise为`+0.4736%`、4/5，
  capability marginal over uniform OT为`+0.1182%`、5/5。

[Historical Decision] 在当时的validation-only、best-H720 checkpoint规则下停止exact SIFF-v1/MCCA-v1，
不进入confirmation、Phase B或test。后续fair test已把SIFF修正为partial pass；MCCA没有进入70-run公平复评，
因此其当前边界是`historical_validation_negative / fair_test_not_reaudited / inactive`，不能写成formal
test rejection。旧证据反对完整competitive assignment，transport与capability marginal仍只保留为ingredients。

按新的paper-facing评估规则，旧artifacts已在validation H96/H192/H336/H720上回溯重算：SIFF architecture
main effect为`-2.3509%`、8/20 cells、2/5 datasets；MCCA为`-0.1357%`、7/20、1/5；joint vs A6为
`-1.3325%`、14/20、4/5，三项仍fail。SIFF architecture按horizon为H96 `-6.3186%`、H192
`-2.6027%`、H336 `-1.0522%`、H720 `+0.5698%`，所以short/mid-to-long tradeoff在标准论文horizon下仍成立，
不是dense H1单独造成。该结果继承历史best-H720 checkpoint、未访问test；exact pair关闭不变。

SIFF的失败存在明确short-prefix pathology：ETTm2 SIFF+MCCA相对PCSD+MCCA在H1为`-669.49%`，H720却为
`+0.6013%`。dense all-prefix MSE AUC的target measure为：

$$
\frac1T\sum_{H=1}^{T}\frac1H\sum_{t\le H}e_t
=\sum_{t=1}^{T}\left(\frac1T\sum_{H=t}^{T}\frac1H\right)e_t.
$$

code audit确认PCSD/SIFF EQUAL/PCC/MCCA的fused training loss已使用同一exact harmonic target measure，但
error norm为L1；所有checkpoint仍按H720 MSE选择，primary screen是dense MSE AUC。因此architecture paired
failure不能归因于flat training，未决边界收紧为Q2 readout optimization、L1/MSE mismatch或checkpoint selection。
由于H1出现>100%局部恶化，formal status=
`validation_screen_failed_exact_design / diagnostic_invalid_for_direction_rejection`：关闭exact v1 candidate，
但在per-epoch trajectory audit前不关闭ordered scale-coordinate问题类。

当前没有active contribution pair。`SC-D16` external-first audit已确认：NeurIPS 2024 ElasTST直接从uniform
random horizon推导harmonic horizon reweighting，并在官方实现中同时用于training与weighted validation
checkpoint；ICML 2024 Loss Shaping与ICLR 2026 QDF进一步覆盖per-step error shaping与non-uniform future-task
weights。因此`SC2-PHMA`作为standalone contribution已被narrative gate否决。

code audit同时否定了新增HR matrix的必要性：harmonic-L1 training本来就存在。保留的
`SC-D16-CTD`仅为`diagnostic_only_checkpoint_trajectory_audit`：拟先在ETTm2上复跑
PCSD-EQUAL/SIFF-EQUAL/SIFF-CONSTANT/Q1-WIDE四条完全matched trajectories，每epoch保存standard+dense risk，
并离线比较best-standard、best-H720、best-dense-MSE与best-dense-MAE。只有best-standard同时消除pathology并
恢复四horizon architecture/control effect，才扩展five-dataset validation confirmation。

Step5/6 v1.1现已冻结20-epoch no-stop trajectory、four-rule deduplicated state retention与kill gates：SIFF
best-standard H1/PCSD ratio必须$\le2$，并在四标准horizon同时超过PCSD、constant、Q1-wide，long-bin相对其
own best-H720退化不得超过1%。decision=`diagnostic_design_refrozen_v1_1_step7a_local_only`；当前
implementation/remote/test均false。2026-07-17该diagnostic被用户暂停；design保留但不占active cursor。

[Execution Order] D14-A problem confirmed -> CCRL retired -> PCSD-CF Step4-7 -> validation screen fail/credit clue ->
PCC-v0 Step2-5 -> frozen PCSD-CF-v1 test audit fail-with-headroom -> Step6 prior-art rollback -> PCC-v1-TI nested-risk
transport Step5b/6 pass -> Step7A 35/35 pass -> Step7B 45/45 validation screen -> prior specificity/diversity fail ->
Step4 scale-identifiability/competitive-credit redesign -> SIFF/MCCA Step5 10/10 theory pass -> Step6 22/22
source/method/control pass -> Step7A 36/36 implementation pass -> Step7B prelaunch 8/8 pass -> seed2021
validation-only 55/55 complete -> Step9/10 exact pair fail -> short-prefix measure pathology audit ->
SC-D16 source/code audit finds ElasTST prior and existing harmonic-L1 path -> PHMA/HR closed ->
CTD Step5/6 design pass -> user pauses CTD -> test-primary governance ->
PCSD/PCC/SIFF 70-run fair re-audit complete -> SIFF performance partial pass / attribution blocked ->
SIFF return Step6；PCC return Step2/4；CTD remains paused。
test、confirmation false。

### Closed Candidate: PRISM Decoder

`PRISM`（Prefix-Risk Isometric Synthesis Module）保留A6的free coefficient path：

$$
M[B,C,P,D]\rightarrow a[B,C,256]\rightarrow U_\mu[:H,:][H,256]\rightarrow\hat y_H[B,C,H].
$$

$U_\mu^TW_\mu U_\mu=I$，并用prefix family诱导的
$\mathbb E_H\|\operatorname{offdiag}(U^TW_HU)\|_F^2$控制short-prefix locality与global compaction的
Pareto tradeoff。$H$只crop rows，不进入learned path。D12-A未直接证伪其locality hypothesis，但其前置joint
problem gate失败，D12-B取消；status=`retired_without_effectiveness_test`。D6 crossing只保留为历史evidence。

### Closed Candidate: CAPE Frame Learning

`CAPE`（Cross-fitted Adaptive Predictable-Energy frame learning）不再修改future-step loss weights，而是用
train-only out-of-fold predictions估计$\Sigma_m=\operatorname{Cov}(\mathbb E[y\mid x])$。rank-limited frame
应最大化$\operatorname{tr}(U^TW_\mu\Sigma_mW_\mu U)$，避免raw-label PCA把capacity分配给不可预测noise。

`localization on/off × predictable/raw covariance`形成预注册`2x2` factorial。旧`MIPR`因D11不支持conflict、
benchmark measure headroom弱且与Time-o1/QDF/Loss Shaping邻近，降为`retired_as_core_candidate`；
$W_\mu$只保留为exact risk protocol/control。

D12-A-v1因uniform normalized risk mismatch不能方向级否定；修复后的v2复用相同pilots并与raw MSE对齐，
最终仅ETTh1支持，`1/5 < 3/5`。ETTm1/ETTm2/Weather的A6 raw gap@256仅`0.18%-0.34%`；CAPE
status=`failed_problem_gate / closed_as_core_candidate`。

完整source/theory audit与D12 gates见
`analysis/stage_c_d12_predictable_frame_feasibility_20260715/d12_final_result_and_rollback.md`。

### Historical Contribution 1 Record: Projective Forecast Operator Redesign

历史`narrative_ready`候选为`PMFO-RCT`。它从A6 history memory建立future interval tree，按
`90 -> 30 -> 10 -> 5 -> 1`逐层生成scaling/detail coefficients，并用fixed orthogonal contrast保证fine
detail不能改写parent coarse projection。目标性质：

- exact refinement recovery与nested-prefix consistency；
- $H$ 只prune与prefix相交的tree nodes，不进入learned state/query/router；
- parent-to-child shared state transition + orthogonal detail complement + local support；
- contribution来自future-side refinement conservativity与domain execution，不是“又一个wavelet/continuous
  basis decoder”。

PMFO-RCT v1已完成其falsification职责：theory/local invariants成立，但Step 7B相对A6的dense-MSE macro为
`-1.0955%`，三dataset均退化，故不能成为paper core。组件归因并不相同：conservative synthesis相对
no-conservation在三dataset一致改善（macro `+2.3393%`），保留为redesign证据；recursive transition相对
no-transition仅`+0.0486%`且跨dataset不一致，v1 claim撤回；structured decoder相对matched dense的
`+0.7193%`只是弱信号。

[Decision] 关闭范围仅是固定`90/30/10/5/1` mixed-radix partition、v1 state transition和整体替换A6
readout的组合。Contribution 1 slot与projectivity/conservation问题仍开放；回到Step 4重审function-class
containment、future partition与history-to-node interface。Step 7B没有操纵Encoder，不能据此认定Encoder不足。

Step 4 redesign audit已进一步确认：PMFO v1 readout有`212,010` parameters，而覆盖A6 rank-256 affine
operator family至少需要`316,112`维；相同256维latent不能称为capacity preservation。A6 effective operator
在fixed 90/30 boundaries上的jump ratio约`0.989-1.009`，8个PMFO root nodes的history-patch profile
cosine为`0.936-0.994`。因此function-family restriction、unsupported factorization与weak scale-native
interface均进入v1 failure attribution。

新provisional candidate为`Function-Preserving Multiresolution Operator Morphism (FPMO)`：把整个A6 future
operator改写到perfect-reconstruction multiresolution coordinates中，参数空间必须显式包含A6；ordered
memory直接进入scale coefficients，不经过shared recursive state作为唯一history path；$H$只选择与prefix
相交的supports。该候选不是“A6 output + residual patch”，也不能以tree、wavelet、lifting或network morphism
单项作为novelty。

[Fact] Step 5已构造任意正整数$T$的orthonormal interval transform，并在9个$T$、53个prefix cases上验证
exact A6 embedding、perfect reconstruction与native restriction，max algebraic gap=`5.329e-14`。因此
`FPMO-M0`可以在无dense bypass下完全复现A6；但它与A6只是bijective coordinate transform，只能作control。

要让history-to-scale path真正不同，`FPMO-DS`为各tree depth设置独立history factors。该class包含A6，
但T720下各group rank caps之和为720，等价于full affine readout。由此得到no-go boundary：exact包含全部
A6、independent scale states与总latent budget 256不能同时满足。params差异不用于否定方法，但full-affine
capacity必须由同function-class `FPMO-DA` control隔离。

[Fact] Step 6进一步证明，T720下linear DS与DA不仅capacity相近，而是拥有完全相同的full-affine
function class；对任意orthogonal coordinates与任意row grouping也成立。DS增加的是non-identifiable
deep-linear factorization。已有matrix-factorization工作说明这可能改变implicit optimization bias，但该
差异不是future-scale专属机制，也不能直接从其GD理论外推到当前Adam + L1 joint training。

[Decision] `FPMO-DS rejected_by_narrative_gate`。M0、DA与DS-L只保留为control/diagnostic artifacts，不进入
Step 7。普通per-scale nonlinear extension会破坏automatic exact A6 containment，并引入新的activation、
capacity与prior-art问题，必须作为新候选重新通过Step 2-5，不能事后挽救DS。Contribution 1 slot保持开放；
当时的cursor回到Step 2/3，以`SC1-D2`分离rank expansion、generic nonlinearity与true-scale alignment。

[Diagnostic] D2 formal5已完成165/165 frozen-memory fits且invariants pass。true interval basis相对random basis
macro `+3.0635%`，5/5 datasets、15/15 seeds为正；但true depth grouping相对same-basis random grouping只有
`+0.0947%`，仅2/5 datasets达到2/3 seeds为正，未过mandatory gate。因此精确的scale-grouping problem关闭，
rollback Step 2。basis signal当时尚未由完整$2\times2$ factorial识别为独立main effect，因此只授权了
`SC1-D3 crossed basis-group diagnostic`，而未升为decoder contribution。

[Fairness Boundary] 上述“关闭”仅针对frozen A6 representation上的final-head grouping设计，不能作为所有
end-to-end scale grouping的方向级否定；当前PLGO也不依赖该已测试grouping。

[Diagnostic] D3已补齐`random basis × random group` cell并形成15个dataset-checkpoint primary units。
basis main MSE reduction为`+2.9174%`，在true groups与random groups下分别为`+3.1164%/+2.7181%`；
5/5 datasets均通过方向一致性与interaction guard，MAE为`+2.3098%`。因此basis geometry作为独立probe
main effect获得支持，并授权返回Step 4。但exact depth grouping仍为false；balanced interval basis本身也受
Haar/wavelet/whitening prior art约束，尚不是Contribution 1。当前问题转为识别conditioning、energy
compaction、local-support或prefix compatibility中的真实机制，并证明其原生服务unified horizons。

[Diagnostic] D4已完成315/315 frozen-memory fits。balanced相对permuted interval的八horizon macro为`+1.6324%`，说明
contiguous locality在A6 representation/probe family中形成稳定conditional signal；但相对DCT-II与fit-only PCA分别为`-0.8609%/-1.5050%`，且相对random
interval tree仅`+0.2742%`。因此exact midpoint balancing与best-accuracy claim在该conditional probe中不成立，decision=
`standard_structured_basis_explains_gain_return_step2`。这不否定把balanced interval basis用于forecast
generation的组件级创新，而是要求paper-core novelty来自更完整的组合：future-prefix local support、
horizon-agnostic restriction、predictive conditioning与实际selective synthesis共同成立。

[Current Problem] 新问题暂命名`SC1-CLG`（Conditioning-Locality Gap）。D4的descriptive geometry显示
log MSE与coefficient covariance off-diagonal ratio的平均Spearman为`+0.8405`，与top-16 energy capture为
`-0.8357`；interval basis用约55个active atoms覆盖H48，而DCT/PCA需720个。下一步先判断local-support
orthogonal family能否在conditioning与prefix sparsity之间形成稳定Pareto improvement，而不是直接实现新head。
该problem现由`SC1-D5`检验：在同一frozen-memory head下，以fit-only geometry从预注册的block-local
DCT/PCA families选择`H48 active atoms <= 96`的basis，并对balanced、global DCT与global PCA做五dataset、
三checkpoint、八horizon比较。D5是problem diagnostic，不是新decoder候选。

[Diagnostic] D5 primary selector在15/15 units选择b96 PCA，但相对balanced仅`+0.0322%`，primary gate fail。
然而预注册的`block_dct2_b144`相对global DCT在short horizons约`+1.05%`、long horizons约`-1.15%`，
11/15 primary units同时short-positive与long-negative；其H48 active atoms为144而非720。因此按failure
attribution rule，D5只能否定`<=96 + offdiag selector`，不能否定local/global co-design。当前由D6在未使用的
validation batches 8-15确认support-scale × horizon interaction；该确认现已完成，但仍无已通过theory gate的
paper-core method。

[Strong Evidence] D6在disjoint validation window完成225/225并通过全部gates：b144相对global DCT的short
MSE为`+1.1964%`、long MSE为`-1.2675%`，12/15 primary units crossing；short-positive与long-negative
分别覆盖4/5和5/5 datasets。该结果支持A6 representation下存在support-scale interaction，并据此将SC1
problem收紧为：同一future function需要local-prefix synthesis与global-domain coherence，但requested H只
定义domain，不能成为learned semantic condition；它本身不证明任意end-to-end decoder都存在同样强度。

[Provisional Candidate] `SC1-PLGO`（Projective Local-Global Operator）通过Step 4 conditional narrative gate。
它不是“首次basis/wavelet forecast”：N-BEATS、N-HiTS、BasisFormer、FBM、WaveToken、Implicit Forecaster与
FlowState已占据相关单项。可辩护边界是global smooth atoms、interval-local supports、domain-only restriction与
selective synthesis的组合。balanced interval basis保留为local support scaffold；Step 4当时只授权进入
Step 5 stable reconstruction/function-class/capacity no-go audit。

[Fact] PLGO Step 5已构造Restricted-Global Nested Basis (`RGNB`)：root保留global DCT subspace，每个balanced
interval的detail是children scaling union相对parent的orthogonal complement。stable local Chebyshev chart修复了
raw restricted-DCT最高`3.110e17`的conditioning pathology；12个$(T,r_g)$、101个selected prefixes与
3,731个all-$H$ active-bound cases通过，max algebraic gap=`2.141e-13`。因此stable global-local synthesis、
arbitrary-prefix restriction与无dense bypass的A6 morphism均可行。

[Decision] 该结果只让mathematical scaffold通过。square `PLGO-ONB-M0`与A6是isometric
reparameterization；direct global/local frame有$r_g$维coefficient kernel；independent support-group maps在
T720的rank caps sum=720并退化为full affine。三者分别只能作为control、overcomplete control与rejected
capacity-confounded design。support pruning也不等于generator-level speedup，效率claim继续撤销。

[Hypothesis] Step 6唯一保留的问题是：一个不读取$H$的shared atom-conditioned generator，能否利用
support/scale/global-local descriptors原生生成active coefficients，并超过matched dense与random-descriptor
controls。prior-art primitive overlap只收紧claim，不自动否决该task-specific组合；但generator尚未通过
descriptor attribution与capacity controls，故PLGO为conditional narrative candidate，禁止直接训练method。

[Fact] Step 6进一步审计后，atomwise PAF tensor contract本身成立：33个prefix cases的coefficient subset、
prefix synthesis与paired-order invariance max gap=`4.547e-13`，且effective rank不超过256。external
source audit显示generic primitives已有DeepONet branch-trunk、NOMAD nonlinear decoder、BasisFormer basis
coefficient attention、TimePerceiver target queries与FlowState functional basis先例；这些先例用于约束
component claim和mandatory comparisons，不自动否决完整contribution。

[Strong Evidence] internal evidence也不允许直接实现：B11 basis-conditioned field的收益被no-basis与
constant-slot controls解释；B14 model-independent retrieval-demand只有`1/6` settings、`0/3` datasets通过。
因此atom-to-memory retrieval被明确删除，narrowed PAF只能读取与A6一致的shared flattened memory。

[Decision] `SC1-PLGO-PAF` Step 6按完整`problem-constraint-mechanism-implementation-claim`链条获得
conditional narrative pass。不把generic atom query、branch-trunk或HyperNetwork单独写成创新；Contribution 1
的候选边界是multi-horizon projective contract、RGNB local/global support geometry与atomwise generation的
组合。D7 frozen-memory diagnostic只负责conditional geometry attribution，不能替代Step 7 end-to-end gate。

[Strong Evidence] D7完成105/105 frozen-memory fits并使用fresh validation batches16-23。canonical RGNB
descriptors相对PERM/RANDOM在compact/matched widths分别提升MSE `+13.8034%/+12.8418%`、MAE
`+9.8581%/+9.3269%`，两个width均覆盖5/5 datasets；gain在H48最强并随horizon增长减弱，与D6 short-prefix
local-support evidence一致。

[Protocol Correction] D7相对free-M0的`-37.3836%/-39.1031%`是frozen A6 representation上的compatibility
gap，不是method-readiness gate。A6 Encoder由A6 decoder共同塑造，free-M0天然兼容该representation；PAF
replacement head没有机会反向塑造Encoder。raw metrics不变，但原“exact PAF v1失败并返回Step4”结论撤销。

[Decision] PAF恢复为`narrative_ready`，下一步是`SC1-D8-E2E`：A6、GEO、PERM、RANDOM compact/matched
七arms全部from scratch端到端joint training，五dataset seed2021先screen，再对decisive arms做三seed确认。
只有stable E2E PAF仍失败，才返回Step4 capacity-preserving redesign；frozen cross-swap不再作为primary gate。

[Fact] D8 Step7A已通过：五profiles × 七arms的210个shape-prefix与35个gradient cases全部通过；full-prefix
max gap=`2.384e-6`，flatten/patch-block-sum max gap=`5.722e-6`。runner dry-run生成35 jobs，analyzer
synthetic gate通过，method screening在CLI层禁止test。该证据只授权Step7B，不构成effectiveness结果。

[Strong Evidence] D8 Step7B完成35/35 validation-only runs。GEO-c256相对same-run A6 dense MSE macro
`-28.10%`、5/5 datasets均负，MAE macro `-20.54%`，因此exact shared-latent PAF不能成为paper-core方法。
但GEO相对PERM/RANDOM median为`+14.33%`、5/5为正；m694下geometry effect仍为`+14.71%`，而width
扩展只比c256回收`+0.58%`。geometry mechanism成立，width/capacity不是主要失败原因。

[Failure Attribution] GEO五dataset均hit epoch cap，但最后5 epochs validation只改善`0.02%–0.49%`；没有
divergence、NaN或>100% degradation。patch entropy在4/5 datasets下降，但ETTh1 entropy几乎不变仍退化
`35.62%`。因此关闭的是$\alpha_j=\psi(d_j)^TAh$这一exact shared/separable readout，而非RGNB、
projectivity或PLGO方向。当前回Step4审计patch-level intervention与readout function class；不做三seed或
无边界longer-epoch sweep，SC2继续held。

[Tensor Boundary] `memory [B,C,P,D] -> hidden [B,C,R]`是$R=PD$的bijective flatten，不是pooling；patch
identity没有在这一步丢失。A6与PAF都随后执行$R\rightarrow256$投影。PAF的真实风险是shared latent与
descriptor-generated atom map构成的separable history-atom interaction，而不是shape从四维变三维本身。D8
强制报告patch-block contributions与atom-patch Jacobian；B14未支持的atom retrieval不会未经新Step4-6直接加入。

[Step 4 Redesign Decision] source-informed audit进一步证明，单纯增加geometry-only branches不能解除该失败：
任何

$$
\alpha_j=\sum_e\pi_e(d_j)\psi_e(d_j)^TA_eh
$$

都可把加权trunks与branch matrices拼接为一个更宽的PAF。固定总rank时function class不变；扩rank时收益可由
capacity解释。因此“scale/atom experts”本身不进入paper core。直接atom-to-patch cross-attention也不推进：
flatten是bijective reshape，且history patch与future atom没有已证实的canonical alignment，B14与
OFormer/GNOT/BasisFormer/TimePerceiver均对该shortcut形成压力。

[Method Candidate] 当前只保留`SC1-JAPO`（Joint Atom-History Projective Operator）进入Step8 effectiveness screen。
它用free RGNB expert maps生成atom coefficients，但gate必须同时依赖history context与atom support geometry；
requested $H$仍只选择active atoms，不进入router。与geometry-only mixture不同，history-dependent gate不能吸收为
fixed temporal table，因此有机会解除D8的fixed separability。令所有experts表示同一A6-equivalent RGNB map时，
任意convex gate仍复现A6，故候选原则上无需dense bypass即可包含A6。

[Theory Result] Step5已在4个$T$、22个prefix cases上验证A6 containment与exact projectivity，最大误差分别为
`1.137e-13/1.172e-13`；constructive $f(h)=h\tanh(h)$证明joint gate严格超出fixed affine PAF，而
geometry-only mixture仍collapse为fixed operator。requested $H$不进入learned path。

[Optimization Boundary] exact containment只是function-class guarantee。identical experts会令router gradient为0并
保持expert symmetry，因此首版必须independent from-scratch initialization，不能把containment构造用作复制初始化。

[Step6 Design] JAPO固定为两个independent rank-256 RGNB coefficient experts，history与8维atom geometry分别
投影到$G=32$后以multiplicative feature形成expert-only softmax。identical copy、warm-start、hard top-k、active-atom
normalization、explicit H和auxiliary routing loss均禁止。五profiles design checker的projectivity最大误差
`3.331e-16`，initial entropy最低`0.999855`，所有joint gradient paths通过。

[Step7A Implementation] production `JAPOReadout`、six same-bank modes、checkpoint invariants与validation-only
runner/analyzer已落地。五profiles × 七arms的210/210 prefix与35/35 gradient cases通过；最大prefix gap
`4.768e-7`，patch-block rewrite gap `5.722e-6`；Encoder与expert-bank paired initialization hashes通过。

[Step8 Evidence] seed2021的35/35 validation-only runs、paired from-scratch initialization与全部artifact/invariant
audit均通过。JOINT相对A6的dense MSE macro为`-1.3754%`、0/5 datasets正向；相对same-bank control median为
`-0.0780%`、2/5正向。该结果没有达到严重失败阈值，也没有达到provisional-pass阈值，因此是
`inconclusive`而不是pass或方向级fail。router normalized entropy最低`0.993263`，提示当前训练可能未形成明显
expert specialization，但单seed不能区分optimization variance与exact design weakness。

[Step8 Decision] seed2022 unchanged matrix完成后，70/70 artifacts与paired contracts全部通过。two-seed mean下
JOINT相对A6为`-1.2435%`、0/5，且相对same-bank median为`-0.1175%`、仅1/5；UNIFORM/HISTORY/ATOM均在
macro上优于JOINT，触发`capacity_control_explains` hard gate。两个seed的router entropy都接近1，说明weak
specialization可重复。`SC1-JAPO exact v1`因此降为`failed_as_core_candidate`，seed2023停止。

[Rollback Boundary] 本结果否定当前`two free RGNB experts + factorized softmax weak mixing`作为paper-core实现，
不否定A6 containment、RGNB projectivity、canonical geometry的PERM/RANDOM小幅正向信号，亦不否定conditional
projective operator方向。Contribution 1回Step4 source-informed redesign；新candidate过Step4-6前不实现，test、
SC2-MIPR与joint factorial继续held。

[Systematic Review] 全阶段证据支持的不是“再加一个basis/router”，而是更窄的问题：A6已具有强free operator与
domain-only prefix consistency，RGNB提供future-side local/global support坐标；尚未证明的是history memory中是否
存在与这些support尺度可识别对应的operator structure。flatten `[B,C,P,D] -> [B,C,PD]`为bijective reshape，
所以当前不把信息压缩当作失败原因，而把“multiscale structure是否可访问、是否值得显式建模”作为待检验问题。

[Next Diagnostic] `SC1-D9 History-Support Operator Evidence Audit`预注册为`diagnostic_only`。D9-A先精确恢复
A6 memory-to-future operator，把history侧分成global/coarse/mid/local scale coordinates，把future侧分成global
root与local support/detail coordinates，并与atom-label permutation和random orthogonal history bases作matched
comparison；只有A通过才做sample-dependent input-Jacobian确认。D9通过只说明新local-global operator具有
existence evidence，不能证明method effectiveness；失败则回Step2/3，而不是继续叠加MoE、router或training loss。详细复盘见
`analysis/stage_c_sc1_post_japo_systematic_review_20260715/systematic_stage_review.md`。

[D9-A Result] 15/15 exact operator audits与Parseval invariant通过，但ordered scale hypothesis未过gate：
five-dataset macro rho=`0.173810`，positive effect datasets=`2/5`，atom-label permutation=`1/5`，
random-history-basis=`0/5`。details相对global root整体更偏高频在15/15 units出现，但details depth 0-5内部不
单调；该binary现象是post-hoc clue，不能挽救primary result。Contribution 1因此回Step2/3，D9-B取消。

[New Problem Boundary] 下一步只设计`SC1-D10 Raw History–Future Scale Identifiability`，在独立raw-data evidence
上区分binary global/detail、monotone multiscale与no-scale三种hypotheses。只有problem存在性与matched controls
通过后，才允许重新形成Step4 architecture candidate；当前new model、test、SC2与factorial均不授权。

[D10 Frozen Design] history使用七个DCT frequency bands，future使用RGNB global root与六层details；两侧group
sizes天然相同，但每个cell仍whiten并固定为16→16 sketched ridge以排除capacity差异。binary gate使用独立2×2
global/detail interaction，monotone gate只检查details内部6×6 diagonal，防止global/detail粗二分伪造多尺度证据。
fit/holdout按train时间区间隔离，final evidence只用official validation；paired history/future permutations为mandatory
controls。当前授权仅限diagnostic，不是model implementation。

[D10 Result] binary hypothesis所有五项gate均失败：effect/control只有2/5，两个directional selectivities同时为正
为0/5。detail-only diagonal相对median与paired controls在4/5 datasets为正，但canonical band从未在任何dataset的
至少4/6 rows成为最佳，6! mapping也仅2/5通过。ETTh1/ETTh2、ETTm1/ETTm2呈现不同off-diagonal patterns，
Weather近零；不存在可支撑unified method的cross-dataset mapping。decision为
`raw_aligned_scale_not_supported_rollback_step2`。

[Boundary Reset] D9 learned-operator层与D10 raw-data层共同关闭history-scale aligned routing。future-side RGNB、
projectivity与D6 horizon-support crossing仍保留。下一Step2问题暂定为future global/local components在不同prefix
losses下的error/gradient responsibility；在D11 source/theory audit前，不恢复SC2，也不实现adaptive router、new
decoder或loss。

[D11 Step2/3] source audit确认Time-o1已经直接提出transformed label alignment、label autocorrelation与
forecast-step task overload；FreDF、DBLoss及withdrawn Hybrid Loss进一步覆盖frequency/component loss与动态调权。
因此论文不能把“分解future再加component loss”作为创新。D11将边界收紧为prefix measure下的exact
future-component gradient responsibility与intervention-point diagnosis。

[D11 Frozen Diagnostic] 对complete orthogonal projectors有
$\sum_gJ^TP_gv=J^Tv$，故可在不假设prefix mask与basis commute的情况下，把MSE/L1 output gradient精确归因到
RGNB groups。五dataset × 三A6 checkpoints使用train/validation replication，比较RGNB、DCT与三个random bases，
并分离strict negative conflict、low positive alignment、norm imbalance与coordinate artifact。任何positive结果只
返回Step4，不直接授权decoder、loss、optimizer或SC2。

[D11 Result] accepted v2完成五dataset × 三checkpoints。strict short/long directional conflict为`0/5`，所有
validation MSE total paths/batches均为positive dot；support-specific component gate仅`2/5`，且同一component跨
short/long的negative fraction为0。formal decision=`transform_generic_pressure_sc2_only`，含义不是SC2通过，
而是SC1 conflict-aware decoder问题关闭并回Step2暂停。

[Coverage Boundary] RGNB responsibility distribution在3/5 datasets随prefix measure变化；short measure对最后两个
projective groups严格zero-gradient，而long measure对二者的平均share约为`0.064107/0.020441`。这只建立
`nested support -> unequal update opportunity` observation。Time-o1、per-step loss shaping与generic task
weighting/sampling已有直接邻近工作，所以下一步仅允许Contribution 2 Step1-3 external novelty/problem audit；
`SC2-MIPR`、coverage normalization、PCGrad、new loss、test与joint factorial仍不授权。

[Narrative Boundary] nonlinear decoder、operator MoE、geometry gating、structure-guided time-series MoE与
step-specific representation均已有直接prior art。可辩护边界只能是joint history-atom conditional operator、
RGNB exact projectivity与multi-horizon domain-only execution的完整组合。JAPO status更新为`narrative_ready`；
UNIFORM/HISTORY/ATOM/PERM/RANDOM same-bank controls与staged three-seed gates已冻结。当前只授权不改变design的
seed2022五dataset × 七arm validation-only confirmation；two-seed mean未过gate则停止exact JAPO并归因，只有
通过才授权seed2023。test与SC2-MIPR仍不授权。

### Historical Contribution 2 Record: Measure-Induced Projective Risk

SC2保留`PIR` slot ID，formal objective收紧为`MIPR`。raw horizon measure的exact risk为
$e^TW_\mu e$；MIPR定义$\widetilde W_\mu=\sum_lQ_lW_\mu Q_l$，在PMFO refinement blocks上保留
within-scale weighting并删除cross-scale coupling。它是decoder-aligned structured surrogate，不是比raw
risk“更measure-aligned”的等价改写。

历史状态曾为`narrative_ready / effectiveness_pending / held_after_SC1_rollback`；post-D11现已更新为
`retired_as_core_candidate`。L2下quadratic algebra成立；
Huber/L1没有exact block-metric等价，首轮不实现。`log_uniform_h` off-block energy为`0.205154`，
`uniform_h/benchmark_h`只有`0.003456/0.002480`，因此贡献主场景必须是continuous dense-horizon
deployment，不能只靠四个benchmark horizons。

[Diagnostic status] D1-v2 aggregate PIR problem gate通过，但证据具有measure boundary：log-uniform强、
uniform弱而跨dataset、benchmark projected excess 0/3。该历史边界已在Step4-6收紧为MIPR与
same-measure raw control。

## Frozen Baseline Evidence

natural profile：

- Weather: `patch_num=12, d_model=64, d_ff=128`；
- ETTm1: `patch_num=24, d_model=32, d_ff=64`；
- ETTh2: `patch_num=12, d_model=64, d_ff=128`。

contract hash:
`254d85d47a9e5b7c212f8a8b88decf17a0328a1ea1df324c9cc65be4c672a50c`。
profile 与 checkpoint 均由 validation 预先冻结，test 不参与选择。完整表见
`analysis/stage_c_natural_baseline_test_20260713/natural_baseline_test_report.md`。

## Contribution Boundary

[Current Boundary] 新主线不claim首次提出Direct/MIMO/DIRMO continuum、future query、block-wise output、
multi-scale decoder、dynamic ensemble、regret supervision或cross-validation。Stratify、CATS、MQTransformer、
TimePerceiver、Implicit Forecaster、MQF2、Multi-output Ensembles与TimeRouter均为mandatory controls。

[Current Boundary] 可探索的贡献单位仅是完整
`fixed past -> exact-prefix neural decoder -> point-to-global coupling spectrum -> sample/target-region
counterfactual policy -> no requested-H semantics -> no external strategy search`链条。

[Theory Boundary] 对deterministic separable MSE，future covariance不是Bayes point predictor的必要输入。
coupling只作为有限样本、有限capacity下的parameter-sharing inductive bias；不得宣称joint output具有
population-risk必然优势。

[Fact] A6先生成`coeff [B,C,256]`，再使用`basis[:H] [H,256]`直接计算H步输出；它已经满足domain-only
horizon、exact prefix equality与output-side $O(HK)$ computation。A6在新mainline中是global MIMO-like
coupling endpoint，不是待替换的弱head。

[Current Boundary] D6 short/local与long/global crossing属于basis-support evidence，只能间接提出output-coupling
hypothesis。新D14必须用neutral carrier、matched point/block/global heads、random partition与capacity controls
直接检验；frozen A6只作sensitivity。

[Decision] ordered patch memory降为optional Encoder–Decoder interface ablation。未来若PCSD需要更丰富history
access，可运行`D14-P`；其positive/negative均不能单独通过或拒绝PCSD-CF，也不能建立当前open SC2 slot。

[Decision] 旧 StageB coefficient conditioning、STBO、GRU future composition、unit-specific retrieval 与
encoder repair 均不再是 active candidate。历史失败只按各自 failure attribution 使用，不能被扩大为未经
测试的方向级结论，也不能因为 archive 中代码仍存在而自动复活。

[Decision] Step 7B将“结构正确”与“预测有效”明确分离：15/15 trained invariants通过说明实现与algebra无误，
但不补偿三dataset performance gate失败。当前归因为exact v1 `readout_or_head_design_wrong`，而非
`optimization_or_numeric_pathology`、Encoder方向失败或conservation方向失败。

[Decision] Step 4 source audit排除了三条捷径：不采用LeapTS式learned horizon/scale scheduling，不采用
PRISM式history tree + fixed-H dense heads，不采用Asymmetric-MMF式global low-rank + hierarchy residual作为
paper core。lifting、nested basis与network morphism只作为构造和proof evidence。

[Decision] Step 5进一步排除“function-preserving transform本身就是创新”：M0没有新function，direct atom
版本与dense affine正交等价，DS则有capacity expansion。Contribution 1必须在Step 6给出并验证
`DS > matched DA`所对应的scale-native inductive bias，否则FPMO不能成为paper core。

[Decision] Step 6 narrative audit已关闭该路径：DS与DA的function class相同，且factorization对random
orthogonal/group controls同样成立；requested prefix虽可少生成inactive coefficients，但dense $D_l$仍要求
先生成全部720维scale latents。由此否决的是当前linear DS design，而不是“future multiscale structure不存在”。
后续D2/D3 frozen-memory diagnostics在A6 representation上不支持depth grouping、支持basis main effect；它们
已经推动Step4机制审计，但不能作为end-to-end grouping direction的普遍否定。

[Decision] 新PLGO Step5没有重复旧FPMO结论，而是把global smooth root与interval-local complements组成了
stable square basis；但同样确认“fixed invertible transform本身不是method”。Contribution 1的新增机制必须
位于coefficient generation path，并由matched dense/random-descriptor controls隔离。Step6若无法做到，回滚
Step4，不以训练性能包装RGNB。

## Main Experiment Logic

1. 固定 natural A6 baseline 与 test reference；
2. D1-A验证label/residual nested structure，D1-B验证当前A6 memory存在可访问forecast information，D1-C验证
   learned basis geometry，同时审计measure/projected gradients；
3. PMFO-RCT与MIPR曾分别通过初版Step 4-6 narrative/theory gate；
4. Step 7A local invariants通过；Step 7B使用固定full-H720 pointwise L1、所有model parameters端到端训练，
   完成15-run architecture controls；
5. PMFO-RCT v1 effectiveness失败，回滚Step 4；MIPR、factorial与full matrix全部暂停；
6. Step 4 redesign audit已解释A6 function class、fixed partition与interface问题，并只把FPMO推进到Step 5；
7. FPMO Step 5 embedding/restriction通过但capacity no-go使其仅partial pass；
8. Step 6已判定DS claim无法脱离full-affine factorization解释，故FPMO不进入实现；
9. SC1-D2 core3 partial只支持basis geometry、不支持depth grouping；先冻结ETTh1/ETTm2 profile，再以拆分的
   random-group/random-basis controls完成formal5；
10. SC1-D3已确认basis main effect但否定grouping叙事；先以structured-basis/whitening controls和external
    prior art完成Step 4 mechanism audit；
11. SC1-D4确认locality但由DCT/PCA解释accuracy，故回Step 2/3诊断SC1-CLG；只有新SC1重新通过Step 4-6并完成screening后，才恢复MIPR、
    `2x2` factorial与3-seed full matrix；第二 backbone与official native baselines最后做generality gate。
12. SC1-D5 primary selector失败但b144出现short/long crossed interaction；D6使用disjoint validation window确认，
    pass只返回Step 4，不直接实现operator。
13. D6全部gate通过；PLGO在external source audit后conditional进入Step5，method implementation仍false。
14. PLGO Step5通过RGNB algebra/prefix/A6 morph，但ONB、frame与independent-group variants均被function/control
    no-go限制；只进入Step6 generator design，method implementation仍false。
15. PLGO Step6的PAF tensor/rank gate通过；external primitive overlap不再自动否决task-specific贡献，
    B11/B14促成D7 conditional attribution；D7现已完成并通过geometry gate。
16. D7在frozen A6 memory上确认conditional geometry effect；free-control gap因Encoder-Decoder co-adaptation
    不能判定method readiness。PAF重新开放。
17. D8-E2E Step7A已通过七arms、五profiles、projectivity、gradient与patch-interface gates；Step7B固定为
    35-run validation-only screen，结果返回前不启动三seed或MIPR。
18. D8 Step7B已完成：exact PAF effectiveness fail、geometry attribution pass、m694不能救回A6 gap；
    Contribution 1回Step4 redesign，三seed/MIPR/joint factorial继续暂停。
19. Step4排除flatten压缩、patch retrieval与geometry-only expert shortcuts，只保留joint history-atom JAPO。
20. JAPO Step5通过A6 containment、exact projectivity与strict non-collapse；identical initialization被symmetry
    audit禁止。
21. JAPO Step6已冻结E2/K256/G32、independent initialization、seven-arm matrix与staged seeds；candidate成为
    `narrative_ready`。
22. JAPO Step7A通过210 prefix、35 gradient、paired hashes与runner/analyzer gates；
23. JAPO seed2021完成35/35且无protocol/numeric pathology，但vs A6 macro `-1.3754%`、0/5，仅构成
    stable/inconclusive evidence；按冻结gate只补seed2022，不调architecture/hyperparameters，test/SC2仍暂停。
24. JAPO two-seed 70/70 gate最终失败：vs A6 `-1.2435%`、0/5，same-bank hard gate触发；exact v1关闭，
    seed2023停止，Contribution 1回Step4 operator-intervention redesign，projective direction本身不作否定。
25. post-JAPO系统复盘完成；下一步只执行SC1-D9 history-support operator diagnostic。D9过gate后才允许形成新的
    Step4-5 candidate，当前不授权model implementation、test、MIPR或joint factorial。
26. D9-A exact operator gate失败且无numeric/protocol pathology；ordered history-scale alignment关闭，D9-B取消，
    rollback Step2/3。binary global/detail只作D10 hypothesis，不作Contribution 1 evidence。
27. D10 Step2/3 design冻结binary、detail-monotone与no-aligned-scale三选一gate；通过只返回Step4，失败则继续
    rollback Step2，不允许从exploratory off-diagonal matrix事后生成method。
28. D10 primary gate失败且protocol有效；history-scale routing从Contribution 1 mainline关闭。下一步只审计
    future-component responsibility problem，不把generic adaptive multiscale mixing改名为新贡献。
29. D11 strict future-component conflict为0/5，support-specific gate为2/5；conflict-aware decoder/loss关闭，
    只把projective coverage observation交给Step1-3 problem audit。
30. D12 risk-aligned predictable-frame support仅1/5；CAPE关闭、PRISM joint route retired、D12-B取消，
    两个contribution slots回到Step2。
31. post-D12系统复盘提出NIFRO/IARL：基本对象由single forecast row改为nested-information
    forecast-revision surface；Forking-Sequences与generic stability loss被列为mandatory prior-art controls。
32. 用户确认forecast revision应作为下一篇独立SCI问题；已转移到`New-idea.md`，D13改为
    `deferred_next_paper`，不再是当前active cursor。
33. CADMO/CPGA曾作为fixed-past compression pair提出，但用户指出ordered patch memory只属于decoder interface，
    不能服务multi-horizon核心叙事；两项改为`rejected_by_narrative_scope`，未进入method implementation。
34. 新主线把multi-horizon矛盾定义为future-output coupling strategy：Direct/query、block-MIMO与global MIMO
    固定不同sharing scopes，而unified model不应依赖per-dataset/horizon external strategy selection。
35. provisional `PCSD`在一个exact-prefix decoder内表示point-to-global coupling spectrum；provisional `CCRL`
    用train-OOF sample × target-region regret监督coupling policy。primitive-level DIRMO/MoE/regret不计创新。
36. 下一步只执行新D14-A/B：A先验证matched coupling-scale crossing与oracle headroom；A pass后B才验证
    history+target regret predictability。neutral raw-history carrier为primary，frozen A6只作sensitivity；test=false。
37. D14-A fail关闭pair；A pass/B fail只让PCSD回Step4并重找SC2；A/B pass也只返回formal Step4-6，不能直接
    实现method或启动remote training。
38. D14-A最终通过；D14-B1在implementation前因cross-fit teacher/student mismatch退出paper core。PCSD-CF
    已完成native Step4-6、D15-A Step7A与Step7B prelaunch gate；seed2021 remote screen已在3×3090启动，SC2 slot保持
    open，test=false。

未来candidate screening固定扩展到ETTh1、ETTh2、ETTm1、ETTm2、Weather。五dataset用于cross-dataset
generality，seeds2021/2022/2023用于stochastic confirmation；两者不能互相替代。ETTh1/ETTm2必须先完成
validation-only natural profile freeze。

任何 candidate 若在 problem或narrative gate失败，回滚 Step 2/3；不得通过叠加 Encoder、MoE、auxiliary
loss 或更多 tuning 来掩盖失败。

## Canonical Active Artifacts

- `analysis/stage_c_post_ccsf_step24_reset_20260719/d18_step9/d18_step9_four_layer_diagnostic.md`
- `analysis/stage_c_post_ccsf_step24_reset_20260719/post_d18_step2_mainline_viability_audit.md`
- `analysis/stage_c_post_ccsf_step24_reset_20260719/d19_step45_source_theory_control_audit.md`
- `analysis/stage_c_post_ccsf_step24_reset_20260719/d19_step6_control_design.md`
- `analysis/stage_c_post_ccsf_step24_reset_20260719/d19_step6_contract_repair.md`
- `analysis/stage_c_post_ccsf_step24_reset_20260719/d19_step7a_local/step7a_implementation_gate_report.md`
- `configs/stage_c_d19_if_control_step6_v1_1.json`
- `configs/stage_c_d19_if_control_step7a.json`
- `configs/stage_c_d19_if_control_step6.json`
- `Papers/implicit-forecaster-neurips2025.md`
- `analysis/stage_c_sc2_pcc_step7b_seed2021_20260717/step9_10_result_and_failure_attribution.md`
- `analysis/stage_c_post_pcc_step4_redesign_20260717/source_informed_redesign_audit.md`
- `analysis/stage_c_post_pcc_step5_theory_20260717/step5_theory_feasibility.md`
- `docs/experiments/stage-c-post-pcc-siff-mcca.md`
- `analysis/stage_c_pcsd_cf_step7b_seed2021_20260716/step9_10_result_and_failure_attribution.md`
- `analysis/stage_c_sc2_projective_coupling_credit_step24_20260716/source_theory_audit.md`
- `analysis/stage_c_sc2_pcc_step5_theory_20260716/step5_theory_feasibility.md`
- `analysis/stage_c_multi_horizon_coupling_mainline_reset_20260715/multi_horizon_coupling_mainline_reconstruction.md`
- `Papers/multi-horizon-output-coupling-audit.md`
- `docs/experiments/stage-c-d14-output-coupling-granularity.md`
- `docs/stage-ledgers/stage-c-unified-forecasting-redesign.md`
- `docs/research-roadmap.md`
- `analysis/stage_c_fixed_past_mainline_reset_20260715/fixed_past_mainline_reconstruction.md`（superseded CADMO/CPGA record）
- `docs/experiments/stage-c-d14-conditional-patch-memory-headroom.md`（D14-P auxiliary，not scheduled）
- `docs/experiments/stage-c-pmfo-pir-problem-diagnostic.md`
- `analysis/stage_c_contribution_research_reset_20260713/stage_c_contribution_deep_audit.md`
- `analysis/stage_c_natural_baseline_test_20260713/natural_baseline_test_report.md`
- `analysis/stage_c_d1_pmfo_pir_offline_20260713/`（v1 invalid audit evidence）
- `analysis/stage_c_d1_pmfo_pir_offline_v2_20260713/research_interpretation.md`
- `analysis/stage_c_step46_pmfo_pir_theory_gate_20260713/step46_design_and_prior_art.md`
- `analysis/stage_c_step7a_pmfo_rct_local_20260713/step7a_local_gate_report.md`
- `analysis/stage_c_step7b_pmfo_rct_20260713/step7b_screening_report.md`
- `analysis/stage_c_step7b_pmfo_rct_20260713/failure_attribution_addendum.md`
- `analysis/stage_c_step4_source_informed_redesign_20260713/step4_source_informed_redesign_audit.md`
- `analysis/stage_c_step5_fpmo_theory_20260713/step5_theory_feasibility.md`
- `analysis/stage_c_step6_fpmo_narrative_control_20260713/step6_narrative_control_gate.md`
- `analysis/stage_c_sc1_d4_structured_basis_20260714/research_interpretation.md`
- `analysis/stage_c_sc1_d5_conditioning_locality_20260714/research_interpretation.md`
- `analysis/stage_c_sc1_d6_horizon_support_interaction_20260714/research_interpretation.md`
- `analysis/stage_c_sc1_step4_projective_local_global_audit_20260714/source_informed_audit.md`
- `analysis/stage_c_sc1_plgo_step5_theory_20260714/step5_theory_feasibility.md`
- `analysis/stage_c_sc1_plgo_step6_design_20260714/step6_design_gate.md`
- `analysis/stage_c_sc1_d7_descriptor_sufficiency_20260714/research_interpretation.md`
- `analysis/stage_c_sc1_plgo_step4_redesign_20260714/step4_source_informed_redesign.md`
- `analysis/stage_c_sc1_japo_step5_theory_20260714/step5_theory_feasibility.md`
- `analysis/stage_c_sc1_japo_step6_design_20260714/step6_method_control_design.md`
- `analysis/stage_c_sc1_japo_step7a_local_20260714/step7a_local_gate_report.md`
- `analysis/stage_c_sc1_post_japo_systematic_review_20260715/systematic_stage_review.md`
- `analysis/stage_c_sc1_d9_history_support_operator_audit_20260715/d9_result_and_rollback.md`
- `analysis/stage_c_sc1_d10_raw_scale_identifiability_20260715/d10_step23_diagnostic_design.md`
- `analysis/stage_c_sc1_d10_raw_scale_identifiability_20260715/d10_result_and_rollback.md`
- `Papers/stage-c-external-decoder-objective-audit.md`
- `docs/experiments/stage-c-five-dataset-validation-policy.md`
- `docs/code-explanation/stage-c-pmfo-rct-step7a.md`
- `docs/code-explanation/stage-c-sc1-japo-step5-theory.md`
- `docs/code-explanation/stage-c-sc1-japo-step6-design.md`
- `docs/code-explanation/stage-c-sc1-japo-step7a.md`

2026-07-13 reset 前主线完整 snapshot 位于
`docs/archive/pre-stage-c-reset-20260713/`，仅作历史审计。
