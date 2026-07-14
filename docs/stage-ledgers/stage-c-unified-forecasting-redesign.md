# StageC Unified Varied-Horizon Forecasting Ledger

## Stage Scope

| Field | Content |
| --- | --- |
| `stage_id` | `StageC-UVHF` |
| `paper_role` | decoder/operator 与 training principle 两项相互支撑的 paper-core innovations |
| `active_question` | canonical RGNB descriptors是否比permuted/random descriptors提供跨dataset的coefficient-row inductive bias？ |
| `source_evidence` | historical/source-faithful `A6-LBF-r256` |
| `mechanism_control` | frozen `A6-LBF-natural-baseline` |
| `active_candidates` | `SC1-PLGO-PAF conditional`；`SC1-D7 diagnostic implementation_ready`；`SC2-MIPR` held |
| `future_validation_suite` | ETTh1, ETTh2, ETTm1, ETTm2, Weather；five profiles frozen |
| `stage_exit` | 两项分别过 narrative/effectiveness gate，`2x2` joint gate显示独立主效应与联合收益 |
| `stage_rollback` | problem/novelty不跨 dataset -> Step 2；禁止直接堆叠 method |

## Decision Cursor

| Field | Content |
| --- | --- |
| `current_11_step` | Step 3 `SC1-D7` attribution，feeding Step 6 closure before Step 7 |
| `current_candidate` | `SC1-PLGO-PAF partial_pass`；`SC1-D7 diagnostic_only`；`SC2-MIPR` held |
| `latest_decision` | Step6 tensor/rank pass；external overlap非自动否决；task-specific narrative conditional pass，D7 mandatory |
| `next_required_action` | commit/push后在3090启动D7五dataset × 三checkpoint × 七arm matrix |
| `method_training_authorized` | `false`；head-only D7 remote diagnostic已通过local gate |
| `rollback_point` | D7 fail -> return Step 4 or close PAF descriptor route；RGNB可保留component |

## 11-Step Record

| Field | Current Record |
| --- | --- |
| `current_step` | Step 3 SC1-D7 attribution；pass returns Step 6 closure |
| `problem` | single fixed support scale无法同时服务short-prefix locality与long-domain global coherence |
| `existence_evidence` | D6 disjoint window：short +1.1964%、long -1.2675%；12/15 crossed units；4/5与5/5 directions |
| `idea` | `SC1-PLGO`：shared projective local-global synthesis；H只限制domain，不进入learned path |
| `theory_check` | PAF 33 cases max gap `4.547e-13`、rank≤256；prior-art primitive overlap不自动否决完整贡献链 |
| `design` | PAF保留为provisional candidate；D7冻结GEO/PERM/RANDOM × width256/694 + free-M0 controls |
| `narrative_gate` | conditional pass；D7 attribution required；method false；SC2 held |
| `effectiveness_gate` | v1 fail：macro vs A6 `-1.0955%`，worst ETTm1 `-2.0844%` |
| `artifacts` | D1-v2、historical Step4-6 closure、D2-D6、PLGO source/Step5/Step6 audits、D7 protocol、five-profile contract |
| `decision` | `conditional_narrative_pass_d7_required`；D7 pass返回Step6冻结method后才可进Step7 |

## Frozen Carrier Contract

| Dataset | Profile | patch_num | d_model | d_ff |
| --- | --- | ---: | ---: | ---: |
| Weather | `r2b_p12_d64_ff128_medium` | 12 | 64 | 128 |
| ETTm1 | `r2b_p24_d32_ff64_narrow` | 24 | 32 | 64 |
| ETTh2 | `r2b_p12_d64_ff128_medium` | 12 | 64 | 128 |
| ETTh1 | `r2b_p24_d64_ff128_medium` | 24 | 64 | 128 |
| ETTm2 | `r2b_p48_d64_ff128_medium` | 48 | 64 | 128 |

历史三dataset contract hash为
`254d85d47a9e5b7c212f8a8b88decf17a0328a1ea1df324c9cc65be4c672a50c`；新的five-dataset contract为
`configs/stage_c_five_dataset_natural_profiles.json`，hash
`80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a`。

Governance：dataset之间允许不同自然偏好；params差异不参与选择；同一dataset后续所有机制共用同一
profile；test、candidate identity与per-mechanism tuning不得改变profile。

## Candidate Queue

| ID | Status | Hypothesis | Narrative Gate | Effectiveness Gate | Next Action |
| --- | --- | --- | --- | --- | --- |
| `A6-LBF-natural-baseline` | `control_only` | validation-frozen natural profiles可作为稳定共同起点 | not required | 72/72 test；3 seeds；dense horizons | `frozen_test_reference_ready`；只作固定reference |
| `SC1-PMFO-RCT-v1` | `failed_as_core_candidate` | fixed mixed-radix conservative future tree | narrative/local pass | Step7B三dataset均不优于A6；no numeric pathology | archived as evidence；rollback Step 4 |
| `SC1-FPMO-M0` | `control_only` | shared-latent exact A6 morph验证function preservation/restriction | not required | exact equality required | mandatory Step6 morphism control |
| `SC1-FPMO-DA` | `control_only` | direct atom full-affine head隔离capacity/orthogonal coordinate effect | not required | matched function-class control | mandatory Step6 dense-equivalence control |
| `SC1-FPMO-DS` | `rejected_by_narrative_gate` | independent scale factors提供native history-scale maps并包含A6 | fail：与DA同为full affine，factorization不依赖真实scale coordinates | not started | controls/evidence only；不进入Step 7 |
| `SC1-D2` | `closed_formal_fail` | true-scale nonlinear grouping是否超越rank expansion、generic nonlinearity与random grouping | not a method gate | basis gate pass；random-group mandatory gate fail | rollback Step 2；不得实现method |
| `SC1-D3` | `diagnostic_only` | basis geometry是否为独立main effect而非basis-group interaction | not required | pass：main +2.9174%；both conditionals + 5/5 interaction guard | evidence complete；return Step 4 only |
| `SC1-D4` | `diagnostic_only_closed` | balanced basis是否超越standard bases，收益来自locality还是exact balancing | not required | locality pass；global noninferiority与balance specificity fail | return Step 2/3；exact balanced basis不升method |
| `SC1-CLG` | `problem_gate_passed` | local prefix synthesis与long-domain coherence是否存在horizon-dependent support tradeoff | not required | D6 pass | evidence feeds PLGO Step4-5 |
| `SC1-D5` | `diagnostic_only_design_fault` | fit-only selected local DCT/PCA能否改善balanced并接近global controls | not required | primary fail；b144 arm出现11/15 crossing | direction rejection invalid；design D6 |
| `SC1-D6` | `diagnostic_only_pass` | b144 short-positive/long-negative interaction能否在disjoint validation window复现 | not required | all gates pass；12/15 crossing | return Step4 only；evidence complete |
| `SC1-PLGO` | `partial_pass` | projective local-global synthesis能否同时服务short prefix与long domain | task-specific narrative conditional pass；descriptor attribution pending | not started | 等D7；no method implementation |
| `SC1-D7` | `diagnostic_only` | true RGNB descriptors是否超越permuted/random descriptors | not required | local gate pass；remote pending | launch 105-fit validation-only matrix |
| `SC2-MIPR` | `held` | measure-induced block metric去除decoder scales之间的cross coupling | pass for L2；log measure primary，benchmark weak | log off-block `0.205154`；benchmark `0.002480`；performance未测 | 等新SC1 problem/method contract；不得先实现 |
| `SC3-JOINT` | `deferred` | decoder与objective co-design存在非冗余interaction | SC1/SC2分别通过后评估 | `2x2` factorial独立主效应 | 不得提前实现 |
| `SC4-XBG` | `deferred` | mechanism不依赖TimeAlign-derived encoder | generality gate | second backbone | 等full matrix |

## Historical Failure Attribution Boundary

- Phase1 target-set decoder只在旧 PatchEncoder 上弱负，不是方向级否决；
- B10 frozen target-specific readout有 numerical/readout pathology，不能否定 target-aware direction；
- B13只否定当前 GRU transition，no-transition control解释其收益；
- B14对跨 dataset unit-specific patch retrieval problem形成负证据；
- B12 STBO被rank/capacity confound阻断，不能被改名为PMFO evidence。

[Decision] 尽管 explicit-H 未被严格方向级否决，StageC仍禁止以 horizon embedding/router 为 paper core：
其连续性shortcut风险与prior-art压力均高。历史代码存在不构成重新授权。

## Experiment Ledger

| Experiment | Role | Result | Decision | Artifact |
| --- | --- | --- | --- | --- |
| Natural profile calibration R2A/B/C | validation-only control | Weather=P12/D64；ETTm1=P24/D32；ETTh2=P12/D64；9/9 stability pass | contract frozen | `analysis/stage_c_dap_r2c_stability_20260712/` |
| SC0 checkpoint gap | diagnostic | validation 31.63%-44.95%不是test值；H720 mean test last-vs-best +6.11% | mechanism control使用best-val；source reproduction保留native last | `analysis/stage_c_sc0_checkpoint_test_gap_20260712/` |
| Natural baseline test | post-freeze reference | 72/72；test未参与选择；ETTh2 H48 MSE CV=5.30% | `frozen_test_reference_ready` | `analysis/stage_c_natural_baseline_test_20260713/natural_baseline_test_report.md` |
| SC1/SC2 deep reset | Step1-3 research audit | explicit-H与simple HML均不够；提出PMFO/PIR并定义falsification | 只授权D1 diagnostic | `analysis/stage_c_contribution_research_reset_20260713/stage_c_contribution_deep_audit.md` |
| A6/PMFO architecture audit | Step2 correction | A6使用`coeff [B,C,256] × basis[:H] [H,256]`；旧“总生成H720”表述不准确 | PMFO问题收紧；扩展D1-A/B/C | `docs/experiments/stage-c-pmfo-pir-problem-diagnostic.md` |
| D1-v1 offline diagnostic | diagnostic-invalid | ETTh2 full-hidden R2=-39.7831却被旧gate误判；Weather/ETTh2 normalized residual≈label | `diagnostic_invalid_for_direction_rejection`；保留raw evidence，运行v2 | `analysis/stage_c_d1_pmfo_pir_offline_20260713/` |
| D1-v2 offline diagnostic | Step2-3 problem gate | structure/frozen-memory/aggregate PIR均3/3；ETTh2 linear probe fail；benchmark PIR excess 0/3 | SC1进入Step4-6；SC2 measure-conditional进入Step4-6 | `analysis/stage_c_d1_pmfo_pir_offline_v2_20260713/research_interpretation.md` |
| SC1/SC2 Step4-6 theory gate | external prior-art + algebra audit | generic basis/wavelet/reweight claims被排除；PMFO invariants pass；MIPR measure geometry明确 | SC1/SC2均`narrative_ready`；remote training仍false | `analysis/stage_c_step46_pmfo_pir_theory_gate_20260713/step46_design_and_prior_art.md` |
| SC1 Step7A local gate | model implementation + invariant audit | 90/90 shape-prefix；float32 max gap `4.172e-7`；conservation `2.682e-7`；locality `0` | `implementation_gate_passed`；effectiveness pending；remote training false | `analysis/stage_c_step7a_pmfo_rct_local_20260713/step7a_local_gate_report.md` |
| SC1 Step7B remote screen | architecture-only matched controls | 15/15 complete；PMFO vs A6 macro `-1.0955%`；conservation +`2.3393%` vs ablation；transition +`0.0486%` vs control | v1 `failed_as_core_candidate`；`readout_or_head_design_wrong`；rollback Step 4 | `analysis/stage_c_step7b_pmfo_rct_20260713/step7b_screening_report.md` |
| SC1 Step4 source-informed redesign | external primary sources + 3-dataset checkpoint operator geometry | PMFO params 212,010 < A6-family dim 316,112；90/30 boundary ratios≈1；root patch-profile cosine 0.936-0.994 | v1 structural attribution strengthened；`SC1-FPMO` only advances to Step5 | `analysis/stage_c_step4_source_informed_redesign_20260713/step4_source_informed_redesign_audit.md` |
| SC1-FPMO Step5 theory feasibility | arbitrary-T constructive proof + function-space/no-go audit | 9 lengths/53 prefixes max gap `5.329e-14`；T720 DS rank caps sum720 and equals full-affine class | `partial_pass_step6_design_only`；efficiency claim withdrawn；training false | `analysis/stage_c_step5_fpmo_theory_20260713/step5_theory_feasibility.md` |
| SC1-FPMO Step6 narrative/control gate | external factorization prior art + function/control audit | DS=DA full affine；true/random grouping class不变；DS需全部720维scale latents | `rejected_by_narrative_gate`；rollback Step 2/3；D2 diagnostic next | `analysis/stage_c_step6_fpmo_narrative_control_20260713/step6_narrative_control_gate.md` |
| SC1-D2 core3 precheck | frozen-memory head-only 3 datasets × 3 seeds × 11 arms | rank `-0.5661%`；dense nonlinear `-6.4492%`；true vs random-group `-0.2212%`；vs random-basis `+2.3137%` | `partial_core3_basis_geometry_signal_only`；formal5 mandatory | `analysis/stage_c_sc1_d2_core3_precheck_20260713/research_interpretation.md` |
| Five-profile extension | validation-only 14-run A/B/C | ETTh1=P24/D64；ETTm2=P48/D64；mean/max CV均pass | five-dataset contract frozen；formal5 ready | `analysis/stage_c_five_profile_extension_20260713/profile_extension_report.md` |
| SC1-D2 formal5 | 5 datasets × 3 checkpoints × 11 arms | basis +3.0635% 5/5 pass；group +0.0947% 2/5 fail；165/165 + invariants pass | exact depth-grouping hypothesis false；rollback Step 2 | `analysis/stage_c_sc1_d2_formal5_20260714/research_interpretation.md` |
| SC1-D3 crossed diagnostic | 5 datasets × 3 checkpoints × 3 missing-cell controls | basis main +2.9174%；true/random-group +3.1164%/+2.7181%；interaction 5/5 pass | independent probe main effect supported；return Step 4，not method | `analysis/stage_c_sc1_d3_crossed_20260714/research_interpretation.md` |
| SC1-D4 structured-basis diagnostic | 5 datasets × 3 checkpoints × 3 grouping seeds × 7 bases | locality +1.6324% pass；balanced vs DCT/PCA -0.8609%/-1.5050%；vs random interval +0.2742% fail | `standard_structured_basis_explains_gain_return_step2`；建立SC1-CLG问题 | `analysis/stage_c_sc1_d4_structured_basis_20260714/research_interpretation.md` |
| SC1-D5 conditioning-locality frontier | 5 datasets × 3 checkpoints × 3 grouping seeds × 13 bases | primary b96 selector fail；b144 vs DCT short +1.05%、long -1.15%，11/15 crossing | `design_fault_suspected`；不可方向否决；D6 disjoint confirmation | `analysis/stage_c_sc1_d5_conditioning_locality_20260714/research_interpretation.md` |
| SC1-D6 support interaction confirmation | validation batches8-15；5 datasets × 3 checkpoints × 15 arms | short +1.1964%、long -1.2675%；12/15 crossing；all gates pass | problem supported；return Step4；PLGO conditional narrative pass | `analysis/stage_c_sc1_d6_horizon_support_interaction_20260714/research_interpretation.md` |
| SC1-PLGO Step5 theory feasibility | 12 RGNB cases；101 selected prefixes；3,731 all-H bounds；frame/function no-go | max gap `2.141e-13`；M0=A6 reparameterization；frame kernel16；group caps sum720 | `partial_pass_step6_design_only`；training false | `analysis/stage_c_sc1_plgo_step5_theory_20260714/step5_theory_feasibility.md` |
| SC1-PLGO Step6 design gate | external primary sources + B11/B14 audit + 33 PAF tensor cases | projectivity max `4.547e-13`、rank≤256；overlap非自动否决；internal attribution pending | `conditional_narrative_pass_d7_required` | `analysis/stage_c_sc1_plgo_step6_design_20260714/step6_design_gate.md` |
| SC1-D7 local implementation gate | worker/analyzer smoke + runner dry-run | seven arms、parameter/descriptor/projectivity/gradient/analyzer contracts pass | remote diagnostic authorized；method false | `analysis/stage_c_sc1_d7_local_gate_20260714/local_gate_report.md` |

## Pending Tasks

| Task | Status | Next Action |
| --- | --- | --- |
| Freeze natural carrier | `completed` | 不再调 profile |
| Establish dense test reference | `completed` | 后续统一对比 |
| Archive closed routes and clean active entrypoints | `completed` | archive只作证据 |
| Implement D1 offline analyzer | `completed_v2` | evaluation-space source/gradient + strict probe + frozen decoder counterfactual |
| Run D1 problem diagnostics | `completed_v2` | v1 invalid evidence与v2 accepted evidence分离 |
| PMFO/PIR Step4-6 gate | `completed` | narrative-ready decisions与paper-mainline已同步 |
| PMFO-RCT Step7A local implementation | `completed` | 四variants、local gate与code explanation已落地 |
| PMFO-RCT Step7B architecture screening | `completed_rollback` | v1 closed；do not continue seeds/tuning |
| SC1 Step4 redesign audit | `completed` | FPMO source boundary与kill gates已冻结 |
| SC1-FPMO Step5 theory feasibility | `completed_partial_pass` | M0/DA/DS boundary与no-go theorem已冻结 |
| SC1-FPMO Step6 narrative/control design | `completed_rejected` | DS降为diagnostic control；不实现、不训练 |
| SC1-D2 rank/nonlinearity/scale diagnostic | `completed_formal_fail` | formal5 165/165；basis signal retained，depth grouping rejected |
| ETTh1/ETTm2 natural profile calibration | `completed_frozen` | 14/14 validation-only；ETTh1 P24/D64，ETTm2 P48/D64；stability pass |
| SC1-D2 formal5 | `completed_rollback` | `scale_alignment_not_supported_reformulate_step2` |
| SC1-D3 crossed basis-group diagnostic | `completed_pass` | 45/45；15 primary units；all gates pass；只授权Step 4 source-informed audit |
| SC1 basis mechanism Step 4 | `source_audit_completed` | component novelty accepted；first-basis-generation claim rejected；组合novelty provisional |
| SC1-D4 structured-basis diagnostic | `completed_rollback` | 315/315；locality成立，standard basis与random interval controls阻断exact claim |
| SC1-CLG conditioning-locality problem | `reformulated` | 从single Pareto point收紧为horizon-support crossed interaction |
| SC1-D5 frontier diagnostic | `completed_design_fault` | 585/585；primary fail但pre-registered b144阻止方向否决 |
| SC1-D6 interaction confirmation | `completed_pass` | 225/225；all gates pass；problem evidence frozen |
| SC1-PLGO source-informed audit | `completed_conditional` | prior-art/rejected shortcuts已冻结；进入Step5 theory only |
| SC1-PLGO theory feasibility | `completed_partial_pass` | RGNB scaffold pass；M0/frame/independent-group均非method；进入Step6 only |
| SC1-PLGO Step6 design gate | `completed_conditional` | tensor/rank pass；task-specific narrative conditional；D7前不进入Step7 |
| SC1-D7 descriptor sufficiency | `implementation_ready` | commit/push；GPU preflight；launch 105-fit remote diagnostic |

## Paper Mainline Sync Log

| Date | Trigger | Paper Section | Change Type | Decision |
| --- | --- | --- | --- | --- |
| 2026-07-13 | Step4-6 narrative/theory gate完成 | Current Position、Contribution Slots、Contribution Boundary、Main Experiment Logic | contribution boundary + experiment order | PMFO收紧为RCT；PIR收紧为MIPR；Step7 local implementation next |
| 2026-07-13 | Step7A local gate通过 | Current Position、Contribution 1、Main Experiment Logic | implementation evidence + screening scope | 三数据集Step7B固定；effectiveness仍pending |
| 2026-07-13 | Step7B effectiveness失败 | Current Position、Contribution 1/2、Boundary、Experiment Logic | candidate closure + rollback | v1关闭；conservation保留；MIPR held；回Step 4 |
| 2026-07-13 | Step4 source-informed redesign完成 | Current Position、Contribution 1、Boundary、Experiment Logic | new provisional candidate + theory-only advance | `SC1-FPMO`进入Step5；implementation/training仍禁止 |
| 2026-07-13 | FPMO Step5 partial pass | Contribution 1、Boundary、Experiment Logic | theorem pass + capacity trilemma | M0/DA降为controls；DS只进入Step6 design；training仍禁止 |
| 2026-07-13 | FPMO Step6 narrative gate失败 | Contribution 1、Boundary、Experiment Logic | candidate closure + Step2/3 rollback | DS与DA同class且scale claim不成立；D2先诊断rank/nonlinearity/scale alignment |
| 2026-07-13 | D2 core3 precheck完成 | Current Position、Contribution 1、Experiment Logic | diagnostic partial result + gate repair | basis geometry positive；depth grouping unsupported in core3；formal5 pending |
| 2026-07-14 | D2 formal5完成 | Current Position、Contribution 1、Experiment Logic | exact problem closure + rollback | basis geometry retained；depth grouping rejected；回Step 2设计crossed diagnostic |
| 2026-07-14 | D3 crossed diagnostic通过 | Current Position、Contribution 1、Experiment Logic | existence evidence + Step 4 authorization | basis independent main effect成立于probe；不等于basis method或novelty |
| 2026-07-14 | D4 structured-basis diagnostic完成 | Current Position、Contribution 1、Boundary、Experiment Logic | exact claim closure + Step 2/3 rollback | locality保留；balanced specificity与global accuracy不成立；转向SC1-CLG |
| 2026-07-14 | D5 frontier diagnostic完成 | Current Position、Contribution 1、Boundary | primary fail + diagnostic redesign | b96 selector fail；b144出现support×horizon crossing；D6确认前不升method |
| 2026-07-14 | D6 disjoint confirmation通过 | Current Position、Contribution 1、Boundary | problem pass + Step4 candidate reformulation | PLGO conditional narrative pass；Step5 theory next；training false |
| 2026-07-14 | PLGO Step5 theory完成 | Current Position、Contribution 1、Boundary、Experiment Logic | constructive proof + function/no-go boundary | RGNB scaffold通过；actual method仍partial；Step6 design only |
| 2026-07-14 | PLGO Step6按新overlap规则重判 | Current Position、Contribution 1、Boundary、Experiment Logic | governance revision + candidate restoration | task-specific narrative conditional pass；D7 attribution mandatory |

## Continuation Rules

1. 每次继续研究先读本 ledger 与Step6 gate report；
2. old analysis可引用，archive脚本不得直接启动；
3. diagnostic failure必须区分 hypothesis、intervention、readout、numeric与capacity control；
4. D2 formal5已关闭depth grouping problem；不得通过调profile、改group size或叠加method重开；
5. test reference只用于最终对比，不能参与设计选择。
6. future mechanism screen固定使用ETTh1/ETTh2/ETTm1/ETTm2/Weather；五dataset不能替代三seed确认。
7. D3只支持probe-family basis main effect；Step 4必须先排除standard basis、whitening与static regularization解释。
8. D4已完成上述排除：不得把fixed balanced midpoint basis单独升为Contribution 1；可将其保留为generation component，并研究conditioning-locality co-design。
9. PLGO Step5只通过RGNB mathematical scaffold；不得把ONB换基、overcomplete union或full-affine group maps升为method。
10. prior-art primitive overlap不自动否决task-specific contribution；PLGO的claim必须落在multi-horizon
    projectivity、RGNB support geometry与atomwise generation组合上。B14仍禁止atom retrieval；D7前不得实现
    PAF forecast model。
