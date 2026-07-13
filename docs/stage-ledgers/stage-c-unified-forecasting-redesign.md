# StageC Unified Varied-Horizon Forecasting Ledger

## Stage Scope

| Field | Content |
| --- | --- |
| `stage_id` | `StageC-UVHF` |
| `paper_role` | decoder/operator 与 training principle 两项相互支撑的 paper-core innovations |
| `active_question` | shared future function representation 与 aligned risk 是否能实现连续、统一、可细化的任意 horizon forecasting？ |
| `source_evidence` | historical/source-faithful `A6-LBF-r256` |
| `mechanism_control` | frozen `A6-LBF-natural-baseline` |
| `active_candidates` | `SC1-FPMO-DS` partial pass；`SC2-MIPR` held |
| `future_validation_suite` | ETTh1, ETTh2, ETTm1, ETTm2, Weather；ETTh1/ETTm2 profiles pending |
| `stage_exit` | 两项分别过 narrative/effectiveness gate，`2x2` joint gate显示独立主效应与联合收益 |
| `stage_rollback` | problem/novelty不跨 dataset -> Step 2；禁止直接堆叠 method |

## Decision Cursor

| Field | Content |
| --- | --- |
| `current_11_step` | Step 5 partial pass；Step 6 narrative/control design |
| `current_candidate` | `SC1-FPMO-DS` partial pass；M0/DA controls frozen；`SC2-MIPR` held |
| `latest_decision` | exact morph/restriction可行；shared morph只是坐标变换；independent scales在T720等价full-affine class |
| `next_required_action` | isolate scale-native mechanism from dense-equivalence/capacity before implementation |
| `method_training_authorized` | `false`；new SC1 implementation需重新通过Step 4-6 |
| `rollback_point` | Step 2/4 if Step6 cannot isolate structure from capacity；不改Encoder、不叠加MIPR/MoE |

## 11-Step Record

| Field | Current Record |
| --- | --- |
| `current_step` | Step 5 complete；Step 6 active |
| `problem` | exact containment、independent scale maps与same latent budget存在no-go trilemma |
| `existence_evidence` | Step4 operator audit + 9 lengths/53 prefixes constructive algebra + T720 budget audit |
| `idea` | M0 exact morph control；DA full-affine control；DS direct-scale factor candidate；MIPR held |
| `theory_check` | embedding/restriction/arbitrary-T pass；efficiency claim withdrawn；DS capacity confound open |
| `design` | only candidate/control algebra frozen；no model tensor contract or implementation |
| `narrative_gate` | partial pass；Step6必须隔离dense-equivalence与capacity；SC2 held |
| `effectiveness_gate` | v1 fail：macro vs A6 `-1.0955%`，worst ETTm1 `-2.0844%` |
| `artifacts` | D1-v2、v1 evidence、Step4 audit、Step5 proof/no-go report、baseline report |
| `decision` | `FPMO-DS partial_pass_step6_design_only`；new implementation/training unauthorized |

## Frozen Carrier Contract

| Dataset | Profile | patch_num | d_model | d_ff |
| --- | --- | ---: | ---: | ---: |
| Weather | `r2b_p12_d64_ff128_medium` | 12 | 64 | 128 |
| ETTm1 | `r2b_p24_d32_ff64_narrow` | 24 | 32 | 64 |
| ETTh2 | `r2b_p12_d64_ff128_medium` | 12 | 64 | 128 |
| ETTh1 | `calibration_pending` | TBD | TBD | TBD |
| ETTm2 | `calibration_pending` | TBD | TBD | TBD |

contract hash:
`254d85d47a9e5b7c212f8a8b88decf17a0328a1ea1df324c9cc65be4c672a50c`。

该hash只覆盖当前三dataset历史contract。未来five-dataset suite按
`docs/experiments/stage-c-five-dataset-validation-policy.md`先校准ETTh1/ETTm2，再生成新hash。

Governance：dataset之间允许不同自然偏好；params差异不参与选择；同一dataset后续所有机制共用同一
profile；test、candidate identity与per-mechanism tuning不得改变profile。

## Candidate Queue

| ID | Status | Hypothesis | Narrative Gate | Effectiveness Gate | Next Action |
| --- | --- | --- | --- | --- | --- |
| `A6-LBF-natural-baseline` | `control_only` | validation-frozen natural profiles可作为稳定共同起点 | not required | 72/72 test；3 seeds；dense horizons | `frozen_test_reference_ready`；只作固定reference |
| `SC1-PMFO-RCT-v1` | `failed_as_core_candidate` | fixed mixed-radix conservative future tree | narrative/local pass | Step7B三dataset均不优于A6；no numeric pathology | archived as evidence；rollback Step 4 |
| `SC1-FPMO-M0` | `control_only` | shared-latent exact A6 morph验证function preservation/restriction | not required | exact equality required | mandatory Step6 morphism control |
| `SC1-FPMO-DA` | `control_only` | direct atom full-affine head隔离capacity/orthogonal coordinate effect | not required | matched function-class control | mandatory Step6 dense-equivalence control |
| `SC1-FPMO-DS` | `partial_pass` | independent scale factors提供native history-scale maps并包含A6 | Step5 algebra pass；Step6 boundary pending | no implementation | prove claim survives DA/capacity control |
| `SC2-MIPR` | `held` | measure-induced block metric去除decoder scales之间的cross coupling | pass for L2；log measure primary，benchmark weak | log off-block `0.205154`；benchmark `0.002480`；performance未测 | 等新SC1 contract冻结；不得先实现 |
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
| SC1-FPMO Step6 narrative/control design | `active` | isolate DS from full-affine capacity and coordinate effects |
| ETTh1/ETTm2 natural profile calibration | `pending_control` | 在下一次SC1 remote screen前按validation-only规则冻结 |

## Paper Mainline Sync Log

| Date | Trigger | Paper Section | Change Type | Decision |
| --- | --- | --- | --- | --- |
| 2026-07-13 | Step4-6 narrative/theory gate完成 | Current Position、Contribution Slots、Contribution Boundary、Main Experiment Logic | contribution boundary + experiment order | PMFO收紧为RCT；PIR收紧为MIPR；Step7 local implementation next |
| 2026-07-13 | Step7A local gate通过 | Current Position、Contribution 1、Main Experiment Logic | implementation evidence + screening scope | 三数据集Step7B固定；effectiveness仍pending |
| 2026-07-13 | Step7B effectiveness失败 | Current Position、Contribution 1/2、Boundary、Experiment Logic | candidate closure + rollback | v1关闭；conservation保留；MIPR held；回Step 4 |
| 2026-07-13 | Step4 source-informed redesign完成 | Current Position、Contribution 1、Boundary、Experiment Logic | new provisional candidate + theory-only advance | `SC1-FPMO`进入Step5；implementation/training仍禁止 |
| 2026-07-13 | FPMO Step5 partial pass | Contribution 1、Boundary、Experiment Logic | theorem pass + capacity trilemma | M0/DA降为controls；DS只进入Step6 design；training仍禁止 |

## Continuation Rules

1. 每次继续研究先读本 ledger 与`stage-c-pmfo-rct-step7-protocol.md`；
2. old analysis可引用，archive脚本不得直接启动；
3. diagnostic failure必须区分 hypothesis、intervention、readout、numeric与capacity control；
4. Step5已partial pass；新model training未授权，先完成FPMO-DS Step6 dense/capacity narrative gate；
5. test reference只用于最终对比，不能参与设计选择。
6. future mechanism screen固定使用ETTh1/ETTh2/ETTm1/ETTm2/Weather；五dataset不能替代三seed确认。
