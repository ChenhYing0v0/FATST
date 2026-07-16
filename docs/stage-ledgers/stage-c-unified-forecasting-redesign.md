# StageC Unified Varied-Horizon Forecasting Ledger

## Stage Scope

| Field | Content |
| --- | --- |
| `stage_id` | `StageC-UVHF` |
| `paper_role` | decoder/operator 与 training principle 两项相互支撑的 paper-core innovations |
| `active_question` | nested-prefix credit transport能否特异性修复test-confirmed same-run arm starvation？ |
| `source_evidence` | historical/source-faithful `A6-LBF-r256` |
| `mechanism_control` | same-run end-to-end `A6-LBF-natural-baseline`；frozen A6只作reference/diagnostic |
| `active_candidates` | `SC1-PCSD-CF-v1` effectiveness rejected；`SC2-PCC-v1-TI` Step6 pass/Step7A next |
| `future_validation_suite` | ETTh1, ETTh2, ETTm1, ETTm2, Weather；five profiles frozen |
| `stage_exit` | 新两项分别过 narrative/effectiveness gate并形成可归因joint story |
| `stage_rollback` | PCC退化为generic skill floor -> Step4；shared-field arm ceiling -> SC1 Step4 redesign |

## Decision Cursor

| Field | Content |
| --- | --- |
| `current_11_step` | SC2-PCC-v1-TI Step6 pass；Step7A local implementation |
| `current_candidate` | `SC2-PCC-v1-TI` narrative/design conditional pass；effectiveness unready |
| `latest_decision` | v0 prior-art overlap；v1 nested-risk transport 19/19 pass；45-run matrix frozen |
| `next_required_action` | 实现objective、nine-arm losses、diagnostics与Step7A invariants |
| `method_training_authorized` | local implementation only；remote/test/confirmation=false |
| `rollback_point` | generic balancing explains -> Step4；shared-field arm ceiling -> SC1 Step4；invalid math -> Step5 redesign |

## 11-Step Record

| Field | Current Record |
| --- | --- |
| `current_step` | PCC-v1-TI Step6 complete；Step7A local implementation next |
| `problem` | 现有unified decoder固定point/block/global coupling scope；最佳future-output sharing是否随target region与history变化 |
| `existence_evidence` | three-seed neutral+A6 5/5 stable crossing；strict 7.1107%/9.1259%；instance 6.7948%/8.5990% |
| `idea` | PCSD field + dense nested-prefix capability + harmonic target-coordinate credit transport |
| `theory_check` | transport identity gap 0；19/19 simplex/floor/stopgrad/schedule/protocol cases pass |
| `design` | one-forward L1 prefix risk；continuous equal-to-capability schedule；9 arms × 5 datasets controls |
| `narrative_gate` | conditional pass；generic expert loss/loss-teacher/gradient balancing excluded；must beat pointwise controls |
| `effectiveness_gate` | official test fail：DIRECT vs A6 -1.3994%、1/5；all control macro gains negative；oracle +2.0197% |
| `artifacts` | `analysis/stage_c_pcsd_cf_test_audit_seed2021_20260716/test_audit_report.md`；`analysis/stage_c_sc2_pcc_step5_theory_20260716/step5_theory_feasibility.md` |
| `decision` | exact PCSD-CF-v1 rejected；PCC-v0 demoted control；PCC-v1-TI Step7A local authorized |

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
| `SC1-D2` | `diagnostic_only` | true-scale nonlinear grouping是否超越rank expansion、generic nonlinearity与random grouping | not a method gate | frozen A6 representation下basis pass、random-group fail | conditional negative retained；不作E2E direction rejection |
| `SC1-D3` | `diagnostic_only` | basis geometry是否为独立main effect而非basis-group interaction | not required | pass：main +2.9174%；both conditionals + 5/5 interaction guard | evidence complete；return Step 4 only |
| `SC1-D4` | `diagnostic_only_closed` | balanced basis是否超越standard bases，收益来自locality还是exact balancing | not required | locality pass；global noninferiority与balance specificity fail | return Step 2/3；exact balanced basis不升method |
| `SC1-CLG` | `problem_gate_passed` | local prefix synthesis与long-domain coherence是否存在horizon-dependent support tradeoff | not required | D6 pass | evidence feeds PLGO Step4-5 |
| `SC1-D5` | `diagnostic_only_design_fault` | fit-only selected local DCT/PCA能否改善balanced并接近global controls | not required | primary fail；b144 arm出现11/15 crossing | direction rejection invalid；design D6 |
| `SC1-D6` | `diagnostic_only_pass` | b144 short-positive/long-negative interaction能否在disjoint validation window复现 | not required | all gates pass；12/15 crossing | return Step4 only；evidence complete |
| `SC1-PLGO` | `geometry_scaffold_retained` | projective local-global synthesis能否同时服务short prefix与long domain | geometry/projectivity retained；exact PAF boundary withdrawn | D8 exact carrier fail | scaffold feeds JAPO Step5；不单独训练 |
| `SC1-JAPO` | `failed_as_core_candidate` | joint history-atom operator能否解除fixed separability且保留A6/projectivity | complete contract仍有效 | two-seed vs A6 0/5；same-bank hard gate fail；near-uniform routing replicated | exact v1 stop；rollback Step4，seed2023 false |
| `SC1-D7` | `diagnostic_only` | true RGNB descriptors是否超越permuted/random descriptors | not required | conditional geometry pass；method readiness not evaluated | evidence complete；feeds D8 controls |
| `SC1-D8-E2E` | `failed_exact_design` | joint Encoder-PAF adaptation是否消除frozen compatibility confound并保留geometry effect | geometry retained | vs A6 -28.10%；vs matched +14.33%；m694 no rescue | rollback Step4；no three-seed |
| `SC1-D9` | `diagnostic_only_failed` | A6 learned operator中是否存在超越随机controls的history-scale × future-support结构 | not required；method=false | macro rho 0.173810；2/5 effect；1/5 permutation；0/5 random basis | D9-B canceled；rollback Step2/3；binary clue exploratory only |
| `SC1-D10` | `diagnostic_only_failed` | raw history→future关系支持binary global/detail、detail-monotone还是no aligned scale | not required；method=false | binary all gates fail；monotone best 0/5、mapping 2/5 | aligned scale closed；rollback Step2；off-diagonal exploratory only |
| `SC1-D11` | `diagnostic_only_completed_rollback` | multi-horizon conflict是否位于future component gradient responsibility | not required；method=false | strict 0/5；support-specific 2/5；generic redistribution 3/5；all invariants pass | SC1 conflict hypothesis关闭；只把coverage observation交给SC2 Step1-3 |
| `SC1-PRISM` | `retired_without_effectiveness_test` | prefix-risk-isometric frame能否同时保持global compaction与short-prefix locality | D12 joint prerequisite failed | not started | D12-B canceled；D6 evidence retained，not active |
| `SC2-CAPE` | `failed_problem_gate` | conditional-mean covariance能否比raw label covariance更好分配rank-256 decoder capacity | risk-aligned v2 support 1/5 | not applicable | closed；no rank/pilot rescue |
| `SC2-MIPR` | `retired_as_core_candidate` | measure-induced block metric去除decoder scales之间的cross coupling | historical L2 pass；current problem/novelty fail | not started | raw $W_\mu$保留为protocol/control；MIPR不实现 |
| `SC1-NIFRO` | `deferred_next_paper` | causal patch-level information increments形成forecast-revision surface | independent-paper idea；不占当前slot | not started | 转移到`New-idea.md`；D13未来重启 |
| `SC2-IARL` | `deferred_next_paper` | revision energy由same-target accuracy gain解释 | independent-paper idea；不占当前slot | not started | 转移到`New-idea.md`；D13未来重启 |
| `SC-D13-A/B` | `deferred_next_paper` | rolling-origin revision efficiency与new-patch information | not current paper | not started | protocol保留，不执行 |
| `SC1-CADMO` | `rejected_by_narrative_scope` | full patch memory在global coeff之外是否支持projective target-specific direct access | 与multi-horizon核心问题不直接对齐 | not started | 不占active slot；仅保留历史设计 |
| `SC2-CPGA` | `rejected_with_parent_route` | full-memory path的prediction change能否由conditional predictive gain核算 | 脱离CADMO后退化为generic accounting | not started | 不实现；不占active slot |
| `SC-D14-P` | `auxiliary_not_scheduled` | A6 patch memory是否含超越global coeff的ordered target-specific conditional information | not paper mainline | not started | 未来仅在decoder interface需要时做small probe |
| `SC1-PCSD` | `problem_supported_parent` | one projective decoder是否需同时表示point/block/global coupling scopes | complete-chain novelty only；DIRMO/Stratify/CATS controls mandatory | D14-A1 three-seed dual-carrier pass；GroupedMLP below LBF | parent retained；native child PCSD-CF active |
| `SC1-PCSD-CF` | `rejected_effectiveness_test` | one shared parameter field能否经scope pooling形成skilled point/block/global arms并contain A6 | field-pooling chain retained as evidence | test DIRECT vs A6 -1.3994%、1/5；all control macro gains negative | exact v1 closed；no confirmation |
| `SC2-CCRL` | `retired_as_core_diagnostic_only` | cross-fit relative risk能否增益matched direct fusion | generic overlap high；two-stage teacher/student inconsistency | not implemented | retain report/config as history；not scheduled |
| `SC2-ICC` | `superseded_by_pcc` | same-forward marginal coupling credit能否修复direct policy misallocation | working hypothesis已由D15-A收紧 | not implemented | historical name only |
| `SC2-PCC-v0` | `superseded_pointwise_control` | pointwise same-forward capability + skill floor | expert loss与loss-teacher gate已有直接prior art | 15/15 theory cases；method untested | mandatory pointwise/prior controls only |
| `SC2-PCC-v1-TI` | `step6_pass_step7a_local` | nested-prefix capability能否经harmonic incidence输运为target-coordinate credit | complete-chain conditional pass；必须超过pointwise/prior composition | 19/19 design cases；method untested | Step7A objective/diagnostic implementation；remote false |
| `SC-D14-A/B` | `a_confirmed_b_retired` | coupling crossing与conditional-risk predictability | A pass；B retired before implementation | A confirmed；B not run | A evidence retained；B closed |
| `SC-D15-A` | `completed_training_blocked` | native PCSD-CF representation、direct trainability与credit-problem existence | Step4-6 conditional pass | 60/60；method fail；25/25 arm starvation | no confirmation；feeds PCC Step2-4 |
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
| SC1-D7 remote diagnostic | 5 datasets × 3 checkpoints × 7 arms；105/105 | conditional GEO vs controls +13.80%/+12.84%、5/5；free gap has co-adaptation confound | geometry retained；method readiness unevaluated；D8-E2E required | `analysis/stage_c_sc1_d7_descriptor_sufficiency_20260714/research_interpretation.md` |
| Frozen replacement fairness audit | D1-D7 + PMFO Step7B code/source audit | D7 free gap invalid as E2E gate；Step7B remains fair E2E v1 failure | reopen PAF；rollback to Step6/7A, not Step4 patch | `analysis/stage_c_frozen_replacement_fairness_audit_20260714/fairness_audit.md` |
| SC1-D8 Step7A local gate | 5 profiles × 7 arms；210 shape-prefix + 35 gradient cases | all gates pass；prefix `2.384e-6`；patch-block rewrite `5.722e-6`；test=false | Step7B 35-run remote screen authorized | `analysis/stage_c_sc1_d8_step7a_local_20260714/step7a_local_gate_report.md` |
| SC1-D8 Step7B E2E screen | 5 datasets × 7 arms × seed2021；35/35 validation-only | GEO vs A6 -28.10%；vs matched +14.33% 5/5；m694 vs c256 +0.58%；all primary epoch-cap plateau | exact PAF failed；geometry retained；rollback Step4，direction rejection invalid | `analysis/stage_c_sc1_d8_e2e_20260714/research_interpretation.md` |
| SC1-PLGO Step4 intervention/readout redesign | external primary sources + D8 function-class/patch audit + geometry-only expert no-go | flatten bijective；patch retrieval unsupported；geometry-only experts collapse to wider PAF | `SC1-JAPO theory_pending`；Step5 only；training false | `analysis/stage_c_sc1_plgo_step4_redesign_20260714/step4_source_informed_redesign.md` |
| SC1-JAPO Step5 theory feasibility | 4 lengths/22 prefixes + function-class witness + autograd symmetry + control audit | containment `1.137e-13`；projectivity `1.172e-13`；joint witness `1.523188`；identical-init router grad 0 | `theory_pass_step6_design_only`；implementation/training false | `analysis/stage_c_sc1_japo_step5_theory_20260714/step5_theory_feasibility.md` |
| SC1-JAPO Step6 method/control design | external primary sources + 5-profile tensor/init/gradient/control audit | projectivity `3.331e-16`；entropy min `0.999855`；usage `0.4980–0.5020`；all gradients pass | `narrative_ready_step7a_local_implementation_only`；35-run remote false | `analysis/stage_c_sc1_japo_step6_design_20260714/step6_method_control_design.md` |
| SC1-JAPO Step7A production gate | 5 profiles × 7 arms；210 prefix + 35 gradient + paired hashes + runner/analyzer dry-run | prefix `4.768e-7`；patch rewrite `5.722e-6`；entropy min `0.999944`；all gates pass | `step7a_pass_remote_screen_authorized`；test/SC2 false | `analysis/stage_c_sc1_japo_step7a_local_20260714/step7a_local_gate_report.md` |
| SC1-JAPO Step8 seed2021 screen | 5 datasets × 7 arms；35/35 validation-only；paired from-scratch initialization | JOINT vs A6 macro `-1.3754%`、0/5；vs same-bank median `-0.0780%`、2/5；router entropy min `0.993263` | stable/inconclusive；不作方向否定或调参；只补seed2022 | `analysis/stage_c_sc1_japo_e2e_20260715/research_interpretation.md` |
| SC1-JAPO Step8 two-seed gate | 5 datasets × 7 arms × 2 seeds；70/70 validation-only | vs A6 `-1.2435%`、0/5；vs median `-0.1175%`、1/5；UNIFORM/HISTORY/ATOM均优于JOINT | exact v1 failed；capacity control explains；rollback Step4，direction remains open | `analysis/stage_c_sc1_japo_e2e_20260715/two_seed/research_interpretation.md` |
| Post-JAPO systematic review | full StageC evidence chain + external primary-source audit | geometry/projectivity retained；rigid replacement、fixed separability与weak mixing依次失败；flatten不是信息压缩 | no active method；先执行SC1-D9 existence diagnostic | `analysis/stage_c_sc1_post_japo_systematic_review_20260715/systematic_stage_review.md` |
| SC1-D9-A exact operator audit | 5 datasets × 3 A6 seeds；1024 atom permutations；64 random history bases | macro rho 0.173810；2/5 positive effect；1/5 permutation；0/5 random basis；Parseval pass | scale hypothesis unsupported；D9-B canceled；rollback Step2/3 | `analysis/stage_c_sc1_d9_history_support_operator_audit_20260715/d9_result_and_rollback.md` |
| SC1-D10 raw scale identifiability | 5 datasets；3 sketches × 3 lambdas；3 families；holdout+validation | binary 0/5 directional；monotone effect/control 4/5但best 0/5、mapping 2/5 | aligned scale unsupported；history routing closed；rollback Step2 | `analysis/stage_c_sc1_d10_raw_scale_identifiability_20260715/d10_result_and_rollback.md` |
| SC1-D11 Step2/3 source-theory gate | external primary sources + exact gradient identity + synthetic controls | Time-o1/FreDF/DBLoss收紧component-loss claim；MSE/L1 additivity与strict conflict definitions冻结 | `theory_pass_diagnostic_protocol_frozen`；remote evidence only | `analysis/stage_c_sc1_d11_future_component_responsibility_20260715/d11_step23_source_theory_audit.md` |
| SC1-D11 remote responsibility gate | 5 datasets × 3 A6 checkpoints；train/validation；MSE/L1；RGNB/DCT/3 random | strict conflict 0/5；support-specific 2/5；generic redistribution 3/5；same-component cross-regime negative=0 | `transform_generic_pressure_sc2_only`；SC1回Step2暂停；SC2仅Step1-3 audit | `analysis/stage_c_sc1_d11_future_component_responsibility_20260715/d11_result_and_rollback.md` |
| Post-D12 systematic mainline audit | D3-D12 synthesis + external primary-source audit + theory/narrative review | retained support geometry；closed scale/conflict/frame explanations；Forking-Sequences/stability/operator overlap mapped | provisional NIFRO/IARL pair；D13-A only authorized | `analysis/stage_c_post_d12_revision_surface_mainline_20260715/systematic_review_and_mainline_redesign.md` |
| Fixed-past mainline reset | internal compression/failure audit + external decoder/memory/IB review | revision idea deferred；global compression boundary identified；CATS/query/global-local overlap mapped | provisional CADMO/CPGA；only D14 diagnostic authorized | `analysis/stage_c_fixed_past_mainline_reset_20260715/fixed_past_mainline_reconstruction.md` |

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
| SC1-D7 descriptor sufficiency | `completed_conditional` | geometry evidence retained；free gap只作compatibility statistic |
| SC1-D8 end-to-end co-adaptation screen | `completed_rollback` | exact shared-latent PAF关闭；不进入三seed |
| SC1-PLGO intervention/readout redesign | `completed_step4` | JAPO为唯一保留候选；geometry-only experts与patch retrieval不推进 |
| SC1-JAPO theory feasibility | `completed_pass` | A6 containment、joint non-collapse、projectivity、continuity与symmetry audit完成 |
| SC1-JAPO Step6 method/control design | `completed_pass` | E2/K256/G32、independent init、seven-arm attribution与staged seed gates frozen |
| SC1-JAPO Step7A local implementation | `completed_pass` | production module + 210 prefix/35 gradient + paired hashes + runner/analyzer dry-run |
| SC1-JAPO Step8 seed2021 remote screen | `completed_inconclusive` | 35/35；protocol/numeric pass；只授权seed2022 unchanged |
| SC1-JAPO Step8 seed2022 confirmation | `completed_fail` | 35/35；70/70 combined；seed2023 stopped；exact v1 closed |
| Contribution 1 Step4 systematic review | `completed` | 支持“support-identifiable local-global projective operator”问题类，但尚未形成method candidate |
| SC1-D9 history-support operator evidence audit | `completed_fail` | exact audit 15/15 valid；D9-B按gate取消 |
| SC1-D10 raw scale identifiability | `completed_fail` | all artifacts/invariants valid；binary与detail-monotone primary gates均fail |
| SC1-D11 future-component responsibility | `completed_rollback` | accepted v2 complete；do not implement conflict-aware method |
| Post-D11 paper-mainline redesign | `completed_step2_3_design` | D11不关闭C1；PRISM/CAPE proposed；MIPR retired；D12 next |
| D12 predictable-frame feasibility | `completed_fail_rollback_step2` | v1 invalid for rejection；v2 valid support 1/5；CAPE/joint route closed；D12-B canceled |
| Post-D12 systematic mainline redesign | `deferred_next_paper` | NIFRO/IARL与D13转移到`New-idea.md`；不再占当前slots |
| D13-A/B rolling-origin diagnostics | `deferred_next_paper` | protocol preserved；not active |
| Fixed-past compression mainline reset | `superseded_by_narrative_scope` | CADMO/CPGA与patch-memory D14降为history-interface auxiliary；未执行 |
| Multi-horizon coupling mainline reset | `superseded_by_native_pcsd_reset` | PCSD/CCRL曾proposed；D14-A evidence保留，CCRL后续retired |
| D14 output-coupling granularity | `a_confirmed_b_retired` | A1 dual-carrier 255/255 confirmed；B1未implementation；remote false |
| PCSD-CF native architecture reset | `step10_training_blocked` | 60/60 result；DIRECT vs A6 0/5；25/25 arm starvation；PCC Step5 conditional pass |

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
| 2026-07-14 | D7 descriptor diagnostic完成 | Current Position、Contribution 1、Boundary、Experiment Logic | geometry evidence + exact-readout rollback | geometry pass；PAF free-control fail；return Step4 capacity-preserving redesign |
| 2026-07-14 | frozen replacement fairness audit | Current Position、Contribution 1、experiment protocol | causal-boundary correction + candidate reopen | D7 free gap降为compatibility statistic；PAF进入D8-E2E Step7A |
| 2026-07-14 | SC1-D8 Step7A local gate | Current Position、Contribution 1、experiment protocol | implementation evidence + remote authorization | 七arms与patch diagnostics通过本地gate；进入Step7B validation-only screen |
| 2026-07-14 | SC1-D8 Step7B E2E result | Current Position、Contribution 1、experiment protocol | exact candidate rollback + geometry evidence retained | PAF exact design关闭；PLGO回Step4；SC2继续held |
| 2026-07-14 | SC1-PLGO Step4 redesign audit | Current Position、Contribution 1、Boundary | source/no-go audit + provisional successor | JAPO仅进入Step5；geometry-only experts、patch retrieval、dense bypass不推进 |
| 2026-07-14 | SC1-JAPO Step5 theory feasibility | Current Position、Contribution 1、Boundary | containment/projectivity/non-collapse proof + optimization boundary | theory pass；identical init禁止；只进入Step6 design |
| 2026-07-14 | SC1-JAPO Step6 narrative/control gate | Current Position、Contribution 1、Main Experiment Logic | concrete method + control/seed protocol freeze | JAPO=`narrative_ready`；只进入Step7A local implementation；remote/SC2 false |
| 2026-07-14 | SC1-JAPO Step7A production gate | Current Position、Contribution 1、Main Experiment Logic | implementation evidence + remote authorization | 210/210 + 35/35 + paired hashes通过；seed2021 remote authorized；test/SC2 false |
| 2026-07-15 | SC1-JAPO Step8 remote launch | Current Position、Main Experiment Logic | running-state update | commit90e4164；3×3090；35-run validation-only；SC2/test held |
| 2026-07-15 | SC1-JAPO seed2021 result audit | Current Position、Contribution 1、Main Experiment Logic | inconclusive effectiveness evidence + staged continuation | 35/35 valid；无pathology；不改design，只补seed2022；test/SC2 held |
| 2026-07-15 | SC1-JAPO seed2022 launch | Current Position、Main Experiment Logic | staged confirmation running | commit3d37440；3×3090；35-run unchanged validation-only；test/SC2 held |
| 2026-07-15 | SC1-JAPO two-seed gate | Current Position、Contribution 1、Boundary | exact candidate closure + Step4 rollback | 70/70 valid；capacity controls explain；exact v1 failed；projective direction retained |
| 2026-07-15 | Post-JAPO systematic review | Current Position、Contribution 1、Boundary、Experiment Logic | evidence synthesis + next diagnostic boundary | 不直接提出新method；SC1-D9先验证history-scale × future-support operator evidence |
| 2026-07-15 | SC1-D9-A exact operator gate | Current Position、Contribution 1、Boundary、Experiment Logic | hypothesis failure + Step2/3 rollback | ordered scale coupling unsupported；binary clue仅exploratory；D9-B取消，D10 protocol next |
| 2026-07-15 | SC1-D10 Step2/3 design | Current Position、Contribution 1、Experiment Logic | raw-data hypotheses + matched probe freeze | binary/detail-monotone/no-scale gates冻结；只授权diagnostic remote evidence |
| 2026-07-15 | SC1-D10 result gate | Current Position、Contribution 1、Boundary、Experiment Logic | aligned-scale closure + Step2 rollback | partial cross-scale signal不构成unified mapping；history routing mainline关闭；D11 question proposed |
| 2026-07-15 | SC1-D11 Step2/3 design gate | Current Position、Contribution 1/2 Boundary、Experiment Logic | prior-art boundary + exact responsibility identity + frozen gates | component loss/gradient surgery不作novelty；strict negative、magnitude与basis controls后再判问题 |
| 2026-07-15 | SC1-D11 remote result gate | Current Position、Contribution 1/2 Boundary、Experiment Logic | conflict hypothesis closure + Step1-3 pivot | strict 0/5、component 2/5；coverage observation只允许SC2 novelty/problem audit |
| 2026-07-15 | Post-D11 paper-mainline redesign | Thesis、Contribution 1/2、Experiment Logic | joint Step2-3 reset + old SC2 closure | D11不否定C1；PRISM/CAPE proposed；MIPR retired；D12 diagnostic next |
| 2026-07-15 | D12 v1/v2 final gate | Current Position、Thesis、Contribution Slots、Experiment Logic | risk mismatch repair + candidate closure + Step2 rollback | v2 valid 1/5；CAPE closed；PRISM joint route retired；D12-B canceled；two slots open |
| 2026-07-15 | Post-D12 systematic mainline redesign | Thesis、Contribution 1/2、Boundary、Experiment Logic | new provisional problem chain + D13 gate | NIFRO/IARL proposed；forecast grid/stability penalty不计创新；只授权D13-A |
| 2026-07-15 | Fixed-past mainline restoration | Thesis、Contribution 1/2、Boundary、Experiment Logic | next-paper archive + current-paper Step2/3 reset | NIFRO/IARL转入`New-idea.md`；CADMO/CPGA provisional；只授权D14 |
| 2026-07-15 | Multi-horizon narrative correction | Thesis、Contribution 1/2、Boundary、Experiment Logic | CADMO/CPGA narrative rejection + Step2/3 reconstruction | ordered patch降为auxiliary；PCSD/CCRL provisional；新D14-A/B active |
| 2026-07-15 | D14-A0 remote result and failure attribution | Current Position、Contribution Slots、Experiment Logic | exact probe closure + direction-rejection correction | crossing 0/5；oracle 0.0586%；DoF/contrast不足；one A1 theory audit；B/method/test held |
| 2026-07-15 | D14-A1 source/theory and Step7A | Current Position、Experiment Logic、failure attribution | neutral-first nonlinear E2E diagnostic | 80+20 local cases pass；neutral remote only；A6 fail不能方向拒绝 |
| 2026-07-16 | D14-A1 three-seed confirmation | Current Position、Contribution Slots、Experiment Logic | problem gate confirmation + next-step authorization | dual-carrier 5/5 stable crossing；strict 7.11%/9.13%；D14-B仅返回Step4-6 |
| 2026-07-16 | D14-B1 Step4-6 source/theory/design | Contribution 2、Boundary、Experiment Logic | novelty tightening + objective correction + local authorization | TimeFuse/TimeRouter controls mandatory；hybrid risk auxiliary；Step7A local only |
| 2026-07-16 | CCRL consistency audit + PCSD-CF reset | Contribution 1/2、Boundary、Experiment Logic | SC2 retirement + SC1 native Step4-6 | CCRL diagnostic-only；shared coupling field conditional pass；D15-A Step7A next |
| 2026-07-16 | PCSD-CF D15-A Step7A local gate | Current Position、Contribution 1、Experiment Logic | production implementation + theory contract | 9/9 local gates pass；effectiveness/remote/SC2/test held；Step7B tooling next |
| 2026-07-16 | PCSD-CF Step7B prelaunch gate | Current Position、Contribution 1、Experiment Logic | controls + runner + diagnostics + authorization | 60/60 contracts pass；seed2021 validation-only remote authorized；test/SC2 held |
| 2026-07-16 | PCSD-CF Step7B remote launch | Current Position、Experiment Logic | running-state + exact launch provenance | commit b9693ec；GPU 0/1/2；60-run validation-only matrix；test/SC2/confirmation held |
| 2026-07-16 | PCSD-CF Step9/10 + PCC Step2-4 | Current Position、Contribution 1/2、Experiment Logic | effectiveness failure attribution + training-candidate reset | DIRECT vs A6 0/5；25/25 arms starved；PCC Step5 local only |
| 2026-07-16 | PCC Step5 theory feasibility | Contribution 2、Experiment Logic | algebra/projectivity/synthetic gate | 15/15 pass；Step6 design only；implementation/remote/test held |
| 2026-07-16 | milestone test policy + SC-D15-T1 authorization | Evaluation Rule、Experiment Logic | test becomes primary effectiveness gate | frozen v1 12×5 audit；no retraining；PCC Step6 held |
| 2026-07-16 | SC-D15-T1 complete test audit | Current Position、Contribution 1/2、Evaluation Rule | exact v1 closure + test-informed rollback | 60/60；DIRECT vs A6 -1.3994%、1/5；oracle +2.0197%；PCC Step6 design only |
| 2026-07-16 | PCC Step6 source-informed redesign | Contribution 2、Boundary、Experiment Logic | v0 prior-art demotion + v1 projective transport | 19/19 pass；9×5 controls frozen；Step7A local only |

## Continuation Rules

1. 每次继续研究先读本 ledger 与active protocol；remote不得静默改变frozen profiles、rank、init、controls或gates；
2. old analysis可引用，archive脚本不得直接启动；
3. diagnostic failure必须区分 hypothesis、intervention、readout、numeric与capacity control；
4. D2 formal5只在frozen A6 representation/head family下不支持depth grouping；当前PLGO不使用该设计，若未来
   重新提出end-to-end grouping method，必须作为新候选通过Step2-6；
5. test reference只用于最终对比，不能参与设计选择。
6. D14-A0只匹配factor storage count而未匹配rank-manifold DoF；其exact negative不得关闭PCSD方向。A1必须
   先证明effective-capacity matching与minimum function contrast；A1仍失败则关闭pair，不继续换head。
6. future mechanism screen固定使用ETTh1/ETTh2/ETTm1/ETTm2/Weather；五dataset不能替代三seed确认。
7. D3只支持probe-family basis main effect；Step 4必须先排除standard basis、whitening与static regularization解释。
8. D4已完成上述排除：不得把fixed balanced midpoint basis单独升为Contribution 1；可将其保留为generation component，并研究conditioning-locality co-design。
9. PLGO Step5只通过RGNB mathematical scaffold；不得把ONB换基、overcomplete union或full-affine group maps升为method。
10. prior-art primitive overlap不自动否决task-specific contribution；PLGO的claim必须落在multi-horizon
    projectivity、RGNB support geometry与atomwise generation组合上。B14仍禁止atom retrieval。
11. D7只证明canonical geometry在frozen A6 representation上优于matched descriptors；free-M0 gap不能证明PAF
    function class失败。primary method gate必须是D8 from-scratch end-to-end joint training。
12. 以后freeze/replace默认只作conditional diagnostic；不得据此拒绝paper-core method或强制架构redesign。
13. D8之后不得把flatten本身写成信息压缩；真正失败边界是fixed descriptor-generated separable operator。
14. geometry-only linear expert mixture可吸收到更宽PAF；固定rank无新class，扩rank需capacity control，不得直接升method。
15. JAPO seed2021完整但inconclusive；只授权原协议seed2022，不得因near-uniform routing临时加入loss、改初始化或
    调E/K/G。two-seed mean未过冻结gate则停止exact JAPO并做failure attribution；只有通过才授权seed2023。
16. two-seed gate已失败：不得补seed2023、tune exact v1或提前启动SC2。JAPO containment/projectivity只保留为
    theory evidence；新的paper-core route必须回Step4，解决weak expert mixing与operator intervention问题。
17. 系统复盘后，下一步固定为`SC1-D9 diagnostic_only`：先从A6 learned operator检验history-scale与
    future-support是否存在canonical coupling。D9通过不等于method有效，只授权新候选进入Step4-5；若跨dataset/
    seed不优于matched permutation/random controls，则回Step2/3，不实现local-global operator。
18. D9-A已失败且protocol有效：不得启动D9-B或用post-hoc binary contrast挽救primary decision。下一步只能在
    Step2/3设计D10 raw-data identifiability，先区分binary、monotone与no-scale hypotheses。
19. D10的binary 2×2与detail-only 6×6 gates已在结果前冻结；off-diagonal matrix没有primary gate，不得在同批
    artifacts上事后升级。D10通过只返回Step4，失败只关闭raw linear aligned-scale evidence。
20. D10已失败且invariants完整：不得从ETT子集off-diagonal patterns事后提出adaptive router。D9+D10共同关闭
    history-scale aligned routing mainline；下一步回Step2审计future-component responsibility，不自动恢复SC2。
21. D11只把negative inner product称为directional conflict；low positive cosine、norm imbalance与generic
    transformed-component pressure必须分别归因。Time-o1/FreDF/DBLoss使generic component loss不可作论文创新；
    D11 positive只返回Step4，不授权PCGrad、new decoder、loss或SC2。
22. D11 accepted v2的total gradient在所有validation MSE dataset/path/batch上均无negative dot；formal
    support-specific gate仅2/5，same-component跨regime negative为0。不得把within-regime group cancellation或
    RGNB JS重新命名为SC1 conflict evidence。
23. short measure对RGNB最后两个groups的zero responsibility只支持nested supervision coverage observation。
    下一步仅授权SC2 Step1-3审计；在证明其不等价于generic importance sampling、step weighting、GradNorm、
    transformed-label loss或existing MIPR前，不实现任何training strategy。
24. post-D11 external audit已经完成上述审计：generic coverage sampling不是足够的新贡献，且完整label可用时raw
    $W_\mu$ risk可精确计算；MIPR正式retired，不再实现。
25. D12启动时只允许diagnostic：先验证cross-fitted predictable covariance，再验证PRISM localization Pareto；
    当时PRISM/CAPE均为`proposed_step2_3`，未因论文需要两个贡献而跳过problem/narrative gate。
26. D12 CAPE frame由train-only OOF predictions构造；final model不复用pilot weights。任何frozen A6
    memory probe只作symmetric conditional diagnostic，不能充当method effectiveness gate。
27. D12-v1因uniform normalized risk mismatch不能方向级否定；risk-aligned v2复用相同pilots后仅1/5支持，
    因此CAPE与joint PRISM route关闭、D12-B取消并回滚Step2。不得通过改rank、换pilot或PRISM-only probe绕过gate。
