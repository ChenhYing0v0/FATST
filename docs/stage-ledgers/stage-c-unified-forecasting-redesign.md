# StageC Unified Varied-Horizon Forecasting Ledger

## Stage Scope

| Field | Content |
| --- | --- |
| `stage_id` | `StageC-UVHF` |
| `paper_role` | decoder/operator 与 training principle 两项相互支撑的 paper-core innovations |
| `active_question` | trajectory-level structured decoder相对strong A6_MEASURE learned-basis control是否仍有可测headroom？ |
| `source_evidence` | historical/source-faithful `A6-LBF-r256` |
| `mechanism_control` | same-run end-to-end `A6-LBF-natural-baseline`；frozen A6只作reference/diagnostic |
| `active_candidates` | `SC1-SIFF-v2-EQ-ATTR-v1` frozen parent；D18 closed；`SC-D19-IFC-control-v1.1 control_only` Step7B prelaunch pass / Step8 launch next；no method；CTD paused |
| `future_validation_suite` | ETTh1, ETTh2, ETTm1, ETTm2, Weather；five profiles frozen |
| `paper_facing_scorecard` | validation/test H96,H192,H336,H720 MSE/MAE；dense默认diagnostic |
| `stage_exit` | 新两项分别过 narrative/effectiveness gate并形成可归因joint story |
| `stage_rollback` | Contribution 1回Step2/4重构decoder problem；Contribution 2回Step2；CTD remains paused |

## Decision Cursor

| Field | Content |
| --- | --- |
| `current_11_step` | D18 Step9/10 closed；Contribution 1/2 Step2；D19 control Step7B prelaunch pass / Step8 launch next |
| `current_candidate` | immutable SIFF-v2 parent；`SC-D19-IFC-control-v1.1 control_only` Step8 remote launch next；no method |
| `latest_decision` | specialists vs A6_MEASURE仅+0.1659%、7/15；measure explains；soft projectivity closed |
| `next_required_action` | commit/push后在3090执行preflight、Weather IF + ETTm2 direct resource smoke并后台启动15-run matrix |
| `method_training_authorized` | false；D19 Step7B=31/31；seed2021 control remote/test=true；confirmation/paper method=false |
| `rollback_point` | D19 control无headroom → fixed-past decoder paper viability review；Contribution 2=Step2 |

## 11-Step Record

| Field | Current Record |
| --- | --- |
| `current_step` | Contribution 1 Step2；Contribution 2 Step2；D19 control Step8 launch next |
| `problem` | strict projectivity cost不成立后，forecasting phase是否仍存在超越A6 learned-basis的trajectory-structure headroom |
| `existence_evidence` | D18 specialists vs A6_MEASURE仅+0.1659%；A6_MEASURE vs A6_FULL +1.798%、15/15 |
| `idea` | source-informed IF作为control，与A6_MEASURE及matched MLP做from-scratch E2E比较 |
| `theory_check` | full-T implicit decoding可保持crop projectivity；Fourier IF与harmonic weighting均已有直接prior art |
| `design` | v1.1修复为same 720-history；A6/IF/IF-no-skip/matched-direct × five datasets；Step7A 114/114 |
| `narrative_gate` | no active child candidate |
| `effectiveness_gate` | CCSF false；SIFF-v2 parent为performance-near但attribution blocked |
| `artifacts` | D18 report + IF note + D19 v1.1 repair + Step7A + Step7B config/manifest/gates/runner/analyzer |
| `decision` | D19 v1 superseded before training；v1.1 Step7B 31/31；一次seed2021 control remote/test已授权，no method/confirmation |

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
| `SC1-PCSD-CF` | `rejected_effectiveness_test_representation_signal_retained` | one shared parameter field能否经scope pooling形成skilled point/block/global arms并contain A6 | field-pooling chain retained as evidence | fair test DIRECT vs A6 -0.8562%；arms高度失衡；oracle +4.41% | exact direct v1 closed；representation/credit-starvation retained |
| `SC2-CCRL` | `retired_as_core_diagnostic_only` | cross-fit relative risk能否增益matched direct fusion | generic overlap high；two-stage teacher/student inconsistency | not implemented | retain report/config as history；not scheduled |
| `SC2-ICC` | `superseded_by_pcc` | same-forward marginal coupling credit能否修复direct policy misallocation | working hypothesis已由D15-A收紧 | not implemented | historical name only |
| `SC2-PCC-v0` | `superseded_pointwise_control` | pointwise same-forward capability + skill floor | expert loss与loss-teacher gate已有直接prior art | 15/15 theory cases；method untested | mandatory pointwise/prior controls only |
| `SC2-PCC-v1-TI` | `rejected_effectiveness_test_fair` | nested-prefix capability能否经harmonic incidence输运为target-coordinate credit | exact claim fail；generic/prior explains most gain | fair test vs prior PCSD +0.0806% fail；on SIFF vs equal -0.2663% | exact v1 closed；return Step2/4 |
| `SC1-SIFF-v1` | `partial_pass_attribution_blocked` | coupling scale能否作为internal coordinate生成可辨识且连续共享的history modes | complete-chain conditional pass | EQUAL vs PCSD +0.5906% pass；vs A6 +1.6436%；prior/PCC/independent specificity fail | return Step6；EQUAL-context controls before seeds |
| `SC1-SIFF-v2-EQ-ATTR-v1` | `frozen_performance_near_candidate_attribution_blocked` | EQUAL-trained ordered scale field能否同时超过A6/PCSD与matched EQUAL-context specificity controls | conditional；完整claim未成立 | main 2/3；controls 3/4；internal 7/7 | immutable parent；Step4 source-informed redesign，不补v1 seeds |
| `SC1-SIFF-v2-CCSF-v1-tau25` | `failed_effectiveness_and_attribution_closed` | target-free scope contrast能否让policy识别relative competence；relative teacher只作co-designed弱监督 | prelaunch narrative conditional；post-result complete claim fail | 50/50、200/200；vs A6 -0.8567%；vs v1 -0.6159%；architecture/objective/ordered specificity fail | no confirmation；exact contrast-policy route closed；return Step2/4 |
| `SC1-CCSF-D2` | `diagnostic_only_closed` | region aggregation能否把contrast competence转成更强mixture utility | not a method gate | expected-arm signal 4 widths/5 datasets稳定；相对pointwise mixture margin不足，2/3 gates | region retained as analysis scale only；do not train |
| `SC1-CCSF-D3` | `diagnostic_only_closed` | residual covariance/cancellation是否解释best-arm teacher与fusion不一致 | not a method gate | simplex vs best-arm仅约+1.34%–1.38%；0 widths过dataset gate | covariance-aware redesign unsupported；return Step2 |
| `SC1-CCSF-D4` | `diagnostic_only_closed` | soft policy是否只需global sharpening或hard routing | not a method gate | best native arm仍-0.0186%、1/5；hard routing更差 | softness not primary；close contrast-policy route |
| `SC-D17-PFC-v1` | `diagnostic_only_conditional_negative_closed_exact_protocol` | frozen full-domain draft的ordered prefix context能否超越pointwise与row-shuffled controls并跨validation→test transfer | not a method gate；dual-carrier conditional evidence only | causal vs pointwise -3.0356%；vs shuffled -2.3616%；pointwise vs parent -28.7314%；1/7 gates | exact post-hoc protocol closed；direction unresolved；no E2E rescue |
| `SC-D18-SPC` | `diagnostic_only_closed_problem_false` | horizon-specific loss是否稳定超过A6_MEASURE，从而证明exact projectivity有accuracy cost | not a method gate | 25/25；vs A6_MEASURE +0.1659%、7/15；2/7 gates | no seeds；soft route closed；rollback Step2 |
| `SC-D19-IFC-control-v1.1` | `control_only_step7b_prelaunch_pass_step8_launch_next` | source-informed implicit trajectory decoder是否超过A6 learned-basis control | not a contribution；IF prior mandatory | Step7A 114/114；Step7B 31/31；effectiveness not started | 3090 resource smoke + 15-run launch；confirmation false |
| `SC2-MCCA-v1` | `historical_validation_negative_fair_test_not_reaudited` | same total scope skill mass能否竞争性分配而避免per-target homogenization | complete-chain conditional pass | old best-H720 validation four-H -0.1357%、1/5；not in 70-run audit | inactive；reuse前回Step4重审相对EQUAL/MEASURE的必要性 |
| `SC-D16-CTD` | `deferred_paused_by_user` | H720 checkpoint是否丢弃healthy SIFF four-H epoch | diagnostic only；weighted checkpoint prior-covered | not implemented | design retained；resume only after user authorization |
| `SC-RETRO-FAIR-v1` | `completed_partial_pass_attribution_blocked` | PCSD/PCC/SIFF在新checkpoint与test-primary规则下是否仍成立 | retrospective audit；not a method | 70/70；280/280；joint pass；two-contribution attribution fail | archive result；SIFF Step6/PCC Step2-4 |
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
| SIFF/MCCA Step7A production gate | production modules + 36 algebra/numeric/gradient/matched-rank cases | 36/36；Q1/A6 gap 0；float32 marginal `4.47e-8`；test=false | implementation pass；Step7B prelaunch | `analysis/stage_c_post_pcc_step7a_local_20260717/step7a_implementation_gate_report.md` |
| SIFF/MCCA Step7B prelaunch | 55 CLI + model constructors + runner/evaluator/analyzer dry-run | 8/8 categories；dataset-major 3-worker matrix；test=false | seed2021 validation remote authorized | `analysis/stage_c_post_pcc_step7b_prelaunch_20260717/prelaunch_report.md` |
| SIFF/MCCA Step9/10 result | 55 new + 25 matched references；factorial/control/horizon attribution | SIFF main -1.5015%；MCCA -0.0250%；joint -0.5621%；ETTm2 H1 pathology | exact pair closed；SIFF direction rejection invalid；MCCA hypothesis false；return Step4 | `analysis/stage_c_post_pcc_step7b_seed2021_20260717/step9_10_result_and_failure_attribution.md` |
| SIFF/MCCA four-H reevaluation | retrospective validation scorecard | SIFF -2.3509%；MCCA -0.1357%；joint -1.3325%；decision unchanged | dense降diagnostic；CTD按best-standard重冻结 | `analysis/stage_c_post_pcc_standard_horizon_reevaluation_20260717/standard_horizon_reevaluation_report.md` |
| SC-D16-CTD Step5/6 | source/code correction + trajectory theory/design | 4 ETTm2 arms；20 epochs；4 checkpoint rules；standard+dense gates | `diagnostic_design_refrozen_v1_1_step7a_local_only` | `analysis/stage_c_d16_ctd_step56_20260717/step56_diagnostic_design.md` |
| SIFF_EQUAL attribution Step6 | test-informed mechanism attribution freeze | 10 arms × 5 datasets；50 runs/200 test cells；16/16 static checks | narrative-ready；Step7A local only | `analysis/stage_c_siff_equal_attribution_step6_20260718/step6_attribution_protocol.md` |
| SIFF_EQUAL attribution Step7A | production wiring + component intervention + four-layer analyzer | 13/13 categories；50 CLI；35 constructors；10 gradients；5 component cases | local implementation pass；Step7B prelaunch next | `analysis/stage_c_siff_equal_attribution_step7a_20260718/step7a_implementation_gate_report.md` |
| SIFF_EQUAL attribution Step7B prelaunch | formal authorization + remote environment + command gate | 9/9；3 GPUs idle；50 runs/200 cells；confirmation held | seed2021 Phase A launch authorized | `analysis/stage_c_siff_equal_attribution_step7b_prelaunch_20260718/prelaunch_report.md` |
| SIFF_EQUAL attribution Step8 launch | remote execution | commit c4c4730；dry-run/smoke pass；GPU0/1/2 jobs1-3 active | Phase A running；no confirmation | `analysis/stage_c_siff_equal_attribution_step8_remote_20260718/remote_launch_record.md` |
| SIFF_EQUAL attribution Step9 | 50 runs/200 test cells + four-layer attribution | vs A6_MEASURE -0.2366%；vs independent +0.2580%；internal 7/7 | exact v1 closed；return Step4；no confirmation | `analysis/stage_c_siff_equal_attribution_step9_20260718/step9_four_layer_diagnostic.md` |
| SIFF v1 candidate freeze + Step4 audit | immutable candidate manifest + existing-artifact routing/fusion audit + external sources | policy match 29.24%；static convex +2.2112%；affine extra +0.1203% | v1 retained/attribution blocked；CCSF provisional Step5；no training | `analysis/stage_c_siff_candidate_step4_source_audit_20260718/source_informed_improvement_audit.md` |
| CCSF Step5 theory feasibility | target-free contrast cross-fit + teacher geometry + projectivity/inclusion/source audit | 5/5 gates；contrast vs coordinate +1.8348%；vs shuffled +1.7085%；A6 anchor rejected | conditional pass to Step6；implementation/remote false | `analysis/stage_c_siff_ccsf_step5_theory_20260718/step5_theory_feasibility.md` |
| CCSF Step6 narrative/control gate | 2×2 architecture/objective + teacher/capacity/semantic/field controls | 10 arms；50 Phase-A runs/200 cells；static 5/5；parameter gap <0.5% | narrative-ready；Step7A local only；validation pilot/remote/test false | `analysis/stage_c_siff_ccsf_step6_20260718/step6_narrative_control_gate.md` |
| CCSF Step7A local implementation | production contrast path + 2 objectives + 10-arm adapters + construction gate | 18/18；50 CLI；30 constructors；10 gradients；4 projectivity；two-step correction pass | local pass；Step7B prelaunch next；pilot/remote/test false | `analysis/stage_c_siff_ccsf_step7a_20260718/step7a_implementation_gate_report.md` |
| CCSF Step7B temperature-pilot prelaunch | shared selection config + 15-job runner + completeness analyzer + remote preflight | 14/14；15 runs/60 validation cells；synthetic tie与no-test boundary pass；3 GPUs idle | pilot remote authorized；formal Phase A/test/confirmation false | `analysis/stage_c_siff_ccsf_step7b_prelaunch_20260718/prelaunch_report.md` |
| CCSF Step8 temperature-pilot launch | commit sync + dry-run + resource smoke + 3-GPU background launch | commit`06d0ffc`；driver PID654232；first Weather jobs active；test=false | pilot running；do not monitor；formal Phase A held | `analysis/stage_c_siff_ccsf_temperature_pilot_step8_remote_20260718/remote_launch_record.md` |
| CCSF Step8 first-attempt failure + repair | 0/15 completion audit + traceback + zero-contrast gradient reproducer | three Weather runs NaN；pre-fix 7200 NaN grads；post-fix 0；three-temp nine steps finite | numeric implementation fault；direction rejection invalid；retry smoke next | `analysis/stage_c_siff_ccsf_runtime_repair_20260718/runtime_failure_and_repair_report.md` |
| CCSF Step8 repaired retry1 launch | three-batch Weather smoke + 3-GPU background launch | train/val finite；checkpoint/metrics pass；commit`7045c80`；driver PID683945 | retry1 running；do not monitor；formal test held | `analysis/stage_c_siff_ccsf_temperature_pilot_retry1_step8_remote_20260718/remote_relaunch_record.md` |
| CCSF pilot retry1 result + candidate freeze | 15-run/60-cell validation audit + selection stability | 9/9；tau0.25；MSE0.568165；17/20 cells、4/5 datasets、4/4 horizons | formal candidate frozen；Step7B formal prelaunch next；test false | `analysis/stage_c_siff_ccsf_temperature_pilot_retry1_result_20260718/pilot_result_and_candidate_freeze.md` |
| CCSF tau0.25 formal Phase-A prelaunch | 50-job runner + official-test evaluator + CCSF internal artifacts + four-layer analyzer | 15/15；50 runs/200 cells；10 hard comparisons；runtime/nonmutation/test metadata pass | Phase A/test true；remote smoke/launch next；confirmation false | `analysis/stage_c_siff_ccsf_v1_tau25_phase_a_20260718/formal_phase_a_prelaunch_report.md` |
| CCSF tau0.25 formal Phase-A launch | commit/pull + 3-GPU preflight + three-batch smoke + background driver | commit`604e1b8`；GPUs0/1/2；first Weather jobs active | Step8 running；do not monitor/change；confirmation false | `analysis/stage_c_siff_ccsf_v1_tau25_phase_a_20260718/remote_launch_record.md` |
| CCSF tau0.25 formal Step9–10 | 50-run official-test four-layer audit | 50/50、200/200；vs A6 -0.8567%；vs v1 -0.6159%；10项hard comparisons仅permuted control为正 | exact candidate failed；confirmation canceled；post-E2E attribution only | `analysis/stage_c_siff_ccsf_v1_tau25_phase_a_20260718/step9_four_layer_and_redesign_audit.md` |
| CCSF D2 granularity | horizon-agnostic widths + row cross-fit | region expected-arm specificity稳定；best native mixture margin over pointwise仅+0.1478pp；2/3 gates | region route不升method | `analysis/stage_c_siff_ccsf_v1_tau25_phase_a_20260718/d2_granularity_diagnostic/` |
| CCSF D3 mixture-risk decomposition | best-arm vs simplex oracle | simplex相对best-arm约+1.34%–1.38%；0 eligible widths pass | covariance不是主矛盾；route closed | `analysis/stage_c_siff_ccsf_v1_tau25_phase_a_20260718/d3_mixture_risk_decomposition/` |
| CCSF D4 readout sharpness | exponents + hard argmax | best native -0.0186%、1/5；hard更差 | softness不是主矛盾；Contribution 1回Step2/4 | `analysis/stage_c_siff_ccsf_v1_tau25_phase_a_20260718/d4_readout_sharpness_diagnostic/` |
| post-CCSF Step2/4 reset | projectivity proof + external primary-source audit + historical boundary | exact-projective requested-H adaptation形成no-go；future-coordinate context为唯一待诊断family | D17 diagnostic only；Contribution 2 held | `analysis/stage_c_post_ccsf_step24_reset_20260719/step24_problem_and_source_audit.md` |
| D17-v0 protocol audit | same-test 256-row cross-fit + pointwise/shuffled controls | 表面6/6，但pointwise +21.27%；flattened sample×channel跨fold，protocol invalid | result prohibited；改validation-fit→test | `analysis/stage_c_post_ccsf_step24_reset_20260719/d17_projective_future_context/INVALID_PROTOCOL.md` |
| D17-v1 validation-to-test | 10 checkpoint-preserving validation exports + existing test probes | causal-pointwise -3.0356%；causal-shuffled -2.3616%；pointwise-parent -28.7314%；prefix gap 0 | exact protocol fail；transfer pathology；direction unresolved | `analysis/stage_c_post_ccsf_step24_reset_20260719/d17_result_and_failure_attribution.md` |
| D18 soft-projectivity problem audit | no-go relaxation + primary-source boundary + 15-run oracle design | protocol design complete；problem untested | Step3/static freeze next；remote/test false | `analysis/stage_c_post_ccsf_step24_reset_20260719/soft_projectivity_step2_problem_audit.md` |
| D18 Step3 prelaunch | machine-readable 25-unit matrix + gradient/probe/authorization gate | 11/11；15/15 gradient；prefix/tail gaps=0；CLI/probe smoke pass | 15-run diagnostic remote/test authorized；method=false | `analysis/stage_c_post_ccsf_step24_reset_20260719/d18_prelaunch/step3_prelaunch_report.md` |
| D18 Step8 launch | remote pull + 11/11 recheck + Weather SPEC96 smoke + 3-GPU launch | commit`c843178`；GPUs0/1/2；first three specialists healthy | running；no config/pull/selection；method=false | `analysis/stage_c_post_ccsf_step24_reset_20260719/d18_remote_launch_record.md` |
| D18 Step9/10 four-layer diagnostic | 25/25 units + 15 own-H test cells + validation/checkpoint/probe attribution | specialists vs A6_MEASURE +0.1659%、7/15；A6_MEASURE vs FULL +1.798%、15/15；2/7 gates | projectivity-cost hypothesis false；soft route closed；rollback Step2 | `analysis/stage_c_post_ccsf_step24_reset_20260719/d18_step9/d18_step9_four_layer_diagnostic.md` |
| post-D18 Step2 viability audit | internal closure + external primary-source search + next-control boundary | IF supports forecasting-phase research；ElasTST/QDF/Loss Shaping block simple objective claims | D19 IF control Step4/5 next；method/remote/test false | `analysis/stage_c_post_ccsf_step24_reset_20260719/post_d18_step2_mainline_viability_audit.md` |
| D19 IF control Step4/5 | official paper/code + full-T projectivity proof + A6 function-class audit | upstream already generates720 then crops；IF adds nonlinear polar synthesis + spectrum skip | conditional pass；Step6 control design only；implementation/remote/test false | `analysis/stage_c_post_ccsf_step24_reset_20260719/d19_step45_source_theory_control_audit.md` |
| D19 IF control Step6 | four-arm exact contract + parameter-matched direct/no-skip controls + decision map | 9/9 static；15 new runs + 5 reused A6；80 official-test cells frozen | Step7A local implementation only；remote/test/paper method false | `analysis/stage_c_post_ccsf_step24_reset_20260719/d19_step6_control_design.md` |
| D19 Step6 v1.1 contract repair | A6 natural seq_len audit + same-history fairness correction | v1的96-history/49-bin contract不匹配A6 720 history；改为361 bins并重新parameter-match | v1 superseded before training；v1.1进入Step7A | `analysis/stage_c_post_ccsf_step24_reset_20260719/d19_step6_contract_repair.md` |
| D19 IF control Step7A | three production readouts + 15 CLI + shape/projectivity/gradient/init/source gates | 114/114；max prefix gap 0；IF/no-skip hash相同；all gradients finite/nonzero | Step7A pass；Step7B prelaunch next；remote/test/paper method false | `analysis/stage_c_post_ccsf_step24_reset_20260719/d19_step7a_local/step7a_implementation_gate_report.md` |
| D19 IF control Step7B prelaunch | frozen runner + checkpoint-preserving test evaluator + amplitude/phase artifacts + four-layer analyzer | 31/31；15 new + 5 reused A6；80 test cells；all contracts/CLI/smokes pass | seed2021 control remote/test true；confirmation/paper method false；Step8 launch next | `analysis/stage_c_post_ccsf_step24_reset_20260719/d19_step7b_prelaunch/prelaunch_gate_report.md` |

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
| CCSF formal Phase-A and D2–D4 | `completed_rollback` | exact contrast-policy closed；no confirmation；new method training false |
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
| PCC-v1-TI Step7A local implementation | `completed_pass` | 35/35；one-forward objective、CLI与adapter smoke通过；Step7B prelaunch next |
| PCC-v1-TI Step7B prelaunch | `completed_pass` | 8/8；45 CLI；dataset-major runner、gradient/analyzer smoke通过；remote seed2021 authorized |
| SIFF/MCCA Step7A production implementation | `completed_pass` | 36/36；production tensor/objective/control contracts通过 |
| SIFF/MCCA Step7B remote screen | `completed_fail_rollback_step4` | 55/55+25/25；exact pair closed；no test/seeds |
| SC-D16-CTD trajectory diagnostic | `paused_by_user` | design retained；Step7A/remote均不继续 |
| PCSD/PCC/SIFF fair test re-audit | `completed_partial_pass` | 70/70；SIFF_EQUAL best +1.6436%；PCC harms SIFF；checkpoint false-failure corrected |
| Fair re-audit internal mechanism health | `completed_diagnostic` | DIRECT arms失衡；PCC oracle增大但policy近均匀；SIFF_EQUAL未collapse；MCCA不在矩阵 |
| SIFF_EQUAL EQUAL-context attribution freeze | `completed_exact_v1_closed` | 50/50；main 2/3、controls 3/4、internal 7/7；回Step4，不补confirmation |
| SIFF v1 portfolio freeze | `completed_retained_attribution_blocked` | immutable manifest complete；作为当前best candidate与v2 parent，不改Step9 failure |
| SIFF contrast-calibrated redesign | `formal_candidate_frozen_prelaunch_next` | 实现50-run formal prelaunch tooling；Phase A/test仍禁止 |
| D18 soft-projectivity cost diagnostic | `completed_fail_rollback_step2` | 25/25；2/7 gates；不补seeds；soft route closed |
| D19 source-informed implicit decoder control | `step7b_prelaunch_pass_step8_launch_next` | commit/push；3090 preflight + dual resource smoke + 15-run background launch；confirmation false |

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
| 2026-07-16 | PCC-v1-TI Step7A local gate | Current Position、Contribution 2、Experiment Logic | production objective + training-path invariants | 35/35 pass；inference unchanged；Step7B prelaunch next；remote/test false |
| 2026-07-17 | PCC-v1-TI Step7B prelaunch gate | Current Position、Contribution 2、Experiment Logic | 45-run tooling + remote authorization | 8/8 pass；seed2021 validation remote authorized；test/confirmation/Phase B false |
| 2026-07-17 | PCC-v1-TI Step8 remote launch | Current Position、Contribution 2、Experiment Logic | running-state + exact launch provenance | commit282b96c；GPU0/1/2；45-run validation matrix running；test/confirmation/Phase B false |
| 2026-07-17 | PCC-v1-TI Step9/10 + Step4 rollback | Current Position、Contribution 1/2、Experiment Logic | specificity/diversity failure attribution + source-informed redesign | 45/45；vs A6 +0.9627%；prior/diversity fail；new pair Step5 theory only |
| 2026-07-17 | SIFF/MCCA Step5 theory feasibility | Contribution 1/2、Experiment Logic | containment/projectivity + balanced competitive allocation | 10/10 pass；Step6 only；implementation/remote/test false |
| 2026-07-17 | SIFF/MCCA Step6 source/method/control design | Contribution 1/2、Boundary、Experiment Logic | Q2 field + same-mass I-projection + $2\times3$ factorial | 22/22 pass；Step7A local only；remote/test/confirmation false |
| 2026-07-17 | SIFF/MCCA Step7A implementation | Current Position、Contribution 1/2、Experiment Logic | production code + exact/numeric/gradient evidence | 36/36 pass；effectiveness pending；Step7B prelaunch next |
| 2026-07-17 | SIFF/MCCA Step7B prelaunch | Current Position、Experiment Logic | 55-run tooling + validation-only remote authorization | 8/8 pass；seed2021 remote launch next；test/confirmation false |
| 2026-07-17 | SIFF/MCCA Step8 remote launch | Current Position、Experiment Logic | exact launch provenance + running state | commit `7a9e5c7`；GPU0/1/2；55-run validation matrix；test/confirmation false |
| 2026-07-17 | SIFF/MCCA Step9/10 result | Current Position、Contribution 1/2、Experiment Logic | exact candidate closure + measure-mismatch rollback | 55/55；SIFF/MCCA main effects fail；short-prefix pathology blocks direction rejection；SC-D16 source audit next |
| 2026-07-17 | SC-D16 Step4 source/code audit | Current Position、Contribution Boundary、Experiment Logic | attribution correction + diagnostic narrowing | harmonic-L1已在coupling training；PHMA/HR关闭；CTD four-run进入Step5/6 design only |
| 2026-07-17 | SC-D16-CTD Step5/6 design | Current Position、Experiment Logic | trajectory/checkpoint gate freeze | ETTm2 four-run、20 epochs、three rules；Step7A local only；remote/test false |
| 2026-07-17 | paper-facing evaluation governance + retrospective audit | Evaluation Protocol、Current Position、Experiment Logic | four-H default + dense diagnostic boundary | SIFF/MCCA four-H仍fail；CTD v1.1改为best-standard primary；test false |
| 2026-07-17 | test-primary governance + fair re-audit Step7A | Current Position、Candidate Queue、Evaluation Protocol | validation职责收缩 + CTD暂停 + 70-run matrix freeze | 9/9 local categories pass；remote test matrix authorized |
| 2026-07-17 | fair re-audit Step8 launch | Current Position、Experiment Ledger | resource smoke + launch provenance | commit d294aab；GPU0/1/2；70-run test-primary matrix running |
| 2026-07-18 | fair re-audit Step9/10 | Current Position、Candidate Queue、Experiment Ledger | effectiveness + attribution correction | SIFF partial pass；PCSD/PCC close；joint performance pass but paper-pair fail |
| 2026-07-18 | fair re-audit mechanism-health correction | Candidate Queue、Failure Attribution、MCCA boundary | internal arm audit + evaluation-scope correction | DIRECT training-blocked；PCC headroom unused；MCCA fair-test unaudited |
| 2026-07-18 | SIFF_EQUAL attribution Step6 | Current Position、Candidate Queue、Experiment Logic | four-layer rule + 10-arm EQUAL-context freeze | 16/16；Step7A local only；remote/test/confirmation false |
| 2026-07-18 | SIFF_EQUAL attribution Step7A | Current Position、Candidate Queue、Experiment Logic | production path + component artifact + analyzer | 13/13；Step7B prelaunch next；remote/test/confirmation false |
| 2026-07-18 | SIFF_EQUAL attribution Step7B prelaunch | Current Position、Candidate Queue、Experiment Logic | formal test authorization + remote launch freeze | 9/9；seed2021 Phase A authorized；confirmation false |
| 2026-07-18 | SIFF_EQUAL attribution Step8 launch | Current Position、Experiment Ledger | commit/resource/process provenance | c4c4730；3 workers；50-run Phase A running；confirmation false |
| 2026-07-18 | SIFF_EQUAL attribution Step9 | Current Position、Candidate Queue、Experiment Ledger | four-layer result + exact candidate closure | A6_MEASURE/independent explain；internal healthy；rollback Step4 |
| 2026-07-18 | SIFF v1 freeze + Step4 improvement audit | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | portfolio retention + fusion bottleneck localization + source boundary | v1 immutable；calibration primary；CCSF Step5 next；training false |
| 2026-07-18 | CCSF Step5 theory feasibility | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | contrast identifiability + tensor/loss/proof/source boundary | 5/5；A6 anchor removed；Step6 next；training false |
| 2026-07-18 | CCSF Step6 narrative/control gate | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | contribution split + 10-arm factorial/control freeze + rollback map | static 5/5；Step7A local only；validation pilot/remote/test false |
| 2026-07-18 | CCSF Step7A local implementation | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | production tensor/objective/control paths + construction evidence | 18/18；Step7B next；pilot/remote/test false |
| 2026-07-18 | CCSF Step7B temperature-pilot prelaunch | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | shared hyperparameter selection + remote authorization boundary | 14/14；pilot remote true；formal Phase A/test/confirmation false |
| 2026-07-18 | CCSF Step8 temperature-pilot launch | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | commit/resource/process provenance | `06d0ffc`；3 GPUs；15-run validation pilot running；formal test false |
| 2026-07-18 | CCSF Step8 failure audit + Step7A repair | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | numeric failure attribution + regression gate | first attempt 0/15；zero-RMS derivative repaired；3/3 + recheck15/15；retry smoke next |
| 2026-07-18 | CCSF Step8 repaired retry1 launch | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | repair smoke + commit/resource/process provenance | `7045c80`；3 GPUs；15-run validation retry running；formal test false |
| 2026-07-18 | CCSF retry1 result + formal freeze | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | validation selection + formal identity freeze | 15/15、9/9；tau0.25；Step7B formal prelaunch next；test false |
| 2026-07-18 | CCSF tau0.25 formal Phase-A prelaunch | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | 50-run matrix + test authorization + four-layer artifact gate | 15/15；Phase A/test true；confirmation false；remote launch next |
| 2026-07-18 | CCSF tau0.25 formal Phase-A launch | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | commit/resource/process provenance | `604e1b8`；3×3090；50-run matrix running；confirmation false |
| 2026-07-19 | CCSF formal Step9–10 + D2–D4 closure | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | four-layer result + granularity/risk/sharpness attribution | 50/50、200/200；CCSF negative；contrast-policy family closed；return Step2/4 |
| 2026-07-19 | D18 Step9–10 + post-D18 viability audit | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | problem closure + Step2 rollback + next control boundary | 25/25；2/7 gates；measure explains；soft route closed；D19 control audit next |
| 2026-07-19 | D19 Step4–6 source/theory/control freeze | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | official-code boundary + four-arm matched control design | Step6 9/9；Step7A local only；remote/test/paper method false |
| 2026-07-19 | D19 v1.1 repair + Step7A | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | same-history fairness repair + production implementation | v1 superseded；v1.1 114/114；Step7B prelaunch next；remote/test false |
| 2026-07-19 | D19 Step7B formal prelaunch | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | 15-run runner + 80-cell audit + internal IF diagnostics + four-layer analyzer | 31/31；seed2021 remote/test true；confirmation/paper method false |

## Continuation Rules

1. 每次继续研究先读本 ledger 与active protocol；remote不得静默改变frozen profiles、rank、init、controls或gates；
2. old analysis可引用，archive脚本不得直接启动；
3. diagnostic failure必须区分 hypothesis、intervention、readout、numeric与capacity control；
4. D2 formal5只在frozen A6 representation/head family下不支持depth grouping；当前PLGO不使用该设计，若未来
   重新提出end-to-end grouping method，必须作为新候选通过Step2-6；
5. official test是正式机制评估与paper-facing primary gate；不得选择checkpoint或用于逐dataset/horizon/cell调参。
   test后新candidate必须标记`test_informed`并重新冻结完整矩阵。
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
28. SIFF/MCCA旧Phase-A已完成55/55 new + 25/25 references；它使用best-H720 checkpoint且validation-only。
    当时停止exact pair是合理的development decision，但按当前治理不能写成fair-test rejection。SIFF已由后续公平
    test修正为partial pass；MCCA未进入70-run audit，状态为inactive/fair-test unaudited，不补seed或机械调Sinkhorn。
29. `ordered > permuted` 5/5只证明scope order含信息；由于ordered未超过Q1-wide/independent的macro gate，
    不得把该control单独升级为architecture contribution。
30. ETTm2 SIFF-vs-PCSD在H1为-669.49%而H720为+0.60%，属于>100% short-prefix
    `checkpoint_or_readout_pathology`；因此结果可关闭exact Q2 candidate，但trajectory audit前不得关闭方向。
31. `SC-D16` source audit确认ElasTST已覆盖harmonic reweighting/weighted checkpoint；code audit确认coupling
    arms已使用exact harmonic-L1 fused loss。HR rerun属于重复，只保留`SC-D16-CTD` per-epoch trajectory；
    Step5/6冻结前不得实现或remote。
32. validation与test均使用H96/H192/H336/H720 MSE/MAE，但职责不同：validation只选checkpoint、
    hyperparameter与做diagnostic；test统一决定正式机制effectiveness与paper main/ablation。dense H1..720默认
    只作diagnostic。后续candidate均为`test_informed`，但禁止逐dataset/horizon/cell反向调参。
33. 后续所有机制结论必须依次报告paper-facing effectiveness、matched mechanism attribution、internal mechanism
    health与failure attribution。performance是必要但不充分条件；internal diagnostic不能挽救negative effectiveness，
    正向test但归因失败只能标记`partial_pass_attribution_blocked`。
34. `SC1-SIFF-v2-EQ-ATTR-v1`已冻结10-arm EQUAL-context矩阵。Phase A七项hard comparisons全部通过前不得补
    seeds、修改controls或升级paper claim；Step7A construction失败必须回Step6，remote/test需新的prelaunch授权。
35. Step7A只证明50-job production path、gradient与component intervention artifact可执行。random-initialized
    component RMS/contrast不得写成trained mechanism health；Step7B必须先完成resource smoke与test metadata
    authorization，正式artifacts返回后才应用四层gate。
36. Step7B只授权`SC1-SIFF-v2-EQ-ATTR-v1`的seed2021完整50-run/200-cell Phase A。不得跳过resource smoke、
    筛选arm/dataset/horizon或补confirmation；seeds2022/2023只有四层Phase-A gate全部通过后才能解锁。
37. Step8已从commit `c4c4730`启动。运行中不得pull新代码、修改config/gates或追加arms；50/50完成后必须先同步
    完整metrics、invariants与component artifacts，再执行四层Step9，不能只查看macro MSE。
38. Step9已完成且exact v1未通过：不得补seeds2022/2023、调当前rank/scale/policy或从内部oracle选择有利arm。
39. Step9后的portfolio decision将exact v1保留为`frozen_performance_near_candidate`，但不推翻其attribution failure。
    v1 manifest、checkpoints与结果不可变；任何改进均建立新`test_informed` candidate，并先完成Step4-6。
    后续新候选必须回Step4，显式超过A6_MEASURE与independent-scope control，并解释如何把conditional headroom
    转成learned fused forecast；order/equal-skill局部信号只能作为design evidence。
40. CCSF Step5只通过target-free contrast identifiability与theory gate，不授权implementation。旧PCC已覆盖同步
    error-supervised routing，所以calibration loss不得单独claim创新；Step6必须分离loss-only、architecture-only与
    full interaction，并加入generic correction、shuffle/zero、independent field和A6_MEASURE controls。
41. CCSF Step6已冻结`SC1-SIFF-v2-CCSF-v1-preimplementation`的10-arm contract。Step7A只能实现local
    forward/objective/adapters与construction gates；不得启动15-run validation temperature pilot、remote或formal
    test。以后不得删减A6_MEASURE、loss-only、architecture-only、standardized-teacher、zero/permuted-contrast或
    matched-independent controls。joint claim要求10项hard comparisons和internal health共同通过；部分失败必须按
    architecture/objective/field/numeric decision map回滚，不能用oracle headroom补救。
42. CCSF Step7A已18/18通过，但`tau=0.1`只是local smoke值，不得视为正式选择。下一步Step7B必须先冻结并授权
    15-run five-dataset shared-temperature validation pilot、temperature选定后的formal candidate identity，以及正式
    runner/evaluator/internal-artifact completeness。当前remote template只允许dry-run manifest；非dry-run必须
    exit 3，且不得以Step7A local pass为由访问official test。
43. CCSF Step7B temperature pilot已通过14/14 prelaunch gate并只授权15-run validation-only remote training。
    selection必须等待60/60 cells完整后，以五dataset×四horizon macro validation MSE选择一个shared temperature；
    tie取更大值。不得per-dataset/test选择，不得复用pilot checkpoint。选择完成后必须生成formal candidate version并
    重新完成Phase-A prelaunch audit；当前formal Phase A、official test与confirmation仍未授权。
44. CCSF temperature pilot已从commit`06d0ffc`启动。运行中不得pull、改变temperature grid/profile/selection、追加
    arms或访问test；用户明确不要求值守。完成后必须先验证15/15 runs、60/60 cells和selection artifacts，pilot
    checkpoint不得复用。只有formal candidate identity与新prelaunch audit冻结后，才可讨论Phase A授权。
45. 首次CCSF pilot completion audit为0/15，不得称为completed result或选择temperature。三temperature一致NaN已归因
    于zero-contrast group RMS的0点反向，属于`optimization_or_numeric_pathology`；只否定旧implementation。
    epsilon repair通过3/3 local runtime gate，Step7B recheck为15/15。retry前必须通过三batch真实Weather smoke，且
    retry使用独立external root；协议、test=false与formal authorization不得改变。
46. repair commit`7045c80`的三batch真实Weather smoke已finite，retry1已启动。运行中不得pull、修改协议或访问test；
    用户明确不要求值守。retry完成后仍只允许选择shared temperature，不能从validation pilot宣称paper-core
    effectiveness；formal Phase A必须另行冻结candidate identity、完整10-arm matrix与四层gate。
47. retry1已15/15并选择shared tau0.25。该选择只固定ordinary training hyperparameter，validation margin不得写成
    mechanism gain。`SC1-SIFF-v2-CCSF-v1-tau25`的50-run Phase A必须全部from-scratch，不复用pilot checkpoints；
    下一步只授权formal prelaunch tooling。remote Phase A、official test与confirmation仍为false，直到新的完整
    runner/evaluator/internal-artifact gate通过并记录test metadata。
48. formal Phase-A prelaunch现为15/15，50-job/200-cell、10 hard comparisons、official-test authorization、
    checkpoint non-mutation、CCSF internal artifacts与four-layer analyzer均通过。Phase A/test现已授权；正式启动前
    仍必须commit/push、remote pull、`nvidia-smi`与Weather CCSF_RELCAL三batch smoke。confirmation seeds保持false。
49. commit`604e1b8`已完成remote pull、15/15复核与Weather CCSF_RELCAL三batch smoke；50-run formal Phase A于
    `2026-07-18T17:27:08+08:00`在GPU0/1/2启动。运行期间不pull或改协议，不高频值守；完成后先做50/50、
    200/200、test date/checkpoint hash与internal artifacts audit，再进入Step9，confirmation仍为false。
50. exact projectivity下，requested horizon不能改变shared prefix；任何新method不得同时claim exact crop
    invariance与requested-H adaptive shared-prefix prediction。显式或隐式按crop长度归一化future coordinate均视为
    horizon leakage。
51. D17-v0的same-test 256-row two-fold结果因flattened sample×channel rows跨fold而protocol invalid。其
    pointwise +21.27%、causal +3.42%等数值不得用于problem promotion、method设计或paper claim。
52. D17-v1只允许从既有checkpoints导出validation probes，在validation labels上fit固定diagnostic ridge，并在
    既有authorized test probes评估；不训练模型、不新增test access、不改checkpoint。即使全gate通过，也只把
    prefix-safe future-context标记为`problem_supported`并进入Step4，不能直接升级Contribution 1。
53. D17-v1正式结果只通过prefix invariance，pointwise与causal correction均发生严重validation→test reversal。
    exact post-hoc route关闭；因frozen carrier与>100%局部退化，future-context方向保持unresolved，不得启动E2E
    refiner作为结果抢救。
54. D18只测exact projectivity的accuracy cost。separate horizon-specific arms是oracle/problem controls，不是
    contribution；只有相对A6_MEASURE的own-H gates全部通过，才允许进入controlled soft-projectivity Step4。
55. D18 Step3固定25个artifact units，其中只新增15个specialist训练并复用10个A6 controls。所有arms保持
    T720 A6 architecture与matched initialization；只允许loss support和validation selector变化。11/11
    prelaunch通过只授权problem diagnostic，不授权soft-projective method或Contribution claim。
56. D18已从commit`c843178`启动。运行中不得pull、改gates、改checkpoint selector或按局部test cells调参；
    25/25后必须完整报告15个own-H cells、A6_MEASURE/A6_FULL comparisons、prediction NRMSE与所有protocol
    invariants，再决定Step4或Step2 rollback。
57. D18正式结果为specialists vs A6_MEASURE `+0.1659%`、7/15 cells、2/7 categories；A6_MEASURE vs
    A6_FULL为`+1.7980%`、15/15。不得用相对A6_FULL的正值重启soft projectivity、补seeds或做
    consistency-penalty sweep；failure scope固定为stable projectivity-cost hypothesis false。
58. `SC-D19-IFC`只能作为source-informed control。IF已有NeurIPS 2025 amplitude/phase/frequency-pool decoding
    prior；通过也只证明trajectory-level decoder headroom，不构成本项目创新。Step7A前必须完成official code、
    full-T crop projectivity、input-spectrum skip、matched MLP与function-class audit；当前remote/test=false。
59. D19 Step6冻结`A6_MEASURE/IF_MEASURE/IF_NOSKIP_MEASURE/DIRECT_NONLINEAR_MATCHED_MEASURE`
    四arm、five datasets、seed2021与four-horizon validation selector。IF与matched-direct参数差必须不超过
    0.1%；15个新训练run与5个复用A6组成20个artifact units。当前只授权Step7A local implementation；
    Step7A/7B重新过gate前不得启动remote、读取official test或把IF control升级为paper method。
60. Step7A发现v1把upstream 96-point lookback误作本地contract，而A6 natural实际`seq_len=720`。v1在任何
    training/test前被supersede；v1.1要求Encoder、IF skip与matched direct读取同一720-point history，并把
    history spectrum改为361 bins。该修复不是结果后调参。
61. v1.1 Step7A为114/114：15 CLI、60 projectivity、12 parameter、10 gradient及governance/init/numeric/source/model
    gates全部通过。local pass不证明accuracy；Step7B必须完成real-batch finite/resource smoke、runner、
    completeness/analyzer与正式authorization，当前remote/test/paper method仍false。
