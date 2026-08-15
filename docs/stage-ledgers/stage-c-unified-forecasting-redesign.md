# StageC Unified Varied-Horizon Forecasting Ledger

## Stage Scope

| Field | Content |
| --- | --- |
| `stage_id` | `StageC-UVHF` |
| `paper_role` | problem-first unified multi-horizon research；不再预设两项机制形式 |
| `active_question` | 用户显式重开PatchTST Decoder-Transfer修补：先做decoder-only HPO v2；若validation gate失败，再回Step 4--6设计iTransformer-style carrier；Figure 5暂后移 |
| `source_evidence` | historical/source-faithful `A6-LBF-r256` |
| `mechanism_control` | Core-Ablation five matched end-to-end variants；historical `ISCF-EQUAL`只作旧diagnostic |
| `active_candidates` | architecture family frozen；`ISCF-BSCA-v1`=exact ablation anchor；`ISCF-BSCA-MAIN-v1`=8-dataset tuned main candidate；Introduction v0.9、Section 2 v0.2、Section 3 v0.7、Section 4 v0.7与Sections 5--7 v0.2 structure temporarily frozen usable；Figure 4 visual design temporarily fixed |
| `future_validation_suite` | Main I dense/Main II v1=ETTh1, ETTh2, ETTm1, ETTm2, Weather, ECL, Solar；Exchange=companion/deferred extension；ablation=original five |
| `paper_facing_scorecard` | validation/test H96,H192,H336,H720 MSE/MAE；dense默认diagnostic |
| `restart_handoff` | `docs/stage-ledgers/stage-c-iscf-bsca-paper-writing-restart-handoff-20260731.md` |
| `experiment_handoff` | `docs/stage-ledgers/stage-c-iscf-bsca-paper-experiments-restart-handoff-20260731.md` |
| `paper_table_registry` | `docs/iscf-bsca-paper-table-registry.md`；machine contract=`configs/iscf_bsca_paper_table_registry.json` |
| `stage_exit` | 新problem先通过existence/narrative gate，再决定一项或两项可归因contributions |
| `stage_rollback` | exact BSCA-v1 negative且无pathology则回Step4；不否定fixed ISCF architecture |

## Decision Cursor

| Field | Content |
| --- | --- |
| `current_11_step` | PatchTST Decoder-Transfer HPO v2 Step 8 remote train/validation；formal test blocked |
| `current_candidate` | `ISCF-BSCA-DECODER-TRANSFER-PATCHTST-HPO-v2`；v1 PatchTST负结果保持有效且不回写 |
| `latest_decision` | `patchtst_decoder_hpo_v2_50_run_remote_train_validation_active_test_zero` |
| `writing_latest_decision` | `main_i_main_ii_author_corrected_20260815_complete_hash_frozen` |
| `next_required_action` | 完成50/50 PatchTST BSCA HPO training artifacts与50 unique hashes；仅按four-H validation mean MSE选profile；通过gate后请求新formal-test授权 |
| `method_training_authorized` | 2026-08-15用户授权继续PatchTST decoder调参并在失败时更换backbone；当前只启动50-run train/validation，formal test与table mutation=false |
| `rollback_point` | data mismatch->H0；HPO instability->H1/H2；frozen-budget test-tuned optimum non-SOTA->report/narrow claim or new candidate gate；no per-H/cell tuning |

## Main-Table Author Correction Record (2026-08-15)

Main I与Main II已建立新的author-corrected canonical freeze；旧hash-frozen版本只保留为historical snapshots。Main I修正scope=`ISCF + TimeAlign all 7x4, SimpleTM Solar, TVNet ETTh2`；Main II修正scope=`ISCF + TimeAlign all 7x4, SimpleTM Solar, PatchTST ETTh2`。未列出cells保持既有source role。

作者只提供三位小数结果，因此canonical CSV不伪造额外精度，Main II corrected rows也不沿用被替换checkpoint hashes。完整重算后Main I=`44/56 best + 9/56 second`，Main II=`50/56 best + 6/56 second`。Canonical audit=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_tables_author_corrected_20260815/result_and_freeze_audit.md`；freeze manifest=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/main_tables_author_corrected_20260815/freeze_manifest.json`。该writing-side freeze不授权或触发PatchTST HPO v2 formal test。

## 11-Step Record

| Field | Current Record |
| --- | --- |
| `current_step` | Efficiency Step 9--10 complete；Figure 5 Step 6 contract next |
| `problem` | one-model varied-horizon service相对four-model horizon-specific families的deployment cost实际如何变化 |
| `existence_evidence` | 35/35 service units、77/77 immutable checkpoint objects；all finite/CV gates pass；test access=0 |
| `idea` | 同一独占RTX3090、FP32/batch1 synthetic-input contract下比较完整deployed services |
| `theory_check` | one-model服务一次H720 forward+prefix views；fixed-H families顺序执行四个native models；current ISCF full-domain materialization如实测量 |
| `design` | 5 systems × 7 datasets；30 warmups；5×100 CUDA-event iterations；params/storage/training/latency/memory/CHPC |
| `narrative_gate` | 只有实际测量支持时才允许efficiency verb；storage与compute必须分开报告 |
| `effectiveness_gate` | trade-off：ISCF相对TimeAlign/QDF减少model count、params与storage，但training/latency不领先 |
| `artifacts` | result=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/efficiency_20260814/formal_results/result_and_table_audit.md`；table=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/efficiency_20260814/formal_results/table/table_iscf_bsca_efficiency.tex` |
| `decision` | `efficiency_complete_tradeoff_supported_no_uniform_compute_advantage`；不宣称faster training/inference或prefix-bounded speedup |

## PatchTST Decoder-Transfer HPO v2 Record (2026-08-15)

| Field | Current Record |
| --- | --- |
| `current_step` | Parent v2 Step 9 audit complete；v2.1 Step 7--8 matched training/formal gate |
| `problem` | v1 PatchTST `+ISCF-BSCA`比`+ISCF`更好但未超过Original Decoder；需判断decoder optimization/capacity profile是否不匹配 |
| `existence_evidence` | v2 50/50 training complete、validation macro gain=0.8126%且5/5 datasets改善；但仅40/50 unique hashes，parent artifact gate失败 |
| `idea` | encoder profile完全冻结；只搜索readout LR multiplier、readout-only weight decay及两个rank边界点 |
| `theory_check` | optimizer parameter group只作用`pcsd_readout.*`；forward、scope集合、policy、objective与four-H selector不变 |
| `design` | Parent v2保留50-run完整审计；v2.1复用5 selected BSCA并补训5 matched ISCF，10-hash manifest后formal-test 40 new cells，合并v1 controls为120 cells |
| `narrative_gate` | v2是test-informed rescue candidate；不得删除或改写v1负结果 |
| `effectiveness_gate` | v2 validation gate已pass；v2.1 PatchTST BSCA vs Original需macro MSE/MAE均正向且dataset MSE wins>=3/5 |
| `artifacts` | parent design=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/decoder_transfer_patchtst_hpo_v2_20260815/design_and_prelaunch_gate.md`；formal gate=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/decoder_transfer_patchtst_hpo_v2_20260815/training_result_and_v2p1_formal_gate.md`；v2.1 config=`configs/iscf_bsca_decoder_transfer_patchtst_v2p1_formal.json` |
| `decision` | `parent_v2_uniqueness_fail_v2p1_manifest_gated_formal_authorized` |

Remote launch：commit=`7480ffc4`，PID=`3782929`，GPU0/1/2，50 jobs，start=`2026-08-15T01:39:23+08:00`。27/27 remote prelaunch与3/3 resource smoke通过；首批三个Weather profiles均进入epoch1且finite，formal test=0。Launch record=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/decoder_transfer_patchtst_hpo_v2_20260815/remote_launch.md`。

Training closure：50/50 complete、validation gate pass；但两组profile pairs在五datasets上collapse，unique hashes=40/50，parent gate保持FAIL。v2.1只冻结五个互异selected BSCA checkpoints；用户已授权五个matched ISCF training及10-hash manifest后的单次formal access。Gate record=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/decoder_transfer_patchtst_hpo_v2_20260815/training_result_and_v2p1_formal_gate.md`。

v2.1 launch：commit=`9cf0e8e8`，3/3 resource smoke pass；GPU0/1/2于14:43启动5个matched ISCF runs（PID=`663673`）。Guarded pipeline PID=`665646`严格执行`training -> 10-hash/5-pair manifest -> formal test -> 120-cell build`，任何gate failure均在test前停止。Launch record=`analysis/iscf_bsca_paper_experiment_consolidation_20260731/decoder_transfer_patchtst_v2p1_20260815/remote_launch.md`。

## Exact Ablation Anchor Contract

下表仅约束`ISCF-BSCA-v1` exact ablation anchor，不是Main I/II的最终超参数。
`ISCF-BSCA-MAIN-v1`需按v2 H0--H4在8 datasets上独立完成test-tuned HPO。

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
| `INTRO-CHPD-v1` | `integrated_in_section3_v0_7_temporarily_frozen_usable` | independently trained horizon models能否给出清晰overlap inconsistency | selected ETTh2 trajectory + all-validation heatmap；aggregate-CHPD selection retained in caption | shared-96 raw mean differences 2.16--2.51；long--short NCHPD 0.0365--0.0406 | retain frozen wording；formal family/seed prevalence deferred |
| `SC-UVHF-FRSD-D1` | `integrated_in_section3_v0_7_temporarily_frozen_usable` | matched fixed sharing extent的future-region prediction-risk optimum是否随future region变化 | selected ETTm2 validation example；label-selected oracle role retained | five scopes win 2--3 regions；10/10 crossings；8.112% descriptive headroom | retain frozen wording；formal CFH remains architecture-level deferred control |
| `ISCF-BSCA-v1` | `core_ablation_complete_partial_attribution` | BSCA objective、allocation、projection与multi-scope是否各自贡献 | exact five-control narrative frozen；all-component claim contingent on four controls | 100/100 cells；3/4 pass；allocation vs equal fusion fail；Full macro 0.308549/0.346278 | retain three passing claims；allocation accuracy claim unsupported；no automatic rerun |
| `ISCF-BSCA-MAIN-v1` | `author_corrected_main_i_and_main_ii_hash_frozen` | frozen architecture在8 datasets上经test-tuned HPO后能否达到SOTA-competitive并支持main claims | validation选checkpoint；test four-H aggregate选dataset profile；ECL/Solar test-informed expanded budget；seed2021 first | Main I 44/56 best、53/56 top-2；Main II 50/56 best、56/56 top-2；macro MSE/MAE均0.260714/0.306107 | do not retrain；use exact ablation/transfer for attribution and portability |
| `ISCF-BSCA-PAPER-EXP-v2` | `main_tables_and_core_ablation_complete_transfer_efficiency_pending` | 完整main/ablation/transfer/efficiency evidence能否兑现provisional claims | Main I frozen；corrected Main II complete；Core-Ablation claim split by control | Core-Ablation 100/100且3/4 controls pass；transfer/efficiency missing | retain allocation negative boundary；freeze transfer/efficiency prelaunch separately |
| `SC-ISCF-UPA-D2` | `superseded_by_bsca_v1` | information-free uniform train-time anchor能否复现ARMERR/SHUFFLED gain | user chose outcome-first method route | not executed as separate diagnostic | retain design history only |
| `SC-ISCF-PSA-D1` | `control_complete_h2_supported` | contemporaneous no-route EQUAL能否解释new ARMERR/SHUFFLED公共gain | not a method；只隔离H2/H3 | exact EQUAL tie；controls +0.6577/+0.6557%；H2 pass | feeds UPA-D2 only |
| `SC-ISCF-PSA-D0` | `diagnostic_only_closed_h1_not_supported` | EQUAL frozen policy向uniform收缩是否存在stable held-out frontier | diagnostic only；generic shrinkage不是paper claim | L1/MSE -0.2431%/-0.1218%；1/5 datasets；2/15 runs | no alpha/temperature rescue；retain joint-training unresolved |
| `SC-ISCF-RSCC-v1` | `closed_control_attribution_fail` | 保留EQUAL reliability时，exact coalition policy credit能否兑现已有arm complementarity | pre-result conditional；post-result binding claim fail | vs EQUAL +0.5189% pass；vs ARMERR/SHUFFLED -0.1414%/-0.1394%；alignment fail | no formal test/rescue；rollback Step2/4 |
| `SC-ISCF-SCC-v0` | `closed_intervention_point_wrong` | fused-only + coalition KL能否改善scope coordination | narrative coherent；exact training intervention failed | vs EQUAL -3.1750%；headroom +18.08% -> -14.93% | no seed/lambda rescue；evidence feeds RSCC only |

Historical and control queue:

| ID | Status | Hypothesis | Narrative Gate | Effectiveness Gate | Next Action |
| --- | --- | --- | --- | --- | --- |
| `A6-LBF-natural-baseline` | `control_only` | validation-frozen natural profiles可作为稳定共同起点 | not required | 72/72 test；3 seeds；dense horizons | `frozen_test_reference_ready`；只作固定reference |
| `SC-ISCF-FRSC-v0` | `closed_validation_continuation_gate_fail` | full-rank scope-conditioned synthesis/gradient能否保留carrier capacity并诱导useful specialization | conditional narrative retained；exact method not promoted | vs identity -1.2745%；vs best global +0.0703%；vs random +0.1781%；same-alpha +0.7215%；no pathology | no formal test/rescue；rollback Step4；retain ISCF prior |
| `SC-ISCF-SPS-v0` | `closed_validation_hard_capacity_loss` | hard scope subspaces能否诱导specialization | narrative coherent but exact readout failed | vs identity -2.3123%；vs global +0.9041%；no pathology | no rescue；rollback Step4；feeds FRSC |
| `ISCF-v0` | `carrier_only_sac_temporal_scope_fail` | independent future-output coupling scopes是否超越near-matched shared-width与exact random grouping | exact narrative fail；generic independent branch claim prohibited | Q1 +0.8496% pass；RANDOM -0.1990% fail；vs A6 +1.3584% | no rescue；rollback Step2/4 portfolio consolidation |
| `ISCF-v1-CPSI` | `closed_material_effectiveness_fail` | common scope state应在native synthesis前非线性调制private deviation | design valid；exact mechanism falsified | vs ISCF -2.2128% MSE；vs A6 -0.7775%；health pass；LINEAR tie | no seeds/rescue；return Step4/5 |
| `SC-D22-HFA` | `completed_target_access_supported` | target-coordinate-specific access是否超越matched generic与shuffles | problem evidence pass；not method effectiveness | ordered vs generic +2.5228%；15/20；4/5；all five controls pass | handoff D23 Step4 |
| `SC-D23-FCMI` | `closed_capacity_control_explains` | generic main与coordinate interaction能否可识别分解并原生fallback | conditional pass | FCMI vs A6 -21.7343%；capacity/order fail；internal pass | no seeds/rescue；return Step2/3 |
| `SC-D24-CTB` | `diagnostic_only_closed_exact_negative` | strong fixed trajectory synthesis是否留下ordered-history可识别coarse deformation | not method gate | v1.1 ordered loses all primary controls；test=0 | no rescue；return Step2/4 consolidation |
| `SC-MNB` | `absorbed_into_paper_matrix_protocol_blocked` | modern native baselines在对应source-faithful role下提供何种accuracy context | ElasTST/TimePerceiver/SRSNet保留；CATS按当前paper claim预先排除 | minimal external surface=45 runs/60 cells；execution false | Tier A repair test hygiene/metric equivalence/config semantics |
| `SC1-SIFF-v3-TSAF-v1` | `closed_effectiveness_and_attribution_fail` | target-scale allocation能否修复SIFF learned fusion而不依赖sample-wise competence | prelaunch conditional；post-result complete claim fail | 45/45；vs A6_MEASURE -1.2854%；vs parent -1.0422%；four attribution questions fail；health pass | no confirmation/rescue；return Step2/4 |
| `SC-SIFF-POST-TSAF-2x2` | `diagnostic_only_completed_weak_lead_not_supported` | independent field与target-only policy是否存在stable positive interaction | not a method gate；latest prior pressure high | same-rank test interaction MSE/MAE `-0.3097%/-0.1175%`；full positive由rank-confounded subset主导 | no candidate/seed/rank rescue；SIFF-v2 claim consolidation |
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
| `SC1-SIFF-v2-FCC-v1` | `performance_pass_attribution_blocked_closed` | immutable SIFF能否以future-output coupling scale coordinate构成single architecture contribution | narrative conditional；post-result contribution fail | vs A6_FULL +1.2497%；vs independent -0.1272%；health pass | no promotion/rescue；paper portfolio decision |
| `SC1-SIFF-v2-CCSF-v1-tau25` | `failed_effectiveness_and_attribution_closed` | target-free scope contrast能否让policy识别relative competence；relative teacher只作co-designed弱监督 | prelaunch narrative conditional；post-result complete claim fail | 50/50、200/200；vs A6 -0.8567%；vs v1 -0.6159%；architecture/objective/ordered specificity fail | no confirmation；exact contrast-policy route closed；return Step2/4 |
| `SC1-CCSF-D2` | `diagnostic_only_closed` | region aggregation能否把contrast competence转成更强mixture utility | not a method gate | expected-arm signal 4 widths/5 datasets稳定；相对pointwise mixture margin不足，2/3 gates | region retained as analysis scale only；do not train |
| `SC1-CCSF-D3` | `diagnostic_only_closed` | residual covariance/cancellation是否解释best-arm teacher与fusion不一致 | not a method gate | simplex vs best-arm仅约+1.34%–1.38%；0 widths过dataset gate | covariance-aware redesign unsupported；return Step2 |
| `SC1-CCSF-D4` | `diagnostic_only_closed` | soft policy是否只需global sharpening或hard routing | not a method gate | best native arm仍-0.0186%、1/5；hard routing更差 | softness not primary；close contrast-policy route |
| `SC-D17-PFC-v1` | `diagnostic_only_conditional_negative_closed_exact_protocol` | frozen full-domain draft的ordered prefix context能否超越pointwise与row-shuffled controls并跨validation→test transfer | not a method gate；dual-carrier conditional evidence only | causal vs pointwise -3.0356%；vs shuffled -2.3616%；pointwise vs parent -28.7314%；1/7 gates | exact post-hoc protocol closed；direction unresolved；no E2E rescue |
| `SC-D18-SPC` | `diagnostic_only_closed_problem_false` | horizon-specific loss是否稳定超过A6_MEASURE，从而证明exact projectivity有accuracy cost | not a method gate | 25/25；vs A6_MEASURE +0.1659%、7/15；2/7 gates | no seeds；soft route closed；rollback Step2 |
| `SC-D19-IFC-control-v1.1` | `control_only_closed_negative_return_step2_4` | source-informed implicit trajectory decoder是否超过A6 learned-basis control | not a contribution；IF prior mandatory | IF vs A6 -3.6117%；vs direct -0.8075%；skip positive；health pass | no seeds/sweep；retain skip evidence；rollback Step2/4 |
| `SC-D20-CST` | `diagnostic_only_closed_failed_transfer_weak_specificity` | IF内的compact history-spectrum信息能否transfer到A6并超过同维random history projection | diagnostic only；generic concat与frequency primitive非method | SPEC-vs-A6 -0.7614%；vs random +0.1412%；health 11/11；val/test reversal | no seeds/sweep；rollback Step2/4；direction not rejected |
| `SC-D20-D1-CONTRIB` | `diagnostic_only_completed_coadaptation_explains` | D20失败是否只是contribution direction/scale错误 | posthoc test oracle；not method | SPEC +26.89% vs co-adapted base/39 of40；RANDOM +9.04%/35 of40；alpha medians >1 | scalar fix rejected；within-model importance non-incremental；return Step2/3 |
| `SC-D21-EVS` | `diagnostic_only_closed_split_unstable` | route validity是否为past × future-region non-separable surface | prelaunch pass；post-result paper necessity fail | oracle 7.64%/10.41%；neutral HGB vs additive +0.0347%；A6 -0.0069%；0/2 readouts pass | no seeds/rescue；joint rollback Step2 |
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
| D19 IF control Step8 launch | remote pull + dual resource smoke + 3-GPU background driver | commit`da011c8`；Weather IF/ETTm2 direct finite；first 3 jobs entered training | running；no monitoring/config changes；confirmation false | `analysis/stage_c_post_ccsf_step24_reset_20260719/d19_remote_launch_record.md` |
| D19 IF control Step9/10 | 20-unit/80-cell test scorecard + validation/checkpoint + polar health + capacity audit | IF vs A6 -3.6117%；vs direct -0.8075%；skip vs no-skip +1.6191%；health pass | exact control closed；no seeds/sweep；`readout_or_head_design_wrong`；rollback Step2/4 | `analysis/stage_c_post_ccsf_step24_reset_20260719/d19_step9/d19_step9_deep_audit.md` |
| Post-D19 Contribution 1 Step2/4 audit | external primary sources + algebraic function-class boundary + D19 evidence synthesis | smaller IF/history-phase atoms narrative fail；compact spectrum transfer/specificity未验证 | D20 diagnostic_only Step6 next；method/remote/test false | `analysis/stage_c_post_ccsf_step24_reset_20260719/post_d19_step24_compact_statistic_viability_audit.md` |
| D20 compact-statistic Step6 | exact q64 Fourier/random projections + paired initialization + matrix/gates + static checker | 14/14；orthogonality <=3.763e-15；init/prefix gap 0；gradient/deformation nonzero | `step6_pass_step7a_local_only`；remote/test/confirmation false | `analysis/stage_c_post_ccsf_step24_reset_20260719/d20_step6/step6_diagnostic_design.md` |
| D20 compact-statistic Step7A | production readout/buffers/CLI + paired init + shape/projectivity/gradient/hash audit | 9/9；15 CLI；60 shape/prefix；10 gradient；init/prefix gap 0 | production pass；Step7B prelaunch next | `analysis/stage_c_post_ccsf_step24_reset_20260719/d20_step7a/step7a_implementation_gate_report.md` |
| D20 compact-statistic Step7B prelaunch | 15-run runner + checkpoint-preserving evaluator + D20 probe + four-layer analyzer | 10/10；15 runs/60 test cells；all hashes/syntax/dry-run/smokes pass | seed2021 remote/test true；confirmation/paper method false；Step8 next | `analysis/stage_c_post_ccsf_step24_reset_20260719/d20_step7b/prelaunch_report.md` |
| D20 compact-statistic Step8 launch | remote pull + dual resource smoke + 3-GPU background driver | commit`9573cd7`；smokes finite；first three jobs entered epoch1 | running；no monitoring/config changes；confirmation false | `analysis/stage_c_post_ccsf_step24_reset_20260719/d20_remote_launch_record.md` |
| D20 compact-statistic Step9/10 | 15-run/60-cell scorecard + validation/checkpoint/dense/internal attribution | vs A6 -0.7614%；vs random +0.1412%；health 11/11；validation/test reversal | exact design closed；direction not rejected；rollback Step2/4 | `analysis/stage_c_post_ccsf_step24_reset_20260719/d20_step9/d20_step9_deep_audit.md` |
| Post-D20 Contribution 1 Step2/4 reassessment | external robustness/frequency refresh + result synthesis + candidate screening | scalar repair/generic robustness fail narrative；support family provisional | D20-D1 posthoc next；new training/method false | `analysis/stage_c_post_ccsf_step24_reset_20260719/post_d20_step24_reassessment.md` |
| D20-D1 contribution diagnostic design | saved-probe base/contribution recovery + actual/oracle scale by future bins | design/tool frozen；result not started | remote artifact read only；test oracle diagnostic；no checkpoint mutation/training | `analysis/stage_c_post_ccsf_step24_reset_20260719/d20_d1_contribution_diagnostic/design.md` |
| D20-D1 contribution diagnostic result | 90-row actual/oracle alpha + within-model base removal | SPEC +26.89%/39 of40；RANDOM +9.04%/35 of40；median alpha >1 | scalar rescue false；co-adaptation explains importance；return Step2/3 | `analysis/stage_c_post_ccsf_step24_reset_20260719/d20_d1_contribution_diagnostic/result.md` |
| D21-EVS Step2/3 + Step7A | external prior-art boundary + validation→test past × region interaction diagnostic | problem definition/narrative/design frozen；192-feature descriptor；synthetic policy recovery pass | only frozen D14 checkpoint evaluation authorized；new training/method false | `analysis/stage_c_d21_evidence_validity_surface_20260720/step23_problem_and_design_audit.md` |
| D21-EVS Step8/9/10 | 100 frozen-checkpoint exports + validation-fit/test interaction controls + posthoc forward attribution | oracle 7.64%/10.41%；neutral HGB vs additive +0.0347%；A6 -0.0069%；validation forward +0.3092%/+0.4406% | exact split-stable EVS closed；no seeds/rescue；rollback joint Step2 | `analysis/stage_c_d21_evidence_validity_surface_20260720/step9/deep_audit.md` |
| D22-A/B Bayes + frontier audit | primary-source refresh + D18 H1..720 dense curves + A6_MEASURE/A6_FULL + checkpoint/seed sensitivity | own-H +0.1659%、7/15；SPEC96 +1.2748%/5 of5但H192/H336负；0/15 arm-dataset Pareto；measure five bins均5/5正 | `finite_capacity_frontier_not_supported`；D22-C design-only；method/remote/test false | `analysis/stage_c_post_d21_unconstrained_reset_20260720/d22_ab_bayes_frontier_audit.md` |
| D22-C target-access problem gate | neutral/raw-history six-arm matched diagnostic | ordered vs generic +2.5228%；15/20 cells、4/5 datasets；Weather -1.0900% | target-coordinate access supported；not method effectiveness | `analysis/stage_c_post_d21_unconstrained_reset_20260720/d22c_result_and_step4_handoff.md` |
| D23-FCMI Step4-6 | source boundary + main/interaction decomposition + contained controls | generic/standard containment；matched dual/order controls | conditional narrative pass；Step7A local only | `analysis/stage_c_post_d21_unconstrained_reset_20260720/d23_fcmi_step46_design_audit.md` |
| D23-FCMI Step7A | production layer + 35 CLI + morph/gradient/parameter/order gates | 11/11；morph `6.33e-8`；dual params exact；A6 gap 83%–95% | local pass；dense control mandatory；remote/test false | `analysis/stage_c_post_d21_unconstrained_reset_20260720/d23_fcmi_step7a_implementation_audit.md` |
| D23-FCMI Step7B prelaunch | 40-run matrix + dense capacity control + evaluator/analyzer/runner refusal | 21/21；dense gap `0.0914%–0.1321%`；160 test + 160 val cells frozen | prelaunch pass；等待独立remote/test授权 | `analysis/stage_c_post_d21_unconstrained_reset_20260720/d23_step7b_prelaunch/prelaunch_report.md` |
| D23-FCMI Step8 launch | commit-pinned pull + 3-GPU preflight + dual resource smoke + background driver | commit `4ff439c`；smokes finite；first Weather jobs active | running；40/40后four-layer analyzer；confirmation false | `analysis/stage_c_post_d21_unconstrained_reset_20260720/d23_step8_remote/remote_launch_record.md` |
| D23-FCMI Step9/10 | 40 runs + 160 val/test cells + matched controls + internal health + conditional complementarity | FCMI vs A6 -21.7343%；DENSE vs STD +15.4825%；internal 5/5；FCMI/A6-DENSE allocation split-unstable | FCMI-v1 closed；capacity explains；direct successor blocked；return Step2/3 | `analysis/stage_c_post_d21_unconstrained_reset_20260720/d23_step8_remote/d23_step9_10_result_and_rollback.md` |
| D24-CTB Step2/3 prelaunch | A6/DENSE frozen validation inference + chronological transfer + ordered/marginal/sorted/shuffled controls | v1 10/10但ridge unnormalized design fault；v1.1 normalized grid frozen | diagnostic only；remote training/test false | `analysis/stage_c_post_d21_unconstrained_reset_20260720/d24_ctb_step23_design_audit.md` |
| D24-CTB v1.1 result | 10 frozen runs + 840 metrics + 720 comparisons + zero test access | ordered vs marginal -8.60%；vs sorted about -9%；vs shuffled about -14%；0/4 horizons | exact probe closed；broader direction unsupported/unresolved；Step2/4 consolidation | `analysis/stage_c_post_d21_unconstrained_reset_20260720/d24_ctb_result_and_rollback.md` |
| SIFF-v3 TSAF Step4-7A | primary-source/narrative audit + production allocation path | Step4-6 conditional pass；Step7A 26/26；history-free allocation、history-dependent arms | Step7B prelaunch only；remote/test false | `analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v2_reactivation_and_tsaf_step46_audit.md` |
| SIFF-v3 TSAF Step7B prelaunch | 9 effective arms；20 references + 25 new runs；capacity/init/gradient/runner/analyzer gates | 15/15 cases、10/10 categories；20/20 remote reference hashes；max capacity gap 0.3619%；3 GPUs idle | prelaunch pass；waiting independent remote/test authorization；no performance result | `analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v3_tsaf_step7b_prelaunch_report.md` |
| SIFF-v3 TSAF Step8 authorization | user authorization + evaluator matrix contract + split training/test runner | 15/15 authorization gate；25 training + one complete 180-cell test；confirmation false | authorized；remote preflight/resource smoke next | `analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v3_tsaf_step8_remote_authorization_and_launch.md` |
| SIFF-v3 TSAF Step8 launch | commit-pinned pull + dual resource smoke + three-worker launch | `6cef063`；smokes finite/no-OOM；first Weather jobs active；test 0/25 | training active；25/25 before formal test | `analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v3_tsaf_step8_remote_authorization_and_launch.md` |
| SIFF-v3 TSAF Step9/10 | 25 new E2E runs + 45 effective-run/180-cell four-layer audit | vs A6_MEASURE `-1.2854%`；vs parent `-1.0422%`；four attribution gates fail；health pass | exact TSAF-v1 closed；no confirmation/rescue；return Step2/4 | `analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v3_tsaf_step9_10_result_and_rollback.md` |
| SIFF post-TSAF 2x2 Step2 audit | four existing E2E arms + field/policy interaction + same-rank/split sensitivity + latest primary sources | same-rank test interaction MSE/MAE `-0.3097%/-0.1175%`；full positive rank-confounded | weak lead not supported for Step4；SIFF-v2 narrow claim consolidation next | `analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_post_tsaf_independent_factorial_audit.md` |
| SIFF-v2 final claim Step4-6 | tensor/theory/prior/evidence audit + single-contribution boundary + three-seed FCC design | narrative conditional pass；main and independent gaps remain blockers；30 new runs frozen | FCC waiting authorization；method unchanged；modern baselines remain blocked | `analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v2_final_paper_claim_and_confirmation_design.md` |
| SIFF-v2 FCC Step7B prelaunch | A6_FULL comparator freeze + 30-job runner + 15-reference audit + three-seed analyzer | 25/25 checks；30/30 jobs；15/15 references complete/unique/init-paired；A6_MEASURE absent | remote/test authorized；commit-pinned preflight/resource smoke next | `analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v2_fcc_v1_prelaunch/prelaunch_report.md` |
| ISCF-v1-CPSI Step7B prelaunch | five new arms + ISCF/A6_FULL references + test-first controls | 19/19；25 new/35 effective；10/10 hashes；remote scanner fallback repaired | seed2021 remote + 25/25后single test true；confirmation false | `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v1_cpsi_step7b_prelaunch_20260721/prelaunch_report.md` |
| ISCF-v1-CPSI Step8 launch | commit-pinned pull + 3-GPU preflight + dual resource smoke + supervisor | `5d2330e`；smokes finite/no-OOM；initial training/test 0/25 | training active；25/25前test hard-blocked | `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v1_cpsi_step8_remote_20260721/remote_launch_record.md` |
| ISCF-v1-CPSI Step9/10 | 25 new + 10 references；formal test + four-layer audit | vs ISCF -2.2128% MSE；LINEAR +0.0217% tie；health 25/25 | exact v1 closed；active method none；rollback Step4/5 | `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v1_cpsi_step9_10_20260721/step9_10_result_and_rollback.md` |
| ISCF-v0 post-CPSI Step4/5 | latest primary-source audit + output-coupling tensor/theory boundary + matched attribution design | carrier stable；generic multi-scale/multi-branch prior crowded；Q1-WIDE and RANDOM gaps identified | conditional paperization candidate；25-control-run SAC designed；remote/test false | `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_post_cpsi_step45_20260721/step4_5_scope_independence_narrative_gate.md` |
| ISCF-v0 SAC Step7B prelaunch | 25-job manifest + 35-reference hash audit + exact partition contract + runner/analyzer | 18/18；60 effective runs；Q1 gap max 0.464638%；unauthorized launch exit 3 | prelaunch pass；remote/test false；explicit authorization next | `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_sac_step7b_prelaunch_20260721/prelaunch_report.md` |
| ISCF-v0 SAC Step8 authorization | user-scoped training authorization + frozen launch sequence | 25 new trainings true；formal test false；candidate/gates unchanged | commit/pull/GPU/dual smoke next；25/25后stop | `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_sac_step8_remote_20260721/remote_authorization_and_launch.md` |
| ISCF-v0 SAC Step8 launch | commit-pinned pull + 3-GPU preflight + dual resource smoke + background runner | `78cbcf4`；smokes finite/no-OOM；PID2383292；first Weather jobs active | training active；formal test 0/25 and unauthorized | `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_sac_step8_remote_20260721/remote_authorization_and_launch.md` |
| ISCF-v0 SAC Step8 validation audit | 25 new checkpoints + 35 historical references；validation-only protocol/health audit | training 25/25；60/60 runs；240/240 rows；health 15/15；Q1 +1.0704%、RANDOM -0.1823% MSE | `formal_test_ready_pending_user_authorization`；not an effectiveness decision | `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_sac_step8_remote_20260721/validation_artifact_audit_and_test_handoff.md` |
| ISCF-v0 SAC Step9 authorization | independent user authorization for one frozen official-test matrix | 25 checkpoints fixed；retraining/mutation/tuning false；access count 1 | `step9_formal_test_authorized`；commit-pinned launch next | `configs/stage_c_iscf_v0_scope_attribution_confirmation.json` |
| ISCF-v0 SAC Step9 runtime repair | first three jobs stopped before test loader due missing diagnostic bins | test 0/25；checkpoint unchanged；not model/result failure | add 8-bin contract + runner assertion + val smoke；then unchanged relaunch | `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_sac_step8_remote_20260721/remote_authorization_and_launch.md` |
| ISCF-v0 SAC Step9/10 | 25 new tests + 35 historical refs；two primary matched attribution gates | 60/60；Q1 +0.8496% pass；RANDOM -0.1990% fail；health/nonmutation pass | temporal scope unsupported；ISCF carrier-only；rollback Step2/4 | `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_v0_sac_step9_10_20260721/step9_10_result_and_rollback.md` |
| ISCF-SPS Step4–7A | source audit + synthesis bottleneck audit + scope/global/identity/random implementation | identity gap 8.34e-7；projector errors <=3.22e-15；five gradients nonzero；production pass | conditional architecture pass；remote/test false；freeze Step7B validation design | `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_sps_step46_20260721/step4_6_design_and_step7a_audit.md` |
| SIFF-v2 FCC Step8 launch | commit-pinned pull + 3-GPU preflight + dual resource smoke + background driver | `87bea35`；smokes finite/no-OOM；first three Weather jobs active；test 0/30 | training active；30/30 before one formal test | `analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v2_fcc_v1_prelaunch/remote_launch_record.md` |
| SIFF-v2 FCC Step9/10 | 30 new E2E runs + one formal test + 45-run/180-cell three-seed audit | vs A6_FULL MSE/MAE +1.2497%/+0.7549%；vs independent -0.1272%/-0.1733%；health 6/6 | performance pass but attribution fail；stop promotion；portfolio decision | `analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v2_fcc_v1/step9_10_result_and_portfolio_decision.md` |
| ISCF-SCC Step9 / RSCC Step5–7 | 20-run SCC result + reliability failure attribution + exact hybrid implementation | SCC vs EQUAL -3.1750%；EQUAL headroom +18.08% vs SCC -14.93%；RSCC skill identity/regression pass | SCC closed；RSCC resource smoke next；test false | `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_frsc_step26_20260722/scc_step9_result_and_rscc_step5_6_design.md` |
| ISCF-RSCC Step8 launch | same-init Weather RSCC/SHUFFLED smoke + 15-run three-arm validation matrix | skill loss exact match；route/nonzero gradients finite；commit `020eea3`；first Weather jobs active | validation running；full-matrix Step9 only；formal test false | `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_frsc_step26_20260722/rscc_step8_remote_launch.md` |
| ISCF-RSCC Step9 | 20 effective runs + 80 validation cells + controls/internal health | vs EQUAL +0.5189%；vs ARMERR/SHUFFLED -0.1414%/-0.1394%；alignment 0.1539 < 0.2052 | control attribution fail；exact route closed；return Step2/4 | `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_frsc_step26_20260722/rscc_step9_result_and_rollback.md` |
| ISCF-BSCA-v1 confirmation Step7B | frozen two-seed runner/analyzer/checker + 10 reused EQUAL references | 10-job dry-run、reference 10/10、test guard与local checker pass | remote resource smoke next；objective/gates frozen | `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_bsca_v1_confirmation_prelaunch_20260722/confirmation_design_and_prelaunch.md` |
| ISCF-BSCA-v1 confirmation Step8 launch | commit-pinned pull + GPU audit + Weather smoke + three-worker launch | `72e3356`；GPU0/1/2；first Weather/ETTm1 jobs in epoch1；test 0/10 | training active；10/10 before one formal test | `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_bsca_v1_confirmation_prelaunch_20260722/remote_launch_record.md` |
| ISCF-BSCA-v1 confirmation Step9/10 | 10 new candidate + 10 reused EQUAL；three-seed 60-cell audit | MSE/MAE +0.3541/+0.3073%；3/3 seeds、4/5 datasets、4/4 horizons；all health/nonmutation pass | paper-core pass；paper consolidation next；new training/test false | `analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_bsca_v1_confirmation_step9_10_20260722/step9_10_three_seed_result_and_paper_handoff.md` |
| Paper experiment v2 single-seed correction | seed2021 primary matrix + optional non-selective seed extension + TimeAlign Exchange source adapter | 233 primary slots；15 reusable/218 new；488 cells；Exchange four-H script local/static only | scoped local patch complete；remote/test false；remaining Tier A required | `analysis/iscf_bsca_paper_experiment_consolidation_20260731/design_and_prelaunch_gate.md` |
| ISCF-BSCA post-HPO Main I published audit | 8 selected checkpoints + TimeAlign Table 6 transcription + competitiveness comparison | 53 HPO trials；32 ISCF cells；140/140 published rows；vs TimeAlign MSE +2.199%、MAE -0.066%；5 source Avg anomalies | aggregate-MSE competitive；full SOTA pending；baseline remote/test false | `analysis/iscf_bsca_paper_experiment_consolidation_20260731/post_hpo_main_i_published_audit_and_next_gate.md` |

## Pending Tasks

| Task | Status | Next Action |
| --- | --- | --- |
| ISCF-BSCA paper architecture | `core_ablation_evidence_synced` | draft Section 5.5 with 3/4-control boundary；Figure 5/transfer/efficiency still pending |
| ISCF-BSCA paper experiment protocol | `v2_single_seed_scoped_exchange_patch_complete` | request remaining 8-dataset/HPO/official-baseline local patches only；B1/B2/B3/C false |
| ISCF-BSCA-v1 three-seed confirmation | `completed_exact_ablation_anchor` | reuse Full evidence；retain Equal as historical BSCA diagnostic only，不将其冒充新冻结的prefix-only w/o BSCA control |
| ISCF-RSCC-v1 validation matrix | `completed_control_attribution_fail` | exact route closed；retain artifacts/control clue；return Step2/4 |
| Freeze natural carrier | `completed` | 不再调 profile |
| ISCF-v0 SAC formal test | `completed_attribution_fail` | no rerun/rescue；use complete negative result in portfolio decision |
| ISCF-SPS Step7A local implementation | `completed_pass` | freeze validation-first Step7B matrix；do not launch remote yet |
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
| D19 source-informed implicit decoder control | `completed_negative_rollback_step2_4` | exact control closed；retain skip evidence；no seeds/width/LR sweep |
| Post-D19 compact statistic viability audit | `completed_step2_4` | D20 diagnostic family保留；smaller IF/history-phase method routes关闭 |
| SC-D20-CST transfer/specificity diagnostic | `completed_negative_rollback_step2_4` | exact design closed；保留weak specificity与split-mismatch evidence |
| SC-D20-D1 contribution direction/scale | `completed_coadaptation_explains` | no D20-v2；evidence feeds Step2/3 only |
| SC-D21-EVS problem audit | `completed_closed_split_unstable` | no seeds/representation rescue；evidence retained；joint Step2 |
| SC-D22-HFA D22-A/B | `completed_frontier_not_supported` | pure-request Bayes boundary冻结；D22-C conditional design only |
| SC-D22-HFA D22-C | `completed_problem_supported` | full artifacts；ordered > all controls；Weather generic negative retained |
| SC-D23-FCMI | `completed_fail_return_step2_3` | 40/40；FCMI vs A6 -21.7343%；capacity/order fail；no rescue |
| SC-D24-CTB | `completed_exact_negative` | v1/v1.1 artifacts保留；no feature/bin/lambda/nonlinear rescue |
| SC1-SIFF-v3-TSAF Step7B | `completed_prelaunch_then_executed` | frozen 45-run/180-cell matrix executed once；no gate modification |
| SC1-SIFF-v3-TSAF Step8-10 | `completed_fail_rollback_step2_4` | 25/25 new training/test；45/45 effective audit；effectiveness/attribution fail；health pass；no confirmation |
| SC-SIFF-POST-TSAF-2x2 | `completed_weak_lead_not_supported` | existing artifacts only；do not promote independent control；next is SIFF-v2 claim consolidation |
| SC1-SIFF-v2-FCC-v1 | `completed_performance_pass_attribution_fail` | no SIFF rescue；do not launch modern baselines/formal ablations；paper portfolio decision |

## Paper Mainline Sync Log

| Date | Trigger | Paper Section | Change Type | Decision |
| --- | --- | --- | --- | --- |
| 2026-08-14 | 用户授权按冻结计划完成Core-Ablation与ablation table | Section 5.5、Table 4、architecture/governance sync | complete matched scorecard + claim-boundary update | 100/100 cells；3/4 controls pass；Target-Adaptive Allocation control fail；table/PDF/hash frozen；no automatic extra experiment |
| 2026-08-12 | Author确认Section 5--7总体结构并收紧ablation、allocation analysis与qualitative scope | Sections 5--7 v0.2、table registry、architecture/governance sync | author-fixed experiment-writing contract + evidence-boundary freeze | Core-Ablation仅五variants；Fixed Scope=$s=144$；realized allocation value删除；performance-selected qualitative并入Figure 5且必须披露selection；standalone Discussion确认；无implementation/training/test授权 |
| 2026-08-11 | Author要求基于已完成Sections 1--4设计后续章节但暂不填具体内容 | Sections 5--7 provisional architecture、table/figure routing、claim-evidence map | evidence-ladder design + Discussion proposal + Appendix routing | v0.1 pending author discussion；不替换既有六章共识、不填结果正文、不改变H5A或其他实验授权 |
| 2026-08-10 | Author要求替换2.3 opening并确认其余Related Work内容可暂时固定 | Section 2.3 sentence、draft status、architecture/governance sync | sentence-level academic polish + temporary manuscript freeze | `Beyond shallow output projections`替代`A smaller body of work`；Section 2 v0.2 temporarily frozen usable；Sections 1/3/4与experiment cursor不变 |
| 2026-08-10 | 用户逐节审阅Related Work v0.1并要求重构策略分类与比较逻辑 | Section 2 v0.2 draft、source audit、architecture/governance sync | recursive/direct/MIMO/DIRMO primary-source taxonomy + ElasTST official-paper boundary + output-side literature funnel | v0.2 author structure refinement complete pending review；Introduction/Sections 3--4与experiment cursor不变 |
| 2026-08-10 | 用户确认Section 4暂时敲定并要求设计、调研和起草Related Work | Section 4 freeze、Section 2 initial draft、architecture/governance sync | temporary manuscript freeze + primary-source literature synthesis | Section 4 v0.7 frozen usable；Section 2 v0.1 pending author review；四段结构与ElasTST/multi-scale/MoE boundaries已落地；experiment cursor不变 |
| 2026-08-10 | 用户指出4.5在Uniform-Prefix objective定义前解释`second objective`导致叙事跳跃 | Section 4.5、architecture/governance sync | paragraph-order correction + causal-transition refinement | v0.7 pending author review；首段保留；顺序固定为prefix objective→multi-scope gradient problem→scope-wise loss→balance regularizer；公式、implementation、experiments与claim boundaries不变 |
| 2026-08-10 | 用户要求明确BSCA的dual training roles、重写balance作用并删除4.6 | Section 4.1、4.5、architecture/governance sync | training-objective narrative polish + method-scope reduction | v0.6 pending author review；uniform-prefix处理varied-horizon；scope-wise/balance稳定multi-scope learning；complexity转入Section 5.4；implementation与experiment cursor不变 |
| 2026-08-08 | 用户要求精简4.3并以multi-scope joint optimization问题重构BSCA loss chain | Section 4.3、4.5、architecture/governance sync | academic compression + objective renaming + optimization narrative refinement | v0.5 pending author review；固定`Uniform-Prefix Forecasting Loss`、`Scope-Wise Forecasting Loss`与`Allocation-Balance Regularizer`；uniform KL只作early anti-collapse proxy，不保证equal usage、sufficient training或specialization；implementation与experiment cursor不变 |
| 2026-08-07 | 用户将Method主图visual design暂时固定并要求Section 4与主图统一 | Section 4 Method、Figure 4 caption、architecture/governance sync | main-figure terminology + computation-flow alignment | v0.2 pending author text review；两条ISCF inference paths与15个figure terms同步；BSCA继续train-only且不进入主图；stable vector source pending；Introduction/Section 3、implementation、experiment与claim boundary不变 |
| 2026-08-07 | 用户对Section开头与4.1--4.3提出逐项修改意见 | Section 4 Method、architecture/governance sync | author-guided narrative refinement + implementation-boundary audit | v0.3 pending continued review；decoder-side framing、patch-token Encoder interface、Future Coordinate rationale、per-scope information pool和region-local generation chain已更新；prefix-bounded execution未写成已实现latency evidence；Introduction/Section 3、implementation与experiment authorization不变 |
| 2026-08-08 | 用户要求完善4.1/4.3、重构4.4并联动审计后续内容，同时提供高亮稿 | Section 4 Method、highlighted review、architecture/governance sync | named-path + target-adaptive allocation narrative refinement | v0.4 pending author review；`Scope Forecasting Path`/`Target-Adaptive Allocation Path`固定；Figure 4 caption压缩；4.3 unified-field framing与4.4--4.6同步完成；full-field implementation、efficiency/effectiveness claim boundary不变 |
| 2026-08-05 | 用户要求推进核心Section 4初稿、主架构图、公式与完整章节 | Section 4 Method、Figure 4、architecture/governance sync | computation-flow draft + architecture schematic + claim audit | v0.1 pending author review；六段Method chain与exact objective/CHPC/complexity完成；Figure 4四格式initial bundle生成；Introduction/Section 3不变；new implementation/training/test false |
| 2026-08-04 | 用户确认Section 3 v0.7基本满意并要求暂时固定为论文可用版本 | Section 3 status、architecture/governance sync | temporary manuscript freeze | body/terminology/equations/Figures 2--3 integration/captions frozen；明确矛盾 + author approval才解冻；next=Section 4 pending direction；experiment cursor不变 |
| 2026-08-04 | 用户否决Section 3 v0.6的3.2句式与region-wise MSE命名，同时接受3.1修改 | Section 3.1--3.3、Figure 3a terminology、architecture/governance sync | selective rollback + risk definition + claim-boundary rewrite | v0.7 pending author review；3.1保留；3.2连接DLinear observation与formulation limitation；3.3定义future-region prediction risk；数据、数值、Introduction与experiment cursor不变 |
| 2026-08-04 | 用户复审Section 3 v0.5并询问3.1主语、3.2结论衔接及3.3 `risk`术语 | Section 3.1--3.3、Figure 3a label、architecture/governance sync | terminology precision + local flow refinement | v0.6 pending author review；3.1主语明确；3.2汇合trajectory/aggregate evidence；经验量统一为region-wise MSE；数据、数值、claim boundary、Introduction与experiment cursor不变 |
| 2026-08-04 | 用户逐小节复审Section 3 v0.4并要求重排3.1、重写Figures 2--3叙事、删除3.3/3.5 | Section 3.1--3.3、Figures 2--3 captions、architecture/governance sync | author structure refinement + evidence compression | v0.5 pending author review；CHPC先于model contrast；inconsistency与CHPD统计分离；naive-unified accuracy和Design Requirements移出Section 3；Figure 2 selection与Figure 3 oracle boundary压缩保留；Introduction/figures/experiment cursor不变 |
| 2026-07-31 | 用户指出Section 3公式驱动、缺少what/why与观点链，并质疑projection notation及accuracy disclaimer位置 | Section 3 overview与3.1--3.5 narrative flow | argument-first rewrite + notation simplification + reverse outline | v0.2 pending author review；删除$\Pi_{H_i}$并改为shared-target equality；future-step-indexed先定义；accuracy boundary移至3.3；neutral tensor derivation移出manuscript；evidence/authorization不变 |
| 2026-07-31 | 用户要求推进Section 3并严格审计problem/evidence/method/claim boundary | Section 3.1--3.5、Figures 2--3、architecture/governance sync | clean initial draft + evidence audit + caption integration | v0.1 pending author review；CHPC/CHPD/NCHPD与matched sharing statistics定义完成；naive unified penalty不成立为当前证据；Figure 3只作validation descriptive oracle；Introduction未改；writing update不扩张实验授权 |
| 2026-07-29 | 用户逐项回复Introduction blind review并冻结ElasTST与related-work叙事取舍 | Introduction P1--P6、problem-existence evidence、Figure 1 | author response + provisional rewrite + diagnostic/visualization plan | v0.2-round1；CHPC=basic property；varied-horizon still underdeveloped；unrelated structure routes omitted；P4 neutral evidence pending；new training/test false |
| 2026-07-28 | 用户要求整合Introduction初稿并以首次接触论文的top-journal reviewer视角独立评审 | Introduction P1--P6、novelty boundary、paper consolidation cursor | clean manuscript draft + blind review | draft v0.1 landed；review=`major_revision/weak_reject` 4/10；ElasTST与SRP/SRP++ overlap、field-vs-mixture、problem evidence、terminology与headline results为主要阻塞；new training/test false |
| 2026-07-28 | 用户暂时冻结scope-indexed forecast field框架并要求重写P5 | Core terminology、Introduction P5、Method outline、claim boundary | framework reframing + paragraph rewrite | v0.6 consensus；ISCF=`Independent Scope-Conditioned Forecasting`；single field + target-conditioned allocation + scope-axis contraction；P6 v0.5 superseded；new training/test false |
| 2026-07-28 | 用户要求基于前四段形成Introduction P5--P6 | Introduction P5--P6、primary-source boundary、contribution chain | method overview + contribution draft | v0.5 discussion draft；ISCF chain与BSCA gradient-allocation边界落地；不提前claim未完成main-table results；new training/test false |
| 2026-07-24 | 用户质疑predictive structure定义与sharing推导并确认修订 | Introduction P4、Problem/Motivation Evidence III、source boundary | terminology + mechanism bridge + baseline taxonomy | v0.4；problem=`future-region sharing-demand heterogeneity`；加入finite-capacity bias--variance bridge及DLinear/PatchTST/iTransformer/N-HiTS边界；new training/test false |
| 2026-07-24 | 用户要求区分问题与coupling-scope方法并文档化 | Introduction P4、Problem/Motivation Evidence III、core terminology | problem/evidence/method boundary | v0.3；problem=`future-region predictive-structure heterogeneity`；evidence=`region-dependent sharing-scale preference`；method=`future-step coupling scope`；new training/test false |
| 2026-07-24 | 用户修订future-step与horizon无关generation表述 | Introduction P1--P3、core terminology | formulation + terminology refinement | v0.2；horizon-agnostic step-indexed field；不使用max-T-crop宏观叙事或independent-horizon claim；new training/test false |
| 2026-07-24 | 用户要求全文结构文档级落地并应用术语修订 | Current Position、Introduction、全文结构与实验布局 | paper architecture + terminology boundary | v0.1落地；P1--P3首轮共识；P4--P6及后续章节provisional；new training/test false |
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
| 2026-07-19 | D19 Step8 remote launch | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | commit/resource/process provenance | `da011c8`；3×3090；15-run matrix running；no monitoring/confirmation |
| 2026-07-19 | D19 Step9/10 deep audit | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | four-layer result + validation/checkpoint/capacity attribution | 20/20；IF negative；skip positive；exact control closed；rollback Step2/4 |
| 2026-07-19 | post-D19 Contribution 1 Step2/4 audit | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | external novelty boundary + transfer/specificity problem freeze | smaller IF/phase atoms blocked；D20 diagnostic Step6 next；method/remote/test false |
| 2026-07-19 | D20 compact-statistic Step6 | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | exact tensor/projection/init/matrix/gate freeze + static audit | 14/14；Step7A local true；remote/test/confirmation/paper method false |
| 2026-07-19 | D20 Step7A/7B implementation/prelaunch | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | production coefficient path + 15-run runner/evaluator/analyzer freeze | Step7A 9/9；Step7B 10/10；seed2021 remote/test true；confirmation false |
| 2026-07-20 | D20 Step8 remote launch | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | commit/resource/process provenance | `9573cd7`；3×3090；15-run matrix running；no monitoring/confirmation |
| 2026-07-20 | D20 Step9/10 + Step2/4 reassessment | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | four-layer result + failure correction + external boundary | transfer fail；weak specificity；val/test mismatch；exact D20 closed；D20-D1 next |
| 2026-07-20 | D20-D1 contribution oracle | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | within-model direction/scale attribution | scale not culprit；co-adaptation responsibility relocation；Step2/3 next |
| 2026-07-20 | D21-EVS problem/narrative gate + Step7A | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | replace vague support with past × future-region interaction | external boundaries frozen；192-feature/synthetic checks pass；Step7B next |
| 2026-07-20 | D21-EVS Step8/9/10 | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | full test gate + oracle accounting + split-stability failure attribution | 100/100；0/2 readouts pass；additive explains；exact EVS closed；joint Step2 |
| 2026-07-20 | D22-A/B Bayes + frontier audit | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | theorem/source refresh + existing-artifact dense frontier decision | frontier not supported；H96 local clue retained；D22-C design-only，implementation/remote/test false |
| 2026-07-20 | D22-C static/prelaunch | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | source/code/theory + identical-parameter six-arm contract + synthetic execution | prelaunch pass；seed2021 remote/test true after commit/push and GPU preflight；paper method false |
| 2026-07-20 | D22-C v1 early numeric audit | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | training-loss scale pathology before any complete/test artifact | v1 invalid；stop；v1.1 standardized-scale loss only；fresh rerun |
| 2026-07-20 | D22-C v1.1 Step9/10 | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | complete matched problem gate + heterogeneity/internal audit | target-access supported；generic +2.5228%；Weather negative；return Step4 |
| 2026-07-20 | D23-FCMI Step4-6 | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | prior boundary + main-interaction decomposition + containment/controls | conditional narrative pass；Step7A local only |
| 2026-07-20 | D23-FCMI Step7A | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | production tensor path + morph/gradient/parameter/CLI audit | 11/11 pass；Step7B design freeze next；remote/test false |
| 2026-07-20 | D23-FCMI Step7B prelaunch | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | dense capacity attribution + formal matrix/tooling/authorization boundary | 21/21 pass；等待独立remote/test授权；no training result |
| 2026-07-20 | D23-FCMI Step8 authorization | Current Position、Candidate Queue、11-Step Record | frozen seed2021 40-run/160-cell remote/test scope | user authorized；preflight/smoke next；confirmation/paper method false |
| 2026-07-20 | D23-FCMI Step8 launch | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | commit/resource/smoke/process provenance | `4ff439c`；3×3090；40-run matrix running；confirmation false |
| 2026-07-20 | D23-FCMI Step9/10 | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | full effectiveness + capacity/order attribution + Step4 source rollback | exact v1 closed；dense/function class explains；direct successor blocked；return Step2/3 |
| 2026-07-20 | D24-CTB Step2/3 prelaunch | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | phase closure + raw-history conditional coarse-deformation protocol | local synthetic pass；frozen validation inference only；training/test false |
| 2026-07-20 | D24-CTB v1/v1.1 result | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | v1 design fault correction + v1.1 full validation attribution | exact negative；test=0；no rescue；return Step2/4 consolidation |
| 2026-07-21 | SIFF-v3 TSAF Step4-7A | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | SIFF-first reset + single-contribution allocation design + production path | Step4-6 conditional pass；Step7A 26/26；remote/test false |
| 2026-07-21 | SIFF-v3 TSAF Step7B prelaunch | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | 45-run effective matrix + reference reuse + capacity/init/gradient/tooling freeze | 15/15、10/10；remote/test/confirmation false；waiting authorization |
| 2026-07-21 | SIFF-v3 TSAF Step8 authorization | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | explicit seed2021 remote/test authorization + evaluator/runner contract | 25 training + one complete test true；confirmation false；preflight next |
| 2026-07-21 | SIFF-v3 TSAF Step8 launch | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | commit/resource/smoke/process provenance | `6cef063`；GPU0/1/2；25-run training active；test 0/25 |
| 2026-07-21 | SIFF-v3 TSAF Step9/10 | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | complete official-test effectiveness + four-layer attribution | TSAF-v1 closed；vs A6/parent negative；all attribution gates fail；return Step2/4 |
| 2026-07-21 | SIFF post-TSAF 2x2 audit | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | existing-artifact field × policy interaction + rank/split/prior audit | weak lead not supported；no successor/seed rescue；SIFF claim consolidation |
| 2026-07-21 | SIFF-v2 final paper-claim Step4-6 | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | single contribution/prohibited claims + immutable three-seed FCC design | conditional narrative pass；30-run FCC frozen；remote/test false |
| 2026-07-21 | SIFF-v2 FCC A6_FULL Step7B prelaunch | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | user-selected comparator + exact matrix/gates/tooling/authorization freeze | 25/25 pass；30 new + 15 historical；remote/test authorized；preflight next |
| 2026-07-21 | SIFF-v2 FCC Step8 launch | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | commit/resource/smoke/process provenance | `87bea35`；GPU0/1/2；30-run training active；test 0/30 |
| 2026-07-21 | SIFF-v2 FCC Step9/10 | Current Position、Candidate Queue、11-Step Record、Experiment Ledger | complete package effectiveness + matched attribution + internal health + failure attribution | package pass；ordered attribution fail；stop promotion；portfolio decision |

## Continuation Rules

1. 每次继续研究先读本 ledger 与active protocol；remote不得静默改变frozen profiles、rank、init、controls或gates；
2. old analysis可引用，archive脚本不得直接启动；
3. diagnostic failure必须区分 hypothesis、intervention、readout、numeric与capacity control；
4. D2 formal5只在frozen A6 representation/head family下不支持depth grouping；当前PLGO不使用该设计，若未来
   重新提出end-to-end grouping method，必须作为新候选通过Step2-6；
5. official test是paper-facing hyperparameter selection、正式机制评估与effectiveness primary surface；不得选择
   checkpoint、epoch或seed。允许按dataset的four-H aggregate选择一个shared profile，禁止逐horizon/metric/cell
   调参或选择性报告；结果必须标记`test_tuned/test_informed`。
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
32. validation与test均使用H96/H192/H336/H720 MSE/MAE，但职责不同：validation只选每trial checkpoint并做
    diagnostic；test按每dataset的four-H mean MSE选择main hyperparameter profile，并统一决定正式机制
    effectiveness与paper main/ablation。dense H1..720默认只作diagnostic。后续candidate均为
    `test_tuned/test_informed`；禁止逐horizon/seed/metric/cell选择。
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
62. D19的skip-vs-no-skip正值只证明IF路径内direct history spectrum有用，不能证明其transfer到A6或具有
    frequency specificity。`SC-D20-CST`必须以same-run from-scratch A6、compact spectrum与同维fixed random
    orthogonal projection三臂同时验证transfer/specificity；positive只允许回Step4，generic concat不得升级method。
63. smaller IF与history-phase-continued atoms已被FITS/FBM/IF/PhaseFormer等prior-art和function-class boundary
    挡在narrative gate外。D20 Step6前implementation/remote/test=false；不得以width/LR sweep或删除random control
    抢救frequency claim。
64. D20 Step6固定$q=64$：32组non-DC real Fourier cos/sin modes对比same-dimensional seed20260719
    Gaussian-QR orthogonal projection。二者必须读取同一normalized history、共享head shape与trainable weights，
    且不得做dataset-specific bins、energy ranking或dimension sweep。
65. D20三臂必须from-scratch paired E2E；summary columns zero-init只表示initial function parity，不是preserved
    learned capacity。SPEC必须同时超过A6与RANDOM；positive只建立problem evidence，不能升级generic concat为method。
66. Step6 static 14/14只授权Step7A local implementation。Step7A需验证real-batch shape、normalization source、
    projection buffers、paired hashes、zero-init equality、summary gradients、active prediction deformation与prefix
    crop；通过前remote/test/confirmation继续false。
67. D20 formal result为SPEC-vs-A6 `-0.7614%`、SPEC-vs-RANDOM `+0.1412%`；11项health全过但validation→test
    reversal。exact q64 additive design关闭，不补seeds/width/LR/gate sweep；不得写成history-spectrum方向已否定。
68. D20-D1使用official-test labels计算optimal contribution scale，只是posthoc oracle。它不得选择新超参数、授权
    D20-v2或作为paper effectiveness；within-model base removal也不得冒充fair architecture comparison。
69. future-distance predictive-support只是provisional problem family。必须先证明past-identifiable、split-stable且超过
    generic coordinate/capacity controls，才允许进入Step4 method；Contribution 2不得提前实现。
70. D20-D1中SPEC/RANDOM相对各自co-adapted base均高度有益，但完整models仍差于A6；这证明path importance不等于
    incremental information。不得用within-model ablation、oracle alpha或component norm为D20/SCTO promotion背书。
71. D21-EVS official-test problem gate为0/2 readouts pass。不得把large oracle、region/permutation positives或把
    0.1% threshold事后降为0来升级EVS；neutral/A6 interaction相对additive的test gain分别为+0.0347%/-0.0069%。
72. D21 validation-forward nonlinear interaction存在但到test缩小/反转，说明局部learnability不等于split-stable
    paper necessity。D21不补seed、不做encoder-representation/readout rescue；两个contribution slots共同回Step2。
73. D22-A/B以D18 H1..720 full-test curves复核后，`SPEC96` own-H为`+1.2748%`、5/5 datasets，但
    `SPEC192/SPEC336`均为负，三个specialists在standard horizons上均0/5 dataset Pareto-dominate A6_MEASURE。
    最终状态为`finite_capacity_frontier_not_supported`；不得用H96局部信号恢复soft projectivity、H feature或seed
    rescue。
74. A6_MEASURE相对A6_FULL在H1–48、49–96、97–192、193–336、337–720五个coordinate bins均5/5
    datasets正向。measure是protocol/control，不是Contribution 2；其稳定收益也不能被重新包装成horizon router。
75. D22-C当前只冻结neutral/raw-history diagnostic design。ordered patch memory不是paper mainline；A6 sensitivity
    必须是symmetric frozen conditional evidence。完成独立static/prelaunch gate前，implementation、remote training、
    official test与paper method均false；Contribution 2继续open。
76. D22-C static/prelaunch已完成：六臂共享完全相同module、parameters、seed initialization、optimizer与selector；
    local synthetic execution和machine aggregator通过。仅冻结diagnostic的remote/test获授权，paper method、A6
    sensitivity与Contribution 2仍false。
77. 用户2026-07-20决定暂不pivot出deterministic-MSE fixed-past task。D22-C有效失败只关闭exact v1并回joint
    Step2/3；不得做seed/width/readout/representation rescue，也不得恢复D17-D21。
78. D22-C v1在training-only阶段因RevIN-normalized loss对near-zero variance rows产生$10^3$量级隐式放大而终止；
    没有complete/test result。v1.1只改dataset-standardized loss scale，必须新目录、新checkpoints完整重跑。
79. D22-C v1.1相对generic为test MSE `+2.5228%`、MAE `+1.6484%`，15/20 cells、4/5 datasets、4/4
    horizons；其余四controls均20/20正向。decision只通过problem gate，不是paper method effectiveness。
80. Weather相对generic 4/4 horizons负向。FCMI必须包含generic fallback；不得删Weather或claim universal
    query superiority。
81. CATS/TimePerceiver/MQTransformer/TQNet已覆盖query-to-history primitive。FCMI的provisional claim只在
    `problem evidence -> main/interaction decomposition -> generic/standard containment -> matched attribution`
    完整chain；Step7A前remote/test false。
82. FCMI Step7A为11/11 pass；zero-mean与standard exact morph数值通过，dual controls参数严格相等，四条关键
    gradients均finite/nonzero。FCMI相对A6 active params少83%–95%，Step7B必须增加dense capacity-matched
    control并冻结formal gates；remote/test继续false。
83. FCMI Step7B为21/21 pass。`DENSE_DUAL_MATCHED`以profile-specific rank `234/250/234/241/247`
    匹配A6 active parameters，五个profiles gap为`0.0914%–0.1321%`；zero-init initial function parity、
    coefficient/basis gradient与active residual均通过。该arm只作capacity control，不是method component或
    Contribution 2。
84. formal matrix固定8 arms × 5 datasets × seed2021 = 40 runs。全部8 arms进入160个official-test cells
    和160个validation cells。`TARGET_SHUFFLED_QUERY`原拟validation-only，但因参与方向级attribution，在任何
    test access前修正为formal control。validation只选checkpoint/审计健康，official test才判effectiveness与
    matched attribution。
85. `step7b_prelaunch_pass_waiting_remote_test_authorization`不等于Step8授权。当前config中remote、official
    test与paper-method flags均为false，runner非dry-run固定exit 3；只有用户独立授权后才可commit-pinned
    remote pull、`nvidia-smi`、resource smoke与formal matrix。
86. 用户2026-07-20以“按计划继续推进工作”独立授权`SC-D23-FCMI-v1`的seed2021 40-run/160-cell
    remote/test matrix。该授权不覆盖confirmation seeds、paper-method promotion、H/router/第二loss或任何
    profile/arm/gate修改；Step8必须先通过commit-pinned remote pull、GPU preflight与两项resource smoke。
87. Step8已从commit `4ff439c`启动。运行中不得remote pull、修改config/profile/arms/gates或查看局部
    official-test结果作选择；40/40完整后只能由冻结analyzer生成一次全矩阵Step9/10 decision。
88. D23 40/40与160/160 test cells完整。FCMI vs A6为`-21.7343%`、0/20；DENSE vs
    STANDARD_DUAL为`+15.4825%`、19/20，DENSE vs A6仅`-0.3284%`。internal health 5/5，
    故primary attribution为`effective capacity/function-class explains`，不是numeric pathology。
89. decomposition、generic与target-shuffle controls通过，只能保留为weak-family evidence；order从validation
    `+1.7757%`反转test `-0.4536%`。三项validation-fit frozen complementarity diagnostics也全部test-negative，
    且有cross-model co-adaptation confound，不得升级method或方向拒绝。
90. `SC-D23-FCMI-v1`关闭，不补seed/width/readout/rank/objective rescue。D22-C target-access evidence保留；
    direct dense+FCMI successor因conditional necessity不稳定且与IF/BasisFormer/S2TX primitives重叠而未过
    Step4 narrative gate。A6/DENSE validation-fit blend也发生test反转，固定等权仅有test-only正信号，
    不授权allocation/router。当前回deterministic-MSE fixed-past Step2/3，
    new implementation/remote/test=false。
91. D24 phase/time-warp probe的一阶derivative specificity仅约`+0.03%`，且被curvature/shift controls与直接
    prior削弱；不得设计phase router。`SC-D24-CTB`只允许读取D23 A6/DENSE冻结checkpoint的validation split，
    按chronological thirds检验ordered raw-history coarse deformation；remote training与official test均false。
92. D24-v1 10/10虽protocol/finite/checkpoint pass，但ridge使用$X^\top X+\lambda I$且未按数万fit rows归一化，
    severe extrapolation属于`design_fault_suspected`，不得作problem rejection。v1.1只改为
    $X^\top X+n\lambda I$与normalized grid；data/features/splits/controls/gates不变。
93. D24-v1.1 10/10、840 metrics、720 comparisons完整且test access为0。ordered history相对marginal、
    sorted与target-shuffled在A6/DENSE上全部macro negative，所有primary horizons均0/4；exact coarse
    deformation hypothesis关闭。
94. D24 negative不得方向级拒绝所有nonlinear native synthesis，但也不授权feature/bin/lambda/nonlinear/seed
    rescue。当前没有active method，回Step2/4审计`Bayes boundary -> frontier -> target access -> capacity
    attribution`完整paper chain与modern baseline gap；narrative gate前不得启动D25。
95. Post-D24 consolidation确认该链条scientifically coherent，但只形成problem/design-control principle，没有
    paper-facing positive method。method-paper narrative仍不完整，不得把negative results或A6/MEASURE拆成
    contributions。
96. modern native-baseline gap是下一blocking gate。P0固定ElasTST、CATS、TimePerceiver、SRSNet与
    A6_FULL/A6_MEASURE；single-weight varied-H、per-H fixed model与foundation/pretrained必须分表。
    `SC-MNB`当前仅Step1-3 protocol design，implementation、remote training与official test均false。
97. SC-MNB official commits与65-run/80-cell roles已冻结。ElasTST是唯一single-weight P0；其余三者均per-H。
    CATS/TimePerceiver per-epoch test access、CATS ETTm2-H96 typo、SRSNet file-level license trace/
    metric equivalence与ElasTST
    `limit_train_batches=10`均为launch blockers；source fault不得解释为model failure。
98. 2026-07-21用户明确恢复SIFF-first paperization。`SC1-SIFF-v2-EQ-ATTR-v1`为immutable parent，历史
    effectiveness/attribution failure不变；SC-MNB降为supporting inventory，baseline execution false。
99. provisional `SC1-SIFF-v3-TSAF-v1`删除unsupported history-conditioned allocation freedom，以future-coordinate ×
    ordered-log-scale field融合history-conditioned SIFF arms。equal-skill仍是单一training contract，不设第二loss。
100. TSAF Step4-6 narrative/design conditional pass；Step7A production-local 26/26通过。allocation不读取history/
     requested H，arms仍读取history；参数少于direct parent policy，target/scale/control gradients均通过。
101. TSAF Step7B现15/15 cases、10/10 categories通过；formal matrix为45 effective runs/180 cells，其中20个
     historical end-to-end references经remote SHA256 20/20复核后复用，25个new runs必须joint from-scratch。
102. historical direct-policy independent不得替代target-only independent。新matched ranks为ETTh1/ETTh2/ETTm1/
     ETTm2/Weather=`109/115/115/106/115`，TSAF-active-parameter gap最大0.3619%。
103. 当前runner只生成validation artifacts且normal launch exit 3；remote resource smoke、training、official test与
     confirmation均false。GPU idle与synthetic analyzer不是performance evidence。
104. 下一步必须先取得独立remote/test authorization，再commit-pinned pull与两arm resource smoke。CCSF、D17-D21、
     region/covariance/temperature、seed/width/rank/readout sweep均不得恢复。
105. 2026-07-21用户已授权冻结seed2021的25-run training与一次完整formal test。test只能在25/25 training完整后
     执行，checkpoint nonmutation为硬门；confirmation/paper promotion仍false。
106. generic evaluator已支持`effective_arms`并核对45-run matrix；TSAF config补齐coupling scales、equal-skill
     training contract与future bins。先commit/pull/GPU/resource smoke，smoke通过前不得launch。
107. commit `6cef063`已pull，两项resource smoke finite/no-OOM；25-run training于10:17:06在GPU0/1/2启动。
     训练期间不pull、不改config/gates；training 25/25前formal test保持0/25。
108. TSAF Phase A已25/25 new training、25/25 new formal test与45/45 effective audit完整；formal test commit为
     `4cc96f21e23c159e37757c66ec2e5c68358c5718`，45个checkpoint hashes unique且逐dataset encoder init一致。
109. TSAF相对A6_MEASURE test MSE/MAE为`-1.2854%/-1.3146%`，相对SIFF-v2 parent为
     `-1.0422%/-0.9183%`；两项均0/4 horizon wins，paper-facing effectiveness fail。
110. ordered-field、ordered-scale、target-coordinate与shared-field comparisons分别为MSE
     `-1.0191%/-0.0796%/-0.0405%/-1.2785%`；matched attribution全fail。internal health全过只说明路径活跃，
     不得覆盖negative effectiveness。
111. validation中TSAF相对parent曾为`+0.7700%`，official test反转为`-1.0422%`。不得换selector或按test选epoch。
     independent target-only相对parent的`+0.2383%`仅为single-seed post-result weak lead，低于0.3% primary threshold；
     它仍是control，不能改名或直接补confirmation。
112. Decision=`close_tsaf_v1_shared_field_design_keep_siff_v2_immutable_parent`。关闭TSAF-v1，不做
     seed/rank/width/readout/loss rescue；回SIFF-first Step2/4，当前无active successor method，SC-MNB execution
     与Contribution 2仍false。
113. post-TSAF 2x2审计确认`independent target-only`不是单因素arm：它同时改变Q2 ordered到Q5 independent field、
     direct到static-target policy，并在ETTh2/ETTm1/Weather改变rank 116到115。
114. 全20-cell test interaction MSE/MAE虽为`+0.5265%/+0.4246%`，严格same-rank ETTh1+ETTm2子集却为
     `-0.3097%/-0.1175%`；validation same-rank近零，Weather出现split reversal。weak lead不支持Step4新候选。
115. Decision=`independent_target_only_weak_lead_not_supported_for_step4`。不补seed/rank/router rescue；下一节点只做
     immutable SIFF-v2 final paper-claim consolidation，claim gate前不执行modern baselines或remote training。
116. SIFF-v2 final claim gate现`conditional_pass_as_single_architecture_contribution`：唯一method claim是ordered
     coupling-scale coordinate上的shared history-conditioned full-domain operator field；不claim首次multi-scale/MoE/query，
     也不把equal-skill或evaluation包装为第二method contribution。
117. 原FCC设计曾冻结`A6_MEASURE` comparator；用户于2026-07-21明确将其替换为`A6_FULL`。该旧设计只作
     chronological history，不得用于当前launch。
118. 当前FCC冻结`SIFF_EQUAL/A6_FULL/SIFF_INDEPENDENT_EQUAL × 5 datasets × seeds2022/2023`共30 new runs，
     复用seed2021形成45 effective runs/180 test cells。A6_MEASURE不进入FCC metrics、matrix或machine gate，
     但历史negative evidence继续保留。
119. `SIFF vs A6_FULL`是architecture与objective共同变化的method-package comparison；ordered-field attribution
     只由same-objective、capacity-matched independent control承担。两项均沿用`+0.3%`与dataset/horizon/seed gates。
120. Step7B prelaunch为25/25 pass：30/30 jobs与15/15 historical references完整，checkpoint unique且逐dataset
     initialization paired，runner/analyzer smokes通过。
121. Decision=`step7b_prelaunch_pass_proceed_commit_remote_preflight`。remote training已授权；single formal test只在
     30/30 training后执行。FCC失败则停止SIFF paper-core rescue，通过后才进入modern native baselines与formal ablations。
122. Step8已从commit `87bea35`完成remote fast-forward、三张3090 preflight及两项resource smoke；smokes finite且
     无OOM。30-run training于`2026-07-21T12:54:37+08:00`启动，首批三个Weather jobs active，formal test 0/30。
123. 当前Decision=`step8_training_active_formal_test_not_started`。training期间不得改matrix/gates；30/30完整后才
     允许一次formal test，随后必须联合three seeds执行45-run/180-cell analyzer。
124. FCC已30/30 training、30/30 new formal test、45/45 effective runs与180/180 cells完整。45 checkpoint hashes
     unique，逐dataset/seed initialization paired，test nonmutation与protocol audit全部通过。
125. SIFF相对A6_FULL test MSE/MAE为`+1.2497%/+0.7549%`，5/5 datasets、4/4 horizons、3/3 seeds正向；
     该结果只确认完整package，不归因ordered field。
126. SIFF相对independent control test MSE/MAE为`-0.1272%/-0.1733%`，validation为
     `-0.3224%/-0.5015%`；新seeds2022/2023均为负。internal health 6/6不能覆盖matched attribution failure。
127. Decision=`performance_pass_attribution_blocked_stop_fcc_promotion`，failure=`capacity_control_explains`。
     SIFF-v2不晋升、不rescue、不进入modern baselines/formal ablations；当前`active_method=none`，回paper portfolio
     decision或new Step2/4。
128. 用户于2026-07-22扩大ISCF研究范围：固定ISCF architecture base，但允许探索与其原生耦合的loss、training与
     architecture extension，以形成连贯paper story并提升official-test性能。
129. evidence/code audit确认：ISCF arm已有median `8.5813%` oracle headroom，但fusion仅9/15超过best fixed；
     `equal_skill`实际以uniform individual L1 target loss把所有arm推向同一conditional-median target，coalition
     role signal缺失。
130. primary-source audit排除把generic expert loss、orthogonality/diversity、structural anchor、frequency expert、
     Shapley或counterfactual routing primitive直接claim为创新。
131. working route=`SC-ISCF-SCC-v0 — Scope Coalition Credit`：利用dense fusion闭式leave-one-scope-out risk，
     train-only校准existing direct policy并以fused-only替代uniform individual supervision；inference graph不变。
132. Decision=`scc_problem_diagnostic_proposed_active_method_none`。只允许复用existing artifacts完成D0；credit必须
     nondegenerate、cross-seed stable、不同于standalone arm error并失败于shuffled control，才返回Step5/6。当前
     implementation、remote training、formal test、modern baselines均false。
133. historical 15-run NPZ缺少exact per-coordinate direct policy，只有bin-level usage；arms/fused无法唯一反演five
     policy weights，故不允许approximate reconstruction。
134. 按预注册fallback冻结same checkpoints的15-run validation-only replay；runner只做forward与checkpoint
     nonmutation，输出exact `probe_direct_policy [256,720,5]`，training/test均false。
135. D0 config/analyzer/runner及code explanation完成，local smokes通过；three-GPU preflight均18 MiB、0%。
     Decision=`d0_validation_replay_prelaunch_pass_remote_forward_authorized`；commit/push后remote fast-forward执行。
136. D0完成15-run frozen validation replay：oracle headroom/nondegeneracy/shuffle specificity通过，但fixed-label
     cross-seed topology仅2/5 datasets稳定。Decision=`coalition_credit_unresolved_requires_validation_diagnostic_redesign`；
     该结果不允许进入Step7，也不作ISCF architecture方向级拒绝。
137. 按rollback冻结D0B：只用arms/policy/position的60/40 blocked-row ridge probe，比较coalition、standalone与16个
     horizon-marginal shuffle controls。existing validation artifacts only；forecast training、implementation、test false。
138. 153/103初版split切开multivariate channel group，predecision invalid；修正为147/109后D0B仍完整通过：15/15
     predicted gain positive、14/15 shuffle binding、vs standalone 13/15 positive。Decision=
     `coalition_credit_information_access_supported_return_step5_6`。
139. SCC-v0 Step5–6冻结：harmonic fused L1 + train-only exact coalition KL，route weight前25% ramp至`.1`，credit
     stop-gradient、uniform fallback、inference unchanged。五臂EQUAL/FUSED/ARMERR/SCC/SHUFFLED attribution冻结；
     narrative gate通过到Step7A，remote training/formal test仍false。
140. Step7A实现并验证exact coalition/shuffled objective、dedicated RNG与per-scope gradient logging；existing PCC
     regression 36/36通过，runner dry-run=20 jobs。Decision=`step7a_pass_step7b_remote_validation_authorized`；先
     resource smoke，formal test仍false。
141. resource smoke通过后，commit=`91e466a`的20-run SCC matched validation已在GPU0/1/2启动；repo-external
     root=`stage_c_iscf_scc_v0_step7b`。只允许full-matrix Step9 decision，formal test仍false。
142. SCC-v0 Step9完整但失败：vs EQUAL `-3.1750%` MSE，且不超过FUSED/ARMERR/SHUFFLED；finite与gradients健康，
     coalition headroom却从`+18.08%`反转为`-14.93%`。failure=`intervention_point_wrong`，关闭v0。
143. Step5–6只允许RSCC-v1 exact hybrid：保留EQUAL reliability，加coalition KL；matched EQUAL-ARMERR与SHUFFLED，
     15 new runs。Step7A authorized，remote/test false；若失败关闭coalition route，不再rescue。
144. RSCC Step7A通过：hybrid skill loss与EQUAL逐值相等，shuffled保持skill/credit marginals；PCC regression 36/36，
     runner dry-run=15。只授权Weather resource smoke，通过后才可launch；formal test=false。
145. Weather RSCC/SHUFFLED resource smoke共享initialization hash，skill loss逐值相等，route loss与five scope
     gradients均finite/nonzero。commit `020eea3`的15-run matrix于`2026-07-22T14:12:34+08:00`在GPU0/1/2启动；
     output=`stage_c_iscf_rscc_v1_step7b`，formal test=false。15/15前禁止partial selection或修改gates。
146. RSCC Step9为20/20 effective runs、80/80 validation cells。vs EQUAL MSE/MAE `+0.5189%/+0.3972%`
     primary pass，但vs ARMERR/SHUFFLED为`-0.1414%/-0.1394%`，policy-credit Spearman亦由`0.2052`降至
     `0.1539`。Decision=`rscc_v1_control_attribution_fail_close_exact_route`；no formal test/rescue；回Step2/4。
147. function-level audit确认ARMERR/SHUFFLED fused relative L1仅`0.00138--0.00462`、policy mean L1仅
     `0.00254--0.00830`，且两者policy entropy约`0.986--0.998`；公共效果更符合near-uniform shrinkage clue。
148. EQUAL为historical reference，未contemporaneously retrain；因此H1 inference shrinkage、H2 training
     co-adaptation与H3 run drift仍混杂，不得把ARMERR直接升级为carrier/method。
149. `SC-ISCF-PSA-D0`冻结为existing validation artifact diagnostic：15 replays、fixed grids、147/109 split、LODO
     selection，uniform/marginal/temperature controls；training/test/method=false。negative不得拒绝joint-training方向。
150. PSA-D0 primary convex-uniform macro L1/MSE=`-0.2431%/-0.1218%`，1/5 datasets、2/15 runs joint-positive；
     scope-marginal与temperature也macro negative。Decision=`frozen_inference_shrinkage_not_supported`。
151. 4/5 folds在source-fit选择nonzero alpha却多在held-out反转，故禁止alpha/dataset/position rescue。H1 closed；
     failure=`frozen_probe_negative_joint_training_unresolved`。
152. 下一最小control=`SC-ISCF-PSA-D1`：contemporaneous seed2021 EQUAL × five datasets，validation-only，区分H2
     co-adaptation与H3 run drift。当前proposed/not authorized；active method none，test=false。
153. D1 protocol已冻结：new EQUAL recovery ratio与dataset/horizon gates区分`run_drift_explains`、
     `joint_training_route_regularization_supported_as_carrier_clue`或`unresolved`。config/runner/remote/test均false。
154. 用户明确授权D1 Step7A + five-run validation training。config/checker/runner/analyzer已实现；EQUAL route=0、five
     scope gradients可达、5-job dry-run与two synthetic decision branches通过。
155. Decision=`psa_d1_step7a_pass_proceed_commit_remote_preflight`。full launch conditional on commit-pinned pull、GPU
     preflight与Weather smoke；formal test、confirmation、method promotion false。
156. commit `f5275a4` remote fast-forward；three GPUs均18 MiB/0%，Weather smoke route=0、five gradients nonzero、
     initialization hash匹配three references。
157. five runs于`2026-07-22T16:00:40+08:00`启动，PID=`3975446`，initial 0/5。Decision=
     `psa_d1_five_run_validation_training_active_formal_test_disabled`；5/5前no partial selection/test。
158. ETTh1 training/standard metrics完成后，evaluator因missing `diagnostic_protocol.future_bins`在probe前失败；标记
     `diagnostic_protocol_fault_predecision`，不产生H2/H3 decision。
159. v0.1只补evaluator contracts与SHA-nonmutation validation replay runner；training/checkpoints/gates不变。其余training
     结束前不remote pull；之后补5 diagnostics再full analyzer。
160. 最终D1 5/5 new、20/20 effective runs、80/80 cells、5/5 diagnostics完整。New/historical EQUAL checkpoint与
     function逐值相同，H3 run drift关闭。
161. ARMERR/SHUFFLED vs new EQUAL MSE=`+0.6577%/+0.6557%`，均17/20、5/5 datasets、4/4 horizons。
     Decision=`joint_training_route_regularization_supported_as_carrier_clue`，test=false。
162. UPA-D2冻结为唯一next diagnostic：EQUAL + information-free uniform policy KL，weight/schedule匹配controls。
     Generic balancing不具novelty；implementation/remote/test未授权。
