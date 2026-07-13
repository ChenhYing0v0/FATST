# StageC Unified Varied-Horizon Forecasting Ledger

## Stage Scope

| Field | Content |
| --- | --- |
| `stage_id` | `StageC-UVHF` |
| `paper_role` | decoder/operator 与 training principle 两项相互支撑的 paper-core innovations |
| `active_question` | shared future function representation 与 aligned risk 是否能实现连续、统一、可细化的任意 horizon forecasting？ |
| `source_evidence` | historical/source-faithful `A6-LBF-r256` |
| `mechanism_control` | frozen `A6-LBF-natural-baseline` |
| `active_candidates` | `SC1-PMFO`, `SC2-PIR` |
| `stage_exit` | 两项分别过 narrative/effectiveness gate，`2x2` joint gate显示独立主效应与联合收益 |
| `stage_rollback` | problem/novelty不跨 dataset -> Step 2；禁止直接堆叠 method |

## Decision Cursor

| Field | Content |
| --- | --- |
| `current_11_step` | Step 2-3 problem-existence diagnostics |
| `current_candidate` | `SC1/SC2-D1`（diagnostic-only） |
| `latest_decision` | D1-v1因negative-R2 gate漏洞与history-std source-space pathology无效，不得作方向结论；已修订v2 measurement/gate |
| `next_required_action` | 运行D1-v2 evaluation-space structure、frozen decoder counterfactual与PIR gradient diagnostic |
| `method_training_authorized` | `false` |
| `rollback_point` | D1 gate fails -> Step 2 problem redefinition |

## 11-Step Record

| Field | Current Record |
| --- | --- |
| `current_step` | Step 2-3 |
| `problem` | A6已domain-only且按H直接求值，但single dense basis没有nested refinement；Encoder是否保留多尺度信息未知；simple horizon reweighting缺少novelty |
| `existence_evidence` | A6 memory=`[B,C,P,D]`、hidden width=768、basis=`[720,256]`；D1-v1表面structure/PIR通过但measurement invalid，尚无可接受跨dataset结论 |
| `idea` | H只作为output domain；PMFO学习nested function coefficients，PIR在同一projection increments上定义risk |
| `theory_check` | restriction consistency可由deterministic basis restriction保证；PIR对L2可建立正交分解，对L1尚无exact等价 |
| `design` | D1 offline diagnostics -> Step4-6 prior-art/theory gate -> minimal implementation -> 2x2 -> full matrix |
| `narrative_gate` | refinement algebra、tensor/gradient path、prior-art boundary与mandatory controls全部清楚后才可实现 |
| `effectiveness_gate` | frozen profiles、multi-seed、dense horizons、matched capacity/FLOPs、cross-dataset |
| `artifacts` | active protocol、deep audit、baseline report |
| `decision` | 只授权 diagnostic；PMFO/PIR 均未成为 accepted contribution |

## Frozen Carrier Contract

| Dataset | Profile | patch_num | d_model | d_ff |
| --- | --- | ---: | ---: | ---: |
| Weather | `r2b_p12_d64_ff128_medium` | 12 | 64 | 128 |
| ETTm1 | `r2b_p24_d32_ff64_narrow` | 24 | 32 | 64 |
| ETTh2 | `r2b_p12_d64_ff128_medium` | 12 | 64 | 128 |

contract hash:
`254d85d47a9e5b7c212f8a8b88decf17a0328a1ea1df324c9cc65be4c672a50c`。

Governance：dataset之间允许不同自然偏好；params差异不参与选择；同一dataset后续所有机制共用同一
profile；test、candidate identity与per-mechanism tuning不得改变profile。

## Candidate Queue

| ID | Status | Hypothesis | Narrative Gate | Effectiveness Gate | Next Action |
| --- | --- | --- | --- | --- | --- |
| `A6-LBF-natural-baseline` | `control_only` | validation-frozen natural profiles可作为稳定共同起点 | not required | 72/72 test；3 seeds；dense horizons | `frozen_test_reference_ready`；只作固定reference |
| `SC1-PMFO` | `diagnostic_only` | nested multiresolution future function在A6已有domain-only H之上提供refinement/local support | prior art/refinement proof未完成 | structure、encoder sufficiency至少2/3 datasets通过 | D1-A/B/C offline diagnostic |
| `SC2-PIR` | `diagnostic_only` | operator-aligned increments提供raw horizon weights之外的risk/gradient信息 | L2/Huber理论边界与TransDF/QDF/ElasTST区分未完成 | 至少2/3 datasets nontrivial measure-gradient evidence | D1 offline gradient diagnostic |
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

## Pending Tasks

| Task | Status | Next Action |
| --- | --- | --- |
| Freeze natural carrier | `completed` | 不再调 profile |
| Establish dense test reference | `completed` | 后续统一对比 |
| Archive closed routes and clean active entrypoints | `completed` | archive只作证据 |
| Implement D1 offline analyzer | `completed_v2` | evaluation-space source/gradient + strict probe + frozen decoder counterfactual |
| Run D1 problem diagnostics | `v1_invalid_v2_pending` | no method training；独立output root |
| PMFO/PIR Step4-6 gate | `blocked_on_D1` | problem evidence通过后再启动 |

## Continuation Rules

1. 每次继续研究先读本 ledger 与 active diagnostic protocol；
2. old analysis可引用，archive脚本不得直接启动；
3. diagnostic failure必须区分 hypothesis、intervention、readout、numeric与capacity control；
4. method implementation 前必须完成 narrative gate；
5. test reference只用于最终对比，不能参与设计选择。
