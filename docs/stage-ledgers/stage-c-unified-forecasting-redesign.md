# StageC Unified Varied-Horizon Forecasting Ledger

## Stage Scope

| Field | Content |
| --- | --- |
| `stage_id` | `StageC-UVHF` |
| `paper_role` | decoder/operator 与 training principle 两项相互支撑的 paper-core innovations |
| `active_question` | shared future function representation 与 aligned risk 是否能实现连续、统一、可细化的任意 horizon forecasting？ |
| `source_evidence` | historical/source-faithful `A6-LBF-r256` |
| `mechanism_control` | frozen `A6-LBF-natural-baseline` |
| `active_candidates` | `SC1-PMFO-RCT`, `SC2-MIPR` |
| `stage_exit` | 两项分别过 narrative/effectiveness gate，`2x2` joint gate显示独立主效应与联合收益 |
| `stage_rollback` | problem/novelty不跨 dataset -> Step 2；禁止直接堆叠 method |

## Decision Cursor

| Field | Content |
| --- | --- |
| `current_11_step` | Step 7A local implementation/invariant gate passed；Step 7B pending |
| `current_candidate` | `SC1-PMFO-RCT`（implementation_gate_passed）、`SC2-MIPR`（narrative_ready）；effectiveness均pending |
| `latest_decision` | PMFO-RCT四个variants已实现；90/90 shape-prefix与refinement/conservation/locality gate通过 |
| `next_required_action` | 准备ETTm1/ETTh2/Weather Step7B runner；launch前commit/push并检查remote GPU |
| `method_training_authorized` | `false` |
| `rollback_point` | capacity/no-transition control explains -> Step 4；numeric/readout fault -> Step 6 |

## 11-Step Record

| Field | Current Record |
| --- | --- |
| `current_step` | Step 7A complete；Step 7B pending |
| `problem` | A6已domain-only但single dense basis没有nested refinement；有序memory可用但ETTh2线性重组风险仍在；simple horizon reweighting缺少novelty |
| `existence_evidence` | D1-v2: structure 3/3、frozen ordered-memory use 3/3、raw/PIR gradient aggregate 3/3；benchmark-measure projected excess 0/3 |
| `idea` | PMFO-RCT用future tree作conservative refinement；MIPR把measure metric投影到相同scale blocks |
| `theory_check` | mixed-radix invariants≤`1.33e-15`；MIPR对L2成立但不是exact raw risk；Huber/L1首轮拒绝 |
| `design` | local invariants passed -> 3-dataset SC1 matched controls -> SC2 raw/random-projector controls -> 2x2 -> full matrix |
| `narrative_gate` | SC1 pass；SC2 pass with L2/measure boundary；两者均effectiveness pending |
| `effectiveness_gate` | frozen profiles、multi-seed、dense horizons、matched capacity/FLOPs、cross-dataset |
| `artifacts` | D1-v2、Step4-6 prior-art/theory report、Step7A local gate、baseline report |
| `decision` | SC1 implementation gate通过但effectiveness pending；Step7B三数据集已预注册，remote training未授权 |

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
| `SC1-PMFO-RCT` | `implementation_gate_passed` | conservative future-tree refinement提供dense basis之外的scale identity与domain-local execution | pass；medium novelty confidence，禁止generic wavelet/arbitrary-H claim | D1/algebra/local implementation pass；performance未测 | Step7B三数据集matched controls |
| `SC2-MIPR` | `narrative_ready` | measure-induced block metric去除decoder scales之间的cross coupling | pass for L2；log measure primary，benchmark weak | log off-block `0.205154`；benchmark `0.002480`；performance未测 | 等SC1 contract冻结后实现raw/random-projector controls |
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
| PMFO-RCT Step7B architecture screening | `pending` | 固定ETTm1/ETTh2/Weather、seed2021；先准备runner，不引入MIPR |

## Paper Mainline Sync Log

| Date | Trigger | Paper Section | Change Type | Decision |
| --- | --- | --- | --- | --- |
| 2026-07-13 | Step4-6 narrative/theory gate完成 | Current Position、Contribution Slots、Contribution Boundary、Main Experiment Logic | contribution boundary + experiment order | PMFO收紧为RCT；PIR收紧为MIPR；Step7 local implementation next |
| 2026-07-13 | Step7A local gate通过 | Current Position、Contribution 1、Main Experiment Logic | implementation evidence + screening scope | 三数据集Step7B固定；effectiveness仍pending |

## Continuation Rules

1. 每次继续研究先读本 ledger 与`stage-c-pmfo-rct-step7-protocol.md`；
2. old analysis可引用，archive脚本不得直接启动；
3. diagnostic failure必须区分 hypothesis、intervention、readout、numeric与capacity control；
4. narrative与Step7A implementation gate已完成；Step7B launch仍需单独remote preflight，且不得实现MIPR；
5. test reference只用于最终对比，不能参与设计选择。
