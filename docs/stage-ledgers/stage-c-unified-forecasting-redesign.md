# StageC Unified Forecasting Redesign Stage Ledger

本文档是 StageC 的 active stage ledger。StageC 不延续 StageB 的局部 mechanism queue，而是从论文级
Step 1-3 重新定义 unified varied-horizon forecasting、decoder contract、training principle 与标准化
mechanism-control protocol。StageB 的实验和负结果继续保存在 `phase5-timealign-interface.md` 与
`analysis/`，但不再作为 active cursor。

## Stage Scope

| Field | Content |
| --- | --- |
| `stage_id` | `StageC-UVHF` |
| `paper_mainline_role` | 重建两项相互支撑的 paper-core innovation：decoder/operator 与 training principle |
| `active_question` | 一个参数共享模型如何对任意 requested horizon 产生 projectively consistent、无需逐 horizon 特调且可与 specialists 竞争的预测？ |
| `source_evidence_carrier` | `A6-LBF-r256` on source-faithful TimeAlign presets；仅保留历史 performance evidence |
| `mechanism_control_carrier` | frozen natural dataset profiles：Weather=P12/D64、ETTm1=P24/D32、ETTh2=P12/D64；contract `254d85d4...a50c` |
| `entry_evidence` | A6 clean rerun；StageB mechanism failures；C0/C1 carrier controls；2024-2026 decoder/training prior-art audit |
| `stage_exit_condition` | decoder 与 training strategy 分别通过 narrative/effectiveness gate，并在 frozen protocol 下显示独立主效应和 joint gain |
| `stage_rollback_condition` | 若 standardized carrier 不成立、问题存在性不跨 dataset，或 prior art 无法形成清晰 novelty boundary，则回到 Step 2/3，不实现 paper-core method |

## Decision Cursor

| Field | Content |
| --- | --- |
| `current_11_step` | StageC Step 1-3：SC1-PFO/SC2-HML prior-art and problem-existence diagnostics |
| `current_candidate` | `SC1-PFO` / `SC2-HML`（analysis_pending；method implementation未授权） |
| `latest_decision` | SC1原projective-inconsistency假设在A6上为false；SC2 measure mismatch是problem candidate但当前full-720 control不存在旧nested-prefix重复加权 |
| `next_required_action` | tensor-level prior-art boundary + frozen-batch horizon-measure gradient diagnostic；不训练method |
| `rollback_point` | 若问题存在性或novelty boundary不成立，回Step 2重定义problem；禁止直接堆叠method |

## 11-Step Stage Record

| Field | Current StageC Record |
| --- | --- |
| `current_step` | Step 1-3 |
| `problem` | 现有 evidence 混入 TimeAlign dataset/horizon-specific preset；A6 仍是 fixed 720-row trajectory slicing，且 two-contribution paper story 未闭环 |
| `existence_evidence` | ETTh2/ETTm1/Weather unified presets 分别使用 `patch_num=48/1/48`、不同 width/dropout/LR；A6 multi-prefix 还存在 early/tail exposure imbalance |
| `idea` | 用统一 mechanism-control carrier 隔离 protocol confound，再研究 projective decoder 与 horizon-measure learning |
| `theory_check` | 同一 prediction problem 的 mechanism attribution 要求 architecture budget、optimization、checkpoint selection 与 data exposure 可比；source reproduction 与 causal mechanism control 必须分层 |
| `design` | SC0 calibration -> SC1/SC2 problem diagnostics -> individual method gates -> factorial joint gate -> cross-backbone/full matrix |
| `narrative_gate` | SC1/SC2 在 Step 4-6 前必须分别证明 novelty、tensor/gradient path、贡献边界和 required controls |
| `effectiveness_gate` | frozen protocol、multi-seed、dense seen/unseen horizons、cross-dataset、matched capacity/control；阈值由 SC0 protocol noise 预注册 |
| `artifacts` | 本 ledger；`docs/experiments/stage-c-standardized-mechanism-control-protocol.md`；后续 analysis/code artifacts |
| `decision` | StageC active；当前只授权 SC0 control-only work |

## Protocol Separation

StageC 强制区分三类 protocol：

1. `source-faithful reproduction`：复现 TimeAlign/A6 历史结果，保留 upstream dataset-specific presets；
2. `standardized mechanism control`：每个dataset使用一次性validation冻结的profile；同dataset的所有mechanism/control共用该profile；
3. `native external baseline`：ElasTST、TimePerceiver、DAM、FlowState、Implicit Forecaster 等先在其官方
   repository/native protocol 中复现，再讨论适配或横向比较。

任何 source-faithful result 都不得单独用来归因 StageC mechanism；任何 standardized result 也不得改写成
TimeAlign official reproduction。

## Candidate Queue

| ID | Status | Hypothesis | Narrative Gate | Effectiveness Gate | Blocking Or Next Action | Artifacts |
| --- | --- | --- | --- | --- | --- | --- |
| `SC0-MCP` | `failed_exact_protocol` | 一个非 `patch_num=1`、capacity-matched、跨 dataset 共用的 TimeAlign-derived carrier profile 可以为后续 mechanism attribution 提供稳定基线 | `not_required`；protocol control | best-val regret通过，但 last/best winner必须一致 | 9/9、0 errors、validation-only；best=`p24/d64`、last=`p48/d32`；固定20 epoch在ETTh2产生31.63%-44.95%退化，回滚Step 2/3 | protocol、config、runner、`analysis/stage_c_sc0_carrier_calibration_20260711/` |
| `SC0-R1` | `passed_global_control_superseded` | 一个全局profile能否作为更严格control | `not_required`；control-only | 27/27下global winner=`p24/d64` | 保留历史证据；用户并不要求跨dataset同profile，不再作为active约束 | R1 result；uniform frozen config |
| `SC0-CPA` | `passed_diagnostic` | SC0 validation degradation是否对应test degradation | `not_required`；post-freeze test diagnostic | 同一fixed20 best/last checkpoints；test不得反向选profile | H720 last 0/9 wins，mean test MSE +6.11%；dense last 29/72 wins | checkpoint test report |
| `SC0-DAP` | `capacity_control_only` | fixed active budget下dataset偏好不同token allocation | `not_required`；diagnostic control | 三臂约72万active params | 保留P12/P48/P24 mapping为诊断；不再决定active carrier | old dataset-aware config；governance revision |
| `SC0-DAP-R2A` | `passed_patch_selection` | 不匹配params时，dataset可从自然小grid选择patch granularity | `not_required`；control-only | 固定D64/ff128；dense validation regret选择 | Weather=P12、ETTm1=P24、ETTh2=P12 | R2A report/config/artifacts |
| `SC0-DAP-R2B` | `passed_width_selection` | 在selected patch下dataset可选择自然representation width | `not_required`；control-only | D/ff={32/64,64/128,128/256}；复用medium run | Weather/ETTh2选D64；ETTm1选D32 | R2B report/artifacts |
| `SC0-DAP-R2C` | `passed_and_frozen` | selected profiles在三seed下具有absolute validation stability | `not_required`；control-only | dataset mean/max dense MSE CV <=3%/5% | Weather 0.323/0.749%；ETTm1 0.707/1.405%；ETTh2 2.094/4.867%；不重证relative winner | R2C report/frozen contract |
| `SC1-PFO` | `problem_redefinition_required` | unified forecasting 应表示满足 projective consistency 的 forecast family，而不是 benchmark-horizon heads 或 full-trajectory clipping | failed as stated：A6 prefix slicing已exact consistent；ElasTST/FlowState/TimePerceiver构成直接prior art | not evaluated | 回Step 2定义horizon-adaptive/refinable operator价值与novelty；不实现 | SC1/SC2 prior-art audit |
| `SC2-HML` | `problem_candidate_diagnostic_required` | training risk 应对应声明的 horizon measure | partial：数学问题成立，但simple harmonic weighting被ElasTST覆盖；当前full-720 control无旧nested-prefix pathology | frozen-batch measure-gradient diagnostic先过Step3 | 实现offline gradient diagnostic，不训练method | diagnostic protocol/prior-art audit |
| `SC3-JCO` | `deferred` | projective decoder 与 horizon-measure learning 有可解释、非冗余的 interaction | 只有 SC1/SC2 分别通过后才评估 | `2x2` factorial 必须显示两项独立主效应，joint arm 不能只由单项解释 | 不得提前实现 | none |
| `SC4-XBG` | `deferred` | 核心 mechanism 不依赖 TimeAlign-derived encoder | generality gate，不单独作为 novelty | 至少第二 backbone、same mechanism direction、matched protocol | 等 SC1/SC2 小门通过 | none |

## SC0 Standardized Carrier Boundary

SC0 不是新 Encoder 创新，也不是为了重新寻找 SOTA preset。它只选择一个后续持续冻结的 research
instrument。初始 calibration 固定 `seq_len=pred_len=720`、A6 learned-basis readout、`basis_rank=256`、
two token-MLP layers 与 `P*D=1536`，比较：

| Arm | `patch_num` | patch length | `d_model` | `d_ff` | Active-forward params | Role |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sc0_p12_d128` | 12 | 60 | 128 | 256 | `718,672` | coarse common-patch control |
| `sc0_p24_d64` | 24 | 30 | 64 | 536 | `719,168` | middle granularity control |
| `sc0_p48_d32` | 48 | 15 | 32 | 1072 | `718,576` | fine common-patch control |

上述 count 已用当前 `TimeAlign.Model` 实例化核对；三臂 unused `proj_x` 均为 `1,106,640`，必须单独
报告，不得计入 active capacity。实现后的 local checker仍须重新复核，表中数值不是跳过 verification 的依据。

## Experiment Ledger

| Experiment | Candidate | Role | Result Summary | Decision | Full Report |
| --- | --- | --- | --- | --- | --- |
| Clean A6 rerun | StageC entry evidence | source evidence | vs fixed TimeAlign `-4.13%`, `9/12`; vs official unified `-1.75%`, `11/12` | 保留为 source-evidence anchor，不视为 standardized proof | `analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706/clean_a6_rerun_report.md` |
| C0 ETTm1 protocol gate | protocol evidence | diagnostic | P5 matched controls 全部输给 P1；不能证明 inherited `P=1` 是 performance defect | 不继续修 C0；但 StageC 不继承 dataset-specific P1 | `analysis/phase5_stage_b_c0_ettm1_carrier_protocol_gate_20260710/c0_ettm1_encoder_control_deep_analysis.md` |
| C1 multipatch carrier | protocol evidence | control | 两个 scales 相对 same-run A6 均明显退化 | 只否定 exact C1 design；不否定 standardized token-MLP calibration | `analysis/phase5_c1_global_anchored_multipatch_gate_20260710/c1_global_anchored_multipatch_deep_analysis.md` |
| SC0 local semantic gate | `SC0-MCP` | implementation verification | 3 arms structural gate与3 datasets × 3 arms one-batch CPU smoke通过；validation-only、dense horizons、dual reload、analyzer complete均通过 | `local_gate_passed`；允许commit/push后remote launch | `scripts/check_stage_c_sc0_carrier_local.py`; `docs/code-explanation/stage-c-sc0-mechanism-control.md` |
| SC0 seed2021 launch | `SC0-MCP` | validation-only carrier calibration | commit `31730cd`；GPU 0/1/2 preflight均free；Weather three arms先行；9 runs | `completed`；14:06-14:19，9/9、0 errors | `analysis/stage_c_sc0_carrier_calibration_20260711/launch_record.md` |
| SC0 seed2021 result | `SC0-MCP` | validation-only carrier calibration | 9/9、0 errors；best winner `p24/d64`、macro regret 0.4051%、max regret 1.2153%；last winner `p48/d32`；ETTh2 last较best恶化31.63%-44.95% | `exact_sc0_fixed20_protocol_not_frozen`；归因optimization/checkpoint pathology，回滚Step 2/3；不启动seeds 2022/2023 | `analysis/stage_c_sc0_carrier_calibration_20260711/sc0_failure_attribution_and_rollback.md` |
| SC0-R1 stopping policy gate | `SC0-R1` | offline protocol diagnostic | patience3保留7/9 best；patience5/7均9/9；选择最小满足者5，预计节省105 epochs | `protocol_gate_passed`；只授权预注册R1，不作为performance evidence | `analysis/stage_c_sc0_r1_protocol_gate_20260711/` |
| SC0-R1 local semantic gate | `SC0-R1` | implementation verification | opt-in early stopping、restore-best、27-run analyzer integration、三臂structure与ETTh2 validation-only smoke通过 | `local_gate_passed`；允许remote launch | `scripts/check_stage_c_sc0_r1_local.py`; `docs/code-explanation/stage-c-sc0-r1-training-control.md` |
| SC0-R1 launch | `SC0-R1` | validation-only multi-seed carrier calibration | commit`15b391b`；profile`3ebd07d6...f31a`；GPU 0/1/2 free；27 runs | `completed`；16:22-16:42，27/27、0 errors | `analysis/stage_c_sc0_r1_carrier_calibration_20260711/launch_record.md` |
| SC0-R1 result/freeze | `SC0-R1` | multi-seed standardized carrier gate | 27/27、0 errors；pooled/median均`p24/d64`；seed winners`p24/p12/p24`；regret gates全通过；test=false | `global_profile_selected_and_frozen`；SC0 blocker关闭 | `analysis/stage_c_sc0_r1_carrier_calibration_20260711/sc0_r1_deep_analysis_and_freeze.md` |
| SC0 checkpoint test gap | `SC0-CPA` | post-freeze diagnostic | H720 last在9/9均差于best；mean test MSE +6.11%，远小于validation +14.72%；dense last有29/72 wins | A6 mechanism-control用best-val有证据；不能把validation 32%-45%写成test值；official TimeAlign仍遵循native last | `analysis/stage_c_sc0_checkpoint_test_gap_20260712/` |
| Dataset-aware carrier revision | `SC0-DAP` | protocol governance | 用户允许dataset结构偏好；从同一三臂capacity-matched grid用三seed validation一次性冻结 | active mapping=`Weather:P12, ETTm1:P48, ETTh2:P24`；禁止test/per-mechanism重选 | `configs/stage_c_mechanism_control_dataset_aware.json` |
| Dataset-profile parameter correction | `SC0-DAP-R2` | protocol governance | 用户明确params差异不应进入profile选择；旧`d_ff=536/1072`是capacity-match artifact | 旧mapping降级diagnostic；打开natural two-stage grid，params只报告 | `configs/stage_c_dataset_profile_calibration_r2.json` |
| SC0-DAP-R2A launch | `SC0-DAP-R2A` | validation-only patch screen | D64/ff128下P12/P24/P48，active params差异自然保留；9 runs | `completed`；9/9、0 errors | `analysis/stage_c_dap_r2a_patch_screen_20260712/launch_record.md` |
| SC0-DAP-R2A result | `SC0-DAP-R2A` | validation-only patch selection | 9/9、0 errors；Weather=P12、ETTm1=P24、ETTh2=P12；ETTh2 P12 8/8 horizons最优 | `phase_a_patch_selected`；进入Phase B，不回调patch | `analysis/stage_c_dap_r2a_patch_screen_20260712/r2a_patch_screen_report.md` |
| SC0-DAP-R2B result | `SC0-DAP-R2B` | validation-only width selection | 9/9 profiles完整；Weather=P12/D64、ETTm1=P24/D32、ETTh2=P12/D64；params/test未参与 | `phase_b_width_selected`；进入selected-only stability | `analysis/stage_c_dap_r2b_width_screen_20260712/r2b_width_screen_report.md` |
| SC0-DAP-R2C result/freeze | `SC0-DAP-R2C` | selected-only absolute stability | 6 new + 3 reused runs；72 dense metrics；mean/max CV均过3%/5% gate，ETTh2 max=4.867%接近边界 | `dataset_profiles_stable_and_frozen`；SC0 blocker关闭 | `analysis/stage_c_dap_r2c_stability_20260712/r2c_stability_and_freeze_report.md` |
| SC1/SC2 prior-art/problem audit | `SC1-PFO` / `SC2-HML` | Step 1-3 research audit | A6已exact prefix-consistent；current full-720 loss无14.39x nested-prefix pathology；latest decoders与ElasTST压缩novelty空间 | SC1回Step2；SC2停Step3并只授权gradient diagnostic | `analysis/stage_c_sc1_sc2_prior_art_problem_audit_20260712/sc1_sc2_prior_art_problem_audit.md` |

## Pending Tasks

| Task | Owner | Trigger | Status | Next Action |
| --- | --- | --- | --- | --- |
| Freeze SC0 mathematical/config contract | Codex | StageC creation | `completed` | 以 protocol 文档为实现依据 |
| Implement SC0 config profile and runner | Codex | user continues StageC execution | `completed` | config、profile overrides、runner、sync、analyzer已完成 |
| Verify SC0 local semantics | Codex | implementation complete | `completed` | 9-run one-batch smoke与analyzer integration通过；profile hash `79a037f7...fd900` |
| Run SC0 validation-only calibration | Codex | local gate + commit/push + GPU preflight | `completed_failed_gate` | 9/9已同步；exact fixed-20 protocol不冻结 |
| Confirm and freeze global profile | Codex | seed2021 gate passes | `blocked_by_gate` | 禁止启动原confirmation；SC0-R1重新过protocol gate后再决定 |
| Design SC0-R1 training control | Codex | SC0 selector instability + optimization pathology | `completed` | max20/patience5/restore-best；全臂三seed gate已冻结 |
| Run SC0-R1 validation-only calibration | Codex | protocol/local gate + commit/push + GPU preflight | `completed_pass` | `p24/d64`与frozen contract已落地 |
| Freeze mechanism-control profile | Codex | parameter-governance correction | `completed` | natural dataset-profile contract hash `254d85d4...a50c` |
| Run SC0-DAP-R2A patch screen | Codex | protocol/local gate | `completed_pass` | patch mapping已冻结 |
| Run SC0-DAP-R2B width screen | Codex | R2A pass | `completed_pass` | selected profiles已冻结 |
| Run SC0-DAP-R2C stability confirmation | Codex | R2B pass | `completed_pass` | 9/9、72/72；ETTh2 boundary-close风险保留 |
| Build StageC prior-art matrix | Codex | R2可并行 | `pending` | carrier冻结后恢复active cursor |
| Run SC1/SC2 problem diagnostics | Codex | prior-art matrix complete | `pending` | 先Step 2/3，后Step 4-6 narrative gate |

## Paper Mainline Sync Log

| Date | Trigger | Paper Section | Change Type | Decision |
| --- | --- | --- | --- | --- |
| 2026-07-11 | 用户要求暂停 StageB 并新开 StageC | Current Position；StageC Research Reset | stage、working title、contribution boundary、active ledger、protocol | StageC 成为 active research stage；A6 降为 source-evidence/prototype；SC0 先行 |

## Notes For Next Continuation

- 每次继续研究先读本 ledger，再读 standardized protocol；旧 Phase5 ledger 只用于历史 evidence。
- `patch_num=1` 可存在于 TimeAlign source reproduction，不得进入 StageC mechanism-control profile。
- SC0 winner 一经 multi-seed freeze，后续 mechanism experiments 不得逐 dataset、逐 candidate 或根据 test
  result 修改 `patch_num/d_model/d_ff/dropout/LR/epoch/selector`。
- dataset profile使用预注册natural coarse grid；params/FLOPs只报告、不参与winner排序；允许dataset之间
  不同，但禁止根据test或每个新mechanism重新选择。
- SC1/SC2 是两个 paper innovation slots，不是预先接受的方法名称。任何 candidate 都必须先过 narrative gate。
