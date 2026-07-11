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
| `mechanism_control_carrier` | 尚未冻结；SC0 fixed-20 protocol selector不稳定，回滚 Step 2/3设计 `SC0-R1` |
| `entry_evidence` | A6 clean rerun；StageB mechanism failures；C0/C1 carrier controls；2024-2026 decoder/training prior-art audit |
| `stage_exit_condition` | decoder 与 training strategy 分别通过 narrative/effectiveness gate，并在 frozen protocol 下显示独立主效应和 joint gain |
| `stage_rollback_condition` | 若 standardized carrier 不成立、问题存在性不跨 dataset，或 prior art 无法形成清晰 novelty boundary，则回到 Step 2/3，不实现 paper-core method |

## Decision Cursor

| Field | Content |
| --- | --- |
| `current_11_step` | StageC Step 7/8：SC0-R1 control implementation complete，等待remote effectiveness gate |
| `current_candidate` | `SC0-R1`（control-only，protocol/local gate passed） |
| `latest_decision` | patience离线gate选择5并保留9/9旧best；opt-in stopping、27-run analyzer与ETTh2 semantic smoke通过 |
| `next_required_action` | commit/push后执行3090 preflight，启动三臂×三dataset×三seed validation-only calibration |
| `rollback_point` | SC0-R1 effectiveness gate失败则回Step 2/3重审carrier topology，禁止逐dataset恢复特调presets或再改stopping追结果 |

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
2. `standardized mechanism control`：所有 StageC mechanism diagnostics/ablations 使用同一个冻结 profile；
3. `native external baseline`：ElasTST、TimePerceiver、DAM、FlowState、Implicit Forecaster 等先在其官方
   repository/native protocol 中复现，再讨论适配或横向比较。

任何 source-faithful result 都不得单独用来归因 StageC mechanism；任何 standardized result 也不得改写成
TimeAlign official reproduction。

## Candidate Queue

| ID | Status | Hypothesis | Narrative Gate | Effectiveness Gate | Blocking Or Next Action | Artifacts |
| --- | --- | --- | --- | --- | --- | --- |
| `SC0-MCP` | `failed_exact_protocol` | 一个非 `patch_num=1`、capacity-matched、跨 dataset 共用的 TimeAlign-derived carrier profile 可以为后续 mechanism attribution 提供稳定基线 | `not_required`；protocol control | best-val regret通过，但 last/best winner必须一致 | 9/9、0 errors、validation-only；best=`p24/d64`、last=`p48/d32`；固定20 epoch在ETTh2产生31.63%-44.95%退化，回滚Step 2/3 | protocol、config、runner、`analysis/stage_c_sc0_carrier_calibration_20260711/` |
| `SC0-R1` | `ready_to_launch` | 统一 validation-controlled training/checkpoint policy可消除固定20 epoch过训练，使common carrier成为稳定研究工具 | `not_required`；control-only；protocol audit通过：max20/patience5/restore-best，同一规则跨dataset | 27 runs；mean/median winner一致；至少2/3 seed wins；pooled regret<=3%；任一seed-dataset<=5% | local gate通过；允许commit/push与GPU preflight后launch | SC0 failure report；stopping audit；R1 config/runner/analyzer/code explanation |
| `SC1-PFO` | `proposed` | unified forecasting 应表示满足 projective consistency 的 forecast family，而不是 benchmark-horizon heads 或 full-trajectory clipping | 尚未通过；必须区别 DAM/FlowState/ElasTST/TimePerceiver，并明确 A6 special-case relation | dense seen/unseen horizons；exact consistency；matched A6/fixed-basis/query controls；跨 dataset 和 seed | 等待 SC0；完成 full paper/code prior-art matrix 与 problem diagnostic | none |
| `SC2-HML` | `proposed` | training risk 应对应声明的 horizon measure，而不是由 `{96,192,336,720}` nested prefixes 隐式产生 early-step overweighting | 尚未通过；必须超越 ElasTST uniform-horizon harmonic reweighting与 generic task balancing | exposure/gradient mechanism 先过 Step 3；再验证 worst-horizon regret、AUC 与 unseen-H generalization | 等待 SC0；先做 gradient/exposure causal diagnostic | none |
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

## Pending Tasks

| Task | Owner | Trigger | Status | Next Action |
| --- | --- | --- | --- | --- |
| Freeze SC0 mathematical/config contract | Codex | StageC creation | `completed` | 以 protocol 文档为实现依据 |
| Implement SC0 config profile and runner | Codex | user continues StageC execution | `completed` | config、profile overrides、runner、sync、analyzer已完成 |
| Verify SC0 local semantics | Codex | implementation complete | `completed` | 9-run one-batch smoke与analyzer integration通过；profile hash `79a037f7...fd900` |
| Run SC0 validation-only calibration | Codex | local gate + commit/push + GPU preflight | `completed_failed_gate` | 9/9已同步；exact fixed-20 protocol不冻结 |
| Confirm and freeze global profile | Codex | seed2021 gate passes | `blocked_by_gate` | 禁止启动原confirmation；SC0-R1重新过protocol gate后再决定 |
| Design SC0-R1 training control | Codex | SC0 selector instability + optimization pathology | `completed` | max20/patience5/restore-best；全臂三seed gate已冻结 |
| Run SC0-R1 validation-only calibration | Codex | protocol/local gate + commit/push + GPU preflight | `pending` | 27 runs；output root使用repo-external StageC路径 |
| Build StageC prior-art matrix | Codex | SC0 可并行准备 | `pending` | 逐项记录 task definition、decoder contract、training distribution、official code defaults与 novelty boundary |
| Run SC1/SC2 problem diagnostics | Codex | SC0 frozen | `pending` | 先 Step 2/3，后 Step 4-6 narrative gate |

## Paper Mainline Sync Log

| Date | Trigger | Paper Section | Change Type | Decision |
| --- | --- | --- | --- | --- |
| 2026-07-11 | 用户要求暂停 StageB 并新开 StageC | Current Position；StageC Research Reset | stage、working title、contribution boundary、active ledger、protocol | StageC 成为 active research stage；A6 降为 source-evidence/prototype；SC0 先行 |

## Notes For Next Continuation

- 每次继续研究先读本 ledger，再读 standardized protocol；旧 Phase5 ledger 只用于历史 evidence。
- `patch_num=1` 可存在于 TimeAlign source reproduction，不得进入 StageC mechanism-control profile。
- SC0 winner 一经 multi-seed freeze，后续 mechanism experiments 不得逐 dataset、逐 candidate 或根据 test
  result 修改 `patch_num/d_model/d_ff/dropout/LR/epoch/selector`。
- 如果统一 profile 在任一 dataset 上超过预注册 regret gate，不得退回 dataset-specific presets；应回到
  StageC Step 2/3 重审 common carrier contract。
- SC1/SC2 是两个 paper innovation slots，不是预先接受的方法名称。任何 candidate 都必须先过 narrative gate。
