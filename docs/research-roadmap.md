# Research Roadmap

## Current Cursor

| Field | Content |
| --- | --- |
| `stage` | `StageC-UVHF` |
| `current_step` | Step 5 partial pass；Step 6 narrative/control design |
| `active_question` | FPMO scale-native factorization能否在full-affine matched control之外提供独立机制？ |
| `active_candidates` | `SC1-FPMO-DS` partial pass；M0/DA controls；`SC2-MIPR` held |
| `future_validation_suite` | ETTh1, ETTh2, ETTm1, ETTm2, Weather |
| `active_protocol` | `analysis/stage_c_step5_fpmo_theory_20260713/step5_theory_feasibility.md` |
| `method_implementation` | `PMFO-RCT v1` frozen as failed evidence；new implementation unauthorized |
| `rollback_point` | Step6无法隔离structure/capacity -> Step 2/4；不改Encoder |

## Completed Foundation

### SC0 natural carrier

[Decision] dataset 可有自然结构偏好，但不得为每个新机制重新精调。使用 validation-only 两阶段小 grid
一次性冻结：Weather=P12/D64/ff128、ETTm1=P24/D32/ff64、ETTh2=P12/D64/ff128。params 差异只报告，
不参与选择。9 profile-seed validation stability gate 已通过。

### Natural baseline test reference

[Fact] 2026-07-13 完成 3 datasets × 3 seeds × 8 horizons，72/72 test metrics；checkpoint/profile 均在
test 前冻结，`selection_used_test=false`。该 reference 只用于后续对比，不允许反向修改 protocol。

[Risk] ETTh2 H48 test MSE CV=`5.30%`，后续必须报告三 seed；这与训练期 validation best-vs-last
`31.63%-44.95%` 恶化不是同一统计。

### Research reset and archive

[Decision] StageB 不再是 active cursor。旧 scripts、local candidates、configs 与 protocol/code docs 已移入
archive；`analysis/` 作为不可变 evidence store 保留。活动入口只保留 natural A6 carrier、baseline test 与
PMFO/PIR diagnostic。

## Step 1: Prior-Art Audit

已确认的 novelty pressure：

- ElasTST：horizon-invariant placeholders 与 horizon reweighting；
- TimePerceiver：target timestamp queries；
- FlowState：functional basis + dynamic horizon/resolution；
- Implicit Forecaster：implicit future waves；
- TransDF/QDF：label decorrelation与task covariance weighting。

[Decision] explicit horizon conditioning、continuous coordinate query、simple functional basis、simple harmonic
step weighting 都不能单独成为 paper core。wavelet/refinement/neural-operator专项审计已在2026-07-13
Step 4-6完成，并进一步排除了generic hierarchical interpolation与learnable lifting claim。

## Step 2-3: Completed Problem Diagnostics

[Decision] D1-v2已完成：PMFO structure与frozen ordered-memory gate均3/3；PIR aggregate gate 3/3。
SC1通过problem gate；SC2以measure-conditional形式通过。以下内容转为已完成problem record。

### SC1-PMFO

问题：A6 已按`basis[:H]`直接计算H步输出，但只提供single dense rank-256 future subspace。是否存在稳定
的nested coarse-to-fine future structure，A6 `memory: [B,C,P,D]`是否保留该信息，以及新的operator能否在
不读取horizon ID的前提下提供refinement/local-support computation？

Gate：至少2/3 datasets、3 seeds支持evaluation-space future deviation与baseline residual的stable increment
structure；frozen A6必须优于zero-deviation baseline，且patch shuffle/collapse必须产生至少1%的SSE恶化。
Linear probe只作辅助量，negative R2之间的差值不得形成pass。
learned basis geometry用于区分“容量足够但缺层次”与“subspace本身不足”。若失败，rollback Step 2；
不得用同步更换Encoder与decoder掩盖归因。

### SC2-PIR

问题：deployment horizon measure 的变化是否产生跨 dataset 的非平凡 gradient/risk差异，并且 nested
increments 是否提供 raw step reweighting之外的解释量？

Gate：至少 2/3 datasets 显示稳定 gradient direction变化；projected risk必须超越 ElasTST-style harmonic
weights 的必然结果。若失败，关闭 PIR；horizon measure 只保留为 protocol/evaluation定义。

## Step 4-6: Completed Design Gate

2026-07-13已完成：

1. external primary-source matrix表明arbitrary horizon、functional basis、hierarchical interpolation、
   learned lifting与raw harmonic weighting均不能单独成文；
2. SC1收紧为`PMFO-RCT`：future interval tree的detail位于父尺度正交补，H只做domain pruning；
3. mixed-radix `(90,30,10,5,1)` orthogonality/refinement/prefix invariants均在`1.33e-15`内通过；
4. SC2收紧为`MIPR`：$\widetilde W_\mu=\sum_lQ_lW_\mu Q_l$，是L2 measure-induced
   block-diagonal surrogate，不是exact raw risk；
5. 预注册dense/no-transition/no-conservation与raw/random-projector controls；
6. SC1/SC2均标记`narrative_ready`，但SC2实现必须等待SC1 operator contract。

## Step 7-10: PMFO-RCT v1 Result

1. Step 7A已完成：90/90 shape-prefix cases及refinement/conservation/locality gate通过，不训练；
2. Step 7B完成ETTm1+ETTh2+Weather、seed2021的15-run matched-control screen；
3. PMFO-RCT相对A6 macro `-1.0955%`且三dataset均退化，effectiveness gate失败；
4. conservation相对no-conservation macro `+2.3393%`，保留；recursive transition相对no-transition仅
   `+0.0486%`，v1 claim撤回；
5. decision=`rollback_step4`；SC2-MIPR与joint factorial暂停，不得建立在失败operator上。

禁止在最小 gate 前加入 Encoder innovation、MoE、router、auxiliary reconstruction 或 per-horizon tuning。

## Step 4 Redesign Audit: Completed

2026-07-13 source-informed redesign audit完成：

1. A6 effective operator为$W=BA\in\mathbb R^{720\times768}$；覆盖rank-256 affine family至少需要
   `316,112`维，而PMFO v1 readout只有`212,010` parameters。因此同为256维latent不构成functional
   containment；
2. A6 operator在fixed block90/30 boundaries上的jump ratio仅`0.989-1.009`，没有跨dataset regime-change
   证据；block90 rank16 capture又从ETTh2 `0.4595`到Weather `0.8025`，不支持统一激进local rank；
3. PMFO 8 root nodes的history-patch profile cosine为`0.936-0.994`、entropy为`0.976-0.995`；nodes学习了
   不同signed projection，但没有清晰history-region specialization；
4. PRISM与LeapTS进一步占据generic multiresolution tree与adaptive scale scheduling；nested basis、lifting、
   Net2Net/Network Morphism只可作为数学工具；global low-rank + hierarchy residual也已有Asymmetric MMF压力；
5. 新候选暂定`SC1-FPMO`：future-domain function-preserving multiresolution operator morphism。它只通过
   Step 4 source-level conditional gate，未通过Step 5 theory或Step 6 design gate。

详细统计定义、source matrix与failure attribution见active protocol。PMFO-RCT v1继续关闭，不做调参复活。

## Step 5 FPMO Theory Feasibility: Partial Pass

1. 采用任意正整数$T$可构造的balanced unbalanced-Haar interval basis；orthogonality、perfect
   reconstruction、A6 morphism与native prefix restriction在9个$T$、53个$(T,H)$ cases通过，max gap
   `5.329e-14`；
2. shared-latent `FPMO-M0`与A6 function class完全相同，只能作exact morph control；
3. independent-scale `FPMO-DS`可逐depth factorize A6 effective map，因此exact containment成立；
4. T720的group sizes/rank caps均为
   `[1,1,2,4,8,16,32,64,128,256,208]`，sum=720，所以DS class等价full affine；
5. exact containment、independent scale maps与总latent budget 256不能同时成立；这是Step5 no-go boundary；
6. native restriction成立，但全部scale latents仍可能对任意$H$执行，故撤销“比A6更快”claim；
7. decision=`partial_pass_step6_design_only`；M0与direct-atom DA降为controls，DS尚未narrative-ready。

## Next Concrete Action

进入`SC1-FPMO-DS Step 6`，只做narrative/control design：

1. 冻结A6、M0、direct-atom full-affine DA、independent-scale DS的function-class mapping；
2. 找到DS相对DA的非冗余mechanism claim；若只能归因于capacity或optimizer coordinates，关闭DS；
3. 给出prefix tensor path与全部scale-latent overhead，不claim inference speedup；
4. 预注册`DS <= DA -> architecture claim fail`与`DS只超过A6 -> capacity_control_explains`；
5. Step 6通过前不实现model，不恢复MIPR。Encoder、MoE继续冻结。

并行control prerequisite：按validation-only natural grid校准ETTh1与ETTm2 profile。未来broad screen固定为
五dataset全arms seed2021；通过后对五dataset全部decisive arms运行seeds2021/2022/2023。增加dataset降低
cross-dataset偶然性，multi-seed才降低training stochasticity。协议见
`docs/experiments/stage-c-five-dataset-validation-policy.md`。

Step 7B证据见`analysis/stage_c_step7b_pmfo_rct_20260713/step7b_screening_report.md`。

## Historical Boundary

reset 前完整路线保存在 `docs/archive/pre-stage-c-reset-20260713/`。历史实验结果位于 `analysis/`，只有在
active ledger明确引用其 failure attribution 时才可用于新决策。
