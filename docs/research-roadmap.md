# Research Roadmap

## Current Cursor

| Field | Content |
| --- | --- |
| `stage` | `StageC-UVHF` |
| `current_step` | frozen replacement fairness correction complete；PLGO Step 6/7A next |
| `active_question` | conditional RGNB signal能否转化为E2E收益，且shared latent是否保留patch-specific usage？ |
| `active_candidates` | `SC1-PLGO-PAF narrative_ready`；`SC1-D8-E2E proposed`；`SC2-MIPR` held |
| `future_validation_suite` | ETTh1, ETTh2, ETTm1, ETTm2, Weather |
| `active_protocol` | `docs/experiments/stage-c-sc1-d8-end-to-end-coadaptation-screen.md` |
| `method_implementation` | D8 Step7A pending；remote training unauthorized |
| `rollback_point` | stable E2E PAF fail -> Step4 redesign；pathology -> Step7 repair only |

## Completed Foundation

### SC0 natural carrier

[Decision] dataset 可有自然结构偏好，但不得为每个新机制重新精调。使用 validation-only 两阶段小 grid
一次性冻结：Weather=P12/D64/ff128、ETTm1=P24/D32/ff64、ETTh2=P12/D64/ff128、
ETTh1=P24/D64/ff128、ETTm2=P48/D64/ff128。params 差异只报告，不参与选择。新增ETTh1/ETTm2的14-run
validation-only extension与3-seed stability gate已通过；five-dataset contract已冻结。

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

## Step 6 FPMO Narrative / Control Gate: Rejected

1. T720下每个group满足$k_l=n_l$，所以$D_lA_l$可表示任意block map；linear DS与DA拥有完全相同的
   full-affine function class；
2. 该等价对任意orthogonal coordinates与任意row grouping成立，当前factorization没有scale-specific
   function constraint；
3. deep linear/matrix factorization prior art支持“factorization可能改变implicit optimization bias”，
   但这不构成新的future-scale operator；当前Adam + L1 joint training也不满足直接移植现有定理的条件；
4. 加入per-scale nonlinearity会成为新候选：automatic exact A6 containment、matched dense/random controls
   与prior-art boundary都需重新审计，不能作为DS的implementation detail；
5. DS可少写出inactive atom coefficients，但dense scale factors仍需先构造全部720维scale latents，故
   prefix algebra不产生独立efficiency claim；
6. decision=`rejected_by_narrative_gate`；M0/DA/DS-L只作controls，rollback Step 2/3，MIPR继续held。

## SC1-D4 Structured-Basis Audit: Completed And Rolled Back

1. 315/315 frozen-memory fits完成，test未使用，PCA只由fit targets构造，315 fits均finite；
2. D3 signal复现：H720 balanced相对random orthogonal `+2.7181%`，5/5 datasets通过；
3. locality成立：balanced相对permuted interval八horizon macro `+1.6324%`，8/8 horizons为正；
4. exact midpoint balancing不特异：相对random interval tree仅`+0.2742%`，未过0.5% gate；
5. standard structured bases解释accuracy：balanced相对DCT-II/PCA-fit分别`-0.8609%/-1.5050%`；
6. decision=`standard_structured_basis_explains_gain_return_step2`。fixed balanced basis可作generation component，
   但不能以独特accuracy claim单独成为Contribution 1。

## SC1-PLGO Step 5 Theory Feasibility: Partial Pass

1. 构造Restricted-Global Nested Basis：root保持global DCT subspace，balanced intervals递归生成children
   scaling union相对parent的orthogonal local details；
2. direct restricted-DCT QR暴露最高`3.110e17` condition number；stable local Chebyshev chart保持同span并将
   最大condition降至`1.784e3`；
3. 12个$(T,r_g)$、101个selected prefixes与3,731个all-$H$ bounds通过，max algebraic gap
   `2.141e-13`；
4. square `PLGO-ONB-M0`可exact morph A6，但只是isometric reparameterization，无新function；
5. naive global/local union虽有frame bounds$[1,2]$，却有$r_g$维coefficient kernel；
6. T720、$r_g=16$ independent-group rank caps sum=720且等价full affine，capacity control解释收益；
7. native support pruning成立，但H1需102个active atoms，generator-level speedup未证明，效率claim撤回；
8. decision=`partial_pass_step6_design_only`；RGNB只冻结为mathematical scaffold，method/training仍false。

## Next Concrete Action

`SC1-D7-RGNB-descriptor-sufficiency`已完成105/105 fits与本地独立复算：compact/matched GEO相对controls为
+13.80%/+12.84%、均5/5 datasets，作为conditional geometry evidence保留。相对free-M0的
-37.38%/-39.10%受A6 Encoder-Decoder co-adaptation影响，不能评估end-to-end method readiness。decision更正为
`conditional_geometry_supported_end_to_end_gate_required`。下一步进入D8 Step7A，而非先做Step4 patch。

## SC1-PLGO Step 6 Design Gate: Conditional Pass, D7 Required

1. `PLGO-PAF`的atomwise tensor contract在$T=16/96/720/721$共33个prefix cases通过，max gap
   `4.547e-13`；$H$不进入descriptor/generator，rank上界仍为256；
2. generic branch-trunk、nonlinear query decoder、HyperNetwork、basis coefficient attention、timestamp query与
   functional basis decoder已有直接先例；overlap用于收紧component claim，不自动否决task-specific组合；
3. internal B11 basis-conditioned field被no-basis/constant-slot controls解释，B14 retrieval demand只有1/6
   settings、0/3 datasets通过；新PAF不得复活atom-specific history retrieval；
4. narrowed PAF只读取shared flattened memory，并以RGNB descriptors生成free temporal table的受限替代；
5. compact width256参数仅为A6 readout的0.696-0.880，可能capacity-restricted；near-budget width694约为
   0.9996-0.9998，却可能memorize descriptors而失去geometry attribution；
6. decision=`conditional_narrative_pass_d7_required`。PAF保留为provisional contribution candidate；D7通过并
   返回Step6冻结method contract前不进入Step7。

## SC1-D6 Confirmation And Step 4 Outcome

D6在未使用的validation batches8-15完成225/225：b144相对global DCT short `+1.1964%`、long
`-1.2675%`，12/15 primary units crossing，short-positive/long-negative分别覆盖4/5与5/5 datasets。
problem gate通过。external primary-source audit确认basis generation、wavelet coefficients、multiscale
interpolation与dynamic target length均已有先例；provisional `SC1-PLGO`只以projective local-global co-synthesis
进入Step5。balanced interval保留为local support scaffold，不claim exact midpoint novelty。

## SC1-D2 Core3 Precheck: Partial

1. 99/99 head-only runs完成，test/freeze/validation/basis/Parseval invariants通过；
2. full affine相对rank256 macro `-0.5661%`，不支持rank expansion是统一瓶颈；
3. strongest dense nonlinear相对full affine macro `-6.4492%`；ETTh2虽fit/inner-holdout更低，official
   validation恶化约19%-24%，属于temporal generalization failure而非未优化；
4. true scale相对strongest dense macro `+4.0358%`被ETTh2 dense overfit放大，且只2/3 datasets为正；
5. true interval basis相对random basis macro `+2.3137%`，3/3 datasets、9/9 seeds为正；
6. true depth grouping相对同basis random grouping macro `-0.2212%`，仅Weather稳定为正；
7. 初版combined random median会隐藏第6项，已在formal5前拆成random-group与random-basis两个mandatory gates；
8. decision=`partial_core3_basis_geometry_signal_only`；不进入Step4，先完成两套profile calibration与formal5。

并行control prerequisite：按validation-only natural grid校准ETTh1与ETTm2 profile。未来broad screen固定为
五dataset全arms seed2021；通过后对五dataset全部decisive arms运行seeds2021/2022/2023。增加dataset降低
cross-dataset偶然性，multi-seed才降低training stochasticity。协议见
`docs/experiments/stage-c-five-dataset-validation-policy.md`。

## SC1-D2 Formal5: Closed

1. five-dataset profiles已冻结；formal5完成165/165 fits，test/freeze/validation/basis/Parseval invariants pass；
2. full affine相对rank256 macro `+0.6780%`，只3/5 datasets达到2/3 seeds为正；rank不是统一瓶颈；
3. strongest dense相对full affine macro `-6.4715%`，ETTh1/ETTh2存在fit/holdout改善但official validation恶化的
   temporal generalization gap；
4. true scale相对strongest dense `+4.5202%`，但该值被上述dense gap放大，不能单独支持scale机制；
5. true basis相对random basis `+3.0635%`，5/5 datasets、15/15 seeds为正；
6. true grouping相对same-basis random grouping仅`+0.0947%`，只有2/5 datasets通过方向一致性，平均只击败
   `1.53/3` controls；
7. exact hypothesis=`hypothesis_false`；否定边界仅为final frozen-memory head上的balanced-depth independent
   nonlinear grouping；
8. decision=`scale_alignment_not_supported_reformulate_step2`；basis main effect因缺失factorial cell仍未识别。

完整解释见`analysis/stage_c_sc1_d2_formal5_20260714/research_interpretation.md`。

Step 7B证据见`analysis/stage_c_step7b_pmfo_rct_20260713/step7b_screening_report.md`。

## Historical Boundary

reset 前完整路线保存在 `docs/archive/pre-stage-c-reset-20260713/`。历史实验结果位于 `analysis/`，只有在
active ledger明确引用其 failure attribution 时才可用于新决策。
