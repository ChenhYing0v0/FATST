# Research Roadmap

## Current Cursor

| Field | Content |
| --- | --- |
| `stage` | `StageC-UVHF` |
| `current_step` | PCSD-CF D15-A Step7A local passed；Step7B design/runner audit next，remote held |
| `active_question` | fixed-past unified model是否需要在一个projective decoder内自适应future-output coupling scope？ |
| `active_candidates` | `SC1-PCSD-CF` narrative-ready/effectiveness-unready；SC2 slot open，`ICC` hypothesis only |
| `future_validation_suite` | ETTh1, ETTh2, ETTm1, ETTm2, Weather |
| `active_protocol` | `docs/experiments/stage-c-d15-native-pcsd-direct-control.md` |
| `method_implementation` | PCSD-CF module/readout/CLI/local gate complete；Step7B runner not frozen；remote/effectiveness/SC2/test=false |
| `rollback_point` | local invariant fail -> Step5/6；native arms unskilled -> Step4；credit problem absent -> SC2 remains open |

## Post-D11 Joint Mainline Reset

[Decision] D11只否定short/long directional gradient conflict，未否定D6 locality-coherence crossing、RGNB
geometry或Contribution 1 slot。不能从D11直接跳到Contribution 2；旧MIPR也因problem evidence与prior-art边界
不足而关闭。

历史post-D11主线曾把unified multi-horizon forecasting定义为同一future function在nested prefix-risk family下的
rank-limited逼近：

1. `PRISM`用$W_\mu$-isometric、prefix-localized forecast frame替换A6 unconstrained basis，保留free
   coefficient head与domain-only crop；
2. `CAPE`用train-only cross-fitted predictions估计conditional-mean covariance，使frame优先保留history可预测
   的future energy，而不是raw-label noise；
3. 两者形成`prefix localization on/off × predictable/raw covariance`的`2x2` factorial；
4. D12只进入Step 2-3 diagnostic；不实现method、不读test、不改Encoder。

D12 risk-aligned v2最终只支持1/5 datasets：CAPE关闭，PRISM joint route未进入D12-B，整条forecast-frame
mainline按预注册规则关闭并回滚Step 2。上述内容只作为历史设计，不是当前论文claim。

## Deferred Post-D12 Forecast-Revision Surface Idea

[Strong Evidence] D3-D8只稳定保留future-support geometry、short/local与long/global crossing及A6 free
operator capacity；D9-D12依次关闭history-scale mapping、future-component conflict与predictable-frame
allocation。A6 flatten是bijective reshape，但`PD -> 256`一次global compaction使所有future targets共享
同一coefficient state；patch-direct target access仍未被验证。

[External Boundary] MQ-RNN与Forking-Sequences已覆盖multi-FCD forecast grid；N-BEATS-S、On Forecast
Stability与Forecast AC已覆盖generic revision/stability optimization；forecast rationality literature已给出
conditional-mean revision moment。因此grid、stability penalty、martingale theory与continuous target query
均不能单独claim创新。

[Historical Provisional Mainline] 把基本对象定义为
$F(o,\tau)=E[Y_\tau\mid\mathcal F_o]$：row是multi-horizon forecast，column是same-target revision path。

1. `SC1-NIFRO`：causal patch memory产生`Delta[B,C,P,T]`，沿origin axis prefix scan得到forecast surface；
   $H$只裁剪latest row；linear control必须contain A6 readout。
2. `SC2-IARL`：不压小所有revision，而约束
   $E[e_{new}\Delta]=0$与$E[e_{old}^2-e_{new}^2]=E[\Delta^2]$，使revision energy由accuracy gain解释。
3. 两项status均为`proposed_step2_3`；它们共享surface object，但尚无internal practical headroom evidence。

[Decision] 用户确认该问题适合作为下一篇独立SCI核心。完整idea已转移到根目录`New-idea.md`；D13 protocol
保留为未来restart artifact，当前状态`deferred_next_paper`，不再执行。

## Fixed-Past Mainline Reset: Adaptive Output Coupling

[Accepted Critique] ordered patch memory只描述Encoder–Decoder information interface；即使full patch memory有用，
该问题也同样适用于single-horizon forecasting，不能成为multi-horizon论文主线。旧`CADMO/CPGA`因此标记
`rejected_by_narrative_scope`，原patch-memory D14降为`D14-P auxiliary_interface_probe`且当前不执行。

[Core Problem] Direct、AR、MIMO、DIRMO与future-query decoder的核心差异之一，是future targets共享
predictive function的scope。经典方法通常固定一个strategy或在模型外选择block size；unified multi-horizon
model若只统一输出长度，却固定一种output coupling，仍未统一forecasting strategy。

[Theory Boundary] deterministic separable MSE的Bayes predictor可逐target写成conditional mean，显式future
dependency不是population-risk必要条件。新主线只研究finite-sample/finite-capacity下parameter sharing引起的
bias–variance–flexibility trade-off。

[Internal Evidence]

1. A6是global low-rank/MIMO-like endpoint；
2. D6在disjoint validation上出现short/local `+1.1964%`、long/local `-1.2675%`与12/15 crossing，但该证据
   属于basis support，只有间接意义；
3. D8/JAPO要求新operator contain A6 global function，而不是整体替换；
4. B13/PMFO未支持current recurrent transition，首版不采用AR output feedback；
5. history-conditioned coupling choice尚无直接证据，D9-D10/JAPO形成高风险先验。

[External Boundary] Direct/MIMO/DIRMO与2025 Stratify已覆盖fixed strategy与output-size continuum；CATS、
MQTransformer、TimePerceiver覆盖future/target queries；Implicit Forecaster覆盖global wave decoding；MQF2覆盖
probabilistic future dependency；dynamic ensemble、meta-learning与TimeRouter覆盖expert/model routing。因此任一
primitive都不能单独claim创新。

[Historical Provisional Mainline, Superseded By PCSD-CF Reset]

1. `SC1-PCSD`：Projective Coupling-Spectrum Decoder在同一fixed future domain内表示point、multiple block与
   global sharing scopes；policy依赖history与target coordinate，不读取requested $H$；global arm contain A6；
2. `SC2-CCRL`：Cross-fitted Coupling-Regret Learning用train-only OOF losses为sample × target-region coupling
   policy提供counterfactual supervision；generic cross-validation/regret/routing不计创新；
3. novelty只允许落在完整
   `fixed past -> exact-prefix decoder -> point-to-global coupling spectrum -> counterfactual coupling policy -> no
   external strategy search`链条。

该版本中的CCRL已因two-stage teacher/student inconsistency在Step7A前退出paper core；当前active mainline为
`PCSD-CF direct control first`，见后文`CCRL Retirement And PCSD-CF Reset`。

[Returned Gate] D14-A0在neutral PCA64 carrier上完成5 datasets × 3 folds。carrier skill 4/5且numeric/split
invariants pass，但stable crossing 0/5、sample × bin oracle仅0.0586%、canonical-vs-random -0.1427%。exact
PCA64 + linear RRR evidence失败。

[Failure Correction] A0匹配factor params但未匹配rank-manifold effective DoF，且five-scale full-risk spread
最多0.04036%，没有形成足够function-level contrast。故方向级拒绝无效，归因为
`intervention_point_wrong + capacity_control_incomplete`。

[A1 Design Gate] A1不再调A0 rank，而以E2E grouped nonlinear head改变point/block/global hidden-bank sharing。
所有scales均经GELU正负对构造证明包含full-affine map；80个parameter/partition/affine与20个forward/gradient
local cases通过，最大parameter gap 0.1646%。neutral raw-history carrier是primary direction gate；A6-natural只作
paper-carrier sensitivity。由于A6 architecture/profile围绕global basis decoder形成，A6-negative不能拒绝scale。

[Completed Execution Order] neutral seed2021 -> neutral gate -> A6-natural sensitivity -> seeds2022/2023
dual-carrier confirmation。该D14-A串行协议已完成；D14-B后来在Step7A前被consistency audit取消。

[A1 Neutral Returned] 40/40 complete；function separation、carrier skill、crossing均5/5；oracle macro 7.6753%；
canonical-vs-random 0.8945%且5/5正。sequential row re-evaluation修复了official validation shuffle造成的artifact
alignment fault，未重训checkpoint。neutral只授权A6 sensitivity；single-seed不能直接形成paper claim。

[A1 A6 Returned] 45/45 complete；A6同样5/5 crossing，strict oracle 9.1504%，sample-over-bin 8.5429%，
contiguity 0.6661%且5/5正。neutral strict/sample分别6.9978%/6.7555%，dual-carrier problem evidence一致。
但train-selected/validation-best GroupedMLP相对A6-LBF H720 macro为-2.9435%/-1.6855%，所以fixed grouped head
不是method candidate。该single-seed结果当时只授权seeds2022/2023 confirmation，最终判定见下一段。

[A1 Three-Seed Confirmation] 新增170/170 runs完成，three-seed dual-carrier gate均pass。neutral/A6均为5/5
stable crossing；strict oracle为7.1107%/9.1259%，sample-over-bin为6.7948%/8.5990%。contiguity均为4/5 stable
datasets，故只能claim broad default而非universal law。A6 train-selected/validation-best GroupedMLP相对LBF仍为
-2.6886%/-1.4879%，所以problem confirmed但method仍未ready。该时点只授权D14-B Step4-6
source/theory/narrative audit；最终设计判定见下一段。

[D14-B1 Step4-6] 2026-07-16 external audit确认TimeFuse已覆盖sample-level adaptive fusion，TimeRouter已覆盖
oracle-best labels、context/CV/forecast features与nonlinear router，AME-TS已覆盖structural-prior KL。CCRL novelty
风险上调为high。理论上OOF squared-error differences可识别conditional relative risk，但expert-risk不是mixture
MSE；故CCRL收紧为`actual fused forecast loss + auxiliary cross-fitted centered-risk`。冻结两个gate：B-P检验
history+target predictability，B-C要求hybrid相对matched direct fusion、hard-oracle与in-sample controls有独立增量。
只授权Step7A local implementation；remote/method/test仍false。完整设计见
`analysis/stage_c_d14b_crossfit_regret_20260716/d14b_step46_source_theory_design_audit.md`。

[CCRL Retirement And PCSD-CF Reset] 后续training-consistency audit确认D14-B1需要独立fold × scale teachers、
只覆盖部分training samples的OOF labels，再监督architecture不同且持续更新的joint PCSD arms；因此存在
teacher-student mismatch、stale target和非最终图工程成本。CCRL在Step7A前取消并降为
`diagnostic_only_not_scheduled`。研究返回PCSD Step4-6，提出`PCSD-CF`：一个shared history-to-future mode
field经scope pooling产生全部point/block/global states，使用direct synthesis而非A6 residual，且以构造性映射
exact contain A6。external audit将DeepONet coordinate synthesis、PoU local operator mixture、Soft MoE与
TimeFuse direct fusion列为mandatory boundaries。narrative gate只对local implementation conditional pass；完整
报告见`analysis/stage_c_pcsd_native_reset_20260716/pcsd_cf_step46_source_theory_design_audit.md`。

[Frozen Boundary] neutral raw-history carrier是primary；A6 sensitivity也从头E2E joint training，但其negative只表示
carrier interface/profile不确认。最终paper effectiveness仍须matched E2E，不能用frozen replacement gap通过或拒绝。

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

## PCSD-CF Step 7A Local Gate: Passed

1. `PCSDCouplingFieldReadout`已接入A6-natural active readout，真实forward为
   `memory [B,C,P,D] -> z [B,C,R] -> modes [B,C,4,256] -> arms [B,C,5,720] -> policy -> [B,H,C]`；
2. five profiles × 13 horizons的65个direct prefix cases与5个真实model integration cases全部exact crop，max gap `0`；
3. arbitrary-A6 mapping在$R=768/1536/3072$的float32 maximum output/arm gap为
   `3.815e-6/2.384e-6`，float64为`3.109e-15/5.329e-15`；
4. scope Jacobian-sharing classes严格为`720/15/5/2/1`；canonical/random minimum arm NRMSE为
   `0.130247/0.023056`，equal-zero initial policy gap为`0`；
5. five-profile module与ETTh2真实Encoder-PCSD E2E two-step gradients finite/active；canonical/random trainable
   parameter values与shapes相同，只改变fixed partition buffers；
6. coupling-field core参数为A6 decoder的`3.0291-3.6184x`，含policy为`3.1006-3.7224x`，FLOP静态估算为
   `7.97-13.93x`，故Step7B必须保留dense capacity control与remote resource smoke；
7. decision=`step7a_local_pass_step7b_design_only_next`。该结果只通过implementation/theory contract，不是
   effectiveness evidence；remote、SC2、test均false。

## Next Concrete Action

完成Step7B experiment implementation/prelaunch audit：冻结A6/M0/five fixed/equal/static/direct/random/dense
matched arms的runner与effective-config contracts，补充same-run arm/policy diagnostics、checkpoint invariant checker
及单batch GPU memory/runtime smoke方案。正式3090 launch仍需单独授权，Contribution 2与test保持false。

## SC1-JAPO Step 7A: Production Gate Passed, Step 8 Authorized

1. `memory [B,C,P,D] -> h [B,C,PD]`是可逆reshape，不是pooling；D8失败不能归因于flatten本身；
2. 真正边界是`alpha_j = psi(d_j)^T A h`：自由A6 temporal table被descriptor-generated fixed separable
   operator替代；
3. 直接atom-to-patch cross-attention缺少future-support/history-patch canonical alignment，且B14与OFormer/GNOT/
   BasisFormer/TimePerceiver共同阻断该shortcut；
4. geometry-only linear expert mixture可代数吸收到一个更宽PAF；固定总rank时无新function class，扩rank时由
   capacity control解释，因此不推进；
5. 唯一保留候选为`SC1-JAPO`：free RGNB expert maps生成coefficients，joint gate同时读取history context与atom
   geometry；requested H只选择active atoms；
6. 令所有experts表示同一A6-equivalent RGNB map时，任意convex gate仍精确复现A6；4个$T$ cases最大误差
   `1.137e-13`，无dense bypass containment通过；
7. generic nonlinear decoder、MoE、geometry gating与step-specific representation均已有先例；novelty只允许落在
   joint history-atom operator、RGNB projectivity与multi-horizon domain-only contract的完整组合；
8. 22个prefix cases的shared coefficient/output最大误差`1.172e-13`；requested H只改变active set；
9. scalar construction得到$f(h)=h\tanh(h)$，affine second difference=`1.523188`，证明joint gate严格超出
   fixed affine PAF；geometry-only mixture仍以`8.882e-15`误差collapse为fixed operator；
10. exact containment不是initialization recipe：identical experts使router gradient严格为0，首版必须independent
    from-scratch initialization；
11. uniform/history-only/atom-only/PERM/RANDOM same-bank controls冻结；params差异不用于选择；
12. Step6冻结两个independent full-rank experts（$E=2,K=256$）与factorized multiplicative router（$G=32$）；
13. basis init使用$\sqrt{E/K}$恢复uniform-mixture initial variance；router output std=`0.01`，五profiles entropy
    min=`0.999855`、usage=`0.4980–0.5020`，所有joint gradients nonzero；
14. seven arms固定为A6/JOINT/UNIFORM/HISTORY/ATOM/PERM/RANDOM；所有JAPO arms paired expert bank；
15. seed2021先做35-run validation-only screen；严重失败早停，模糊结果只补seed2022，pass后补seed2023；
16. Step6 decision=`SC1-JAPO narrative_ready_step7a_local_implementation_only`；当时只授权本地编码；
17. production `JAPOReadout`实现六个same-bank modes，requested $H$只选择active atoms；
18. 210/210 prefix与35/35 gradient cases通过；最大gap分别为`4.768e-7`与patch rewrite `5.722e-6`；
19. 七arms Encoder hashes paired，六JAPO arms expert-bank hashes paired且within-bank experts独立；
20. runner/analyzer dry-run固定35 jobs、validation-only、full-H720 L1与best-val；
21. decision=`step7a_pass_remote_screen_authorized`；只授权seed2021 Step8，test/SC2继续held。
22. commit `90e4164`在3090 GPUs0/1/2完成seed2021 35-run matrix；output root固定为repo-external路径；
23. 35/35 artifacts、protocol、from-scratch paired initialization、prefix与patch invariants均通过，无numeric pathology；
24. JOINT vs A6 dense MSE macro=`-1.3754%`、0/5 positive；vs same-bank median macro=`-0.0780%`、2/5；
25. immediate-fail=false、provisional-pass=false，冻结decision=`seed2021_inconclusive_run_seed2022_only`；
26. 五个JOINT router normalized entropy均不低于`0.993263`，提示under-specialization，但单seed不足以区分
    optimization variance与exact design weakness，不能据此拒绝理论方向或临时改loss；
27. two-seed gate固定为先对每个dataset/arm求seed2021/2022 metric mean，再原样执行Step6 provisional threshold；
28. commit `3d37440`于`2026-07-15T11:37:11+08:00`在GPUs0/1/2启动并完成seed2022 unchanged matrix；
29. 70/70 audit通过；two-seed JOINT vs A6=`-1.2435%`、0/5，vs same-bank median=`-0.1175%`、1/5；
30. canonical geometry相对PERM/RANDOM仍为`+0.2229%/+0.1259%`，但JOINT不及UNIFORM/HISTORY/ATOM，
    `capacity_control_explains=true`；
31. 两seed router entropy均接近1，under-specialization复现；这支持exact head/intervention weakness，不构成
    projective conditional operator方向级否定；
32. decision=`two_seed_mean_fail_stop_and_attribute`：JAPO exact v1关闭，seed2023/test/SC2停止，回Step4
    source-informed redesign audit。
33. 2026-07-15系统复盘把正证据收紧为RGNB geometry、exact projectivity、local-support crossing与A6自由算子；
    把fixed tree、shared separable PAF和weak expert mixing关闭为exact designs，而非方向级否定；
34. 下一步为`SC1-D9 History-Support Operator Evidence Audit`：从A6 learned operator验证history-scale ×
    future-support coupling是否超越scale permutation/random controls。该实验预注册为`diagnostic_only`，通过也只
    授权Step4-5候选设计，失败则回Step2/3。
35. D9-A完成15/15 exact audits，Parseval max gap=`7.5381e-16`；macro rho=`0.173810`，positive
    datasets=`2/5`，permutation/random-basis gates=`1/5`与`0/5`，故primary hypothesis失败；
36. global-root与details之间存在15/15正向binary contrast，但它是post-hoc observation且details内部不单调，
    不能挽救D9。D9-B取消，回Step2/3设计D10 raw history–future scale identifiability。
37. D10 Step2/3 protocol已冻结：history DCT与future RGNB使用相同七组sizes，但所有cells进一步固定为16→16；
    binary 2×2与detail-only 6×6 monotone gates分离，paired history/future permutations阻断coordinate/capacity解释；
38. D10使用chronological train fit、20% temporal gap、train holdout与official validation；不读取test，不训练
    forecast model。当前只授权diagnostic implementation与remote evidence。
39. D10 artifacts/invariants完成：binary effect/direction/control=`2/5,0/5,2/5`；detail-monotone
    effect/control=`4/5,4/5`但best-count=`0/5`、mapping permutation=`2/5`；
40. decision=`raw_aligned_scale_not_supported_rollback_step2`。partial off-diagonal signal缺少跨dataset统一mapping，
    不得事后升级adaptive router；D9+D10共同关闭history-scale aligned routing，下一步审计future-component问题。
41. D11 external audit确认Time-o1已覆盖transformed label alignment与task-overload，FreDF/DBLoss覆盖
    frequency/component losses；generic component loss不能成为本项目创新边界；
42. D11 exact identity冻结为`sum_g J^T P_g v = J^T v`，直接分解output gradient而非错误地相加prefix
    component energies；MSE primary、L1 replication；
43. strict directional conflict必须negative dot；low positive cosine与norm ratio分别归为heterogeneity和magnitude
    imbalance。RGNB必须超过DCT/3 random controls才支持future-support-specific problem；remote前method/SC2/test false。
44. D11 accepted v2完成15 checkpoints：strict directional conflict=`0/5 datasets`，support-specific component
    gate=`2/5`，generic responsibility redistribution=`3/5`，magnitude=`2/5`；all invariants pass；
45. 所有validation MSE total paths/batches均为positive dot，same-component跨short/long negative fraction也为0；
    因此SC1 conflict-aware decoder问题为`hypothesis_false`，不是architecture或hyperparameter failure；
46. short measure对RGNB groups 5/6严格zero-gradient，long shares分别约`0.064107/0.020441`；该现象收紧为
    projective supervision coverage observation，只授权Contribution 2 Step1-3 prior-art/equivalence audit；
47. Time-o1、Loss Shaping Constraints与generic task weighting/sampling形成强overlap压力。未经Step1-3证明完整
    `measure -> inclusion probability -> unbiased/controlled risk -> non-equivalence -> falsifiable benefit`链条，
    不实现coverage normalization、MIPR、PCGrad或joint factorial。
48. post-D11 external audit确认：完整T720 label可用时raw horizon-measure risk可一次精确计算，generic
    importance sampling不构成必要机制；MIPR删除cross-scale terms但D11没有支持删除必要性，正式retired；
49. 新主线回到joint Step2-3：PRISM从nested prefix family推导risk-localized frame，CAPE以train-only
    cross-fitted predictions估计predictable covariance；两者先过D12，不直接实现method或读取test。
50. D12-v1暴露uniform normalized risk mismatch；v2以$s_x^2$对齐raw MSE并复用相同pilots，所有invariants
    通过但只1/5 datasets支持。CAPE与joint PRISM route关闭，D12-B取消；回滚Step2并重新开放两个slots。

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
