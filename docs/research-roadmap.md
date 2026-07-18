# Research Roadmap

## Current Cursor

| Field | Content |
| --- | --- |
| `stage` | `StageC-UVHF` |
| `current_step` | `SC1-SIFF-v2-CCSF-v1-preimplementation` Step7B temperature-pilot prelaunch pass；pilot launch next |
| `active_question` | 为什么healthy multi-arm SIFF未把conditional headroom转成超过A6_MEASURE/independent的fused forecast？ |
| `active_candidates` | v1 frozen performance-near parent；CCSF v1 temperature-pilot ready；PCSD/PCC closed；CTD paused |
| `future_validation_suite` | ETTh1, ETTh2, ETTm1, ETTm2, Weather |
| `active_protocol` | `analysis/stage_c_siff_ccsf_step7b_prelaunch_20260718/prelaunch_report.md` |
| `method_implementation` | Step7A 18/18；Step7B prelaunch 14/14；pilot remote=true，formal Phase A/test=false |
| `rollback_point` | pilot不完整只补缺失run；numeric pathology回Step7A；完整选择后进入formal-candidate prelaunch audit |

## SIFF_EQUAL Attribution Step 6 Freeze

`SIFF_EQUAL` 已从“性能正向但归因阻塞”推进为 `SC1-SIFF-v2-EQ-ATTR-v1`：

1. 10-arm matrix同时包含`A6_FULL/A6_MEASURE`、`PCSD_MEASURE/PCSD_EQUAL`、
   `SIFF_MEASURE/SIFF_EQUAL`和四个EQUAL-context SIFF controls；
2. Phase A固定为5 datasets × 10 arms × seed2021，共50 runs与200个standard-horizon test cells；
3. `SIFF_EQUAL`必须逐项超过A6_FULL、A6_MEASURE、PCSD_EQUAL、constant、permuted、Q1-wide与independent；
4. 统一结论分为paper-facing effectiveness、matched mechanism attribution、internal mechanism health与failure
   attribution四层，内部健康度不得挽救negative performance gate；
5. Step6 16/16与Step7A 13/13 categories已通过；下一步只做Step7B prelaunch，remote/test仍未授权。

若`SIFF_MEASURE`未超过`PCSD_MEASURE`但EQUAL comparison通过，claim只能收紧为equal-skill-trained scale
field。只有Phase A七项hard comparisons全部通过，才允许seeds2022/2023 confirmation。

详见`analysis/stage_c_siff_equal_attribution_step6_20260718/step6_attribution_protocol.md`。

Step7A现已完成50-job CLI wiring、35个unique model constructors、10条objective gradient paths、matched
parameter accounting与scale-component intervention artifact。新增component统计固定policy并逐个移除
scale-field component，保存raw-scale `full - ablated`；它是non-additive diagnostic，不改变production forward。
checkpoint evaluator与四层analyzer smoke均通过，remote runner在authorization=false时固定拒绝launch。

Decision=`step7a_local_pass / step7b_prelaunch_next`。详见
`analysis/stage_c_siff_equal_attribution_step7a_20260718/step7a_implementation_gate_report.md`。

Step7B prelaunch进一步以9/9 categories冻结正式授权：50 runs/200 test cells、four-horizon validation
checkpoint、official test primary gate、single formal access与confirmation hold。3090三张GPU预检均约15 MiB
used、无训练进程；远程历史dirty CSV与本次路径不重叠并必须保留。下一步为pull、dry-run、resource smoke与后台
launch。详见`analysis/stage_c_siff_equal_attribution_step7b_prelaunch_20260718/prelaunch_report.md`。

Step8已于2026-07-18 11:12:03在3090 GPU0/1/2启动。remote dry-run与Weather-SIFF_EQUAL resource smoke先行
通过，首批Weather的A6_FULL/A6_MEASURE/PCSD_MEASURE正常进入epoch 3/3/1。当前不高频值守；50/50完成后同步
完整test metrics、invariants与component artifacts，再执行四层Step9。详见
`analysis/stage_c_siff_equal_attribution_step8_remote_20260718/remote_launch_record.md`。

## SIFF_EQUAL Attribution Step 9 Result

Phase A已50/50完成，200/200 standard-horizon test cells、50/50 invariants与paired initialization均通过。
七项hard comparisons中：

1. SIFF_EQUAL超过A6_FULL `+1.6436%`与PCSD_EQUAL `+0.5906%`，但低于A6_MEASURE `-0.2366%`；
2. ordered超过constant `+0.9393%`、permuted `+0.3959%`、Q1-wide `+1.1619%`，但相对independent仅
   `+0.2580%`，未达冻结`0.3%`；
3. internal health 7/7，通过finite、projectivity、oracle、diversity、entropy与component-use gates。

结论不是mechanism未执行，而是simple objective与independent-scope controls解释了paper-facing gain。exact v1关闭，
confirmation保持false，回Step4重新研究conditional headroom到learned fusion的转化问题。详见
`analysis/stage_c_siff_equal_attribution_step9_20260718/step9_four_layer_diagnostic.md`。

## Post-Step9 Candidate Freeze And Step4 Improvement Audit

用户决定把`SC1-SIFF-v2-EQ-ATTR-v1`保留为本阶段最接近论文级performance的候选。该决定只改变portfolio
status，不推翻Step9 attribution failure：v1固定为`frozen_performance_near_candidate / performance_partial_pass`，
其source commit、config/profile hash与五个checkpoint hashes记录于
`configs/stage_c_siff_equal_attribution_v1_candidate_freeze.json`。

复用现有artifacts的diagnostic表明，SIFF_EQUAL policy best-arm match为29.24%、skill alignment为0.0277，
policy-weighted expected arm loss相对uniform仅+0.0762%；two-fold static convex fusion却比learned fusion高
2.2112%且8/10 dataset-fold为正。bounded affine相对convex只多0.1203%，因此首要问题是router看不到/学不到
relative arm competence，而不是softmax convex hull太窄。

决策：保留v1作为当前candidate与mandatory parent reference；研究正式进入Step4 source-informed redesign，
优先将`arm-contrast-aware policy + synchronous competence calibration`送入Step5 theory feasibility，并同时审计
`A6_MEASURE anchor containment`。generic deeper MLP、top-k、entropy loss、直接扩Q/rank/scales不进入下一步。
详见`analysis/stage_c_siff_candidate_step4_source_audit_20260718/source_informed_improvement_audit.md`。

## CCSF Step 5 Theory Feasibility

代码审计确认旧PCC已实现same-forward detached arm-error route supervision，因此Step4中“同步calibration”不能作为
未测试创新。CCSF核心收紧为target-free arm-contrast information path；calibration只作为co-designed weak
supervision，并必须设置loss-only control。

预冻结contrast diagnostic使用5 datasets × 2 row folds。contrast相对coordinate-only expected arm MSE
`+1.8348%`、10/10 folds正；相对shuffled `+1.7085%`、10/10；best-arm accuracy相对existing policy提高
15.31 percentage points。5/5 gates通过，支持contrast identifiability，但仍是test-derived offline evidence。

Step5冻结的provisional operator保留v1 logits，并用scope-shared scorer读取dimensionless pointwise/scope-group
contrast；correction为零时包含v1。full T computation + prefix crop给出strict projectivity。旧PCC teacher除以
cross-arm std，会放大near-equal arms的噪声；provisional relative-regret teacher改按mean error归一化，并用
`1-normalized entropy`降低ambiguous supervision。temperature grid只作geometry evidence，禁止从test选择。

显式A6 anchor branch因capacity/ensemble confound退出method，A6_MEASURE保留为mandatory external baseline。
Decision=`conditional_theory_pass_to_step6`；Step6必须冻结v1、loss-only、architecture-only、full、shuffled/zero、
generic capacity、independent与A6 controls，当前implementation/remote均false。详见
`analysis/stage_c_siff_ccsf_step5_theory_20260718/step5_theory_feasibility.md`。

## CCSF Step 6 Narrative And Control Gate

候选版本冻结为`SC1-SIFF-v2-CCSF-v1-preimplementation`，并通过5/5 static gates。method保留v1 logits，使用
scope-shared六维target-free contrast descriptor产生零初始化logit correction；完整T=720计算后才crop，requested
horizon与benchmark bins不进入model。新增scorer为2,881参数，ordered/independent CCSF在五dataset上的总参数gap
均低于0.5%。

归因矩阵以`SIFF-v1/CCSF × EQUAL/RELCAL` 2×2为核心，另含A6_MEASURE、old standardized teacher、
zero-contrast same-capacity、permuted-contrast与matched-independent controls，共10 arms。temperature只允许从
`{0.05,0.1,0.25}`通过五dataset共同validation macro score选择，不允许per-dataset或test选择；该15-run pilot尚未
授权。正式Phase A冻结为50 runs/200 cells，confirmation为100 runs/400 cells，但remote/test/confirmation均false。

Decision=`step6_pass_step7a_local_only`。下一步只实现production forward/objective/control adapters、prefix/gradient/
parameter tests与remote refusal gate；10项hard comparisons全部通过前不得形成joint claim。详见
`analysis/stage_c_siff_ccsf_step6_20260718/step6_narrative_control_gate.md`。

## CCSF Step 7A Local Implementation

production path已实现`arms [B,C,S,T] -> contrast [B,C,T,S,6] -> shared 43-64-1 correction -> v1 logits +
correction -> projective fusion`。新增参数严格2,881；ordered/independent总参数gap最大0.3833%。相同seed的v1与
三个ordered CCSF controls具有相同base hash和initial forecast，gap为0。true/zero/permuted descriptors及nonzero
intervention、contrast-to-arm gradient均通过，四readout的prefix gap均为0。

relative/standardized calibration objectives、50-job adapters、30 constructors、10 gradient paths、two-step correction
optimization与diagnostic tensor contract全部就绪。local gate=18/18。`tau=0.1`只作synthetic smoke；正式shared
temperature没有选择。remote template在当前authorization下exit 3，即使手工改authorization也因Step7B未冻结而
exit 4。

Decision=`step7a_local_pass_step7b_next`；下一步先设计Step7B validation pilot/prelaunch boundary，不得直接启动
pilot、remote或test。详见
`analysis/stage_c_siff_ccsf_step7a_20260718/step7a_implementation_gate_report.md`。

## CCSF Step 7B Temperature-Pilot Prelaunch

Step7B把temperature selection与formal effectiveness严格拆开。pilot固定为唯一`ccsf_relcal` arm、5 datasets、
`{0.05,0.1,0.25}`和seed2021，共15 runs/60 validation cells。checkpoint仍由H96/H192/H336/H720 validation MSE
平均选择；shared temperature则由完整5×4 validation macro MSE选择，并列取更大temperature。禁止per-dataset、
per-horizon或test-informed selection。

本地prelaunch gate=14/14；runner dry-run=15 jobs，synthetic tie选择器通过。pilot weights/checkpoints不会进入formal
comparison；实际pilot选定temperature后，必须生成新的formal candidate version并重新审核50-run Phase A的runner、
evaluator、internal artifacts与test metadata。当前只授权validation pilot remote，formal Phase A、official test与
confirmation仍为false。Decision=`step7b_temperature_pilot_prelaunch_pass`。详见
`analysis/stage_c_siff_ccsf_step7b_prelaunch_20260718/prelaunch_report.md`。

## Fair Re-audit Step 9–10 Result

70/70 runs与280/280 test cells通过protocol。结论发生三层分化：

1. PCSD_DIRECT相对A6为`-0.8562%`，exact direct candidate继续关闭；内部诊断仍保留representation与
   credit-starvation线索；
2. PCC相对prior仅`+0.0806%`，且在SIFF上相对EQUAL为`-0.2663%`，exact PCC关闭；
3. SIFF_EQUAL相对PCSD_EQUAL为`+0.5906%`并通过预注册gate；相对A6为`+1.6436%` MSE、
   `+0.9084%` MAE，是当前最佳carrier；
4. SIFF+PCC相对A6虽为`+1.3812%`并通过performance gate，但PCC降低SIFF_EQUAL性能，不能形成双贡献归因。

旧best-H720下SIFF_EQUAL vs PCSD_EQUAL四-horizon validation为`-2.3897%`，本次four-H checkpoint下变为
validation `+0.1469%`、test `+0.5906%`。旧SIFF failure包含明确checkpoint假失败。但SIFF在PRIOR/PCC下不稳，
且未超过independent control的macro gate，因此只能标记`partial_pass_attribution_blocked`。

下一步不是直接补seed，而是回Step6冻结EQUAL-context controls与`A6_MEASURE_ONLY`。详见
`analysis/stage_c_fair_reaudit_v1_20260717/step9_10_conclusion.md`。

补充internal mechanism audit确认：PCSD_DIRECT arms的loss CV为`36.11%`，同一run最佳固定arm仍比fused差
`18.05%`；PCSD_PCC/SIFF_PCC的row-bin oracle headroom达到`16.17%/18.17%`，但policy entropy为
`0.965/0.975`，说明PCC增加了未利用的conditional headroom而非形成有效routing。SIFF_EQUAL则同时保持
`+1.6436%` test gain、`6.39%` oracle headroom与非零arm差异。MCCA没有进入本次14-arm矩阵，其状态修正为
`historical_validation_negative / fair_test_not_reaudited / inactive`。

## Test-Primary Fair Historical Re-audit

用户决定暂停CTD，先按最新规则公平评估PCSD、PCC与SIFF。规则现明确：validation只选择checkpoint与支持
debug/diagnostic；正式机制pass/fail统一看official test的H96/H192/H336/H720。历史best-H720 checkpoints不直接
复用，A6与所有candidate/control均from-scratch重训，并由四horizon validation MSE平均选择checkpoint。

冻结矩阵包含14 arms × 5 datasets × seed2021，共70 runs与280 test cells。它同时覆盖A6→PCSD、PCC相对
equal/prior controls、SIFF在equal/prior/PCC下相对PCSD，以及constant/permuted/Q1-wide/independent controls。
Step7A通过70 CLI、40 model construction、prefix identity、profile hash与paired Encoder initialization共9类
gate；Step8于2026-07-18 00:52完成70/70，Step9/10结论见上节。

由于test已成为统一benchmark decision surface，所有后续candidate均标记`test_informed`；项目不再声称test是
untouched holdout。完整协议见
`analysis/stage_c_fair_reaudit_v1_20260717/preregistered_protocol.md`。

## SIFF/MCCA Step 9–10 Result And Measure Rollback

55/55 new validation runs与25/25 matched references通过protocol audit。SIFF architecture main effect为
`-1.5015%`、2/5 datasets；MCCA相对same-mass PCC为`-0.0250%`、2/5；joint相对A6为
`-0.5621%`、4/5。exact SIFF-v1/MCCA-v1均未通过Phase-A，confirmation、Phase B与test保持false。

controls给出两个局部正信号：ordered SIFF相对permuted为`+1.1177%`、5/5；MCCA transport相对pointwise为
`+0.4736%`、4/5，capability marginal相对uniform OT为`+0.1182%`、5/5。但ordered未超过Q1-wide/
independent macro gate，PCSD MCCA相对PCC为0/5，因此这些结果不能挽救exact methods。

failure attribution发现SIFF不是全面fit失败：ETTm2 SIFF+MCCA相对PCSD+MCCA在H1为`-669.49%`，H720却为
`+0.6013%`。dense all-prefix AUC等价于target weight
$w_t=T^{-1}\sum_{H=t}^{T}H^{-1}$。code audit随后确认PCSD/SIFF coupling arms的fused training loss已经使用
exact harmonic target measure（L1），所以不能归因于flat training；未决问题是L1-vs-MSE与H720 checkpoint。

当时decision：停止exact SIFF-v1/MCCA-v1 development；因>100%局部pathology，不作scale-coordinate方向级否决，
MCCA只保留transport/marginal ingredients。按后续test-primary治理，该段不是formal test rejection：SIFF已公平
复评并修正为partial pass，MCCA尚未公平复评且保持inactive。回滚Step4执行`SC-D16` external-first measure audit。

source audit现已完成：NeurIPS 2024 ElasTST直接覆盖harmonic horizon reweighting与weighted checkpoint；
Loss Shaping/QDF进一步覆盖future-step weighting。standalone PHMA narrative fail，新增HR arms也因现有training
已exact harmonic而冗余。只保留`SC-D16-CTD diagnostic_only`：下一步Step5/6冻结ETTm2上的
PCSD/SIFF/constant/Q1四条per-epoch trajectories；implementation/remote/test仍false。

详见
`analysis/stage_c_post_pcc_step7b_seed2021_20260717/step9_10_result_and_failure_attribution.md`。
source audit见
`analysis/stage_c_post_pcc_step4_measure_audit_20260717/source_informed_measure_audit.md`。

## Paper-Facing Four-Horizon Reevaluation

项目规则现固定为validation development与official test main/ablation均使用H96/H192/H336/H720；dense curve/AUC
默认只作mechanism diagnostic与unified-horizon补充证据。旧Step7B artifacts回溯重算后，SIFF architecture为
`-2.3509%`、8/20 cells、2/5 datasets；MCCA为`-0.1357%`、7/20、1/5；joint vs A6为`-1.3325%`、
14/20、4/5。三项均fail，旧decision不反转。

SIFF architecture按horizon为`-6.3186/-2.6027/-1.0522/+0.5698%`，对应H96/192/336/720。说明问题不是
dense H1指标单独制造，而是明确的short/mid-to-long tradeoff。该audit为validation-only retrospective screen，
继承best-H720 checkpoint且未重选epoch；test=false。详见
`analysis/stage_c_post_pcc_standard_horizon_reevaluation_20260717/standard_horizon_reevaluation_report.md`。

## SC-D16-CTD Step 5/6 Design

diagnostic已冻结为ETTm2 pathology-locus四条trajectory：
`PCSD_EQUAL/SIFF_EQUAL/SIFF_CONSTANT_EQUAL/Q1_WIDE_EQUAL`。四arms共享seed2021、equal-skill exact
harmonic-L1 objective、batch32与learning rate $10^{-4}$，固定运行20 epochs而不由H720 early stopping截断；
每epoch一次full validation forward生成paper-facing H96/H192/H336/H720与dense H1..720 MSE/MAE。

同一trajectory离线选择`best_standard_mse`、`best_h720_mse`、`best_dense_mse_auc`与
`best_dense_mae_auc`。只有SIFF best-standard同时移除H1>100% pathology、在四标准horizon超过
PCSD/constant/Q1且不以long-bin >1%退化换取，才允许five-dataset validation confirmation。若所有checkpoint
rules下H1 ratio仍>2，回Step2关闭scale-field方向。

decision=`diagnostic_design_pass_step7a_local_only`；下一步实现per-epoch evaluator与selected-state retention，
remote/test仍false。详见
`analysis/stage_c_d16_ctd_step56_20260717/step56_diagnostic_design.md`。

[Pause] 2026-07-17用户明确暂停CTD。上述design与rollback边界保留，但Step7A、remote与test均不继续；当前cursor
已切换到PCSD/PCC/SIFF公平重评估。

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
   修正fan-in初始化后为`0.131493/0.023079`，equal-zero initial policy gap为`0`；
5. five-profile module与ETTh2真实Encoder-PCSD E2E two-step gradients finite/active；canonical/random trainable
   parameter values与shapes相同，只改变fixed partition buffers；
6. coupling-field core参数为A6 decoder的`3.0291-3.6184x`，含policy为`3.1006-3.7224x`，FLOP静态估算为
   `7.97-13.93x`，故Step7B必须保留dense capacity control与remote resource smoke；
7. decision=`step7a_local_pass_step7b_design_only_next`。该结果只通过implementation/theory contract，不是
   effectiveness evidence；remote、SC2、test均false。

## PCSD-CF Step 7B Step9/10 Result

seed2021 validation-only screen于17:26:52正常结束，60/60 protocol/artifacts与paired initialization通过。DIRECT
相对A6 macro -1.5833%、0/5，method gate失败；相对dense matched +2.3492%、5/5，相对random +0.4499%、3/5，
capacity/random explanations排除。25/25 DIRECT same-run scope arms相对独立fixed E2E training退化，median
89.95%，failure attribution=`design_fault_suspected_joint_credit_starvation`。SC1不跑confirmation，回Step4保留
training-aware representation question；SC2-PCC完成Step2-5 source/theory audit，15/15 local cases通过，下一步只做
Step6 control matrix、optimization与rollback design。Contribution-2 implementation、test与confirmation seeds保持false。

## PCSD-CF Milestone Test Audit: Exact V1 Rejected

2026-07-16冻结的12 arms × 5 datasets official test audit完成60/60，checkpoint hash与no-retraining invariants全过。
DIRECT相对A6为-1.3994%、1/5；相对equal/static/dense/random的macro gain均为负。validation上的dense advantage
发生test reversal，但A6 primary gate在两split均失败，exact PCSD-CF-v1因此在Step10关闭。

same-run oracle test headroom为+2.0197%、3/5，且25/25 DIRECT arms仍under-trained，median 90.6647%。按预注册
decision map归为`test_fail_with_arm_headroom`：PCC可进入test-informed Step6 design，但不得据此宣布training机制成立，
也不得按test dataset/horizon调参。下一rollback point是PCC是否超越measure-only、equal-skill、capability-only、
route-only及generic balancing controls。

## SC2-PCC Step 5 Theory Feasibility

plain fused arm/router与PCC附加credit的四个gradient identities均以float64 autograd验证，最大误差
`5.20e-18`；dense-prefix measure identity误差`4.44e-16`，full-domain prefix crop gap为`0`。history × target
crossed synthetic policy达到capability KL `1.50e-11`与argmax accuracy `1.0`。这只证明output-level skill floor、
router credit与projective measure可同时成立；shared-parameter gradient cancellation、moving target、arm
homogenization与真实capability predictability仍未解决。decision=`conditional_pass_step6_design_only`；下一步必须
冻结`MEASURE_ONLY/EQUAL_SKILL/CAPABILITY_SKILL_ONLY/ROUTE_ONLY/full PCC`等controls后才可讨论Step7A。

## SC2-PCC Step 6 Source-Informed Redesign

fresh external search发现time-series Expert Loss Integration已直接训练expert losses，ICLR 2026 graph MoE也已使用
negative per-expert loss teacher、gate KL与uniform warm-up。因此原pointwise PCC-v0不能承担Contribution 2，降为
closest-prior control。研究显式回滚Step4/5后提出test-informed `PCC-v1-TI`：先计算全部$H=1..720$的scope
prefix risks，再用harmonic prefix-target incidence把capability输运为不含requested-H的target-coordinate credit。

exact nested-risk/transport identity误差`0`，19/19 local design cases通过；全局固定continuous schedule与9 new arms ×
5 datasets validation matrix已冻结。narrative gate仅conditional pass：full transport必须超过A6、plain、pointwise v0
与pointwise prior composition，并显著恢复25个arm pairs；否则按generic control、readout ceiling、shared-gradient或
numeric pathology分别回Step4/5。decision=`step6_pass_step7a_local_authorized`，remote/test false。

## SC2-PCC-v1-TI Step 7A Local Implementation

`layers/PCC.py`已实现nine frozen modes、dense-prefix measure、pointwise/prefix capability、harmonic transport与continuous
schedule；TimeAlign只增加显式training-details path，默认三元组、parameter count与inference output不变。real PCSD batch
中的raw-scale arm fusion gap为`8.88e-16`，arbitrary prefix gap为`0`，五个scope auxiliary gradients均非零。

35/35 gates通过：vectorized/direct loop最大差`2.22e-16`、transport identity gap`0`、nine decompositions 9/9、
adapter optimizer step finite且只访问train/val。该结果只建立implementation correctness，不建立effectiveness。
decision=`step7a_pass_prelaunch_audit_next`；下一步仅做45-run runner/analyzer/resource audit，remote/test仍false。

## SC2-PCC-v1-TI Step 7B Prelaunch

45-run Phase-A matrix已按dataset-major slow-first顺序固定：nine objective modes × Weather/ETTm1/ETTm2/ETTh1/
ETTh2。45个production CLI contracts、frozen hashes、endpoint-mode initialization pairing、validation-only authorization、
shared-gradient evaluator与analyzer synthetic smoke均通过；prelaunch categories为8/8。

新runs只训练PCSD DIRECT架构的nine objectives；A6/plain DIRECT/dense/five fixed scopes复用冻结seed2021 references，
不重训。analyzer同时执行performance、pointwise-prior specificity、25-pair arm recovery、pairwise NRMSE retention、policy
collapse与best-val shared-gradient diagnostics。decision=`step7b_prelaunch_pass_remote_seed2021_authorized`；只授权45-run
validation Phase A，test、confirmation seeds与conditional Phase B继续false。

2026-07-17 remote dry-run和GPU0 resource smoke通过，commit `282b96c`已在GPU 0/1/2后台启动45-run
matrix。单次startup audit确认三个Weather jobs进入training、显存占用正常、runner与workers存活。当前进入11-step
Step8，停止长期值守；45/45返回后再进入Step9/10。launch provenance见
`analysis/stage_c_sc2_pcc_step7b_prelaunch_20260717/remote_launch_record.md`。

## SC2-PCC-v1-TI Step9/10 Result And Step4 Rollback

45/45 PCC runs与15/15 references通过本地复算。full PCC相对A6 macro `+0.9627%`、3/5，相对plain
`+2.4927%`、5/5，25/25 arm pairs改善且median relative reduction `98.01%`。但相对closest prior composed
仅`+0.1050%`，低于`0.2%`门槛；five-dataset pairwise NRMSE retention仅`20.57%-41.13%`，低于50%。

formal decision=`generic_or_pointwise_control_explains_return_step4`。`EQUAL_SKILL`已经解释full PCC相对A6 gain的
88.90%，说明arm recovery主要通过same-label homogenization完成；harmonic transport未形成独立performance/horizon
signature。exact v1不进入Phase B、seeds或test。

Step4 external-first audit进一步确认generic expert loss、structural routing prior、heterogeneous experts、orthogonality/
variance diversity与balanced OT assignment均已有直接prior art。provisional next pair为：(1) coupling scale作为internal
coordinate生成scope-conditioned history modes；(2) projective target-measure rows与scope-skill-budget columns约束的
competitive credit。两者只进入Step5 proof，不实现、不remote。详见
`analysis/stage_c_sc2_pcc_step7b_seed2021_20260717/step9_10_result_and_failure_attribution.md`与
`analysis/stage_c_post_pcc_step4_redesign_20260717/source_informed_redesign_audit.md`。

## SIFF/MCCA Step5 Theory Feasibility

10/10 float64 cases通过：SIFF的Q1 containment gap `3.5527e-15`、prefix gap `0`，current constant-coordinate
scope gap `0`而SIFF witness gap `1.0`；MCCA row/column marginal gaps分别`6.25e-17/1.11e-16`，crossed
best-scope mass相对uniform增加`0.6667`，dominant-arm case minimum scope mass `0.2`，skill/router gradients均finite。

该pass只证明algebraic feasibility。generic wider head、heterogeneous experts与BASE/SSR-style OT仍可能解释完整primitive，
production rank/parameter/marginal也未冻结。decision=`step5_theory_pass_step6_source_design_next`；下一步Step6，不授权
implementation、remote或test。详见
`analysis/stage_c_post_pcc_step5_theory_20260717/step5_theory_feasibility.md`。

## SIFF/MCCA Step6 Source-Informed Design

external-first primary-source audit确认：DirMO/Stratify已覆盖固定block-size strategy；CViT/conditioned neural fields/
HyperDeepONet覆盖coordinate-conditioned operators；BASE/Expert Choice/Selective Sinkhorn覆盖balanced assignment与
expert capacity。因此SIFF不能claim coordinate field primitive，MCCA不能claim OT/Sinkhorn或anti-starvation primitive。

production SIFF-v1固定$Q=2,D=4,K=256$：`hidden [B,C,R] -> component modes [B,C,Q,D,K] -> log-scale
basis [S,Q] -> scale-indexed modes [B,C,S,D,K] -> existing scope pooling/shared synthesis`。Q1 exact containment与
same-parameter constant control成立；Q1-wide和independent-scope integer-rank controls在五profiles上的最大parameter
gap为`0.3893%`。params只作attribution，不作profile/candidate选择。

MCCA-v1把batch-channel-target rows的projective mass与scope columns做log-domain I-projection。column marginal被定义为
current PCC在同一progress下给予每个scope的**完全相同总skill mass**，所以方法差异只剩credit放置位置：PCC逐target
均匀撒floor，MCCA在global coverage下竞争分配。float64/float32 marginal gap为`3.86e-10/1.04e-7`，same-mass gap
`5.55e-17/2.98e-8`，22/22 design cases通过。

Phase A冻结`PCSD/SIFF × EQUAL/PCC/MCCA`的$2\times3$ factorial，加SIFF constant/permuted、Q1-wide、independent、
dense matched与pointwise/uniform-OT controls。narrative gate=`conditional_pass`，只授权Step7A local implementation；
remote、confirmation和test均false。详见
`analysis/stage_c_post_pcc_step6_design_20260717/step6_source_method_control_design.md`。

## SIFF/MCCA Step7A Implementation and Step7B Prelaunch

production `SIFFCouplingFieldReadout`、MCCA objective与dense measure-only control已落地。Step7A 36/36通过：
Q1/A6 containment gap `0`，constant collapse `3.55e-15`，float32 MCCA marginal gap `4.47e-8`，same-mass
PCC gap最大`2.78e-17`，arm/policy gradients均非零。该证据只完成implementation/numeric gate，不等于method
effectiveness。

Step7B prelaunch 8/8通过；冻结11 new arms × five datasets = 55 runs，复用未改变的`PCSD_EQUAL/PCSD_PCC`
及A6/PCSD/dense references。seed2021 validation-only remote现已授权；test、confirmation与conditional Phase B仍为
false。remote resource smoke已通过，55-run matrix于`2026-07-17T14:59:22+08:00`从commit `7a9e5c7`启动；
artifacts返回后进入Step9 attribution，不允许依据partial runs改设计。
详见`analysis/stage_c_post_pcc_step7a_local_20260717/step7a_implementation_gate_report.md`和
`analysis/stage_c_post_pcc_step7b_prelaunch_20260717/prelaunch_report.md`。

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
