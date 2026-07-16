# StageC D14 Output-Coupling Granularity Audit

## Status

| Field | Value |
| --- | --- |
| `stage` | `StageC-UVHF` |
| `current_step` | D14-B1 Step 4-6 conditional pass；Step7A local implementation next |
| `role` | `diagnostic_only` |
| `active_candidates` | `SC1-PCSD` problem-supported；`SC2-CCRL` high-risk diagnostic candidate |
| `method_training` | false |
| `remote_training` | D14-A 255/255 complete；D14-B1 local=true，remote=false，paper method=false |
| `test_access` | false |
| `primary_carrier` | neutral train-only raw-history carrier |
| `sensitivity_carrier` | A6-natural E2E architecture，只有neutral problem pass后才授权 |
| `rollback` | neutral valid fail -> Step 2 close pair；neutral invalid -> repair diagnostic；A6 fail不能拒绝scale hypothesis |

## What We Plan To Test

D14回答两个按顺序执行的问题：

1. `D14-A`：在相同history information与matched capacity下，point/block/global output coupling scopes是否存在
   跨sample、跨future region的稳定performance crossing？
2. `D14-B`：若crossing存在，最佳coupling scope能否仅根据inference可见history与target coordinate预测？

D14不实现PCSD/CCRL，也不访问test。

## Why It Matters

如果一个fixed coupling scale稳定支配其余scales，adaptive coupling只是多余capacity。若oracle headroom存在但
history无法预测，router在inference时也无法兑现该headroom。只有“scale crossing + pre-target predictability”
同时成立，PCSD/CCRL才有进入formal Step 4-6的problem evidence。

## Theory And Diagnostic Boundary

对ordinary Frobenius-regularized multi-output ridge，separable squared loss可能使逐target与joint output得到等价
solution；这种probe无法测试coupling。D14必须使用coupling scope确实改变parameter sharing/function class的
matched head family，首选：

- blockwise reduced-rank regression with matched degrees of freedom；或
- shared-trunk/block-token head with matched total parameters and optimization。

最终family在implementation前需完成source/code audit与synthetic equivalence test。

## Dataset And Split Contract

- datasets：ETTh1、ETTh2、ETTm1、ETTm2、Weather；
- seeds/folds：2021、2022、2023或三个预注册chronological train folds；
- natural profiles：`configs/stage_c_five_dataset_natural_profiles.json`；
- train：feature fitting、cross-fitting与hyperparameter selection；
- validation：唯一decision surface；
- test：禁止读取；
- normalization、feature construction与rank/budget selection全部train-only。

## Common Carriers

### C0 neutral carrier — primary

从normalized raw history构造train-only fixed features；不得使用与任一candidate head共同训练的representation。
可选PCA/DCT/random projection须固定维度并由train fit。

### C1 A6-natural architecture carrier — sensitivity only

复用五数据集natural profile的`timealign-token-mlp` architecture，但不加载A6 checkpoint；encoder与candidate
decoder从头E2E joint training。由于这些profiles与encoder本来围绕global basis decoder形成，即使candidate在
A6上失败，也可能是carrier/head interface或profile inductive bias。故A6-negative不能拒绝scale hypothesis；
A6-positive也不能在neutral invalid时单独通过problem gate。

## D14-A Coupling-Spectrum Headroom

### Frozen A0 implementation (2026-07-15)

source-informed audit排除了ordinary separable multi-output ridge：它与逐target ridge共享normal equations，不能
检验output coupling。A0因此冻结为neutral PCA64 carrier上的closed-form blockwise reduced-rank regression：

$$
\hat Y_{B_j}=XW_j,\qquad \operatorname{rank}(W_j)\le r_s.
$$

scales/ranks固定为`1/1, 48/28, 144/45, 360/55, 720/60`，对应factor params分别为
`46800/47040/46800/46640/47040`，最大relative gap为0.513%。point arm必须等价于unstructured full-affine；
intermediate scales同时运行shifted contiguous与random partition controls。三chronological folds各用512 fit、
128 train-calibration与256 official-validation windows；fit/cal observation gap至少1440。五数据集全部运行，
不使用test。

本地synthetic gate已验证point equivalence、rank上界、partition disjoint cover、parameter budget和analyzer gate
logic。对应实现与定义见：

- `configs/stage_c_d14a_output_coupling_granularity.json`；
- `analysis/stage_c_d14a_output_coupling_granularity_20260715/d14a_source_and_design_audit.md`；
- `docs/code-explanation/stage-c-d14a-output-coupling-granularity.md`。

此前generic A0-A9列表由上述exact family取代；A6 sensitivity不属于本次首轮launch。A0 positive最多授权返回
Step 4-6，不能直接授权D14-B或paper method；D14-B仍需按本protocol的串行gate单独授权。

### A0 returned result and correction

A0完成5 datasets × 3 folds：carrier skill 4/5、numeric/split invariants全部通过，但stable crossing 0/5、
sample × bin oracle macro gain仅0.0586%、canonical-vs-random为-0.1427%。因此exact A0 statistical gate失败。

failure attribution同时发现：预注册匹配的是factor storage params，不是rank-manifold effective DoF；五scale的
aggregate risk spread也只有0.000004%-0.04036%，未形成足够function-level intervention contrast。故A0只能关闭
当前PCA64 + linear RRR probe，方向级拒绝标记`design_fault_suspected`。D14-B取消；A1实现前必须先通过
effective-DoF matched structured-shrinkage source/theory audit。完整报告见
`analysis/stage_c_d14a_output_coupling_granularity_20260715/d14a_result_and_failure_attribution.md`。

### A1 repaired nonlinear E2E diagnostic

A1改用真正改变parameter-sharing topology的grouped nonlinear head。对carrier
$h\in\mathbb R^R$与scale $s$的每个future group $B_g$：

$$
z_g=\operatorname{GELU}(hA_g+a_g),\qquad \hat y_{B_g}=z_gB_g+b_g.
$$

$s=1$为每个future target独立hidden bank，$s=720$为全future共享bank，中间scale为block sharing；
random partition保持相同scale、width与params，只改变future coordinates的membership。requested $H$不进入网络，
只crop full-domain output。

point width固定为4，其余width按总decoder params最近整数匹配。利用exact identity
$\operatorname{GELU}(u)-\operatorname{GELU}(-u)=u$，所有scale均满足
$k_s\ge2\min(R,s)$，因而都包含任意full-affine history-to-future map；linear capacity差异不能解释结果。

Step7A已通过80个parameter/partition/affine cases与20个full forward/gradient cases：最大parameter gap
0.1646%，最大affine witness gap $2.3842\times10^{-7}$，prefix gap 0，五profiles的A6 encoder initialization
pairing通过。由此只授权`neutral_raw, seed=2021`的Step7B；A6-natural仍由neutral gate串行控制。完整设计与gate见：

- `analysis/stage_c_d14a1_dual_carrier_grouped_mlp_20260715/source_theory_design_audit.md`；
- `analysis/stage_c_d14a1_dual_carrier_grouped_mlp_20260715/local_gate/local_gate_report.md`；
- `configs/stage_c_d14a1_dual_carrier_grouped_mlp.json`。

### A1 neutral seed2021 returned gate

neutral完成5 datasets × 8 arms，40/40 complete。本地独立重算与远端一致：function separation、carrier skill、
crossing均5/5；sample × bin oracle macro gain 7.6753%；canonical-vs-random macro gain 0.8945%且5/5为正；
所有invariants通过。train-only fixed scale为ETTh1/ETTm1/Weather=`s360`、ETTh2=`s720`、ETTm2=`s48`。

首次聚合因official validation shuffle触发row-alignment hard failure；随后使用sequential loader从同一checkpoints重算，
未重训、未读取test。该问题只属于artifact alignment fault。neutral decision为
`neutral_problem_pass_authorize_a6_sensitivity`；A6 sensitivity已启动，但confirmation/D14-B/method/test仍false。
完整报告见`analysis/stage_c_d14a1_dual_carrier_grouped_mlp_20260715/neutral_seed2021_result.md`。

### A1 A6-natural returned and strict review

A6 sensitivity 45/45 complete：function separation/carrier skill/crossing均5/5；原oracle 9.9892%；
canonical-vs-random 0.6661%且5/5正。A6 decision=`a6_sensitivity_confirming`。

为排除train-selected fixed scale不稳定造成的oracle膨胀，confirmation前增加更严格的robustness accounting：A6相对
validation-best fixed scale的oracle仍9.1504%，sample oracle相对validation-bin policy仍8.5429%；neutral分别为
6.9978%与6.7555%。因此instance × future-region headroom不是static scale selection解释。

但train-selected GroupedMLP在H720相对exact A6-LBF macro为-2.9435%，validation-best GroupedMLP仍为-1.6855%。
故当前fixed grouped heads只通过problem diagnostic，不是paper method。完整review见
`analysis/stage_c_d14a1_dual_carrier_grouped_mlp_20260715/dual_carrier_seed2021_review.md`。

[Decision] 授权seeds2022/2023 dual-carrier confirmation；multi-seed stable gate通过后才允许D14-B返回Step4-6设计。
D14-B implementation、PCSD/CCRL method与test仍false。

### A1 seeds2022/2023 confirmation returned

本轮新增170/170 runs完成；与seed2021合并后，本地从原始artifacts独立重算three-seed gate。neutral与A6的
function separation、carrier skill、stable crossing均为5/5 datasets。neutral strict oracle / sample-over-bin为
7.1107% / 6.7948%，A6为9.1259% / 8.5990%。contiguity在两carrier均为4/5 stable datasets，macro分别
0.4230% / 0.4667%；neutral ETTh2与A6 ETTh1没有达到2/3-seed稳定，因此contiguous grouping只能作为有根据的
default与control，不能写成universal temporal law。

A6 train-selected / validation-best GroupedMLP相对A6-LBF H720的three-seed macro仍为-2.6886% / -1.4879%。
这确认了coupling-choice problem，而没有确认当前diagnostic head或PCSD method。所有invariants通过，未读取test。

[Decision] `dual_carrier_confirmation_pass_authorize_d14b_design`。D14-A problem gate正式通过；只授权D14-B
返回Step 4-6进行source-informed design、theory与narrative audit。D14-B implementation、paper method与test仍
为false。完整报告见
`analysis/stage_c_d14a1_dual_carrier_grouped_mlp_20260715/multiseed_confirmation_report.md`。

### Coupling scales

首轮候选$\mathcal S=\{1,48,144,360,720\}$；最终需根据parameter accounting与partition divisibility冻结，
不得按validation结果删改。$s=1$为point-like endpoint，$s=720$为global endpoint。

### Mandatory arms

1. `A0_POINT`；
2. `A1_BLOCK_SMALL`；
3. `A2_BLOCK_MEDIUM`；
4. `A3_BLOCK_LARGE`；
5. `A4_GLOBAL`；
6. `A5_RANDOM_PARTITION_MATCHED`；
7. `A6_PERMUTED_CONTIGUITY_MATCHED`；
8. `A7_GENERIC_CAPACITY_MATCHED`；
9. `A8_EQUAL_SCALE_ENSEMBLE`；
10. `A9_BEST_FIXED_SCALE_TRAIN_ONLY`。

### Parameter accounting

对每个arm记录：

- trainable params；
- effective degrees of freedom/rank；
- per-target shared parameter count；
- optimizer steps与wall time；
- carrier dimension；
- block count与block boundary。

params差异不参与dataset profile选择，但coupling attribution必须有same-budget control。不得用更多独立block
networks制造虚假local advantage。

### Statistics

对每个dataset/seed/sample/future bin记录$L_{i,b,s}$，并计算：

$$
\Delta_{oracle}^{sample}
=\frac{L_{best\ fixed}-\sum_{i,b}\min_sL_{i,b,s}}
{L_{best\ fixed}},
$$

以及：

- scale-wise validation MSE/MAE；
- short/mid/long winner；
- pairwise crossing count；
- per-bin oracle gain；
- per-sample × bin oracle gain；
- oracle winner entropy；
- contiguous vs random/permuted partition gap；
- train-validation generalization gap；
- C0/C1 carrier agreement。

### D14-A gate

总体pass需要：

1. 至少3/5 datasets中至少两个real coupling scales稳定交叉；
2. 至少2/3 seeds/folds同方向；
3. five-dataset sample × bin oracle MSE headroom `>=0.5%` over train-selected best fixed scale；
4. real contiguous scopes的主要headroom不能由random/permuted partition或generic capacity control解释；
5. 任一arm不得出现divergence、non-finite或`>100%` pathology；
6. C0为primary；C1只能加强或触发compatibility audit。

D14-A valid fail则取消D14-B并先完成failure attribution；carrier/numeric invalid不得用于方向否决。

## D14-B1 Cross-Fitted Conditional-Risk Predictability

### Step 4-6 source/theory correction

TimeFuse已覆盖sample-level meta-feature fusion；TimeRouter已覆盖oracle-best expert labels、context/CV/forecast
features、nonlinear routing与OOF threshold selection；AME-TS已覆盖structural-prior KL routing。因此generic
history-conditioned router不构成CCRL novelty。

原`softmax(-realized regret)`也不能直接解释为optimal mixture weight。对mixture $f_p=\sum_sp_sf_s$：

$$
\sum_sp_s(Y-f_s)^2-(Y-f_p)^2=\sum_sp_s(f_s-f_p)^2\ge0.
$$

expert-risk objective只是mixture loss上界，会遗漏prediction cancellation。D14-B1因此改为：actual fused forecast
loss是primary，chronological cross-fitted centered risk只作auxiliary identifiable supervision。完整审计见
`analysis/stage_c_d14b_crossfit_regret_20260716/d14b_step46_source_theory_design_audit.md`。

### Leakage-free construction

两个purged forward outer folds使用train windows的`[0.6,0.8)`与`[0.8,1.0)`作为OOF ranges，raw
past+future coverage之间保持1439-window purge。每个fit prefix内部另设purged inner tail选择expert epoch；最终
experts在full train按median inner-best epoch refit。official validation不参与expert checkpoint、policy、temperature
或$\lambda$选择。

对OOF sample $i$、region $b$与scale $s$，primary target为centered risk：

$$
r^{cf}_{i,b,s}=L^{cf}_{i,b,s}-\frac1{|\mathcal S|}\sum_jL^{cf}_{i,b,j}.
$$

centering保留pairwise risk differences；hard oracle只作prior-art control。requested $H$、dataset ID、validation
performance与future truth均不得进入deployed policy。

### Frozen policy arms

1. `B0_OOF_BEST_FIXED`；
2. `B1_EQUAL_MIXTURE`；
3. `B2_TARGET_ONLY_RISK`；
4. `B3_HISTORY_ONLY_RISK`；
5. `B4_HISTORY_TARGET_RISK`；
6. `B5_HISTORY_TARGET_DIRECT_FUSION`；
7. `B6_CCRL_HYBRID`；
8. `B7_HARD_ORACLE_HYBRID`；
9. `B8_IN_SAMPLE_REGRET_HYBRID`；
10. `B9_PERMUTED_CROSSFIT_REGRET`；
11. `B10_RANDOM_HISTORY_HYBRID`；
12. `B11_FORECAST_FEATURE_CCRL_CONTROL`；
13. `B12_ORACLE_UPPER_BOUND`。

primary policy是matched 64-64 GELU MLP；target-only/history-only通过zero mask保持相同input slots。另运行
`HistGradientBoostingRegressor`排除弱regression head造成的假失败。forecast-feature arm是TimeRouter-style
secondary control，不计novelty。

### Gate B-P: predictability problem

`B4_HISTORY_TARGET_RISK`必须：

1. 至少3/5 datasets超过`B0`，macro MSE gain `>=0.3%`；
2. 至少3/5超过`B2`且macro `>=0.2%`；
3. 至少3/5超过`B3`且macro `>=0.1%`；
4. confirmation至少2/3 seeds同方向；
5. MLP fail但tree positive只触发readout redesign，不形成direction pass/fail。

### Gate B-C: contribution-specific value

`B6_CCRL_HYBRID`必须在至少3/5 datasets超过matched `B5_DIRECT_FUSION`，macro MSE gain `>=0.2%`，并在
至少3/5超过hard-oracle与in-sample hybrid；permuted/random controls不得复制收益，且不能出现collapse、non-finite
或validation reversal。B-P通过而B-C失败只支持generic adaptive fusion，CCRL关闭。

[Decision] narrative gate对`diagnostic_only` conditional pass；只授权Step7A local implementation。remote、paper
method与test仍false。冻结设计见`configs/stage_c_d14b_crossfit_regret.json`。

## Decision Matrix

| D14-A | D14-B | Decision |
| --- | --- | --- |
| fail | canceled | PCSD/CCRL关闭；rollback Step 2 |
| pass | B-P fail | PCSD返回Step 4；关闭instance-adaptive coupling claim与CCRL |
| pass | B-P pass / B-C fail | 只支持generic adaptive fusion；CCRL关闭，PCSD返回Step 4 |
| pass | B-P + B-C pass | PCSD/CCRL返回formal method Step 4-6；method/remote/test仍false |
| invalid | any | 修复diagnostic；不得方向级否决 |

## Failure Attribution

- `hypothesis_false`：matched stable diagnostic下一个scale支配，或oracle headroom不足；
- `intervention_point_wrong`：head family没有真正改变output sharing，或carrier抹除相关信息；
- `readout_or_head_design_wrong`：scale arms capacity/accounting不等价；
- `optimization_or_numeric_pathology`：divergence、ill-conditioning、>100% degradation；
- `capacity_control_explains`：generic capacity/random partition复制gain。

只有第一类在C0 neutral carrier、matched controls与stable optimization同时成立时，才可拒绝coupling problem。

## Required Artifacts

1. source-informed head-family audit；
2. effective protocol/config JSON；
3. dataset/profile/split hashes；
4. synthetic equivalence/non-equivalence tests；
5. parameter/DoF/FLOP table；
6. per-sample × bin × scale loss artifact；
7. crossing/oracle/partition-control summary；
8. cross-fit fold and leakage audit；
9. policy calibration/routing diagnostics；
10. Chinese research interpretation；
11. 11-step decision与failure attribution。

## Source Boundary

完整外部审计见`Papers/multi-horizon-output-coupling-audit.md`。mandatory references包括Stratify、
Direct/MIMO/DIRMO review、CATS、MQTransformer、TimePerceiver、Implicit Forecaster、MQF2、Multi-output
Ensembles与TimeRouter。
