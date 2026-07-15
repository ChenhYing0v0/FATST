# StageC D14 Output-Coupling Granularity Audit

## Status

| Field | Value |
| --- | --- |
| `stage` | `StageC-UVHF` |
| `current_step` | D14-A1 Step 7A passed；Step 7B neutral seed2021 authorized |
| `role` | `diagnostic_only` |
| `active_candidates` | provisional `SC1-PCSD` + `SC2-CCRL` |
| `method_training` | false |
| `remote_training` | A0 complete；A1 neutral diagnostic authorized；A6 sensitivity held；paper method=false |
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

## D14-B Cross-Fitted Regret Predictability

### Label construction

仅使用train chronological cross-fitting predictions。对held-out train sample $i$、bin $b$与scale $s$：

$$
R^{cf}_{i,b,s}=L^{cf}_{i,b,s}-\min_jL^{cf}_{i,b,j},
\qquad
q_{i,b,s}=\operatorname{softmax}(-R^{cf}_{i,b,s}/\tau_r).
$$

$\tau_r$只能在train内部选择。validation labels不得用于policy fit或temperature selection。

### Policy arms

1. `B0_CONSTANT_PRIOR`；
2. `B1_EQUAL_MIXTURE`；
3. `B2_TARGET_ONLY`；
4. `B3_HISTORY_ONLY`；
5. `B4_HISTORY_TARGET`；
6. `B5_IN_SAMPLE_PSEUDOLABEL`；
7. `B6_PERMUTED_REGRET_LABEL`；
8. `B7_RANDOM_HISTORY_FEATURE`；
9. `B8_ORACLE_UPPER_BOUND`。

history features必须在inference时可得；禁止future label、requested $H$、validation arm performance与dataset-specific
manual rule。

### Statistics

- validation forecast MSE/MAE of routed/soft-mixture output；
- gain over best fixed scale and equal mixture；
- gain of history+target over target-only；
- regret calibration与top-1 scale accuracy；
- router entropy、scale usage、expert disagreement；
- in-sample vs cross-fitted pseudo-label generalization gap；
- permuted-label falsification。

forecast gain是primary；classification accuracy只是mechanism diagnostic。

### D14-B gate

CCRL problem pass需要：

1. `B4_HISTORY_TARGET`在至少3/5 datasets超过train-selected best fixed scale；
2. five-dataset macro MSE gain `>=0.3%`；
3. 至少3/5 datasets超过`B2_TARGET_ONLY`，证明instance information有增量；
4. 至少2/3 seeds/folds同方向；
5. `B6/B7`不复制收益；
6. 无router collapse或validation reversal解释。

## Decision Matrix

| D14-A | D14-B | Decision |
| --- | --- | --- |
| fail | canceled | PCSD/CCRL关闭；rollback Step 2 |
| pass | fail | PCSD返回Step 4；CCRL关闭并重找SC2 |
| pass | target-only only | 只保留distance policy evidence；instance-adaptive claim关闭 |
| pass | history+target pass | PCSD/CCRL返回formal Step 4-6；method/remote/test仍false |
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
