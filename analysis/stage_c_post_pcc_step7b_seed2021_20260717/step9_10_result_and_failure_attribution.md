# SIFF/MCCA Step 9–10 结果、失败归因与回滚

## 1. 当前节点

| Field | Value |
| --- | --- |
| `current_step` | Step 9 artifact evaluation -> Step 10 candidate decision |
| `candidate` | `SC1-SIFF-v1` + `SC2-MCCA-v1` |
| `matrix` | 11 new arms × 5 datasets = 55；25 historical matched references |
| `seed` | 2021 |
| `evaluation_split` | validation |
| `coupling training loss` | exact dense-prefix harmonic L1 fused loss + arm/router auxiliary |
| `checkpoint_rule` | best validation H720 MSE |
| `primary_screen_metric` | dense prefix MSE AUC over $H=1,\ldots,720$ |
| `test_used` | false |
| `artifact_audit` | 55/55 new + 25/25 references；all protocol checks pass |
| `formal_gate` | fail |

本报告回答三个问题：

1. SIFF是否形成超越generic width、independent scope与错误scale mapping的architecture effect；
2. MCCA在保持与PCC完全相同scope总skill mass时，competitive target assignment是否更优；
3. 失败来自理论假设、intervention/readout、capacity、numeric，还是训练与评价measure不一致。

## 2. 数据与统计量如何构造

### 2.1 Dense multi-horizon MSE AUC

对完整预测$\hat y_{1:T}$，prefix $H$的MSE为

$$
L(H)=\frac{1}{H}\sum_{t=1}^{H}e_t,\qquad
e_t=(\hat y_t-y_t)^2.
$$

本阶段冻结的primary screening metric为

$$
L_{\mathrm{dense}}
=\frac{1}{T}\sum_{H=1}^{T}L(H)
=\sum_{t=1}^{T}w_t e_t,
\qquad
w_t=\frac{1}{T}\sum_{H=t}^{T}\frac{1}{H}.
$$

因此dense MSE AUC不等价于flat full-domain MSE
$T^{-1}\sum_t e_t$。在$T=720$时：

- $w_1=0.00994050$，是flat target weight的$7.1572$倍；
- $w_{720}=1.9290\times10^{-6}$，只有flat target weight的$1/720$；
- 同一单位step error在$t=1$对dense AUC的影响是$t=720$的$5153.16$倍。

这里的“dense horizon”是requested prefix $H$，不是把720个target当成720个独立任务。上述等价式只需一次
完整forward即可精确计算，不需要720次模型调用。

[Code Fact] 重新核查training path后确认：所有PCSD/SIFF EQUAL/PCC/MCCA arms的`fused_loss`已经使用
`prefix_measure()`的exact harmonic target weights，但error norm为L1；A6 reference仍使用flat full-domain L1。
所有arms的checkpoint仍按H720 MSE选择，primary screen则是dense-prefix MSE AUC。

因此：

- SIFF-vs-PCSD architecture effect是same harmonic-L1 objective下的fair paired comparison；
- 不能把SIFF failure归因于“训练完全使用flat loss”；
- 未决confound收紧为`H720 checkpoint vs dense risk`以及`L1 training vs MSE screening`；
- joint-vs-A6同时改变architecture与training objective，只能作为joint gate，不能单独归因architecture。

### 2.2 Relative gain与factorial main effect

所有gain均定义为

$$
G(A,B)=1-\frac{L_A}{L_B},
$$

正值表示candidate优于reference。

- `architecture_main_effect`：分别计算SIFF相对PCSD在EQUAL、PCC、MCCA三个training modes下的gain，
  再先按dataset平均、后按五dataset macro平均；
- `mcca_main_effect`：分别计算PCSD与SIFF carrier上MCCA相对same-mass PCC的gain，再按dataset与macro平均；
- `joint_over_a6`：SIFF+MCCA相对same-seed A6的dense AUC gain；
- horizon-bin统计把prefix horizons分为1–48、49–96、97–192、193–336、337–720，并在bin内先平均MSE。

### 2.3 控制项含义

- `constant`：保留SIFF参数化但去除scale变化；
- `permuted`：保留相同scale values与capacity，仅打乱scope-scale对应；
- `Q1-wide`：不使用scale coordinate的matched-width control；
- `independent`：每个scope使用独立matched-rank modes；
- `pointwise MCCA`：去除prefix transport；
- `uniform OT`：保留balanced transport solver但去除capability-informed scope marginal。

## 3. Primary gates

### 3.1 Architecture与training main effects

| Effect | Macro dense-MSE gain | Dataset wins | Frozen gate |
| --- | ---: | ---: | --- |
| SIFF architecture main effect | -1.5015% | 2/5 | fail |
| MCCA main effect over same-mass PCC | -0.0250% | 2/5 | fail |
| SIFF+MCCA over A6 | -0.5621% | 4/5 | fail |

[Fact] SIFF在三个training modes下呈现相同dataset pattern：ETTh1/ETTh2正，ETTm1/Weather弱负，
ETTm2约-9.5%至-9.8%。因此ETTm2 failure不是MCCA独有，也不能由objective interaction单独解释。

joint逐dataset结果如下：

| Dataset | SIFF+MCCA vs A6 |
| --- | ---: |
| ETTh1 | +0.0830% |
| ETTh2 | +3.8847% |
| ETTm1 | +2.5305% |
| ETTm2 | -10.1209% |
| Weather | +0.8124% |
| Macro | -0.5621% |

[Fact] 若事后删除ETTm2，joint为+1.8277%、4/4 wins。但ETTm2属于冻结matrix，不能因不利而删除；
该leave-one-out只用于定位异质性，不改变formal fail。

[Boundary] A6使用flat L1而coupling arms使用harmonic L1，故`joint_over_a6`不是纯architecture contrast；
formal architecture attribution以三个SIFF-vs-PCSD paired effects为准。

### 3.2 Attribution controls

| Comparison | Macro gain | Wins | 结论边界 |
| --- | ---: | ---: | --- |
| ordered SIFF vs constant | -0.9804% | 4/5 | ETTm2抵消其余四dataset |
| ordered SIFF vs permuted | +1.1177% | 5/5 | scope order含有效信息，但不证明绝对有效 |
| ordered SIFF vs Q1-wide | -1.1880% | 3/5 | generic width不能解释全部dataset，但macro解释 |
| ordered SIFF vs independent | -2.1420% | 2/5 | continuous shared field未优于独立scope |
| ordered SIFF vs dense matched | +2.3246% | 4/5 | dense generic head不是主要解释，ETTm2除外 |
| MCCA transport vs pointwise | +0.4736% | 4/5 | projective prefix transport有稳定局部信号 |
| capability marginal vs uniform OT | +0.1182% | 5/5 | capability-informed marginal有小而一致的信号 |
| PCSD MCCA vs same-mass PCC | -0.1092% | 0/5 | exact competitive assignment在PCSD上被否定 |

[Strong Evidence] `ordered > permuted`的5/5结果排除了“scope labels完全无意义”，但permuted是错误映射control；
它不能替代`ordered > Q1/constant/independent`的absolute method gate。

[Strong Evidence] MCCA的两个组成部分——prefix transport与capability marginal——分别有正信号，但完整
competitive row allocation仍在PCSD carrier上0/5输给PCC。这说明“正确的measure/marginal”与“强制全局竞争”
必须分开判断。

## 4. Horizon signature：真正的失败位置

### 4.1 Macro horizon-bin signature

| Effect | H1–48 | H49–96 | H97–192 | H193–336 | H337–720 |
| --- | ---: | ---: | ---: | ---: | ---: |
| joint over A6 | -16.224% | -6.129% | -2.411% | -0.606% | +0.489% |
| SIFF architecture on MCCA | -22.525% | -9.283% | -3.732% | -1.425% | +0.149% |
| ordered over constant | -16.233% | -7.770% | -3.055% | -0.602% | +0.239% |
| ordered over permuted | +2.532% | +1.612% | +1.516% | +1.358% | +0.778% |
| MCCA over PCC | +0.123% | -0.121% | -0.134% | -0.098% | -0.117% |
| transport over pointwise | +0.314% | +0.543% | +0.491% | +0.691% | +0.410% |
| capability over uniform OT | +0.358% | +0.201% | +0.163% | +0.134% | +0.090% |

[Strong Evidence] SIFF-v1不是均匀变差，而是把error profile向早期targets重新分配：长horizon bin微正，
短horizon bin显著负。由于SIFF/PCSD已共享exact harmonic-L1 training，这个signature不能由“flat training”
解释；更可能位于Q2 readout optimization、L1/MSE mismatch或H720 checkpoint selection。

### 4.2 ETTm2 exact pathology

| Arm | H1 MSE | H48 MSE | H720 MSE |
| --- | ---: | ---: | ---: |
| A6 | 0.036076 | 0.090738 | 0.281152 |
| PCSD+MCCA | 0.034956 | 0.095787 | 0.279216 |
| SIFF+MCCA | 0.268982 | 0.155576 | 0.277537 |
| SIFF constant | 0.101416 | 0.096961 | 0.282118 |
| Q1-wide | 0.042894 | 0.096321 | 0.278901 |
| independent scope | 0.030333 | 0.092567 | 0.274098 |
| dense matched | 0.042311 | 0.093415 | 0.283852 |

SIFF相对PCSD在ETTm2：

- H1 gain = -669.49%，即MSE约为7.69倍；
- H720 gain = +0.6013%；
- best-H720 checkpoint为epoch 1，H720 validation MSE 0.277537；
- 所有training/evaluation values finite，checkpoint也正确恢复。

[Fact] 这不是NaN、OOM、checkpoint未恢复或全面训练失败。

[Strong Evidence] 这是需要进一步定位的`optimization_or_checkpoint pathology`：training target measure已经
harmonic-aligned，但checkpoint只看H720 MSE，且training使用L1、primary screen使用MSE。现有artifacts没有保存
每个epoch的dense curve，无法判断是否存在被H720 rule丢弃的healthy checkpoint。其>100%局部恶化触发项目的
diagnostic failure保护规则，因此该结果只能关闭exact SIFF-v1 candidate，不能关闭更广的scale-coordinate方向。

## 5. Failure attribution

### 5.1 SC1-SIFF-v1

| Attribution class | Decision |
| --- | --- |
| `hypothesis_false` | **部分**：ordered scale含信息；但rigid global-linear Q2 field不是general method |
| `intervention_point_wrong` | [Hypothesis] current field在scope synthesis前以固定线性coordinate强制error redistribution |
| `readout_or_head_design_wrong` | [Strong Evidence] Q2/constant decomposition出现short-prefix pathology；Q1-wide/independent不出现 |
| `optimization_or_numeric_pathology` | **present as short-prefix/checkpoint pathology, not numeric divergence** |
| `capacity_control_explains` | independent/Q1解释ETTm2 recovery；但不能解释ordered>permuted 5/5 |

formal status：

`validation_screen_failed_exact_design / diagnostic_invalid_for_direction_rejection`

这意味着不补seed、不test、不把SIFF-v1写入paper claim；但保留“ordered coupling coordinate”和
“short-vs-long error redistribution”作为下一轮Step4 evidence。

### 5.2 SC2-MCCA-v1

| Attribution class | Decision |
| --- | --- |
| `hypothesis_false` | exact same-mass global competitive assignment不获益，PCSD carrier上0/5 |
| `intervention_point_wrong` | 可能：移除per-target floor后，竞争只改变credit placement但未改善fused forecast |
| `readout_or_head_design_wrong` | 无独立证据 |
| `optimization_or_numeric_pathology` | false；marginal/gradient/finite checks均通过 |
| `capacity_control_explains` | uniform OT不能解释capability marginal小增益；PCC直接解释完整method |

formal status：

`validation_screen_failed_exact_hypothesis`

MCCA-v1停止；不以微调Sinkhorn、temperature或capacity继续追逐。保留prefix transport与capability marginal作为
method ingredients，不保留“global competition必然优于coverage floor”的claim。

## 6. Step 10 decision

1. `SC1-SIFF-v1`与`SC2-MCCA-v1`均未通过frozen Phase-A gate；
2. 不运行confirmation seeds，不进入conditional Phase B，不访问test；
3. joint 4/5 wins不能覆盖macro fail，也不能绕过ETTm2；
4. rollback到Step 4，但两个分支的回滚边界不同：
   - architecture：保留ordered-coordinate problem evidence，先定位per-epoch dense-risk trajectory；
   - training：关闭exact competitive assignment；harmonic fused loss已存在，不能重复提出。

## 7. 下一步：source audit后收紧为checkpoint-trajectory diagnostic

最初提出的`Projective Measure Alignment`经code audit发现已经存在于PCC/MCCA fused loss，经external audit又被
ElasTST直接覆盖。因此它不再进入implementation。下一节点收紧为
`SC-D16-CTD Checkpoint Trajectory Diagnostic`，状态仅为`diagnostic_only_step5_6_design_pending`。

### 7.1 要验证的问题

> 在相同harmonic-L1 training下，SIFF的short-prefix pathology是否只由H720 checkpoint selection造成，
> 还是Q2 readout在全部training epochs都存在？

### 7.2 执行顺序

1. **Step4 source audit**：已完成；ElasTST覆盖HR，Loss Shaping/QDF覆盖step weighting；
2. **Step5/6 diagnostic freeze**：只在ETTm2 pathology locus复跑PCSD/SIFF/Q1/constant matched arms，
   每epoch评估并保存dense-MSE/MAE AUC与H1/H720；
3. **counterfactual checkpoint audit**：同一training trajectory分别选择best-H720、best-dense-MAE与
   best-dense-MSE，不使用test；
4. **判定边界**：若SIFF在任何pre-registered checkpoint rule下都保持>100% short-prefix degradation，则
   readout/optimization negative有效；若dense checkpoint消除pathology，再授权five-dataset matched confirmation。

[Self-critique] per-epoch checkpoint trajectory可能只说明early stopping metric选错，不会自动恢复SIFF的
architecture novelty或effectiveness。即使diagnostic通过，ElasTST-covered horizon reweighting仍只能作protocol。

## 8. Artifacts

- `remote_analysis/run_summary.csv`：80个有效run/reference的aggregate metrics与mechanism diagnostics；
- `remote_analysis/factorial_and_control_effects.csv`：冻结factorial/control gains；
- `arm_scoreboard.csv`：逐dataset-arm dense AUC及关键prefix MSE；
- `horizon_bin_effects.csv`：所有关键comparison的五段horizon signature；
- `training_stability.csv`：55个new runs的best/last H720 validation与finite audit；
- `leave_one_dataset_out.csv`：只用于异质性定位，不用于改gate；
- `step9_attribution.json`：machine-readable Step9/10 decision。

## 9. Subsequent Step4 source-audit outcome

本报告完成后已执行external-first audit。NeurIPS 2024 ElasTST直接覆盖uniform random horizon对应的harmonic
horizon reweighting，并在官方实现中对齐weighted training与validation checkpoint；Loss Shaping Constraints
与QDF进一步覆盖per-step/non-uniform future-task weighting。

随后code audit确认PCC/MCCA training path本来就使用exact harmonic-L1 fused loss，故新增HR arm是重复实验。
standalone `SC2-PHMA`已被narrative gate拒绝；下一步只保留`SC-D16-CTD` per-epoch checkpoint trajectory
diagnostic。详见
`analysis/stage_c_post_pcc_step4_measure_audit_20260717/source_informed_measure_audit.md`。
