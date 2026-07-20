# SC-D23-FCMI Step9/10：Formal Result、Failure Attribution与Rollback

## 1. Decision

| Field | Result |
| --- | --- |
| `candidate_version` | `SC-D23-FCMI-v1` |
| `machine_decision` | `fails_a6_internal_valid` |
| `research_decision` | `fcmi_v1_failed_capacity_control_explains_return_step2_3` |
| `paper_core_status` | closed；不补seed、width、readout或parameter rescue |
| `problem_status` | target-coordinate information access仍由D22-C支持；本结果不否定该问题 |
| `next_step` | deterministic-MSE fixed-past task内回Step2/3；暂无active method |

[Fact] FCMI-v1没有通过paper-facing effectiveness。其main–interaction decomposition在弱query family内部有
正贡献，但A6-capacity-matched trajectory path解释了主要差距；order claim又出现validation/test reversal。
因此不能把FCMI写成论文方法，也不能把dense path直接接回后称为successor。

## 2. Protocol与test audit完整性

- 8 arms × 5 datasets × seed2021 = 40/40 runs；
- 160/160 validation cells与160/160 official-test cells；
- 40/40 protocol、finite、prefix、readout与checkpoint invariants通过；
- test access date：`2026-07-20`；
- user authorization：2026-07-20“按计划继续推进工作”；
- candidate version：`SC-D23-FCMI-v1`；
- checkpoint selection：validation H96/H192/H336/H720 mean MSE；
- 40个checkpoint hashes均唯一并记录在`analysis_seed2021/run_audit.csv`；
- `checkpoint_retrained=true`，所有arms均same-run from-scratch；
- test role：`primary-mechanism-effectiveness-and-paper-benchmark`；
- matrix complete：true；test-informed：true。

不存在missing artifact、NaN/Inf、OOM、checkpoint mutation或frozen replacement。训练epochs为6–20；
达到20 epochs不伴随数值发散，不能用optimization pathology撤销negative result。

## 3. Paper-facing effectiveness

gain定义为

$$
g=100\left(1-\frac{\mathrm{metric}_{candidate}}
{\mathrm{metric}_{reference}}\right).
$$

正值表示candidate更优。macro是20个dataset-horizon cells的gain算术平均；dataset/horizon wins先在对应
4 horizons/5 datasets内平均，再判断是否大于0。

| Comparison | Split | Metric | Macro gain | Cells | Datasets | Horizons |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| FCMI vs A6_MEASURE | validation | MSE | -16.1434% | 0/20 | 0/5 | 0/4 |
| FCMI vs A6_MEASURE | test | MSE | -21.7343% | 0/20 | 0/5 | 0/4 |
| FCMI vs A6_MEASURE | validation | MAE | -9.4475% | 0/20 | 0/5 | 0/4 |
| FCMI vs A6_MEASURE | test | MAE | -10.9242% | 0/20 | 0/5 | 0/4 |

test MSE按dataset为Weather `-4.8830%`、ETTm1 `-18.9626%`、ETTh1 `-72.0753%`、
ETTh2 `-6.3461%`、ETTm2 `-6.4044%`；四个horizons也全部为负。该失败跨split、dataset与horizon，
远超冻结的0.3% gate。

## 4. Matched mechanism attribution

| FCMI comparison | Test MSE | Cells | Datasets | Horizons | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| vs STANDARD_DUAL | +1.3409% | 15/20 | 5/5 | 4/4 | pass |
| vs GENERIC_DUAL | +6.0060% | 16/20 | 4/5 | 4/4 | pass |
| vs ORDER_SHUFFLED | -0.4536% | 8/20 | 2/5 | 0/4 | fail |
| vs DENSE_DUAL | -21.7107% | 1/20 | 0/5 | 0/4 | fail |
| vs TARGET_SHUFFLED | +9.2071% | 16/20 | 4/5 | 4/4 | pass |
| vs STANDARD_QUERY | +2.4168% | 17/20 | 4/5 | 4/4 | descriptive positive |

这组结果支持两个有限结论：

1. [Strong Evidence] main–interaction coordinates、coordinate-specific context与stable target identity在当前
   query family内部有效；
2. [Strong Evidence] ordered history binding并不split-stable：FCMI vs order-shuffled从validation
   `+1.7757%`反转为test `-0.4536%`。

内部positive不能越过A6 effectiveness与dense capacity controls。

## 5. Capacity解释

`DENSE_DUAL_MATCHED`相对A6 active parameter gap只有`0.0914%–0.1321%`，且从standard-dual zero-output
initial morph开始训练。

| Derived comparison | Split | MSE gain | Cells | Datasets | Horizons |
| --- | --- | ---: | ---: | ---: | ---: |
| DENSE vs A6 | validation | -0.9517% | 5/20 | 1/5 | 1/4 |
| DENSE vs A6 | test | -0.3284% | 9/20 | 2/5 | 0/4 |
| DENSE vs STANDARD_DUAL | validation | +12.7127% | 20/20 | 5/5 | 4/4 |
| DENSE vs STANDARD_DUAL | test | +15.4825% | 19/20 | 5/5 | 4/4 |

因此trajectory-wide low-rank synthesis几乎恢复A6 performance，而同参数量standard query family仍显著落后。
FCMI相对standard-dual的`+1.34%`是真实的within-family improvement，但不足以抵消function-class/capacity
缺口。这对应`capacity_control_explains`，并定位为`readout_or_head_design_wrong /
intervention_point_wrong`，不是`optimization_or_numeric_pathology`。

## 6. Internal mechanism health

五datasets全部通过：

- prefix max absolute gap不超过`3.58e-7`；
- paired Encoder/common/main/interaction hash count均为1；
- context coordinate std、main/interaction RMS、attention target dispersion均finite/nonzero；
- FCMI interaction prediction std为`0.0700–0.5695`；
- dense coefficient norm为`1.4010–3.8534`，dense residual std为`0.4721–0.7218`。

故negative effectiveness不能归因于inactive path或numeric collapse。internal health只证明机制被使用，
不证明它有paper-facing价值。

## 7. Frozen conditional complementarity diagnostic

本诊断只复用已保存的相同256 probe rows，不重新访问test、不训练模型。对每个dataset在validation拟合
$\alpha\in[0,1]$：

$$
\alpha^*=\arg\min_{\alpha\in[0,1]}
\|y_{val}-(p_{base}+\alpha d)\|_2^2,
$$

再固定$\alpha^*$用于test。三个$d$分别为：

1. `FCMI - DENSE`（普通convex blend）；
2. `FCMI - STANDARD_DUAL`加到DENSE；
3. `FCMI - STANDARD_DUAL`加到A6。

| Diagnostic | Validation MSE gain | Test MSE gain |
| --- | ---: | ---: |
| dense/FCMI blend | +2.1739% | -3.8507% |
| dense + (FCMI-standard) | +0.7276% | -0.8408% |
| A6 + (FCMI-standard) | +0.6048% | -0.8106% |

所有arms的probe target maximum gap为0。该结果带有cross-model co-adaptation与256-row sampling confound，
只能标记`diagnostic_only`；不得用于方向级拒绝。但三种validation-positive均test-negative，足以阻止
“把dense/A6 main接回FCMI”在没有新problem evidence时直接通过Step4。

### 7.1 A6-DENSE strong-family allocation check

为排除“FCMI失败但A6/DENSE之间存在可学习allocation”的遗漏，又用相同probe rows做了两项
`diagnostic_only`检查：

| Diagnostic | Validation MSE gain | Test MSE gain |
| --- | ---: | ---: |
| per-dataset validation-fit A6/DENSE blend | +0.5127% | -0.1707% |
| fixed 0.5 A6/DENSE blend | -0.4009% | +1.4680% |

validation-fit权重在Weather与ETTh1上选择了明显的DENSE分量，但macro test仍反转；固定等权则出现相反的
validation-negative/test-positive。后者是test-only事后正信号，不能用于选择mixing rule，更不能据此设计
router/MoE。该审计说明目前没有split-stable、validation-identifiable的strong-family allocation evidence。

## 8. Latest primary-source boundary

检索日期：2026-07-20。范围：global trajectory decoding、basis forecasting、global/local cross-attention、
attention attribution与2025–2026 forecasting-phase architectures。来源均为paper/OpenReview或official code；
Zotero coverage未用于判断完整性。

- [Implicit Forecaster, NeurIPS 2025](https://openreview.net/forum?id=gqoeQPhQcE)已明确以global-view
  waveform synthesis解决pointwise decoding不足；
- [BasisFormer, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e150e6d0a1e5214740c39c6e4503ba7a-Abstract-Conference.html)
  已覆盖learned future bases、cross-attention coefficients与basis consolidation；
- [S2TX](https://arxiv.org/abs/2502.11340)已使用cross-attention沟通global long-range与local representations；
- [attention attribution critique](https://openreview.net/forum?id=Mu18gwLAnk)直接警告multi-branch
  mapping/fusion可能解释attention收益。该稿为withdrawn ICLR 2026 submission，只作lower-confidence control
  motivation，不作为已确立事实；
- CATS、TimePerceiver与TQNet继续构成future-query/history-access直接prior。

因此“强trajectory main + coordinate interaction”的primitive组合本身不足以形成新contribution。必须有新的
problem necessity与split-stable conditional gain，不能仅因dense control成功就把它改名为method。

## 9. Failure attribution与rollback

1. `paper_facing_effectiveness`：fail；FCMI vs A6为0/20；
2. `matched_mechanism_attribution`：partial；decomposition/interaction/target通过，order/capacity失败；
3. `internal_mechanism_health`：5/5 pass；
4. `failure_attribution`：
   - primary：`capacity_control_explains`；
   - secondary：`readout_or_head_design_wrong / intervention_point_wrong`；
   - absent：numeric pathology、missing matrix、inactive mechanism。

方向级边界：

- 关闭`SC-D23-FCMI-v1` paper-core candidate；
- 保留D22-C target-coordinate information-access problem evidence；
- 不做FCMI seed、width、readout、representation、rank或objective rescue；
- 不把DENSE control升级method，不把conditional blend升级paper evidence；
- Step4 direct successor narrative gate不通过，继续回Step2/3寻找新的fixed-past finite-capacity problem；
- CTD、D17-D21、H embedding/router与Contribution 2预设继续关闭/paused。

### 9.1 Step 2/3 problem-surface decision

- `dense + FCMI interaction`：conditional gain在validation/test间反转，且global/basis/cross-attention
  primitive已有直接prior；当前不构成新candidate；
- `A6/DENSE allocation`：validation-fit与fixed blend给出相反split结论；当前不能授权router或MoE；
- `ordered coordinate access`：D22-C有信息存在性证据，但D23 order control未通过；这只说明当前
  intervention/readout未兑现order necessity，不足以方向级拒绝，也不足以重开D17-D21式
  representation rescue；
- 因此当前没有active method。下一步只能继续做不访问test的新Step2/3 problem audit；任何新实现都必须
  先给出独立problem necessity、完整prior-art boundary和split-frozen Step4-6 gate。

## 10. Self-critique

[Uncertainty] dense control同时改变parameter count与function class，故“capacity explains”更准确地指
`effective capacity/function-class explains`，不是纯宽度因果证明。另一方面，FCMI在query family内部的多个
controls为正，说明decomposition并非完全无效。若未来独立problem audit证明强trajectory main与coordinate
interaction存在split-stable conditional necessity，可作为新candidate重新过Step4-6；当前证据不足以这样做。
