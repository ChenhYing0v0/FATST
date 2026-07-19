# D19 IF Control Step 9 Deep Audit

## 1. What was tested

`SC-D19-IFC-control-v1.1`不是论文方法，而是一个source-informed control，用来回答：在same-history、
from-scratch E2E与相同four-horizon checkpoint rule下，Implicit Forecaster式trajectory synthesis是否显示
超越A6 learned-basis decoder的headroom。

冻结矩阵包含5 datasets × 4 arms × seed2021：

- `A6_MEASURE`：复用相同natural profile与four-horizon validation selector的A6 control；
- `IF_MEASURE`：amplitude/phase polar spectrum + iFFT + history-spectrum skip；
- `IF_NOSKIP_MEASURE`：与IF相同初始化和参数，只把history spectrum替换为zero；
- `DIRECT_NONLINEAR_MATCHED_MEASURE`：读取相同Encoder state与history spectrum，参数与IF相差低于0.1%，
  但直接生成720-point trajectory。

正式effectiveness使用official test H96/H192/H336/H720 MSE/MAE。validation只用于checkpoint选择与
failure attribution。

## 2. Artifact and protocol audit

[Fact]

- 15/15 new runs完成；加上5个复用A6，共20/20 artifact units；
- 80/80 official-test cells完整；
- 15/15 checkpoints在test audit前后hash不变；
- 15/15 invariants、protocol、finite与projectivity checks通过；
- 无Traceback、OOM、NaN或Inf；
- Encoder initialization在每个dataset的四arms间一致；IF/no-skip decoder initialization一致。

因此，本次结果不是artifact缺失、checkpoint mutation或随机初始化不配对造成的。

## 3. Paper-facing effectiveness

定义gain为

$$
G=100\left(1-\frac{\operatorname{metric}_{candidate}}
{\operatorname{metric}_{reference}}\right).
$$

正值表示candidate更好。`cell_wins`统计20个dataset-horizon cells中$G>0$的数量；
`dataset_wins`与`horizon_wins`分别先在对应轴上取mean gain后判断正负。

| Comparison | Metric | Macro gain | Cell wins | Dataset wins | Horizon wins | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| IF vs A6 | MSE | -3.6117% | 3/20 | 1/5 | 0/4 | fail |
| IF vs A6 | MAE | -3.6519% | 2/20 | 0/5 | 0/4 | fail |
| IF vs matched direct | MSE | -0.8075% | 10/20 | 2/5 | 1/4 | fail |
| IF vs matched direct | MAE | -1.7096% | 4/20 | 1/5 | 0/4 | fail |
| IF vs no-skip | MSE | +1.6191% | 16/20 | 4/5 | 4/4 | pass |
| IF vs no-skip | MAE | +0.8963% | 15/20 | 4/5 | 4/4 | pass |

### IF MSE gain by dataset and horizon

| Reference | Dataset | H96 | H192 | H336 | H720 | Dataset mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A6 | Weather | -5.164% | -5.004% | -4.373% | -5.808% | -5.087% |
| A6 | ETTm1 | -5.369% | -2.662% | -1.832% | -8.535% | -4.599% |
| A6 | ETTh1 | -3.423% | -1.753% | -1.848% | -3.958% | -2.745% |
| A6 | ETTh2 | -1.552% | +1.657% | +6.191% | +2.673% | +2.242% |
| A6 | ETTm2 | -11.262% | -7.593% | -5.081% | -7.540% | -7.869% |
| matched direct | Weather | -3.500% | -3.380% | -3.383% | -3.953% | -3.554% |
| matched direct | ETTm1 | -0.624% | +0.939% | +1.672% | -3.941% | -0.488% |
| matched direct | ETTh1 | +1.953% | +1.095% | +0.305% | -2.413% | +0.235% |
| matched direct | ETTh2 | +0.446% | +0.442% | +2.265% | +1.615% | +1.192% |
| matched direct | ETTm2 | -4.237% | -1.118% | +0.297% | -0.631% | -1.422% |
| no-skip | Weather | +1.995% | +2.517% | +2.160% | +1.725% | +2.099% |
| no-skip | ETTm1 | -5.488% | -3.487% | -2.523% | -3.826% | -3.831% |
| no-skip | ETTh1 | +2.965% | +2.292% | +0.879% | +0.862% | +1.749% |
| no-skip | ETTh2 | +2.238% | +2.024% | +0.871% | +0.011% | +1.286% |
| no-skip | ETTm2 | +9.709% | +8.764% | +6.487% | +2.209% | +6.792% |

[Fact] IF只在ETTh2超过A6；其他四个datasets全部为负。相对matched direct，IF恰好赢10/20 cells，
但负向cells的幅度更大，所以macro MSE与MAE均失败。这不是“几乎全面获胜但gate过严”。

## 4. Validation and checkpoint behavior

| Candidate vs A6 | Validation MSE macro gain | Cell wins |
| --- | ---: | ---: |
| IF | -10.9950% | 0/20 |
| no-skip IF | -11.7751% | 0/20 |
| matched direct | -9.7887% | 0/20 |

IF相对no-skip在validation仍为`+0.6595%`、11/20 cells；相对matched direct为`-1.1188%`、
8/20 cells。因此test上的两项主要关系不是纯validation→test reversal。

15个new arms中12个在epoch 1达到best validation checkpoint，另外3个在epoch 2达到；随后均因patience=5
在epoch 6或7停止。A6的best epochs依次为Weather 18、ETTm1 8、ETTh1 2、ETTh2 2与ETTm2 5。

[Strong Evidence] 新readouts在统一optimizer/LR下表现出明显更早的generalization peak，但没有numeric divergence。
这支持`readout-scale/optimization mismatch suspected`，而不是`optimization_or_numeric_pathology`。

## 5. Internal mechanism health

| Dataset | IF/no-skip prediction NRMSE | IF/direct prediction NRMSE | Amplitude std | Phase radius mean | Max prefix gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Weather | 0.2357 | 0.3937 | 1.4465 | 0.7951 | 3.58e-7 |
| ETTm1 | 0.2410 | 0.3177 | 0.7738 | 0.8710 | 5.96e-8 |
| ETTh1 | 0.4128 | 0.4442 | 0.8845 | 0.6484 | 3.58e-7 |
| ETTh2 | 0.0780 | 0.1087 | 1.0227 | 0.6910 | 2.38e-7 |
| ETTm2 | 0.0990 | 0.1122 | 1.0594 | 0.8108 | 2.98e-8 |

`prediction NRMSE`为两个probe forecasts之差的RMSE除以reference forecast RMS。Amplitude std与phase
radius来自256个固定probe rows的361-bin internal tensors。全部finite且远离冻结的collapse threshold。

[Fact] IF、no-skip与direct产生了实质不同的预测；amplitude/phase没有塌缩；history skip也确实改变并改善了
输出。因此本次失败不能归因为“机制根本没执行”。

## 6. Capacity and design-scale audit

IF总参数相对A6总参数分别为：Weather `10.02×`、ETTm1 `10.29×`、ETTh1 `8.73×`、ETTh2
`10.02×`、ETTm2 `7.94×`。这是source-informed hidden width 2048带来的结果。matched direct与IF的参数差
低于0.1%，所以它能公平回答“polar synthesis vs generic nonlinear full-trajectory head”，但不能消除两者共同
相对A6的large-head generalization mismatch。

[Strong Evidence] generic direct head相对A6也为`-2.7691%` MSE、3/20 cells，只在ETTh2 dataset mean为正。
增加大规模nonlinear decoder capacity本身没有带来收益；当前IF的负结果也不是因为polar结构牺牲了一个本来
很强的matched nonlinear carrier。

## 7. Four-layer decision

1. `paper_facing_effectiveness`: **fail**。IF vs A6为-3.6117% MSE、-3.6519% MAE；
2. `matched_mechanism_attribution`: **fail**。IF未超过matched direct，但超过no-skip；
3. `internal_mechanism_health`: **pass**。所有finite、projectivity、initialization、deformation与polar health
   gates通过；
4. `failure_attribution`: 当前exact implementation为`readout_or_head_design_wrong`，并有
   `readout-scale/optimization mismatch suspected`；不得标记为direction-level `hypothesis_false`。

Decision=`control_negative_return_step2_4`。

## 8. What remains useful

[Strong Evidence] 720-point history-spectrum skip相对no-skip同时在validation与test为正，且test为4/5 datasets、
4/4 horizons。这说明A6 Encoder state之外的normalized-history spectral shortcut提供了可利用的信息。

但这个结论的边界是：

- 它没有使IF超过A6；
- ETTm1上skip稳定为负；
- skip是history-to-decoder information path，不足以单独构成multi-horizon Contribution；
- polar/frequency synthesis本身没有获得matched-control支持。

因此不补seeds2022/2023，不做hidden-width/LR sweep，也不把IF改名为论文方法。下一步回到Step2/4：先判断
是否存在“compact structured generation + direct history statistic”这一新的、真正服务fixed-past unified
multi-horizon问题的必要性与narrative，再决定是否提出新candidate。若只是在D19上缩width或调LR，应视为
engineering rescue，不足以进入paper-core loop。

## 9. Artifact map

- `standard_metrics.csv`: 80个official-test dataset-arm-horizon metrics；
- `comparison_cells.csv`: 三组预注册comparison的逐cell gain；
- `comparison_summary.csv`: macro/cell/dataset/horizon summary；
- `internal_health.csv`: paired initialization、prediction deformation与polar health；
- `artifact_audit.csv`: 20-unit protocol/checkpoint audit；
- `four_layer_decision.json`: machine-readable Step9 decision。
