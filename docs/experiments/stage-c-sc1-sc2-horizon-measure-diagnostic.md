# StageC SC1/SC2 Step 2-3 Diagnostic Protocol

## What We Plan To Test

在frozen natural dataset profiles上，不更新参数，抽取各dataset固定train batches，比较不同horizon risk
measure对同一prediction error与shared parameters产生的梯度。

候选measure不以benchmark四个horizons为本体：

1. `delta_720`: 当前full-720 pointwise mean；
2. `uniform_discrete_horizon`: $H$在1..720均匀，使用解析step weights；
3. `log_uniform_horizon`: 对forecast extent近似scale-balanced；
4. `registered_benchmark_control`: 仅作为`{96,192,336,720}`历史可比control，不作为论文目标measure。

## Why It Matters

SC1当前carrier已exact prefix-consistent，因此不需要再做“输出是否一致”的性能实验；需要判断的是真正
horizon-adaptive computation是否有问题价值。SC2则必须先证明risk measure变化会在shared computation
上形成跨dataset稳定的梯度方向/强度变化，而不是只展示解析权重不同。

## Artifacts And Metrics

- 固定batch indices、seed、resolved frozen contract hash；
- 每个measure的step-weight vector与effective sample size；
- decoder coefficients、basis/readout与encoder shared parameters的gradient norm；
- measure-pair gradient cosine与conflicting-coordinate fraction；
- per-future-region gradient contribution；
- 三dataset符号一致性与bootstrap interval。

所有量必须由同一forward prediction构造；禁止读取test或训练新candidate。

## Gate

只有当至少两个datasets在decoder/readout shared parameters上显示稳定的measure-dependent gradient差异，
且差异不能仅由global scalar norm解释，SC2才进入Step 4。否则回Step 2重定义deployment risk。

SC1只有在完成FlowState/ElasTST/TimePerceiver/Implicit Forecaster的tensor-level boundary后才可提出新core
idea；当前diagnostic不授权decoder implementation。
