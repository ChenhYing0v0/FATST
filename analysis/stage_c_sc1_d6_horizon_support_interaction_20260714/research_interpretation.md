# SC1-D6 Horizon-Support Interaction: Research Interpretation

## Decision

`decision = horizon_support_scale_interaction_supported_return_step4`。

[Strong Evidence] 在D5未使用的official validation batches 8-15上，fixed b144 local DCT再次表现为short
horizons优于global DCT、long horizons反向。该结果通过所有预注册MSE、MAE、cross-unit与cross-dataset gates，
因此support-scale × horizon不是D5 evaluation window偶然现象。

## Validity

- 225/225 fits、15/15 metadata；
- validation batch offset固定为8，D5为0；
- test未加载，A6与training data未修改；
- all metrics finite，orthogonality通过；
- candidate在launch前固定为`block_dct2_b144`，没有使用新window重新选择。

## Gate Results

| Estimand | Result | Gate |
| --- | ---: | --- |
| short MSE reduction vs global DCT | +1.1964% | >= +0.5%，pass |
| long MSE reduction vs global DCT | -1.2675% | <= -0.5%，pass |
| short MAE reduction vs global DCT | +0.5863% | >= 0，pass |
| long MAE reduction vs global DCT | -0.7720% | <= 0，pass |
| short MSE reduction vs balanced | +2.1394% | >= +0.5%，pass |
| crossed primary units | 12/15 | >= 9，pass |
| short-positive datasets vs DCT | 4/5 | >= 3，pass |
| long-negative datasets vs DCT | 5/5 | >= 3，pass |
| short-positive datasets vs balanced | 4/5 | >= 3，pass |

逐horizon相对global DCT从H48 `+1.7121%`、H96 `+1.4640%`、H144 `+0.4083%`，在H192已转为
`-0.2912%`，随后H336/H512/H720为`-1.1992%/-1.3636%/-1.2397%`。crossing约出现在144-192之间，
与block width 144一致。

## Problem Interpretation

[Supported Hypothesis] 一个fixed global basis在long domain具有global view与energy compaction，但其短prefix
restriction仍依赖大量global atoms；fixed local basis在短prefix形成更紧的support，却丢失long-domain coherence。
因此multi-horizon unified forecasting存在一个此前被“所有H共用同一dense basis”掩盖的问题：

> 同一个future function的合适synthesis support scale随被请求的prefix domain变化，但H不应成为learned
> semantic condition。

这不是要求horizon-specific head。目标是让同一operator的domain restriction自然选择local/global supports。

## Balanced-Interval Basis Boundary

balanced interval basis仍保留组件级创新潜力：它提供nested contiguous supports与native prefix intersection；
D4否定的只是exact midpoint balancing具有独特accuracy。后续可将其作为local support scaffold，但paper-core
claim必须是local-prefix与global-domain的projective co-synthesis，而不是“首次用Haar/basis预测”。

## Failure Attribution And Next Step

不存在numeric、optimization或capacity pathology。本轮problem gate pass，但尚未测试任何新operator。
返回Step 4做external source-informed narrative audit；只有明确区别于N-HiTS、BasisFormer、FBM、WaveToken、
FlowState与Implicit Forecaster，并给出function class、perfect reconstruction、prefix restriction与matched
capacity controls后，才可进入Step 5。SC2-MIPR继续held。
