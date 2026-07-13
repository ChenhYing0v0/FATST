# StageC D1 PMFO/PIR Offline Diagnostic Report

## Protocol Audit

- datasets: `Weather, ETTm1, ETTh2`；seed/profile instances: `9`；
- test split: `false`；new forecast-model training: `false`；
- primary space: evaluation-space future deviation / frozen-A6 residual；
- diagnostics: nested energy、frozen-encoder ridge probes、frozen-decoder counterfactual、learned-basis geometry、measure/projected gradients。

## PMFO Evidence

| Dataset | Label adv. | Residual adv. | Full R2 | Shuffle gain | Linear | Frozen R2 | Order effect | Encoder |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Weather | 0.6600 | 0.6115 | 0.1389 | 0.2003 | pass | 0.2164 | 0.3800 | pass |
| ETTm1 | 0.7646 | 0.7462 | 0.3077 | 0.5108 | pass | 0.3339 | 0.6834 | pass |
| ETTh2 | 0.7657 | 0.7410 | -0.9500 | 0.0189 | fail | 0.2284 | 0.3603 | pass |

[Decision] PMFO problem gate: `pass`。Label structure passes=`3/3`，residual structure passes=`3/3`，encoder sufficiency passes=`3/3`。
[Scope] Encoder gate只证明当前frozen decoder确实利用有序patch memory，不等价于Encoder已经提供了完备的multiresolution sufficient statistics。

## Basis Geometry

| Dataset | Learned label@256 | DCT label@256 | Learned residual@256 | Effective rank | Entropy | Support90 |
| --- | --- | --- | --- | --- | --- | --- |
| Weather | 0.8151 | 0.9029 | 0.7478 | 189.9121 | 0.8895 | 0.4459 |
| ETTm1 | 0.8199 | 0.9777 | 0.7359 | 210.2542 | 0.8894 | 0.4459 |
| ETTh2 | 0.7819 | 0.9756 | 0.7201 | 211.9083 | 0.8892 | 0.4454 |

[Fact] 当前 learned basis 无 nested/refinement constraint；本表只判断其容量、条件与局部化几何，不能把高 capture 解释为 PMFO 已存在。

## PIR Evidence

| Dataset | Parseval gap | Parseval cosine | Raw measure sep. | Projected excess | Measure | Projected |
| --- | --- | --- | --- | --- | --- | --- |
| Weather | 0.0000 | 1.0000 | 0.4177 | 0.1235 | pass | pass |
| ETTm1 | 0.0000 | 1.0000 | 0.3417 | 0.1015 | pass | pass |
| ETTh2 | 0.0000 | 1.0000 | 0.3898 | 0.1388 | pass | pass |

[Decision] PIR problem gate: `pass`。Parseval invariant=`pass`，measure passes=`3/3`，projected-excess passes=`3/3`。

### Per-Measure Separation Audit

| Dataset | Raw uniform | Raw log | Raw benchmark | PIR uniform | PIR log | PIR benchmark |
| --- | --- | --- | --- | --- | --- | --- |
| Weather | 0.2064 | 0.7890 | 0.2575 | 0.0102 | 0.3594 | 0.0011 |
| ETTm1 | 0.1891 | 0.6223 | 0.2137 | 0.0117 | 0.2895 | 0.0033 |
| ETTh2 | 0.1354 | 0.8279 | 0.2062 | 0.0066 | 0.4059 | 0.0038 |

[Scope] Aggregate PIR gate必须结合per-measure列解释，不能用log-uniform的强差异替代benchmark measure下的独立证据。

## Overall Decision And Failure Attribution

[Decision] `pmfo_pir_problem_gate_passed`。

若 Parseval invariant失败，本诊断只能标记 `diagnostic_invalid_for_direction_rejection`，因为projection或gradient实现存在问题。若invariant通过但structure/probe/gradient gate失败，结论只针对当前PMFO/PIR problem formulation；不得扩大为所有 multiresolution decoder 或 training strategy 的方向级否决。
