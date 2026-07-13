# StageC D1 PMFO/PIR Offline Diagnostic Report

> [Invalidity notice, 2026-07-13] 本报告的自动v1 decision已作废，最终状态为
> `diagnostic_invalid_for_direction_rejection`。ETTh2 `full_hidden R2=-39.7831`却因shuffled更差被旧gate
> 误判pass；同时Weather/ETTh2的history-std normalized residual近似label，未可靠隔离A6 residual。
> 下列表格保留为原始审计证据，不能支持PMFO/PIR paper claim；修订协议见
> `docs/experiments/stage-c-pmfo-pir-problem-diagnostic.md`。

## Protocol Audit

- datasets: `Weather, ETTm1, ETTh2`；seed/profile instances: `9`；
- test split: `false`；new forecast-model training: `false`；
- diagnostics: label/residual nested energy、frozen-encoder ridge probes、learned-basis geometry、measure/projected gradients。

## PMFO Evidence

| Dataset | Label adv. | Residual adv. | Full R2 | Shuffle gain | Raw retention | Encoder |
| --- | --- | --- | --- | --- | --- | --- |
| Weather | 0.5593 | 0.5591 | 0.0013 | 0.0002 | nan | fail |
| ETTm1 | 0.7612 | 0.7487 | 0.1808 | 0.5062 | 1.6530 | pass |
| ETTh2 | 0.7915 | 0.7914 | -39.7831 | 11.0580 | nan | pass |

[Decision] PMFO problem gate: `pass`。Label structure passes=`3/3`，residual structure passes=`3/3`，encoder sufficiency passes=`2/3`。

## Basis Geometry

| Dataset | Learned label@256 | DCT label@256 | Learned residual@256 | Effective rank | Entropy | Support90 |
| --- | --- | --- | --- | --- | --- | --- |
| Weather | 0.5822 | 0.8785 | 0.5820 | 189.9121 | 0.8895 | 0.4459 |
| ETTm1 | 0.8123 | 0.9744 | 0.7644 | 210.2542 | 0.8894 | 0.4459 |
| ETTh2 | 0.9253 | 0.9987 | 0.9253 | 211.9083 | 0.8892 | 0.4454 |

[Fact] 当前 learned basis 无 nested/refinement constraint；本表只判断其容量、条件与局部化几何，不能把高 capture 解释为 PMFO 已存在。

## PIR Evidence

| Dataset | Parseval gap | Parseval cosine | Raw measure sep. | Projected excess | Measure | Projected |
| --- | --- | --- | --- | --- | --- | --- |
| Weather | 0.0000 | 1.0000 | 0.3143 | 0.1054 | pass | pass |
| ETTm1 | 0.0000 | 1.0000 | 0.3600 | 0.1021 | pass | pass |
| ETTh2 | 0.0000 | 1.0000 | 0.0979 | 0.0326 | pass | pass |

[Decision] PIR problem gate: `pass`。Parseval invariant=`pass`，measure passes=`3/3`，projected-excess passes=`3/3`。

## Overall Decision And Failure Attribution

[Superseded automated decision] `pmfo_pir_problem_gate_passed`；最终decision为
`diagnostic_invalid_for_direction_rejection`。

若 Parseval invariant失败，本诊断只能标记 `diagnostic_invalid_for_direction_rejection`，因为projection或gradient实现存在问题。若invariant通过但structure/probe/gradient gate失败，结论只针对当前PMFO/PIR problem formulation；不得扩大为所有 multiresolution decoder 或 training strategy 的方向级否决。
