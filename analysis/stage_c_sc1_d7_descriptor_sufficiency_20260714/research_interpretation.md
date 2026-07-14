# SC1-D7 RGNB Descriptor Sufficiency Research Interpretation

## Decision Summary

| Field | Decision |
| --- | --- |
| `candidate` | `SC1-D7` |
| `role` | `diagnostic_only` |
| `current_step` | conditional diagnostic complete；return PLGO Step 6/7A end-to-end gate |
| `invariant_gate` | pass；105/105 fits、15/15 metadata、finite/freeze/parameter/projectivity全部通过 |
| `geometry_gate` | pass；compact与matched均超过PERM/RANDOM，5/5 datasets |
| `free_control_observation` | GEO显著落后free-M0，但该比较存在A6 Encoder-Decoder co-adaptation confound |
| `method_readiness_gate` | not evaluated by frozen replacement |
| `decision` | `conditional_geometry_supported_end_to_end_gate_required` |
| `method_training_authorized` | false |

## 1. What D7 Tested

D7没有训练完整forecast model，而是在冻结A6 memory上比较七个head-only arms。GEO/PERM/RANDOM在同一
width下具有相同architecture、parameters、initialization与optimizer；RANDOM descriptor逐列匹配GEO
mean/std。validation固定使用此前未消费的batches16-23，test从未加载。

D7原先试图同时回答两个不同问题：

1. **geometry attribution**：canonical RGNB descriptors是否比permuted/random descriptors更有用？
2. **frozen compatibility**：descriptor-generated table能否接近与该A6 memory原生共适配的free-M0？

第2项不是end-to-end method readiness。free-M0兼容A6训练形成的representation，而PAF没有机会反向塑造
Encoder；因此两项不能被一个总pass/fail混合。

## 2. Overall Results

| Width | GEO vs controls MSE | GEO vs controls MAE | Positive datasets | Fit-holdout gain gap | GEO vs free-M0 MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| compact-256 | +13.8034% | +9.8581% | 5/5 | +0.4627% | -37.3836% |
| matched-694 | +12.8418% | +9.3269% | 5/5 | +0.3937% | -39.1031% |

[Strong Evidence] true geometry不是无效标签。两种width、MSE/MAE和全部五个datasets方向一致，且fit相对
holdout的额外优势小于1 percentage point，不符合“只记住train descriptors”的解释。

[Strong Evidence] PAF不是A6 frozen representation上的drop-in replacement。即使matched-694 readout
parameters约等于free-M0，GEO仍落后39.10%；扩大trunk没有关闭该compatibility gap。

[Boundary] 该gap不能区分descriptor-only function restriction、Encoder-Decoder co-adaptation与joint
optimization的缺失，因此不能用于判定完整PAF architecture失败。

## 3. Cross-Dataset Results

| Dataset | Compact geometry gain | Matched geometry gain | Compact free gap | Matched free gap |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | +6.63% | +6.33% | -40.34% | -40.87% |
| ETTh2 | +16.26% | +14.13% | -21.41% | -24.61% |
| ETTm1 | +18.55% | +16.85% | -73.60% | -77.91% |
| ETTm2 | +11.26% | +10.95% | -30.89% | -30.66% |
| Weather | +16.31% | +15.95% | -20.68% | -21.46% |

15/15 dataset-checkpoint units在两个width下均为positive geometry effect。ETTm1同时给出最大geometry gain和
最大free gap，说明“geometry在A6 memory上有用”与“PAF不兼容该frozen representation”可以同时成立；不能据此
断言descriptor-only function class过窄。

## 4. Horizon Pattern

| Horizon | Compact geometry gain | Matched geometry gain | Compact free gap | Matched free gap |
| ---: | ---: | ---: | ---: | ---: |
| 48 | +36.82% | +35.91% | -41.32% | -43.51% |
| 96 | +20.87% | +18.84% | -50.73% | -54.67% |
| 144 | +15.78% | +14.63% | -46.46% | -48.56% |
| 192 | +12.29% | +11.11% | -42.63% | -44.62% |
| 288 | +8.48% | +7.63% | -36.04% | -37.37% |
| 336 | +7.37% | +6.67% | -33.39% | -34.42% |
| 512 | +5.12% | +4.63% | -26.50% | -27.20% |
| 720 | +3.70% | +3.33% | -21.99% | -22.47% |

[Strong Evidence] geometry gain随horizon增长单调减弱，在H48最强。这与D6所确认的short-prefix/local-support
需求一致，形成跨诊断闭环；它不是一个只在H720 aggregate上出现的偶然head effect。

## 5. Optimization Audit

- free-M0 mean best epoch=`18.3`，range=`5-34`；
- GEO compact/matched mean best epoch=`119.5/118.9`，多数run触及120 epoch cap；
- 全部loss finite，没有divergence、>100% degradation或prefix/numeric invariant failure；
- compact/matched的geometry direction一致，matched width没有性能恢复。

[Uncertainty] factorized PAF收敛慢，optimization convergence尚未被完全排除。更重要的是，head-only训练无法
更新Encoder，所以不能将主要嫌疑指定为descriptor-only function restriction。D7既不能写成方向级拒绝，也
不能写成method pass。

## 6. Failure Attribution

1. `hypothesis_false`：**否**。canonical geometry相对两个matched controls形成强、跨dataset正向证据；
2. `intervention_point_wrong`：**possible**。descriptor进入primary coefficient-row generation，但其上游A6
   memory是为另一Decoder共同学习的；
3. `readout_or_head_design_wrong`：possible，不能与co-adaptation confound分离；
4. `optimization_or_numeric_pathology`：无numeric pathology；epoch-cap提示slow optimization未完全排除；
5. `capacity_control_explains`：free-M0解释frozen compatibility gap，但不能解释GEO相对PERM/RANDOM的收益；
6. `protocol_fairness`：**fail for method readiness**。frozen component曾与control head共同训练。

因此，远端analyzer最初的`close_paf` raw label过度扩大了失败边界。hard method gate仍然失败，但依据项目
failure-attribution rule，正确decision应为：

> `conditional_geometry_supported_end_to_end_gate_required`

## 7. End-to-End Return Question

下一步不继续在frozen A6 memory上调PAF width/epoch，也不直接叠加MoE或MIPR。返回PLGO Step 6修正
method contract，并进入Step 7A/D8-E2E：

> 当Encoder与PAF共同训练时，RGNB geometry能否相对A6与PERM/RANDOM controls形成稳定、跨dataset的
> end-to-end收益？

primary screen必须from-scratch joint training全部Encoder/Decoder parameters；旧A6 checkpoint不得进入PAF
primary arm。capacity-preserving redesign只有在stable end-to-end PAF仍失败后才有依据。

## 8. 11-Step Record

| Field | Record |
| --- | --- |
| `current_step` | D7 conditional diagnostic complete；return Step 6/7A end-to-end gate |
| `problem` | frozen A6 representation compatibility与完整architecture effectiveness被旧gate混合 |
| `existence_evidence` | conditional GEO vs controls +12.84%-13.80%，5/5 datasets；free gap不作method gate |
| `idea` | PLGO-PAF descriptor-generated atom table |
| `theory_check` | projectivity pass；task-specific narrative conditional |
| `design` | seven-arm compact/matched frozen-memory conditional diagnostic |
| `narrative_gate` | geometry boundary strengthened；PAF reopened for fair E2E test |
| `effectiveness_gate` | not evaluated；D8-E2E required |
| `artifacts` | 105 fits、raw logs/metrics/history/metadata、three summaries、JSON/report |
| `decision` | conditional geometry retained；Step 7A/D8-E2E next |
