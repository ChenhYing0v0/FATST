# SC1-D7 RGNB Descriptor Sufficiency Research Interpretation

## Decision Summary

| Field | Decision |
| --- | --- |
| `candidate` | `SC1-D7` |
| `role` | `diagnostic_only` |
| `current_step` | Step 9-10 complete；return Step 4 redesign |
| `invariant_gate` | pass；105/105 fits、15/15 metadata、finite/freeze/parameter/projectivity全部通过 |
| `geometry_gate` | pass；compact与matched均超过PERM/RANDOM，5/5 datasets |
| `free_control_gate` | fail；GEO仍显著落后free-M0 |
| `method_readiness_gate` | fail |
| `decision` | `descriptor_geometry_supported_paf_not_ready_return_step4` |
| `method_training_authorized` | false |

## 1. What D7 Tested

D7没有训练完整forecast model，而是在冻结A6 memory上比较七个head-only arms。GEO/PERM/RANDOM在同一
width下具有相同architecture、parameters、initialization与optimizer；RANDOM descriptor逐列匹配GEO
mean/std。validation固定使用此前未消费的batches16-23，test从未加载。

D7同时回答两个不同问题：

1. **geometry attribution**：canonical RGNB descriptors是否比permuted/random descriptors更有用？
2. **method readiness**：descriptor-generated table能否接近保留完整自由度的free-M0？

这两个问题不能被一个总pass/fail混合。

## 2. Overall Results

| Width | GEO vs controls MSE | GEO vs controls MAE | Positive datasets | Fit-holdout gain gap | GEO vs free-M0 MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| compact-256 | +13.8034% | +9.8581% | 5/5 | +0.4627% | -37.3836% |
| matched-694 | +12.8418% | +9.3269% | 5/5 | +0.3937% | -39.1031% |

[Strong Evidence] true geometry不是无效标签。两种width、MSE/MAE和全部五个datasets方向一致，且fit相对
holdout的额外优势小于1 percentage point，不符合“只记住train descriptors”的解释。

[Strong Evidence] exact PAF v1也没有method readiness。即使matched-694 readout parameters约等于free-M0，
GEO仍落后39.10%；扩大trunk没有关闭gap，反而略弱于compact。这说明问题不是简单parameter count不足。

## 3. Cross-Dataset Results

| Dataset | Compact geometry gain | Matched geometry gain | Compact free gap | Matched free gap |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | +6.63% | +6.33% | -40.34% | -40.87% |
| ETTh2 | +16.26% | +14.13% | -21.41% | -24.61% |
| ETTm1 | +18.55% | +16.85% | -73.60% | -77.91% |
| ETTm2 | +11.26% | +10.95% | -30.89% | -30.66% |
| Weather | +16.31% | +15.95% | -20.68% | -21.46% |

15/15 dataset-checkpoint units在两个width下均为positive geometry effect。ETTm1同时给出最大geometry gain和
最大free gap，说明“geometry有用”与“descriptor-only function class过窄”可以同时成立。

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

[Uncertainty] factorized PAF收敛慢，因而optimization convergence尚未被完全排除。但free gap达到22%-78%，
且width从256增至694没有改善，因此当前主要嫌疑仍是descriptor-only row manifold/function-class restriction，
不是简单增加epoch即可解决。不能把D7写成方向级拒绝，也不能把它写成method pass。

## 6. Failure Attribution

1. `hypothesis_false`：**否**。canonical geometry相对两个matched controls形成强、跨dataset正向证据；
2. `intervention_point_wrong`：未支持。descriptor直接进入primary coefficient-row generation，不是late auxiliary；
3. `readout_or_head_design_wrong`：**strongly suspected**。free-M0显著更强，matched width不能恢复；
4. `optimization_or_numeric_pathology`：无numeric pathology；但epoch-cap提示slow optimization未完全排除；
5. `capacity_control_explains`：free function class解释绝对性能差距，但不能解释GEO相对PERM/RANDOM的收益。

因此，远端analyzer最初的`close_paf` raw label过度扩大了失败边界。hard method gate仍然失败，但依据项目
failure-attribution rule，正确decision应为：

> `descriptor_geometry_supported_paf_not_ready_return_step4`

## 7. Step 4 Return Question

下一步不继续调PAF width/epoch，也不直接叠加Encoder、MoE或MIPR。返回Step 4审计：

> 如何在保留free atom-table function class和A6-level expressiveness的同时，让已被D7证明有效的RGNB
> geometry原生参与coefficient generation，并能被no-geometry/permuted controls清晰归因？

优先审计capacity-preserving geometry-conditioned atom table、geometry-aware parameterization/initialization与
matched no-geometry controls；在source/theory audit前不冻结具体method，也不授权Step 7。

## 8. 11-Step Record

| Field | Record |
| --- | --- |
| `current_step` | D7 Step 9-10 complete；return Step 4 |
| `problem` | descriptor geometry有用，但descriptor-only generator损失过多free-table expressiveness |
| `existence_evidence` | GEO vs controls +12.84%-13.80%，5/5 datasets；free gap -37.38%~-39.10% |
| `idea` | PLGO-PAF descriptor-generated atom table |
| `theory_check` | projectivity pass；task-specific narrative conditional |
| `design` | seven-arm compact/matched frozen-memory diagnostic |
| `narrative_gate` | geometry boundary strengthened；exact PAF v1 not ready |
| `effectiveness_gate` | geometry pass；free-control/method-readiness fail |
| `artifacts` | 105 fits、raw logs/metrics/history/metadata、three summaries、JSON/report |
| `decision` | return Step 4 capacity-preserving geometry redesign；method false |
