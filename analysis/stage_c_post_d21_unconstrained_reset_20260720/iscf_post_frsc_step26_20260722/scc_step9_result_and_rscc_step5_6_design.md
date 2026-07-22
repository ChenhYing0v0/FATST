# ISCF-SCC Step9 Result and RSCC Step5–6 Design

## 1. Long-stage record

| Field | Record |
| --- | --- |
| `current_step` | SCC-v0 Step9 failed；rollback Step5；RSCC-v1 exact hybrid frozen |
| `problem` | v0删除equal-skill后arm reliability/headroom崩溃，coalition calibration没有可用成员可协调 |
| `existence_evidence` | 25/25 artifacts与100 cells完整；SCC输给EQUAL及全部controls；parent仍有正headroom与D0B access |
| `idea` | 保留uniform individual reliability loss，只把exact coalition credit作为existing policy calibration |
| `theory_check` | reliability与coalition utility职责分离；credit仍stop-gradient、inference unchanged |
| `design` | EQUAL parent + EQUAL-ARMERR + RSCC + RSCC-SHUFFLED；3 new arms × 5 datasets |
| `narrative_gate` | `conditional_pass_to_exact_hybrid_step7a_only` |
| `effectiveness_gate` | SCC-v0 failed；RSCC not evaluated |
| `artifacts` | Step9 CSV/JSON、training/internal health、comparison scorecard |
| `decision` | `close_scc_v0_allow_single_reliability_preserving_successor` |

## 2. SCC-v0 four-layer result

### 2.1 Paper-facing effectiveness layer

本轮只使用validation，不建立paper-facing test effectiveness。SCC相对EQUAL的validation MSE/MAE为
`-3.1750%/-1.7742%`，仅2/20 MSE cells正向，dataset wins=`0/5`，horizon wins=`0/4`。因此validation
continuation gate明确失败，seed confirmation和formal test均不授权。

### 2.2 Matched attribution layer

| Comparison | MSE gain | MAE gain | MSE cells won |
| --- | ---: | ---: | ---: |
| SCC vs FUSED | `-0.0150%` | `-0.0323%` | 8/20 |
| SCC vs ARMERR | `-0.1663%` | `-0.3388%` | 2/20 |
| SCC vs SHUFFLED | `-0.0428%` | `-0.1545%` | 8/20 |

coalition calibration没有超过删除equal-skill本身、standalone-error credit或随机binding，故不能把任何局部收益归因SCC。

### 2.3 Internal mechanism health

all artifacts finite，5/5 scopes均有nonzero usage和gradient；不存在OOM/divergence。问题发生在可学习对象本身：

- EQUAL parent median target-visible coalition headroom=`+18.0775%`；
- SCC-v0 median headroom=`-14.9326%`；
- SCC training credit entropy约`.49–.51`，policy entropy仍约`.89–.99`；
- final credit-policy KL约`.39–.49`，argmax alignment仅`.27–.37`。

去掉individual reliability loss后，ETTm1/ETTm2/Weather的target-visible coalition reweighting反而显著变差，说明arms不再形成
可被该credit稳定协调的成员集合。

### 2.4 Failure attribution

Decision=`scc_v0_failed_return_step5_reliability_preserving_design`，failure=`intervention_point_wrong`。不是
`optimization_or_numeric_pathology`；也不能从D0/D0B oracle signal推断v0有效。exact v0关闭，不做seed、lambda、epsilon、
fallback或router-width rescue。

## 3. Why one hybrid successor remains justified

D0/D0B是在EQUAL parent上得到的：该parent既有稳定arm skill，又有`+18%` headroom和target-free可预测credit。v0同时删除
reliability term与加入SCC，虽然FUSED control隔离了平均效果，但也让credit target随arm deterioration失效。

因此唯一未测试的必要组合是：保留anti-starvation/reliability supervision，并额外校准coalition policy。它不是预设第二个
contribution；`equal_skill`只作已有稳定训练基底，method claim仍只能来自SCC是否超过matched controls。

external primary-source audit已确认individual expert loss、global+expert loss和leave-one-expert routing primitives均有prior。
RSCC不能claim这些组件首创；边界仍是ISCF future-output scope fusion algebra下的task-specific reliability/coalition coupling。

## 4. RSCC-v1 exact objective

保留EQUAL parent的harmonic fused L1与uniform individual arm L1：

$$
\mathcal L_{reliability}=\mathcal L_{fused}
+\frac{1}{5}\sum_s\mathcal L_{arm_s}.
$$

使用与v0完全相同的detached leave-one-scope positive credit $q^{coalition}$：

$$
\mathcal L_{RSCC}=\mathcal L_{reliability}
+\lambda(\tau)KL(q^{coalition}\|P),
$$

其中$lambda$前25% updates线性升到`.1`，其余固定；epsilon=`1e-6`、all-negative uniform fallback不变。不做grid。
credit KL只更新policy；reliability/fused losses保持joint E2E。inference graph仍完全等于ISCF-v0。

## 5. Matched validation matrix

| Arm | Objective | Role |
| --- | --- | --- |
| `ISCF-EQUAL` | fused + uniform arm reliability | reused parent |
| `ISCF-EQUAL-ARMERR` | EQUAL + standalone-error route KL | closest credit control |
| `ISCF-RSCC` | EQUAL + coalition route KL | candidate |
| `ISCF-RSCC-SHUFFLED` | EQUAL + shuffled coalition KL | binding control |

3 new arms × 5 datasets × seed2021=`15` runs，parent 5 runs复用；same initialization、profile、optimizer、checkpoint
selector与validation surface。不得复用v0 checkpoints或warm-start。

continuation gate冻结为：RSCC vs EQUAL macro MSE至少`+.3%`、MAE正、3/5 datasets和3/4 horizons正；RSCC vs
EQUAL-ARMERR与SHUFFLED的macro MSE各至少`+.1%`；all finite、5-scope gradients nonzero、policy-credit alignment提高且
headroom保持正。任何一项失败均关闭exact SCC/RSCC route，不再做coefficient/seed rescue。

## 6. Authorization

```text
SCC_v0 = closed
active_method = SC-ISCF-RSCC-v1_narrative_ready_preimplementation
RSCC_step7a_implementation_authorized = true
remote_training_authorized = false_until_step7a_and_resource_smoke
formal_test_authorized = false
```
