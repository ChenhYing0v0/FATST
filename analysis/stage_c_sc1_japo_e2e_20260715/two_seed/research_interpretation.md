# SC1-JAPO Step 8 Two-Seed Effectiveness Gate

## Decision Summary

| Field | Result |
| --- | --- |
| `matrix` | 5 datasets × 7 arms × seeds2021/2022；70/70 valid |
| `split` | validation-only；dense H1..720；test=false |
| `decision` | `two_seed_mean_fail_stop_and_attribute` |
| `JOINT vs A6 macro MSE` | `-1.2435%`；0/5 datasets positive |
| `JOINT vs same-bank median` | `-0.1175%`；1/5 positive |
| `capacity_control_explains` | true |
| `exact_design_rejection_authorized` | true |
| `direction_level_rejection_authorized` | false |
| `next_action` | seed2023/test/SC2停止；Contribution 1回Step 4 redesign |

## 1. What Was Tested And Why

本轮执行Step 6预注册的staged effectiveness gate。seed2021得到稳定但inconclusive结果后，唯一授权动作是保持
E2/K256/G32、independent initialization、profiles、objective、epochs与seven arms不变，补seed2022。目的不是
通过第二个seed“挽救”候选，而是区分single-seed fluctuation与可重复的method weakness。

完整JAPO必须同时证明：

1. `JOINT > A6`：新operator提升forecast effectiveness；
2. `JOINT > UNIFORM/HISTORY/ATOM`：收益不能由双expert容量、单侧conditioning或ensemble解释；
3. `JOINT > PERM/RANDOM`：canonical RGNB geometry确有必要；
4. 上述关系跨五datasets，而不是单dataset偶然性。

two-seed gate先对每个dataset/arm的dense metric求seed2021/2022 mean，再执行冻结threshold；未通过即停止exact
JAPO，不允许继续seed2023或事后调router。

## 2. Artifact Construction And Audit

- 70/70 runs含完整H1..720 `metrics_by_target_horizon.csv`；
- 70/70均为full-H720 pointwise L1、best-val、`final_split=val`、test=false；
- 70/70均from-scratch end-to-end joint training，frozen parameter tensors为0；
- 每个seed/dataset内Encoder hashes paired；六个JAPO arms的expert-bank与basis hashes paired；
- trained prefix、patch-block decomposition、finite metric与checkpoint invariants全部通过；
- 两个seed独立重算的JOINT-vs-A6 macro分别为`-1.3754%`和`-1.1129%`，均为负。

[Fact] 结果不存在frozen replacement、validation/test混用、NaN、divergence或initialization mismatch，因此足以拒绝
当前exact JAPO实现。

## 3. Two-Seed Dataset And Horizon Results

improvement定义为$100(1-\mathrm{candidate}/\mathrm{reference})$，正值更好。表内均为先平均两seed metrics后再
计算ratio。

| Dataset | JOINT vs A6 | vs same-bank median | short H1-96 | middle H97-336 | long H337-720 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | -0.889% | -0.241% | -2.697% | -1.527% | -0.350% |
| ETTh2 | -2.544% | -0.330% | -2.743% | -2.762% | -2.437% |
| ETTm1 | -0.693% | +0.505% | +0.215% | -0.711% | -0.773% |
| ETTm2 | -1.405% | -0.325% | -3.268% | -1.800% | -1.073% |
| Weather | -0.686% | -0.196% | -3.217% | -0.623% | -0.338% |

[Strong Evidence] JOINT相对A6为0/5正向；15个dataset-segment cells中只有ETTm1 short为正。ETTh2在全部区间
稳定差约2.4%–2.8%，ETTm2与Weather的short区间也差约3.2%。这不是由单一long horizon拖累的结果。

[Positive but Insufficient] ETTm1相对same-bank median为`+0.505%`，且short segment相对A6为`+0.215%`；说明
joint routing并非在所有数据分布上都完全无效，但只有1/5 dataset，不能成为paper-core evidence。

## 4. Same-Bank Attribution

| Reference | JOINT macro improvement | Positive datasets |
| --- | ---: | ---: |
| UNIFORM | -0.409% | 1/5 |
| HISTORY | -0.148% | 1/5 |
| ATOM | -0.404% | 1/5 |
| JOINT-PERM | +0.223% | 4/5 |
| JOINT-RANDOM | +0.126% | 3/5 |
| per-dataset control median | -0.118% | 1/5 |

[Strong Evidence] canonical geometry相对PERM/RANDOM保留了小而较一致的正向信号，但JOINT系统性不及
UNIFORM/HISTORY/ATOM。也就是说，geometry signal可能真实，完整history-atom interaction却没有产生超过简单
same-bank机制的收益。

预注册的capacity hard gate要求`vs same-bank median <= 0`且positive datasets不超过1；当前为`-0.1175%`、
1/5，故`capacity_control_explains=true`。这里的“capacity control explains”并非说所有arms完全相同，而是说
JAPO完整机制的收益不能排除双expert bank与简单routing方式的解释。

## 5. Routing And Optimization

seed2022 JOINT normalized entropy为：ETTh1 `0.999346`、ETTh2 `0.999821`、ETTm1 `0.997914`、ETTm2
`0.997629`、Weather `0.993389`；seed2021最低值也为`0.993263`。两个seed均接近uniform routing。

[Strong Evidence] under-specialization可重复，不是seed2021偶然现象。JOINT与UNIFORM/HISTORY/ATOM之间的微小差距
也与“router只做弱扰动”的解释一致。

[Self-Critique] 高entropy仍不等价于router数学上恒定，且不能单独证明intervention point必然错误。但在two-seed
effectiveness fail与same-bank hard gate同时成立时，它足以成为下一轮Step 4 redesign的mechanism clue，而不是
继续对exact v1做temperature或auxiliary-loss tuning的理由。

## 6. Failure Attribution

| Cause | Judgment | Reason |
| --- | --- | --- |
| `hypothesis_false` | not established | D6仍支持horizon-support interaction problem；本轮只测一种operator realization |
| `intervention_point_wrong` | plausible | history-atom gate对expert mixture的实际调制过弱 |
| `readout_or_head_design_wrong` | supported for exact v1 | JOINT 0/5不及A6，且不及三个简单same-bank controls |
| `optimization_or_numeric_pathology` | numeric false；optimization weakness plausible | finite/invariants pass；但两seed router均near-uniform |
| `capacity_control_explains` | true | vs control median `-0.1175%`、仅1/5正向 |

[Decision] `SC1-JAPO exact v1`降为`failed_as_core_candidate`，seed2023停止。该结论不否定A6 containment、RGNB
projectivity、canonical geometry signal或“conditional projective operator”大方向；它否定的是当前
`two free RGNB experts + weak factorized softmax gate`作为paper-core实现。

## 7. Rollback And Next Research Question

回到11-step loop的Step 4，而不是Step 2/3：问题存在性与geometry scaffold仍有证据，失败发生在method
intervention/readout层。下一步先做source-informed redesign audit，研究问题收紧为：

> 如何让history与future-support geometry改变operator本身，而不是只对两个近似同质、近uniform使用的experts
> 做弱convex mixing，同时保持requested horizon只定义active domain与exact projectivity？

在新的Step 4-6 narrative/theory gate通过前，不实现新architecture，不启动test、SC2-MIPR或joint factorial。
