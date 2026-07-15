# SC1-D10 Raw Scale Identifiability: Result And Rollback

## Decision Summary

| Field | Result |
| --- | --- |
| `current_step` | Contribution 1 Step 2/3 diagnostic → rollback Step 2 problem reset |
| `problem` | raw history→future relation是否支持binary或detail-level aligned scale routing？ |
| `existence_evidence` | canonical scale coordinates有partial signal，但不构成binary或detail-monotone mapping |
| `idea` | capacity-matched 16→16 sketched ridge over history DCT × future RGNB |
| `theory_check` | exact transforms、paired controls、fit/holdout temporal separation与all invariants pass |
| `design` | 5 datasets；3 sketches × 3 lambdas；canonical/history-perm/future-perm；holdout + validation |
| `narrative_gate` | not applicable；`diagnostic_only` |
| `effectiveness_gate` | not applicable；no forecast training/test/model replacement |
| `artifacts` | `dataset_matrix_cells.csv`、`dataset_binary_cells.csv`、`replicate_metrics.csv`、`gate.json` |
| `decision` | `raw_aligned_scale_not_supported_rollback_step2`；history-scale routing mainline closed |

## 1. Protocol Validity

[Fact] 五个dataset均完成。每个dataset产生2646个7×7 matrix cell rows与216个binary 2×2 rows；总计
13,230与1,080 rows。aggregate artifacts包含270个family/split replicate metrics与30个dataset-family-split
metrics。

[Fact] 所有invariants通过：

- fit/holdout的history+future observation ranges无重叠；ETTh1最小index gap为1441，合同要求至少1440；
- DCT、RGNB与sketch orthogonality通过，threshold=$10^{-8}$；
- 所有probe严格为16→16；
- no test、no checkpoint、no forecast update/training；
- all ridge solutions与$R^2$ finite。

因此D10可以用于否定预注册的raw aligned-scale hypothesis，不属于numeric/protocol-invalid diagnostic。

## 2. Frozen Gate Result

### Binary global/detail

| Gate | Required | Observed | Pass |
| --- | ---: | ---: | --- |
| interaction $\ge0.01$ datasets | 4/5 | 2/5 | no |
| both directional selectivities positive | 4/5 | 0/5 | no |
| canonical beats paired controls | 4/5 | 2/5 | no |
| positive validation replicates | 36/45 | 32/45 | no |
| holdout+validation positive | 4/5 | 2/5 | no |

[Strong Evidence] binary interaction的正值主要由ETTm1/ETTm2的global selectivity驱动；它们的detail selectivity
分别为`-0.00949/-0.03263`。ETTh1则相反：detail selectivity轻微为正，但global selectivity为负。没有任何dataset
同时满足两个directional terms，因此“global history负责global future、detail history负责detail future”的二分
并未成立。

### Detail-level monotone

| Gate | Required | Observed | Pass |
| --- | ---: | ---: | --- |
| diagonal-vs-median gain datasets | 4/5 | 4/5 | yes |
| canonical history band best in $\ge4/6$ rows | 4/5 | 0/5 | no |
| 6! mapping permutation $p\le0.05$ | 4/5 | 2/5 | no |
| canonical beats paired controls | 4/5 | 4/5 | yes |
| positive validation replicates | 36/45 | 36/45 | yes |

[Strong Evidence] canonical scale band通常比同一future row的median alternative更好，且并非完全由random group
controls解释；但它从未在任何dataset的至少4/6 detail rows中成为最佳输入。mapping permutation只在ETTm1与
ETTm2通过。因此存在某种scale-related predictive structure，但不是论文候选所需的逐层aligned routing law。

## 3. Dataset-Level Results

canonical validation aggregates：

| Dataset | binary interaction | global selectivity | detail selectivity | detail gain | best | mapping p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | -0.001459 | -0.007613 | 0.004695 | 0.044428 | 2/6 | 0.077670 |
| ETTh2 | -0.000691 | 0.000953 | -0.002334 | 0.035374 | 2/6 | 0.147018 |
| ETTm1 | 0.073690 | 0.156866 | -0.009487 | 0.076495 | 2/6 | 0.011096 |
| ETTm2 | 0.053731 | 0.140091 | -0.032628 | 0.042693 | 1/6 | 0.029126 |
| Weather | 0.000006 | 0.000024 | -0.000013 | -0.000016 | 1/6 | 0.843273 |

## 4. Exploratory Off-Diagonal Matrix

该部分在协议中明确无primary gate，只用于决定下一问题。

| Dataset | Stable qualitative pattern |
| --- | --- |
| ETTh1 | future d0/d1主要由history d1预测；future d3-d5更偏history d2 |
| ETTh2 | future d0-d1与较多fine details偏history d1；不是逐层移动 |
| ETTm1 | global/d0可预测性强；future d1更偏history d0；fine details大多接近或低于zero baseline |
| ETTm2 | global/d0主导；future d1也更偏history d0；更细details普遍不可预测 |
| Weather | 所有cells接近0，无可靠linear scale mapping |

[Exploratory] ETTh1/ETTh2之间存在一定共同off-diagonal pattern，ETTm1/ETTm2也形成另一种low-band-dominant
pattern，但两类ETT与Weather不共享统一mapping。该现象可能来自dataset dynamics、sampling frequency或
input-dependent mixing；它不能在当前artifacts上升级为新的router hypothesis。TimeMixer与Pathformer也已经覆盖
generic bidirectional/adaptive multiscale mixing primitives，因此“改成adaptive scale router”既缺少本地
cross-dataset evidence，也缺乏清晰novelty boundary。

## 5. Failure Attribution

### What failed

`hypothesis_false_at_raw_aligned_scale_level`：raw normalized history与future之间没有跨五dataset成立的binary或
detail-level monotone linear mapping，无法支撑history-scale routing作为Contribution 1的problem foundation。

### What remains open

1. nonlinear、sample-specific cross-scale dependence可能存在，但当前没有cross-dataset evidence；
2. future-side RGNB geometry、projectivity、D6 support×horizon interaction仍保留；
3. future components在不同requested-prefix losses下是否产生error/gradient responsibility冲突尚未直接审计；
4. 其他backbone可能学习不同representation，但不能据此恢复当前aligned-scale claim。

### Direction boundary

D9在A6 learned operator层失败，D10又在independent raw-data层失败。两层证据共同关闭“history multi-scale
states按future support depth进行显式aligned routing”作为当前paper mainline。该结论不依赖frozen component
replacement，也不是capacity差异造成。

linear random-sketch probe具有边界：它不能否定所有nonlinear dependence。但在没有raw linear existence、没有
unified mapping且已有强prior art的情况下，继续实现nonlinear adaptive router会变成mechanism-first搜索，不符合
本项目11-step规则。

## 6. Rollback And Next Question

[Decision] Contribution 1回Step 2 problem reset，当前无active method candidate。history-scale architecture
mainline关闭；SC2、test、joint factorial继续held。

下一问题建议收紧为`SC1-D11 Future-Component Responsibility Audit`（暂为`diagnostic_only_proposed`）：

> multi-horizon unified forecasting的主要冲突是否发生在future global/local components跨prefix measure的error与
> gradient responsibility，而不是history scale access？

D11在设计前必须完成source/theory audit，并至少分离：

1. output geometry effect vs learned-basis coordinate artifact；
2. short/long prefix error energy distribution；
3. per-component gradient conflict vs simple loss-scale imbalance；
4. RGNB、DCT与A6 learned basis controls；
5. architecture problem与training-objective problem的贡献边界。

D11 protocol冻结前不恢复SC2、不编写new decoder或loss。
