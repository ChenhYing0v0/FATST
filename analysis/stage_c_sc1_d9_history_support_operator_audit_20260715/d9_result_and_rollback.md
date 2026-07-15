# SC1-D9-A Exact Operator Audit: Result And Rollback

## Decision Summary

| Field | Result |
| --- | --- |
| `current_step` | Contribution 1 Step 4 diagnostic result → rollback Step 2/3 |
| `problem` | A6 operator是否存在history-scale × future-support逐层可识别耦合？ |
| `existence_evidence` | weak binary global-vs-detail pattern；ordered multiscale evidence不足 |
| `idea` | exact $W=BC$ decomposition over future RGNB × history patch DCT |
| `theory_check` | exact linear map与orthogonal Parseval invariant成立 |
| `design` | 5 datasets × 3 frozen A6 seeds；1024 label permutations；64 random bases |
| `narrative_gate` | not applicable；`diagnostic_only` |
| `effectiveness_gate` | not applicable；无训练、无test、无forecast replacement |
| `artifacts` | `raw/unit_metrics.csv`、`raw/group_profiles.csv`、`raw/control_distributions.csv`、`raw/gate.json` |
| `decision` | `operator_scale_hypothesis_not_supported`；D9-B取消；rollback Step 2/3 |

## 1. Protocol Validity

[Fact] 15/15 frozen natural A6 checkpoints均完成，覆盖ETTh1、ETTh2、ETTm1、ETTm2、Weather与
seeds 2021/2022/2023。实验不读取train/validation/test sample，不训练probe，不冻结替换component，也不更新
forecast model。

[Fact] max `parseval_relative_gap=7.5381e-16`，远低于预注册$10^{-8}$阈值。checkpoint/profile/readout contract
完整，故该结果不是numeric、implementation或protocol pathology。

## 2. Pre-Registered Gate Result

| Gate | Threshold | Observed | Pass |
| --- | --- | ---: | --- |
| five-dataset macro `scale_rho` | $\ge0.25$ | `0.173810` | no |
| positive rho + contrast datasets | $\ge4/5$ | `2/5` | no |
| positive-rho checkpoint units | $\ge12/15$ | `11/15` | no |
| atom-label permutation datasets | $\ge4/5$ | `1/5` | no |
| random-history-basis datasets | $\ge4/5$ | `0/5` | no |
| Parseval invariant | all $\le10^{-8}$ | max `7.5381e-16` | yes |

dataset-level aggregates：

| Dataset | scale rho | fine-global contrast | permutation p | random-basis percentile |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | 0.119048 | 0.036083 | 0.281951 | 0.796875 |
| ETTh2 | -0.023810 | 0.037549 | 0.511220 | 0.437500 |
| ETTm1 | 0.059524 | 0.074949 | 0.345366 | 0.562500 |
| ETTm2 | 0.380952 | 0.129929 | 0.013659 | 0.859375 |
| Weather | 0.333333 | 0.032271 | 0.077073 | 0.843750 |

[Strong Evidence] 只有ETTm2通过atom-label permutation，五个dataset均未达到random-history-basis 95th
percentile。即使部分dataset的rho为正，也不能证明ordered DCT history scale比任意orthogonal coordinates更能
解释future support hierarchy。

## 3. What The Seven-Level Profiles Actually Show

three-seed mean history-frequency centroid：

| Dataset | global | d0 | d1 | d2 | d3 | d4 | d5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 0.4663 | 0.5044 | 0.5041 | 0.5088 | 0.5264 | 0.5012 | 0.5024 |
| ETTh2 | 0.4656 | 0.5158 | 0.5304 | 0.5415 | 0.5026 | 0.5026 | 0.5032 |
| ETTm1 | 0.4342 | 0.5531 | 0.5091 | 0.5195 | 0.5179 | 0.5081 | 0.5091 |
| ETTm2 | 0.3772 | 0.4992 | 0.5420 | 0.5107 | 0.5110 | 0.5054 | 0.5071 |
| Weather | 0.4786 | 0.4082 | 0.5323 | 0.5141 | 0.5115 | 0.5104 | 0.5109 |

[Fact] d5-global contrast在15/15 units为正，但进入detail后，d0-d5并不逐层上升，而是在约0.50附近摆动。
把六个detail groups平均后，binary detail-global contrast也是15/15为正，macro=`0.066384`。

[Exploratory] 这提示A6可能形成“global smooth root vs non-global detail”的粗二分，而不是可供decoder逐层路由的
multiscale ladder。该binary statistic是在查看primary result后提出，且没有预注册binary-specific permutation与
random-basis gate；因此它只能生成下一问题，不能升级为confirmatory evidence或挽救D9-A。

## 4. Failure Attribution

### What failed

`hypothesis_false_at_memory_operator_level`：当前A6 exact decoder map不支持“future support越细，就逐级更多读取
history fine-scale coordinates”的跨dataset规律。特别是ETTh2 three-seed mean rho为负，ETTh1/ETTm1很弱；
matched controls也没有支持DCT scale specificity。

### What did not fail

1. RGNB future geometry与projectivity并未被否定；
2. raw history中是否存在global/detail或更复杂cross-scale predictability尚未测试；
3. 其他Encoder是否能产生native multiresolution states尚未测试；
4. binary global-vs-detail decomposition只得到exploratory clue，未获得独立确认；
5. D9不评价任何新method的MSE/MAE。

### Direction-level boundary

关闭“在当前A6 memory上按多层scale一一对齐history states与future RGNB depths”的直接decoder设计。由于exact
operator本身没有该结构，继续实现scale-matched transport会再次把叙事先验强加给carrier。这个结论比frozen
replacement更强，因为D9分析的是A6自身jointly trained decoder，不存在cross-swap公平性问题。

## 5. Rollback And Next Research Question

[Decision] 按预注册gate取消D9-B input-Jacobian confirmation，不用post-hoc binary metric改变原decision。
Contribution 1回Step2/3，当前仍无active method candidate；model implementation、test、SC2与joint factorial
继续false。

下一问题收紧为：

> raw history→future关系究竟支持`global vs detail`二分、支持非单调的cross-scale coupling，还是根本不需要
> history-scale routing？

下一步只允许设计`SC1-D10 Raw History–Future Scale Identifiability`（`diagnostic_only`）：

1. 使用raw normalized history/future validation samples，而不是A6 decoder weights；
2. 将binary global/detail与seven-level monotone scale作为两个预注册、互斥hypotheses；
3. 使用capacity-matched low/high/random history subspaces与disjoint fit/holdout windows；
4. 若raw data支持binary而不支持monotone，才允许Step4重构为two-block non-exchangeable projective operator；
5. 若两者均不支持，停止history-scale architecture叙事，转向future-side projective regularization或重新评估
   Contribution 1 problem。

在D10 Step2/3 protocol冻结前，不编写method code或启动新的forecast training。
