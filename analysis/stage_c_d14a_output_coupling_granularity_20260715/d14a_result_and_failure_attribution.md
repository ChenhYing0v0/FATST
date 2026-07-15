# StageC D14-A0 Result And Failure Attribution

## 0. Decision Card

| Field | Value |
| --- | --- |
| `current_step` | Step 9-11 result evaluation and rollback |
| `role` | `diagnostic_only_neutral_carrier` |
| `statistical_gate` | fail：crossing/oracle/contiguity均未通过 |
| `numeric_protocol` | pass：carrier 4/5；parameter/PCA/condition/finite/split invariants pass |
| `direction_rejection` | invalid；A0 intervention contrast与effective-DoF accounting不足 |
| `failure_attribution` | `intervention_point_wrong + capacity_control_incomplete` |
| `D14-B` | held；不得启动 |
| `rollback` | Step 2-3；最多允许一次source-informed A1 redesign，不直接实现PCSD/CCRL |

## 1. What Was Actually Tested

D14-A0没有使用A6 checkpoint或frozen replacement。五数据集均从official train/validation split构造三个
chronological folds；每fold使用512 fit windows、128 train-calibration windows与256 validation windows。
history经window/channel normalization后，只用fit rows拟合PCA64。所有arms共享同一个carrier、full affine
solution与intercept，只改变future coefficient matrix的blockwise rank projection。

canonical scales为`1/48/144/360/720`，另有intermediate scale的shifted contiguous与random partition controls。
`train_selected_best`只根据train-calibration full MSE选scale。test split与forecast-model training均为false。

首轮remote launch在生成metrics metadata时，把`train_selected_best`误解析成scale arm并一致停止；commit
`df4cb25`将parser收紧为仅接受`canonical|shifted|random_s<int>`并增加synthetic collision test。修复后全部数据
从头重算，结果commit为`df4cb25`，不混用首轮partial metrics。

## 2. Gate Results

| Gate | Threshold | Observed | Result |
| --- | ---: | ---: | --- |
| carrier skill | $\ge3/5$ datasets，gain $\ge0.5\%$ | 4/5 | pass |
| stable scale crossing | $\ge3/5$ datasets | 0/5 | fail |
| sample × bin oracle headroom | macro $\ge0.5\%$ | 0.0586% | fail |
| canonical contiguity | positive $\ge3/5$，macro $\ge0.1\%$ | 0/5，-0.1427% | fail |
| factor parameter gap | $\le1\%$ | 0.5128% | pass |
| PCA orthogonality | $\le10^{-8}$ | max $2.44\times10^{-15}$ | pass |
| standardized feature condition | $\le10^6$ | max 1.0000000000002 | pass |
| finite/split/pathology | all pass | all pass | pass |

### Dataset-level summary

| Dataset | Carrier gain | Oracle gain | Canonical vs random | Stable crossing |
| --- | ---: | ---: | ---: | --- |
| ETTh1 | 13.2350% | 0.0515% | -0.1724% | false |
| ETTh2 | 14.3564% | 0.0740% | -0.1960% | false |
| ETTm1 | 22.1330% | 0.0424% | -0.1403% | false |
| ETTm2 | 15.8844% | 0.1250% | -0.2047% | false |
| Weather | -0.0162% | 0.00003% | -0.00006% | false |

Weather carrier不优于train mean，因此Weather不能提供方向级negative evidence；其余四个dataset carrier具有明显
forecast skill，但仍没有任何stable crossing。

## 3. The Most Important Observation

五个canonical arms的aggregate validation full-MSE spread极小：

| Dataset | Best scale | Worst scale | Relative spread |
| --- | ---: | ---: | ---: |
| ETTh1 | 48 | 1 | 0.00698% |
| ETTh2 | 48 | 1 | 0.01777% |
| ETTm1 | 144 | 1 | 0.01198% |
| ETTm2 | 144 | 1 | 0.04036% |
| Weather | 720 | 1 | 0.000004% |

calibration selected scale会在fold间变化，但这种变化发生在几乎相同的risk surface上；它不能被解释成
sample/region-adaptive coupling evidence。相反，oracle gain很小且random oracle更好，说明A0中可观察到的差异
主要是generic low-rank shrinkage或近似tie，而不是future order-aware block coupling。

## 4. Why This Does Not Yet Reject The Whole PCSD Problem

### 4.1 Factor params are not effective degrees of freedom

A0预注册时匹配的是factor storage count：

$$
P_s=\frac{T}{s}r_s(d+s).
$$

但rank-$r$ matrix manifold的dimension应为$r(d+s-r)$。忽略common intercept后，实际manifold DoF为：

| scale | factor params | rank-manifold DoF | vs point DoF |
| ---: | ---: | ---: | ---: |
| 1 | 46800 | 46080 | 0.00% |
| 48 | 47040 | 35280 | -23.44% |
| 144 | 46800 | 36675 | -20.41% |
| 360 | 46640 | 40590 | -11.91% |
| 720 | 47040 | 43440 | -5.73% |

所以“factor parameter matched”成立，但“effective capacity matched”不成立。random controls能检验同scale下
contiguity是否有用，却不能完全排除不同scales之间的regularization/capacity解释。

### 4.2 The tested rank restriction was functionally too weak

global arm只把carrier-limited coefficient rank从最多64限制到60；intermediate arms虽然formal DoF更低，但
validation risks仍只相差$10^{-5}$到$10^{-4}$量级。A0没有保存coefficient/prediction intervention norm，因此
不能证明各arms在实际fit上形成了足够不同的forecast functions。

### 4.3 Linear output-subspace projection is narrower than the paper hypothesis

PCSD设想的是jointly learned nonlinear decoder中future targets共享predictive computation的scope；A0只改变
同一个linear affine map的output-subspace rank。A0 exact negative可以否定“当前PCA64 + post-OLS RRR足以提供
coupling evidence”，但不能把所有joint nonlinear sharing mechanisms一起否定。

## 5. Failure Attribution

- `hypothesis_false`: **not established**；没有形成足够强且capacity-auditable的intervention contrast；
- `intervention_point_wrong`: **suspected**；post-OLS output projection没有显著改变实际forecast function；
- `readout_or_head_design_wrong`: **partially supported**；linear RRR只表达output subspace shrinkage；
- `optimization_or_numeric_pathology`: **false**；closed-form、finite、condition与split invariants通过；
- `capacity_control_explains`: **unresolved across scales / supported within scale**；factor count matched但manifold DoF
  未匹配；random partition优于canonical，未发现ordered coupling收益。

[Decision] A0的statistical gate确实失败，但其方向级结论标记
`design_fault_suspected / diagnostic_invalid_for_direction_rejection`。不得将其写成“multi-horizon coupling
granularity不存在”，也不得启动D14-B。

## 6. Rollback And Next Research Question

回滚Step 2-3。若继续该吸引人的主线，只允许一次A1 repair，且必须在实现前满足：

1. coupling intervention在function/prediction level有预注册的minimum contrast，不再只比较名义rank；
2. capacity使用effective degrees of freedom或等价统计复杂度匹配，不把factor storage count当作DoF；
3. canonical、shifted、random controls在同一effective capacity下比较；
4. primary carrier在5/5或明确排除无skill dataset后有效；
5. A1仍无stable crossing/headroom则关闭PCSD/CCRL pair，不继续换head死磕。

最值得source-informed评估的A1形式是full-rank structured shrinkage：用block projector/Laplacian改变output
coefficient sharing，同时通过linear-smoother generalized degrees of freedom匹配总complexity。它目前只是
下一步研究问题，不是已授权实现；应先完成source/theory feasibility与closed-form identifiability audit。

## 7. Artifacts

- remote raw：`analysis/stage_c_d14a_output_coupling_granularity_20260715/raw/`；
- independent local reanalysis：`analysis/stage_c_d14a_output_coupling_granularity_20260715/local_reanalysis/`；
- exact gate：`local_reanalysis/gate.json`；
- row-level validation bin losses：每dataset三个`validation_bin_losses_fold*.npz`；
- code/design：commit `978a405`；parser repair：commit `df4cb25`。
