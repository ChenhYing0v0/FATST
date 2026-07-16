# StageC D14-A1 Dual-Carrier Seed2021 Review

## 0. Decision Card

| Field | Value |
| --- | --- |
| `current_step` | Step 9-10 single-seed dual-carrier review complete；Step 8 confirmation next |
| `neutral` | 40/40 complete；`neutral_problem_pass_authorize_a6_sensitivity` |
| `A6-natural` | 45/45 complete；`a6_sensitivity_confirming` |
| `test` | false |
| `problem_decision` | direct positive on both carriers；authorize seeds2022/2023 confirmation |
| `method_decision` | GroupedMLP不是paper method；H720仍落后A6-LBF；PCSD/D14-B remain held |

## 1. What A6 Sensitivity Tested

A6 sensitivity复用五数据集natural profile的`timealign-token-mlp` architecture，但encoder与GroupedMLP从头E2E
joint training。它不使用frozen replacement，因此回答的是：在当前paper carrier architecture上，改变future-output
sharing topology后，scale functions是否仍分离、crossing是否仍存在、temporal-contiguous grouping是否优于random。

五个exact A6-LBF runs只作performance compatibility reference，不参与scale problem gate。

## 2. Returned Problem Evidence

| Dataset | train-selected scale | validation-best fixed | function disagreement median | crossing | strict oracle | sample over bin-policy | contiguous vs random |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| ETTh1 | 720 | 48 | 15.0417% | yes | 8.5894% | 8.4217% | 0.4035% |
| ETTh2 | 720 | 1 | 31.1899% | yes | 8.3792% | 7.1419% | 1.2307% |
| ETTm1 | 720 | 48 | 11.3607% | yes | 7.9485% | 7.5043% | 0.4151% |
| ETTm2 | 720 | 1 | 31.1373% | yes | 13.9898% | 12.8838% | 0.3296% |
| Weather | 360 | 360 | 13.2522% | yes | 6.8452% | 6.7625% | 0.9514% |
| **Macro / count** | — | dataset-dependent | **5/5 pass** | **5/5** | **9.1504%** | **8.5429%** | **0.6661%, 5/5 positive** |

原预注册的train-selected-fixed oracle为9.9892%。为排除static scale-selection gap，review增加两个更严格但不取代
原gate的robustness quantities：

1. 相对validation-best fixed scale的oracle仍为9.1504%；
2. 相对“每个future bin事后固定一个scale”的sample增量仍为8.5429%。

因此主要headroom不是train选错一个global scale，也不只是short/mid/long各自偏好不同scale；它主要位于sample ×
future-region层面。neutral上对应strict oracle为6.9978%，sample-over-bin为6.7555%，方向一致。

## 3. A6-LBF Performance Compatibility

| Dataset | train-selected GroupedMLP vs A6-LBF H720 | validation-best GroupedMLP vs A6-LBF H720 |
| --- | ---: | ---: |
| ETTh1 | -0.3403% | +0.6836% |
| ETTh2 | -7.6451% | -5.4699% |
| ETTm1 | -2.3520% | -1.9315% |
| ETTm2 | -4.2590% | -1.5884% |
| Weather | -0.1213% | -0.1213% |
| **Macro** | **-2.9435%** | **-1.6855%** |

[Fact] fixed GroupedMLP heads没有超过A6-LBF；即使事后选validation-best scale，五数据集macro仍落后1.6855%。

[Interpretation] 这不否定scale problem，因为A6上的function separation/crossing/strict oracle均通过；但它明确否定
“把某个fixed grouped head直接当Contribution 1”的做法。A6 global basis decoder仍是更强的fixed operator，未来
PCSD必须contain或residualize A6能力，而不是用当前GroupedMLP整体替换。

## 4. Failure Attribution

- `hypothesis_false`: **not supported**；neutral与A6均5/5 crossing，strict/instance oracle显著；
- `intervention_point_wrong`: **not supported for the diagnostic**；prediction disagreement 5/5且gradient topology已验证；
- `readout_or_head_design_wrong`: **applies to method readiness**；fixed GroupedMLP相对A6-LBF性能为负；
- `optimization_or_numeric_pathology`: **not observed**；all invariants、finite与severity gates通过；
- `capacity_control_explains`: **not supported**；canonical相对matched random 5/5为正。

因此本阶段结论是`problem_gate_positive_but_method_not_ready`，不是paper Contribution 1通过。

## 5. Confirmation Gate Frozen Before New Seeds

seeds2022/2023开始前冻结以下补充要求：

1. 每dataset的crossing pair、function separation、carrier skill与contiguity至少2/3 seeds复现；
2. 至少3/5 datasets形成stable crossing；
3. five-dataset strict oracle macro至少0.5%；
4. sample-over-bin-policy macro至少0.5%；
5. contiguity至少3/5 datasets在2/3 seeds为正且macro至少0.1%；
6. neutral为primary；A6-LBF performance只报告，不参与problem pass；
7. confirmation pass只授权D14-B返回Step4-6设计，不自动实现PCSD/CCRL、不读test。

## 6. Decision

[Decision] 双carrier seed2021 evidence足以授权2022/2023 confirmation。执行顺序为每seed
`neutral -> A6`，最后运行multi-seed analyzer。D14-B、paper method与test继续held。
