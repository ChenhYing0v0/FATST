# SC0-DAP-R2B Natural Width Screen Report

## What Was Tested

Phase A已经为Weather/ETTm1/ETTh2分别冻结`P=12/24/12`。Phase B在各自selected P下比较
`D/d_ff={32/64,64/128,128/256}`。medium width复用Phase A artifacts，新增6 runs；全部结果只读取
H48/96/144/192/288/336/512/720 validation metrics。

## Result

9个dataset-profile实例完整且配置一致，0 errors。dense normalized-regret selector得到：

| Dataset | Selected profile | Mean regret | Max regret | H720 val MSE |
| --- | --- | ---: | ---: | ---: |
| Weather | `P12/D64/ff128` | 0.1354% | 0.5816% | 0.589831 |
| ETTm1 | `P24/D32/ff64` | 0.1674% | 1.3395% | 0.954085 |
| ETTh2 | `P12/D64/ff128` | 0.1132% | 0.9060% | 0.646329 |

Weather的wide arm在H720更好，但在H48的regret为1.9673%，因此没有赢得跨dense-horizon selector。
ETTm1 narrow arm除H48外在其余七个horizons均为最优；ETTh2 medium arm除H48外在其余七个horizons
均为最优。

## Parameter Boundary

active-forward params分别为Weather/ETTh2 medium `419,216`与ETTm1 narrow `391,408`。这些数值只用于
描述和后续计算成本报告，从未进入selector，也没有对候选施加capacity matching。

## Decision And Failure Attribution

[Decision] Phase B通过并冻结上述三个selected profiles，进入selected-only multi-seed absolute stability
confirmation。该confirmation不能重新证明relative winner。

[Failure Attribution] 当前没有optimization/numeric pathology或artifact缺失。若后续多seed CV gate失败，
只能判定所选profile在当前protocol下绝对稳定性不足；不得通过读取test、扩大grid或微调dataset超参数来
修复，回滚点为StageC Step 2/3 protocol audit。
