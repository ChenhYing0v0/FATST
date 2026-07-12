# StageC SC0-DAP-R2 Phase A Patch Screen Report

## Decision

- `runs`: 9/9
- `errors`: 0
- `selection_split`: validation
- `parameter_count_used_for_selection`: false
- `test_metrics_used_for_selection`: false
- `profile_hash`: `2bfb1f88d38a7dfca691302d166b543db74928813653873ffc1a05e93f285f19`
- `decision`: `phase_a_patch_selected`

固定`D=64,d_ff=128`后，dense-horizon normalized regret选择：

| Dataset | Selected patch | Macro regret | Max horizon regret | H720 MSE |
| --- | ---: | ---: | ---: | ---: |
| Weather | 12 | 0.021% | 0.166% | 0.589831 |
| ETTm1 | 24 | 0.002% | 0.018% | 0.964205 |
| ETTh2 | 12 | 0.000% | 0.000% | 0.646329 |

ETTh2 P12在8/8 horizons全部最优。Weather P12除H48略输0.166%外其余horizons最优。ETTm1 P24
几乎完全支配，P48只在H192/H288与其持平附近。

## Parameter Boundary

P12/P24/P48的active-forward params为419,216/613,904/1,006,160。参数最多的P48没有在任何dataset
被选中；但这不是penalty结果，因为selector没有读取parameter count或latency。

[Decision] Phase A通过，patch mapping冻结为Weather=P12、ETTm1=P24、ETTh2=P12。下一步只允许在该P下
执行Phase B natural width screen；不得根据Phase B结果返回修改P或打开联合grid。
