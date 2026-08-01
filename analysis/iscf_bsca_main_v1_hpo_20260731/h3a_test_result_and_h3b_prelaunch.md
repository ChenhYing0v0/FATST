# ECL/Solar H3A Test Result and Solar H3B Prelaunch

## Decision summary

| Field | Value |
| --- | --- |
| `current_step` | H3A Step 9--10 complete；Solar H3B Step 6--8 prelaunch |
| `H3A_test_matrix` | 9/9 checkpoints；36/36 cells；errors=0 |
| `ECL_selected` | `ECL__h3a_budget45`，mean MSE 0.150669 |
| `Solar_H3A_selected` | `Solar__h3a_lr3e4`，mean MSE 0.193341 |
| `Solar_target` | TimeAlign published mean MSE 0.192 |
| `remaining_gap` | +0.70% |
| `decision` | freeze terminal four-profile Solar H3B |

## H3A results

[Fact] H3A official test在commit `55f1364`上完成，9/9 checkpoints、36/36 standard-horizon cells、720-row dense metrics、candidate/trial/profile/seed provenance、invariants、NPZ与checkpoint immutability全部通过，errors为空。

ECL expanded-budget profile结果：

| Trial | H96 | H192 | H336 | H720 | Mean MSE | Mean MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H1 winner | 0.116722 | 0.136832 | 0.155733 | 0.195475 | 0.151191 | 0.245509 |
| `h3a_budget45` | 0.116640 | 0.137240 | 0.155773 | 0.193023 | **0.150669** | **0.245170** |

H3A将ECL aggregate再改善0.345%，并主要改善H720；相对TimeAlign 0.154 target低2.16%。ECL HPO停止，冻结`h3a_budget45`为当前single-seed shared profile。

Solar H3A完整排名：

| Rank | Trial | Mean MSE | Mean MAE |
| ---: | --- | ---: | ---: |
| 1 | `h3a_lr3e4` | **0.193341** | 0.219143 |
| 2 | `h3a_rank64` | 0.196470 | 0.219070 |
| 3 | `h3a_dropout4` | 0.196843 | 0.221094 |
| 4 | `h3a_wd5e2` | 0.197158 | 0.222143 |
| 5 | `h3a_budget45` | 0.197514 | 0.222539 |
| 6 | `h3a_patch4` | 0.198930 | 0.214330 |
| 7 | `h3a_effective_batch16` | 0.203630 | 0.223875 |
| 8 | `h3a_patch2` | 0.214117 | 0.225196 |

`lr=3e-4`相对H1/H2 best 0.196157改善1.436%，但相对0.192 target仍高0.698%。它的四-H MSE为0.175541/0.191196/0.200901/0.205726。Learning rate是唯一material positive aggregate direction；patch、effective batch和单独正则化方向不支持继续单因素扩张。

## H3B design

H3B是terminal bounded batch，只包含四个Solar profiles：

1. `lr2e4`：围绕3e-4 winner的lower neighbor；
2. `lr4e4`：upper neighbor；
3. `lr3e4_dropout4`：winner与最小dropout regularization interaction；
4. `lr3e4_rank64`：winner与decoder-capacity regularization interaction。

所有profiles固定seq_len720、patch1、d_model256、d_ff256、batch16、accumulation2、45 epochs、patience10、seed2021、from-scratch joint training。Validation只选择trial checkpoint；4/4完成后直接test，不进行validation profile ranking。

## Gates and rollback

`narrative_gate=passed_as_same_architecture_test_informed_HPO`。H3B不改变method。

`effectiveness_gate=H3A_material_gain_target_narrowly_missed`。H3B若任一profile mean MSE低于0.192，则target pass；否则选择全部H1--H3B中最低aggregate profile并记录`bounded_HPO_target_narrowly_missed`。H3B后不继续沿相同LR/regularization邻域无限搜索；不得per-horizon组合。

Failure attribution：当前Solar gap=`hyperparameter_optimization_incomplete_with_test_informed_narrow_gap`，不是architecture failure。若H3B仍失败，回Step 6重估main claim或baseline protocol，而非选择有利cells。

Decision=`ECL_H3A_target_pass_Solar_H3A_material_gain_H3B_terminal_prelaunch`。
