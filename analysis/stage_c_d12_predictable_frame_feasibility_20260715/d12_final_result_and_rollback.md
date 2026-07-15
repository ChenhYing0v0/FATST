# StageC D12 Final Result And Step 2 Rollback

## Decision Summary

| Field | Decision |
| --- | --- |
| `current_step` | D12 Step 9-10 completed；rollback to joint Contribution 1/2 Step 2 |
| `D12-A-v1` | execution valid，但uniform normalized measure错误；不能方向级否定 |
| `D12-A-v2` | risk-aligned protocol valid；support `1/5`，required `3/5` |
| `CAPE` | `failed_problem_gate / closed_as_core_candidate` |
| `PRISM` | 未被单独证伪，但joint D12 route与D12-B关闭；`retired_without_effectiveness_test` |
| `D12-B` | canceled；不得用standalone PRISM probe绕过前置gate |
| `method/test` | false / false |
| `paper impact` | post-D11 predictable-frame主线撤回；两个contribution slots重新开放 |

## 1. What Was Actually Run

D12-A-v1完成五dataset、two-fold purged forward cross-fitting；v2复用完全相同的10个A6 pilot checkpoints，
只将统计risk从uniform normalized rows修复为history-std-squared weighted rows：

$$
w_n=s_{x,n}^2,qquad
\Sigma_w=\frac{\sum_nw_n(z_n-\bar z_w)(z_n-\bar z_w)^T}{\sum_nw_n}.
$$

v2 remote provenance：commit=`e1db10786ff02490c2a6ca8dd0e2ff9accc8127e`；GPU 0/1/2启动时均
15 MiB used；运行区间`2026-07-15T15:40:40+08:00`至`15:41:56+08:00`；output root为
`/home/yingch/exp_outputs/r-2026-fatst/stage_c_d12_predictable_frame_feasibility_v2`。

5/5 datasets均满足：

- v1 pilot checkpoint cache hit；v2没有重新训练A6；
- train split only，validation/test=false；
- official/manual normalized forward max absolute gap=0；
- covariance symmetry/PSD与weight concentration invariants通过；
- Weight ESS fraction在`0.1327-0.5406`，max weight share在`0.00054-0.00307`。

因此v2不存在numeric pathology、data leakage或单row weight domination。

## 2. Frozen v2 Results

| Dataset | A6 OOF $R^2$ | Ridge $R^2$ | A6 trace ratio | A6 fold overlap@32 | A6 raw gap@256 | Ridge raw gap@256 | Primary support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ETTh1 | 0.1258 | 0.1893 | 0.4994 | 0.7145 | 0.016021 | 0.004580 | pass |
| ETTh2 | -0.1669 | 0.0429 | 0.2375 | 0.5796 | 0.018077 | 0.007008 | fail: A6 predictability |
| ETTm1 | 0.3571 | 0.4126 | 0.5326 | 0.9513 | 0.003370 | 0.002474 | fail: rank256 headroom |
| ETTm2 | 0.0002 | 0.1200 | 0.3294 | 0.8530 | 0.002953 | 0.006231 | fail: A6 rank256 headroom |
| Weather | 0.2697 | -10.8126 | 0.3764 | 0.7695 | 0.001767 | 0.162658 | fail: A6 headroom + ridge robustness |

Primary frozen gate只支持ETTh1，得到`1/5 < 3/5`。ETTm1、ETTm2、Weather中A6-optimal frame相对
raw-label PCA frame在rank256下只多捕获`0.18%-0.34%` predictable energy，低于冻结的0.5% practical
headroom。ETTh2虽然subspace gap存在，但A6 forward OOF $R^2<0$，不能把其prediction covariance当作可靠
conditional-mean proxy。

[Strong Evidence, secondary] 没有任何dataset同时满足“A6与ridge均有正OOF $R^2$且两者raw gap@256均
$\ge0.5\%$”。因此即使放松primary-pilot身份，也没有model-bias-robust的cross-dataset CAPE headroom。

## 3. What v1 To v2 Taught Us

risk repair不是无关紧要的patch：

- ETTm1 A6 OOF $R^2$从`0.1618`升至`0.3571`；
- Weather从约`0.0002`升至`0.2697`；
- ETTm2 prediction/label trace ratio从`0.00004`升至`0.3294`。

这验证了v1 failure attribution：uniform normalized covariance确实错误地放大了small-history-std rows。
但修复后CAPE仍失败，而且失败原因变得更有解释力——不是“future不可预测”，而是当前rank256的raw-label
subspace已经覆盖绝大部分A6 predictable variation。

## 4. Failure Attribution

### D12-A-v1

- exact failure：`intervention_measure_wrong`；
- direction status：`diagnostic_invalid_for_direction_rejection`；
- resolved by：v2 exact raw-MSE-aligned weighting。

### D12-A-v2 / CAPE

- numeric/optimization pathology：无；
- capacity control：rank固定256，pilots与folds不变；
- exact failure：`hypothesis_false_for_CAPE_on_current_carrier_and_rank`；
- remaining untested：其他rank、其他backbone、jointly learned nonlinear frame；
- why not continue：论文carrier与rank已冻结，改rank或升级pilot会把problem gate变成post-hoc rescue，且不能
  建立跨dataset practical necessity。

[Decision] CAPE作为“独立training contribution”关闭。conditional-mean covariance仍是可用analysis tool，
但不能成为当前论文创新点。

### PRISM

D12-A没有直接测试prefix-locality Pareto，因此不能写成`hypothesis_false_for_PRISM`。但D12-B被预注册为
CAPE existence通过后的joint frame audit；绕过前置gate继续做PRISM-only probe会违背protocol，并把失去第二
contribution支撑的局部geometry继续堆叠下去。

[Decision] exact PRISM/CAPE forecast-frame mainline关闭；PRISM标记
`retired_without_effectiveness_test / problem_evidence_not_independently_sufficient`。D6 locality crossing保留为历史
evidence，可为未来非frame机制提供动机，但不是active candidate。

## 5. Paper-Narrative Consequence

撤回以下post-D11叙事：

> where accuracy is requested (PRISM) + what history can predict (CAPE)
> jointly allocates rank-limited future-frame capacity.

保留以下稳定论文约束：

1. unified forecasting应预测同一个future function，而不是学习horizon ID；
2. requested horizon只定义nested output domain/crop；
3. A6-LBF-natural-baseline仍是后续matched end-to-end carrier；
4. 五dataset natural profiles与rank256继续冻结；
5. future work仍需decoder/operator与training strategy两项可无缝组合的贡献，但当前两个slots均重新开放。

## 6. Rollback And Next Required Research Step

按11-step loop回滚到Step 2，而不是进入D12-B或直接设计新method：

1. 重新列出D6-D12仍跨dataset成立的problem evidence，区分representation geometry、optimization与evaluation
   measure；
2. 外部primary-source检索新的unified multi-horizon problem formulations，排除已失败的scale routing、
   component conflict、predictable-frame reallocation与generic loss weighting；
3. 先形成至少两个互补但可独立falsify的problem hypotheses，再提出新的decoder/training candidates；
4. 新candidate必须在Step 4-6通过narrative gate后才可实现；当前forecast method、validation/test均不授权。

[Final Decision] `D12 completed_fail_and_rollback_step2`。
