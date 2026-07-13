# StageC Step 7B Failure Attribution Addendum

## Question

Step 7B失败主要来自architecture、hyperparameter，还是PMFO/projective forecasting的整体理论基础？

## Architecture And Optimization Evidence

所有arms使用相同dataset profile、full-H720 pointwise L1、learning rate、batch size、early stopping和
best-H720-validation-MSE checkpoint selection。下表读取各arm `training_log.csv`的best-validation epoch；
`relative improvement`为`(1 - PMFO / A6) * 100`，负值表示PMFO更差。

| Dataset | A6 train loss | PMFO train loss | Train improvement | A6 validation MSE | PMFO validation MSE | Validation improvement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTm1 | 0.430015 | 0.431362 | -0.3133% | 0.954085 | 0.976412 | -2.3402% |
| ETTh2 | 0.529680 | 0.529473 | +0.0392% | 0.646329 | 0.683007 | -5.6749% |
| Weather | 0.389668 | 0.382474 | +1.8462% | 0.589831 | 0.594791 | -0.8409% |

[Fact] 所有training/validation数值finite，early stopping和checkpoint reload正常；15/15 trained invariants
通过。PMFO没有出现不收敛、爆炸或prefix inconsistency。

[Strong Evidence] ETTh2与Weather的PMFO best-epoch train loss不高于A6，但validation仍更差；三个dataset的
validation方向与dense-horizon test AUC方向一致。这不符合“epoch不足或基础optimizer没有把模型训动”的主要
特征，更符合readout function class或inductive bias的generalization问题。

[Uncertainty] 该实验没有为PMFO单独搜索state dimension、initialization、regularization或learning rate。
因此不能证明任何hyperparameter都无法改善v1；尤其PMFO是nonlinear recursive head，而A6 profile与training
contract先在A6上冻结，decoder-specific regularization mismatch仍是可能的次要解释。

## Component Evidence

- [Strong Evidence] PMFO相对no-conservation三dataset均改善，macro `+2.3393%`：conservative synthesis
  有正向跨dataset证据；
- [Strong Evidence] PMFO相对no-transition macro仅`+0.0486%`，且ETTm1全部720 horizons更差：v1
  recursive transition没有独立支持；
- [Weak Evidence] PMFO相对matched dense macro `+0.7193%`并在2/3 datasets更好：structured decoder
  可能优于相近规模generic nonlinear head，但尚未超过A6；
- [Fact] exact prefix、refinement recovery、conservation和locality在trained checkpoints继续成立：
  algebraic theory与实现没有失败。

## Judgment

1. `primary`: `readout_or_head_design_wrong`。被否定的是fixed `90/30/10/5/1` partition、v1 state
   transition和整体替换A6 readout的具体architecture组合；
2. `secondary_uncertainty`: decoder-specific hyperparameter/regularization mismatch不能排除，但当前证据不
   支持先做无诊断的tuning sweep；
3. `not_rejected`: projectivity与conservation的数学基础。结构性质成立，conservation还有正向效果；失败说明
   “结构正确”本身不足以保证forecast accuracy，而不是整体理论坍塌；
4. `empirically_unsupported`: v1 recursive-transition predictive hypothesis。

因此继续回滚Step 4：先诊断function-class containment、future partition和history-to-node interface。只有
诊断指向明确optimization bottleneck时，才设计小型hyperparameter control；不得用大规模调参复活v1。
