# StageC Checkpoint与Dataset-Specific Profile治理修订

## 1. User Correction

本次修订回应两个边界：

1. SC0报告的31.63%-44.95%来自training-time validation，不是test；必须直接检查test后才能判断last的影响。
2. “避免精调”不等于“所有dataset共用完全相同的结构超参数”。允许dataset有有限偏好，但必须限制搜索
   自由度、保持test-blind，并在所有mechanism/control之间冻结。

## 2. TimeAlign Author Evidence

TimeAlign作者在[GitHub issue #2](https://github.com/TROUBADOUR000/TimeAlign/issues/2)中说明：论文主表使用
固定训练轮数后的last checkpoint，而不是validation-best；其理由是长时序预测中validation/test可能存在
distribution shift，validation early stopping有时训练不足。作者也确认`w_align`、`layer_norm`等部分参数会
按dataset调整，ETTh系列尤其敏感。

[Boundary] 该解释是official TimeAlign protocol evidence，不能自动证明我们的pure A6-LBF carrier也应使用
last。因此对SC0已有checkpoint进行独立test诊断。

## 3. What The 31.63%-44.95% Meant

该数字只针对SC0 seed2021、ETTh2三臂的full-720 validation MSE：epoch20相对trajectory内best epoch的
恶化。它说明validation selector发生反转，但在本诊断前不能称为test degradation。

## 4. Frozen-Checkpoint Test Diagnostic

诊断直接读取原SC0 fixed-20的9组`checkpoint_best_val.pt`和`checkpoint_last.pt`，不重新训练。每个
checkpoint在相同test loader、相同official test flag下评估8个dense horizons。test在原carrier calibration
结束后才打开，且不用于profile选择。

### H720结果

| Scope | Last MSE wins | Mean validation last vs best | Mean test last vs best |
| --- | ---: | ---: | ---: |
| All 9 runs | 0/9 | +14.72% | +6.11% |
| Weather | 0/3 | +1.33% | +2.19% |
| ETTm1 | 0/3 | +3.06% | +6.80% |
| ETTh2 | 0/3 | +39.75% | +9.35% |

H720最坏单run的last test MSE恶化13.70%，最好一组仍恶化1.32%。因此：

- [Fact] 31.63%-44.95%不是test数字，原表述必须纠正；
- [Strong Evidence] 对当前A6-LBF/SC0 carrier，validation degradation虽然在test上明显收缩，但并未消失；
- [Decision] StageC mechanism-control继续使用best-validation checkpoint是有当前代码与artifact依据的；
- [Boundary] 这不否定TimeAlign论文使用last的合法性，因为模型路径、objective和dataset behavior不同。

### Dense-horizon反例

全部72个比较中，last在29/72个MSE项更好，平均test MSE仍为+1.63%；MAE则有37/72项更好，平均
-0.18%。这支持作者所述validation/test shift确实存在，也说明checkpoint结论依赖horizon和metric。后续
论文应分别报告best-validation mechanism evidence与source-native last结果，不能混为一套协议。

## 5. Revised Hyperparameter Governance

原`one-global-profile`要求比用户目标更严格，现降为历史control，不再作为active StageC约束。新的规则是
`dataset-aware but low-degree-of-freedom`：

### Globally frozen fields

decoder/readout、basis rank、layers、dropout、LayerNorm、optimizer、LR、objective、effective batch、
max epochs、patience、selector、seeds与dense horizons当前保持全局一致。

### Dataset-specific structural fields

只允许从预注册的三组capacity-matched profiles中选择`patch_num/d_model/d_ff`。三组active parameters spread
约0.08%，不会把明显容量差异混入mechanism comparison。基于已有三seed、validation-only pooled H720
结果冻结：

| Dataset | Profile | P/D/d_ff | Per-seed dataset wins | Selection source |
| --- | --- | --- | ---: | --- |
| Weather | `p12/d128` | 12/128/256 | 3/3 | validation only |
| ETTm1 | `p48/d32` | 48/32/1072 | 2/3 | validation only |
| ETTh2 | `p24/d64` | 24/64/536 | 2/3 | validation only |

Weather seed2021 test实际上偏好P24而不是validation选出的P12；我们仍保留P12，正是为了证明mapping没有
被test反向调节。

## 6. Mechanism Attribution Rule

对任一dataset，baseline、decoder candidate、training candidate和capacity/no-mechanism control必须使用该
dataset同一个冻结profile。允许dataset之间不同，但禁止：

- 每提出一个mechanism就重新为它挑carrier；
- 根据test切换profile；
- 扩大为连续或大规模grid search；
- method arm和matched control使用不同dataset profile；
- 在mechanism-control中恢复`patch_num=1`。

新dataset只允许评估同样三组registered profiles，用standard seeds在validation上一次性冻结，然后所有
后续实验复用。

## 7. Decision And Next Step

- uniform P24 contract保留为历史更严格control，但active mechanism-control改为
  `configs/stage_c_mechanism_control_dataset_aware.json`；
- checkpoint policy在StageC mechanism-control中保持best-validation；
- source-faithful TimeAlign继续遵循作者native last protocol；
- SC1/SC2实验设计必须读取dataset-aware mapping，并把protocol class写入effective config；
- 下一步恢复SC1/SC2 Step 1-3 prior-art与problem diagnostics，不重新做大规模carrier tuning。
