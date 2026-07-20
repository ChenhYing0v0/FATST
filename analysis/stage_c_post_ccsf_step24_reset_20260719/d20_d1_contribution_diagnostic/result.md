# D20-D1 Summary Contribution Direction/Scale Result

## 1. What the diagnostic did

对每个SPEC/RANDOM checkpoint保存的test probe，恢复：

$$
\hat y_{base}=\hat y_{fused}-c,
$$

其中$c$是summary path的prediction contribution。随后比较实际$\alpha=1$与移除path的$\alpha=0$，并使用test
label计算每个dataset/future bin的MSE-optimal oracle scale$\alpha^*$。该实验没有训练、没有修改checkpoint，也没有
选择新candidate；test oracle只用于failure attribution。

## 2. Protocol health

- 2 arms × 5 datasets × 9 regions = 90 rows完整；
- 所有tensor与统计finite；
- `base + contribution`重构fused prediction的maximum absolute gap为`0`；
- full-H720与8个冻结future bins均报告；
- within-model base是co-adapted conditional ablation，不等于A6 architecture control。

## 3. Main result

| Arm | Actual macro gain vs co-adapted base | Helpful region cells | Median optimal alpha | Oracle macro gain | Full-H720 gain mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| SPEC | +26.8928% | 39/40 | 1.2649 | +30.3517% | +21.4730% |
| RANDOM | +9.0422% | 35/40 | 1.4115 | +14.4119% | +8.8474% |

SPEC没有出现“实际$\alpha=1$有害但$0<\alpha^*<1$”的region；只有1/40 region的$\alpha^*\le0$。因此
`summary over-injection / scalar shrinkage`不是D20失败的主要解释。多数区域的oracle alpha反而大于1。

### Full-H720 by dataset

| Dataset | SPEC actual gain | SPEC alpha* | RANDOM actual gain | RANDOM alpha* |
| --- | ---: | ---: | ---: | ---: |
| Weather | +7.1974% | 0.9044 | -0.3686% | 0.1195 |
| ETTm1 | +23.7169% | 1.3093 | +7.0491% | 1.9058 |
| ETTh1 | +15.4350% | 1.5083 | +8.0074% | 2.2455 |
| ETTh2 | +19.1120% | 1.2969 | +4.9938% | 1.1219 |
| ETTm2 | +41.9037% | 1.0584 | +24.5552% | 1.4801 |

## 4. What this means

[Fact] summary path在jointly trained model内部承担了真实且大量的预测责任；SPEC比RANDOM承担得更多。

[Strong Evidence] 这不是独立增量价值。完整SPEC相对same-run A6仍为`-0.7614%`，完整RANDOM为`-0.9028%`；
但移除其各自summary path却分别造成约21.5%和8.8%的full-H720 conditional degradation。RANDOM这个反例直接说明：
一个path在co-adapted model中很重要，并不意味着它给A6增加了新信息。base path会在joint optimization中把预测责任
转移给新增path。

[Inference] D20失败的直接原因不是summary方向普遍错误，也不是scale过强，而是`non-identifiable additive
co-adaptation / redundant responsibility relocation`。新增path学会替代A6已有能力，却没有提升跨split net risk。

## 5. Effect on the provisional future-distance hypothesis

D1没有验证`future-distance predictive support`：SPEC contribution在39/40 regions对其co-adapted base有益，说明
uniform full-trajectory path内部并未普遍出现远端有害。D20中SPEC-vs-RANDOM specificity随distance衰减仍是真实
observation，但它可能来自两个完整模型的co-adaptation差异，而不是一个可直接建模的support envelope。

因此不能以D1为依据实现Support-Calibrated Trajectory Operator。该family继续停在`problem_unverified`。

## 6. Failure attribution and decision

- `hypothesis_false`：不成立；D1无法否定history structure；
- `readout_or_head_design_wrong`：成立，additive paths不可辨识地重分配prediction responsibility；
- `optimization_or_numeric_pathology`：numeric false，但D20 validation/test mismatch仍成立；
- `capacity_control_explains`：RANDOM证明within-model importance主要受co-adaptation解释；
- direction-level rejection：false。

Decision=`d1_complete_scalar_fix_rejected_coadaptation_explains_return_step2_3`。

不实现D20-v2 gate/normalization，不把conditional ablation当作mechanism success。下一步回Contribution 1 Step2/3：
只有在独立证据证明future-distance support可从past识别并跨split稳定后，才允许重新进入Step4；否则该provisional
family关闭。Contribution 2继续Step2。
