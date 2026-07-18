# CCSF temperature pilot runtime failure与Step7A repair报告

## 1. 结论

[Fact] 首次temperature pilot不是“15 runs完成”，而是driver正常退出但3个首批Weather runs全部因NaN失败：
`completed=0/15`，checkpoint=0，metrics=0，selection artifact不存在。official test未访问。

[Strong Evidence] 三个temperature均在第1 epoch进入NaN，5 epochs后因没有finite validation checkpoint抛出
`RuntimeError: training completed without capturing a validation checkpoint`。temperature-independent一致失败说明该批
artifact不能用于temperature比较。

failure attribution=`optimization_or_numeric_pathology`，并具体定位为
`zero_contrast_group_rms_sqrt_derivative`。这是exact implementation fault，不是architecture、training hypothesis或
CCSF理论方向失败。

## 2. Root cause

CCSF descriptor的group RMS原实现为：

$$
r=\sqrt{\operatorname{mean}(x^2)}.
$$

当真实batch中某个arm group的normalized contrast恰为0时，forward仍为0且finite，但$r$在0点的autograd
derivative不定义。correction output以零初始化开始，所以第1次forward/loss有限；optimizer更新后，gradient开始经
descriptor回传，NaN污染参数，随后所有loss与validation均为NaN。

旧Step7A random synthetic two-step没有构造exact zero contrast；旧resource smoke只有一个train batch，因此两者都
漏掉了该failure mode。

## 3. Repair与local evidence

最小修复只把group RMS改为：

$$
r_\epsilon=\sqrt{\operatorname{mean}(x^2)+\epsilon},\qquad\epsilon=10^{-6}.
$$

这与descriptor中已有的disagreement epsilon一致，不改变tensor path、scope semantics、projectivity、参数量或
temperature protocol。

`SC1-SIFF-v2-CCSF-RUNTIME-REPAIR-v1` local gate=3/3：

1. identical arms下descriptor finite；修复前arm gradient有7200个NaN，修复后为0；
2. group RMS minimum为0.001，与$\sqrt{10^{-6}}$一致；
3. degenerate zero input下三个temperatures各执行3个AdamW steps，共9/9 loss/gradient/parameter finite。

remote resource smoke也从1 train batch收紧为3 train batches，以确保至少跨过两次parameter updates。

## 4. Decision boundary

首次失败只否定原始unstabilized implementation，标记为`diagnostic_invalid_for_direction_rejection`。同一冻结pilot允许
在新external root `stage_c_siff_ccsf_temperature_pilot_v1_retry1`重跑；temperature grid、datasets、seed、profiles、
selection score与no-test boundary均不变。

下一步必须先提交/push repair，再执行remote三batch smoke。只有smoke finite才允许relaunch 15-run pilot。formal
Phase A、official test与confirmation继续为false。
