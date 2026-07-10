# Phase5 StageB B14-FURD Step 3 Deep Analysis

## Conclusion

[Decision] `blocked_by_nonrobust_label_patch_evidence`。B14-FURD当前问题表述不进入 trainable retrieval或
Step 4-6。A1 current-gradient mismatch为 `0/6`；修复其 shared-Jacobian circularity后的 A2
model-independent label-patch gate仅 `1/6`。回滚到 11-step loop Step 2/3，下一问题优先收窄为：ETTm1
`patch_num=1` 是否是一个可通过 minimal tokenization change修复的 carrier defect。

这不是对所有 future-unit-aware architecture的方向级否定。它关闭的是：在 ETTh2、ETTm1、Weather上，
U180/U240均存在稳定、跨数据集、可支撑 patch retrieval的 label-history mismatch。

## History Evidence Contract

旁路不再使用 initial right-padded 30th patch，而是 29 个完整 `K48-S24` supports：starts
`0,24,...,672`。三个 datasets各 8 batches，共 24/24 batches满足：

- side-path memory与手工 normalized-history slices max diff `0.0`；
- coverage-corrected overlap-add reconstruction max diff `0.0`；
- explicit normalized-leaf A6 forward与 model forward max diff `0.0`；
- position-to-patch attribution mass error最大 `1.19e-7`。

因此返回负结果不能归因于错误 patch values、末端 padding、overlap double-counting或 forecast-path改变。

## A1：Current-Gradient Contradiction

| Dataset | U | `Delta_cos p05` | `Delta_JS p05` | sensitivity cosine | Support |
| --- | ---: | ---: | ---: | ---: | --- |
| ETTh2 | 180 | 0.0051 | 0.00153 | 0.9917 | no |
| ETTh2 | 240 | 0.0026 | 0.00084 | 0.9933 | no |
| ETTm1 | 180 | -0.0032 | -0.00003 | 0.9943 | no |
| ETTm1 | 240 | -0.0033 | -0.00016 | 0.9933 | no |
| Weather | 180 | -0.0294 | -0.00119 | 0.9486 | no |
| Weather | 240 | -0.0315 | -0.00153 | 0.9447 | no |

A1可靠关闭精确 contradiction：A6 error-conditioned gradient没有比 A6 target-independent Jacobian显露更
unit-specific的 patch profiles。但 $J_{A6}^T r_m$ 与 sensitivity共享 $J_{A6}$，所以 A1没有 broader
direction rejection authority。

## A2：Model-Independent Label-Patch Dependence

A2将 valid history patches与 future units投影到固定 DCT-8 descriptors，以 centered linear CKA构造
label-patch profiles，并用 4-draw shuffled target CKA控制 finite-sample dependence。主 gate比较该 profile与
A6 sensitivity。

| Dataset | U | label `Delta_cos p05` | label `Delta_JS p05` | CKA-shuffle p05 | Support |
| --- | ---: | ---: | ---: | ---: | --- |
| ETTh2 | 180 | -0.0017 | 0.00841 | -0.00880 | no |
| ETTh2 | 240 | -0.0016 | 0.00432 | -0.01191 | no |
| ETTm1 | 180 | 0.0227 | 0.00677 | 0.08976 | no |
| ETTm1 | 240 | 0.0181 | 0.00582 | 0.09148 | no |
| Weather | 180 | 0.0549 | 0.01844 | 0.00629 | yes |
| Weather | 240 | 0.0268 | 0.01147 | 0.00413 | no |

解释：

- ETTh2：label CKA不高于 shuffle，当前 DCT-8 patch dependence不成立；
- ETTm1：存在显著 label-history dependence，但不同 future units的 patch profiles仍高度共享，不能推出
  unit-specific retrieval；
- Weather：U180形成完整 mismatch，U240同向但未达到 cosine gate；证据是 dataset/unit-size specific；
- cross-dataset gate要求至少两个 datasets的 U180/U240同时通过，实际为 `0/3` datasets、`1/6` settings。

## Failure Attribution

- `hypothesis_false`：对“跨数据集 U180/U240均需要 unit-specific patch retrieval”是 strong negative；
- `intervention_point_wrong`：不适用，A2不经过 A6 intervention；
- `readout_or_head_design_wrong`：不适用，未训练 readout；
- `optimization_or_numeric_pathology`：否，全部 finite且 evidence contract exact；
- `capacity_control_explains`：不适用，A2是 fixed-rank statistic；
- remaining untested：nonlinear/high-frequency dependence、不同 unit definition、Weather-specific mechanism。

这些 remaining items不足以授权继续 sweep。按“先诊断、不死磕”，当前 B14-FURD关闭为 paper-core route。

## Rollback And Next Question

- current step：B14 Step 3 decision completed；
- decision：`blocked_by_nonrobust_label_patch_evidence`；
- rollback：Step 2/3；
- do not implement：cross-attention、query retrieval、future-unit router；
- next scoped problem：只改变 ETTm1 inherited `patch_num=1` 是否能保留或改善 A6 performance，并让统一
  Patch-wise carrier不依赖 TimeAlign preset；
- required controls：`patch_num=1` clean A6、minimal `patch_num>1`、parameter/capacity-matched control；不得重新
  引入 contextual Transformer或 future retrieval。
