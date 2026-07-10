# Phase5 StageB B14-FURD Step 3 Cross-Dataset Report

[Decision] `dataset_or_unit_size_specific_mismatch`。

## Gate Results

| Dataset | U | label dCos mean | label dCos p05 | label dJS mean | label dJS p05 | CKA-shuffle p05 | sensitivity cos | support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ETTh2 | 180 | 0.0450 | -0.0017 | 0.0197 | 0.0084 | -0.0088 | 0.9915 | no |
| ETTh2 | 240 | 0.0140 | -0.0016 | 0.0100 | 0.0043 | -0.0119 | 0.9932 | no |
| ETTm1 | 180 | 0.0269 | 0.0227 | 0.0079 | 0.0068 | 0.0898 | 0.9945 | no |
| ETTm1 | 240 | 0.0257 | 0.0181 | 0.0076 | 0.0058 | 0.0915 | 0.9930 | no |
| Weather | 180 | 0.1030 | 0.0549 | 0.0327 | 0.0184 | 0.0063 | 0.9497 | yes |
| Weather | 240 | 0.0855 | 0.0268 | 0.0297 | 0.0115 | 0.0041 | 0.9429 | no |

## Cross-Dataset Gate

Dataset-level support requires both U180 and U240 to pass. Overall support requires at least two datasets.

- ETTh2: no support
- ETTm1: no support
- Weather: no support

## Failure Attribution Boundary

该诊断只判断 model-independent label-patch dependence 是否比 accepted A6 sensitivity更 unit-specific。
正结果只允许进入 parameter-matched B14-B probe；负结果回滚 Step 2，不允许通过实现
cross-attention 来替代 problem evidence。任何 evidence-contract、non-finite 或 mass-conservation
问题只能标记为 `diagnostic_invalid_for_direction_rejection`。
