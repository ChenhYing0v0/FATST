# Phase5 StageB B14-FURD Step 3 Cross-Dataset Report

[Decision] `retrieval_demand_problem_not_supported`。

## Gate Results

| Dataset | U | dCos mean | dCos p05 | dJS mean | dJS p05 | sensitivity cos | support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ETTh2 | 180 | 0.0082 | 0.0051 | 0.0019 | 0.0015 | 0.9917 | no |
| ETTh2 | 240 | 0.0043 | 0.0026 | 0.0011 | 0.0008 | 0.9933 | no |
| ETTm1 | 180 | -0.0023 | -0.0032 | 0.0001 | -0.0000 | 0.9943 | no |
| ETTm1 | 240 | -0.0027 | -0.0033 | -0.0001 | -0.0002 | 0.9933 | no |
| Weather | 180 | -0.0271 | -0.0294 | -0.0003 | -0.0012 | 0.9486 | no |
| Weather | 240 | -0.0289 | -0.0315 | -0.0004 | -0.0015 | 0.9447 | no |

## Cross-Dataset Gate

Dataset-level support requires both U180 and U240 to pass. Overall support requires at least two datasets.

- ETTh2: no support
- ETTm1: no support
- Weather: no support

## Failure Attribution Boundary

该诊断只判断 accepted A6 是否存在 error-conditioned demand 与 existing sensitivity 的 patch-level
mismatch。正结果只允许进入 parameter-matched B14-B probe；负结果回滚 Step 2，不允许通过实现
cross-attention 来替代 problem evidence。任何 evidence-contract、non-finite 或 mass-conservation
问题只能标记为 `diagnostic_invalid_for_direction_rejection`。
