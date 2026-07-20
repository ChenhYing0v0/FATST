# SC-D22-HFA D22-C target-coordinate information-access audit

## 1. Decision

- `decision`: `target_coordinate_information_access_supported`
- `failure_attribution`: `none`
- `candidate_version`: `d22c-neutral-target-access-v1.1`
- `role`: `diagnostic_only_raw_history_primary`
- `test_role`: one-shot `test_informed` problem gate；validation只选择checkpoint。
- ordered patch memory仍是诊断载体，不是paper contribution。

## 2. Complete 20-cell scorecard

| Control | Val MSE gain | Test MSE gain | Test MAE gain | Positive cells | Datasets | Horizons | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `GLOBAL_COMPRESSED` | +20.6453% | +17.2910% | +11.7415% | 20/20 | 5/5 | 4/4 | pass |
| `POOLED_MEMORY` | +21.3815% | +17.5308% | +12.0611% | 20/20 | 5/5 | 4/4 | pass |
| `ORDER_SHUFFLED` | +24.5556% | +17.0826% | +11.6314% | 20/20 | 5/5 | 4/4 | pass |
| `TARGET_SHUFFLED_QUERY` | +13.7007% | +13.7449% | +9.2103% | 20/20 | 5/5 | 4/4 | pass |
| `GENERIC_MATCHED` | +2.5410% | +2.5228% | +1.6484% | 15/20 | 4/5 | 4/4 | pass |

所有gain定义为`(control - ORDERED_TARGET_ACCESS) / control`；正值表示ordered更好。

## 3. Coordinate-bin audit

| Control | Bin | Macro MSE gain | Positive datasets |
| --- | --- | ---: | ---: |
| `GLOBAL_COMPRESSED` | `h1_48` | +26.9722% | 5/5 |
| `GLOBAL_COMPRESSED` | `h49_96` | +20.8323% | 5/5 |
| `GLOBAL_COMPRESSED` | `h97_192` | +14.4680% | 5/5 |
| `GLOBAL_COMPRESSED` | `h193_336` | +11.0766% | 5/5 |
| `GLOBAL_COMPRESSED` | `h337_720` | +9.2747% | 5/5 |
| `POOLED_MEMORY` | `h1_48` | +26.9110% | 5/5 |
| `POOLED_MEMORY` | `h49_96` | +20.2910% | 5/5 |
| `POOLED_MEMORY` | `h97_192` | +14.7869% | 5/5 |
| `POOLED_MEMORY` | `h193_336` | +12.0581% | 5/5 |
| `POOLED_MEMORY` | `h337_720` | +10.2366% | 5/5 |
| `ORDER_SHUFFLED` | `h1_48` | +27.6490% | 5/5 |
| `ORDER_SHUFFLED` | `h49_96` | +20.3563% | 5/5 |
| `ORDER_SHUFFLED` | `h97_192` | +14.2445% | 5/5 |
| `ORDER_SHUFFLED` | `h193_336` | +10.6827% | 5/5 |
| `ORDER_SHUFFLED` | `h337_720` | +8.2396% | 5/5 |
| `TARGET_SHUFFLED_QUERY` | `h1_48` | +21.8158% | 5/5 |
| `TARGET_SHUFFLED_QUERY` | `h49_96` | +16.0617% | 5/5 |
| `TARGET_SHUFFLED_QUERY` | `h97_192` | +12.1890% | 5/5 |
| `TARGET_SHUFFLED_QUERY` | `h193_336` | +8.9919% | 4/5 |
| `TARGET_SHUFFLED_QUERY` | `h337_720` | +6.6135% | 5/5 |
| `GENERIC_MATCHED` | `h1_48` | +4.7875% | 4/5 |
| `GENERIC_MATCHED` | `h49_96` | +2.9119% | 5/5 |
| `GENERIC_MATCHED` | `h97_192` | +1.6719% | 2/5 |
| `GENERIC_MATCHED` | `h193_336` | +1.5583% | 3/5 |
| `GENERIC_MATCHED` | `h337_720` | +0.9407% | 3/5 |

## 4. Static and protocol integrity

- maximum trainable-parameter relative gap: `0.00000000`；
- matrix/load errors: `0`；

## 5. Failure attribution and rollback

[Strong Evidence] raw-history primary在matched capacity下支持target-coordinate-specific retrieval necessity。
返回Step4进行source-informed `lead-time-conditioned evidence operator`设计；本诊断arm不能直接升级为method。
Contribution 2继续保持open，只有首个E2E operator暴露真实瓶颈后才允许定义。
