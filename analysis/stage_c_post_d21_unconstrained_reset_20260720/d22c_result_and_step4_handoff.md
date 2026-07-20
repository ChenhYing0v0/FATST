# SC-D22-HFA D22-C：positive problem gate与Step4 handoff

## 1. Decision

| Field | Result |
| --- | --- |
| `current_step` | D22-C Step9/10 complete；return Step4 |
| `candidate_version` | `d22c-neutral-target-access-v1.1` |
| `role` | `diagnostic_only_raw_history_primary` |
| `decision` | `target_coordinate_information_access_supported` |
| `failure_attribution` | `none` |
| `paper_method_status` | none；diagnostic不能直接升级为method |
| `next_step` | SC-D23-FCMI Step4-6 source/narrative/design gate |

[Strong Evidence] 在fixed past、same raw-history information、same trainable parameters与same initialization下，
future-coordinate-specific ordered retrieval稳定超过五个controls。它回答的是finite computation中的evidence
organization，不改变D22-A的Bayes theorem：同一coordinate的conditional mean仍不依赖requested horizon。

## 2. Protocol integrity

- five datasets × six arms全部完成；
- official test完整`5 × {96,192,336,720}`，每个control 20 cells；
- validation只选择H96/H192/H336/H720平均MSE checkpoint；
- six arms每dataset均为11,553 trainable parameters，maximum relative gap为0；
- ordered best epoch为ETTh1/ETTh2/ETTm1/ETTm2/Weather=`4/6/10/7/11`，均未卡12-epoch上限；
- 30个checkpoint hash、runtime/config/test metadata与全部negative cells已保存；
- v1 normalized-loss pathology在任何complete/test artifact前发现并终止；v1.1从新目录/新checkpoints完整重跑。

## 3. Matched mechanism attribution

gain定义为`(control - ordered) / control`。

| Control | Validation MSE | Test MSE | Test MAE | Test cells | Datasets | Horizons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `GLOBAL_COMPRESSED` | +20.6453% | +17.2910% | +11.7415% | 20/20 | 5/5 | 4/4 |
| `POOLED_MEMORY` | +21.3815% | +17.5308% | +12.0611% | 20/20 | 5/5 | 4/4 |
| `ORDER_SHUFFLED` | +24.5556% | +17.0826% | +11.6314% | 20/20 | 5/5 | 4/4 |
| `TARGET_SHUFFLED_QUERY` | +13.7007% | +13.7449% | +9.2103% | 20/20 | 5/5 | 4/4 |
| `GENERIC_MATCHED` | +2.5410% | +2.5228% | +1.6484% | 15/20 | 4/5 | 4/4 |

`GENERIC_MATCHED`是最关键的hard control：它保留ordered memory、coordinate readout、相同attention/MLP
parameters与prediction diversity，只去除coordinate-specific memory retrieval。因此`+2.5228%`不能由单纯增加
memory、query branch或parameter count解释。

### 3.1 Generic comparison的heterogeneity

| Slice | MSE gain |
| --- | ---: |
| ETTh1 | +0.4606% |
| ETTh2 | +5.4434% |
| ETTm1 | +5.2547% |
| ETTm2 | +2.5453% |
| Weather | -1.0900% |
| H96 macro | +3.9013% |
| H192 macro | +2.6607% |
| H336 macro | +2.1171% |
| H720 macro | +1.4122% |

该heterogeneity是设计约束而不是可隐藏的negative result：

1. 不得声称coordinate-specific retrieval在所有datasets都优于generic；
2. benefit随lead time整体衰减，H96最强、H720仍为正；
3. 新operator必须原生包含query-independent generic fallback，不能用强制specific path；
4. 不做Weather-specific tuning或删dataset。

## 4. Coordinate bins与internal health

相对`GENERIC_MATCHED`的五个bin macro MSE gain为：

- H1–48 `+4.7875%`；
- H49–96 `+2.9119%`；
- H97–192 `+1.6719%`；
- H193–336 `+1.5583%`；
- H337–720 `+0.9407%`。

ordered attention entropy在五datasets为`0.745–0.839`，没有one-token collapse；target dispersion为
`0.0083–0.0370`。`GENERIC_MATCHED`的target dispersion按构造为0，但prediction coordinate dispersion仍为
`0.0179–0.4094`，确认它不是constant-output strawman。`TARGET_SHUFFLED_QUERY` prediction dispersion仅
`0.0032–0.0109`，说明row-specific shuffle成功破坏稳定coordinate identity。

internal health只证明路径active；problem gate由matched performance决定。

## 5. Four-layer boundary

1. `paper_facing_effectiveness`：**missing/not applicable**。这些11,553-param neutral models只用于problem
   diagnostic，未与A6、CATS、TimePerceiver或SOTA作公平E2E paper comparison；
2. `matched_mechanism_attribution`：**pass**。五个controls完整，最关键generic control仍pass；
3. `internal_mechanism_health`：**pass**。attention与prediction均active，shuffles有效；
4. `failure_attribution`：`none`。Weather negative保留为method design constraint，不推翻macro problem result。

因此允许返回Step4，但禁止把D22-C的raw patch cross-attention直接命名为Contribution 1。

## 6. Prior-art correction

positive result必须在以下直接prior下解释：

- [CATS, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/cf66f995883298c4db2f0dcba28fb211-Paper-Conference.pdf)
  已把future horizons作为queries读取input patches，并共享horizon parameters；
- [CATS official code](https://github.com/dongbeank/cats)实现cross-attention-only forecasting；
- [TimePerceiver, NeurIPS 2025](https://arxiv.org/abs/2512.22550)使用target-timestamp queries读取input latents；
- [TimePerceiver official code](https://github.com/efficient-learning-lab/TimePerceiver/blob/main/models/TimePerceiver.py)
  直接实现positioned patch memory与target query cross-attention；
- [MQTransformer](https://arxiv.org/abs/2009.14799)已有forecast-context-dependent history alignment；
- [TQNet, ICML 2025](https://openreview.net/pdf?id=e24CueVty2)使用temporal query与raw input key/value，
  主要解决cross-variable correlation；
- 2026年attention critique指出多branch projection/fusion可能解释attention收益，故后续必须保留
  no-interaction multi-branch control。

所以novelty不能是“future query读取history”。D22-C的价值是给出本项目task boundary下的controlled necessity
证据，并暴露`generic main evidence + coordinate-specific interaction`的heterogeneity。

## 7. Step4 handoff

新的paper candidate必须同时满足：

1. 把generic evidence作为原生contained case，而不是另加router；
2. 显式隔离trajectory-wide evidence main effect与future-coordinate interaction；
3. 相对standard query attention与matched multi-branch/no-interaction control可归因；
4. 使用same-objective from-scratch E2E，与A6_MEASURE比较；
5. 不增加第二loss，不把ordered patch memory或attention primitive写成论文主语；
6. Weather negative必须完整保留。

下一candidate工作名：`SC-D23-FCMI`，即`Future-Coordinate Main–Interaction operator`。
