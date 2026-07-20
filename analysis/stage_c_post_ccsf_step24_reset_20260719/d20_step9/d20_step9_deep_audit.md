# D20 CST Step 9–10 Four-Layer Deep Audit

## 1. Audit identity

| Field | Value |
| --- | --- |
| candidate | `SC-D20-CST-v1 diagnostic_only` |
| remote commit | `9573cd7` |
| test access date | `2026-07-20` |
| user authorization | completed D20 remote matrix；continue experiments |
| checkpoint rule | validation mean MSE over H96/H192/H336/H720 |
| checkpoint retrained | true；15 arms all from scratch |
| matrix complete | 15/15 runs；60/60 official-test cells |
| test role | `test_informed primary-problem-existence diagnostic` |
| confirmation | false |

## 2. What was tested

D20问两个不同问题：

1. `transfer`：64-dimensional low-frequency history summary接入A6 coefficient operator后，是否超过same-run A6；
2. `specificity`：该summary是否超过同维fixed random orthogonal history projection。

三臂共享dataset profile、A6 Encoder、learned basis、objective、optimizer、seed、checkpoint rule与full-T prefix-crop
contract。SPEC/RANDOM均只比A6多16,384个summary-to-coefficient参数，且初始summary weights为0。

## 3. Artifact and protocol audit

- 15/15 runs具有checkpoint、effective config、training log、initialization/model diagnostics与test invariants；
- 15/15 test evaluator均报告`pass`，checkpoint hash在test前后不变；
- 所有输出finite；maximum prefix gap为`3.576e-07`；
- 逐dataset的Encoder与base coefficient operator initialization hash三臂一致；
- SPEC/RANDOM projection hash不同且orthogonality error不超过`3.136e-08`。

因此本次不是缺失artifact、protocol drift或initialization mismatch。

## 4. Layer 1 — Paper-facing effectiveness

gain为$100(1-M_{candidate}/M_{reference})$，正值表示candidate更好。

| Comparison | Metric | Macro gain | Cell wins | Dataset wins | Horizon wins | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SPEC vs A6 | MSE | -0.7614% | 8/20 | 2/5 | 0/4 | fail |
| SPEC vs A6 | MAE | -0.5831% | 3/20 | 0/5 | 0/4 | fail |
| SPEC vs RANDOM | MSE | +0.1412% | 14/20 | 4/5 | 3/4 | fail：低于0.3% |
| SPEC vs RANDOM | MAE | +0.0273% | 12/20 | 3/5 | 3/4 | directional positive |
| RANDOM vs A6 | MSE | -0.9028% | 7/20 | 1/5 | 0/4 | negative control |

SPEC没有通过transfer，且specificity虽大多数cells为正，幅度没有达到冻结gate。formal effectiveness结论为
`failed`，不得补confirmation seeds。

### Dataset decomposition

SPEC-vs-A6 test MSE gain为Weather `+0.4834%`、ETTm1 `+0.2541%`、ETTh1 `-3.0428%`、ETTh2
`-1.3051%`、ETTm2 `-0.1965%`。负结果主要来自ETTh1/ETTh2，但不是单一dataset失败，因为ETTm2也为负且
macro horizon在H96/H192/H336/H720全部为负。

### Horizon decomposition

| Horizon | SPEC vs A6 MSE | SPEC vs RANDOM MSE |
| ---: | ---: | ---: |
| 96 | -0.8422% | +0.2657% |
| 192 | -0.7137% | +0.2539% |
| 336 | -0.7272% | +0.0672% |
| 720 | -0.7624% | -0.0218% |

SPEC相对RANDOM的弱优势随future distance衰减，而相对A6没有任何standard-horizon恢复。

## 5. Validation-to-test transfer audit

| Comparison | Validation MSE | Test MSE | Interpretation |
| --- | ---: | ---: | --- |
| SPEC vs A6 | +0.5755%, 13/20 | -0.7614%, 8/20 | sign reversal |
| SPEC vs RANDOM | +0.7288%, 15/20 | +0.1412%, 14/20 | large attenuation |
| RANDOM vs A6 | -0.1580%, 8/20 | -0.9028%, 7/20 | generic path also worsens transfer |

validation selector没有实现错误：15/15 logs重算的best epoch与reported best epoch一致。CST arms通常更早停止；
例如ETTh2的SPEC/RANDOM均在epoch 1达到best，最后epoch相对best分别恶化9.78%与6.54%，但A6在ETTh1/ETTh2
也有约5% validation恶化且正确恢复best checkpoint。没有NaN、divergence或训练预算普遍卡顶。

[Strong Evidence] 这不是checkpoint选择错误，而是新增history path在validation上看似有用、到shifted test
distribution上不能保持收益的generalization mismatch。

## 6. Dense-horizon diagnostic

Dense H1–H720只作解释，不替代standard gate。

| Future region | SPEC vs A6 MSE | SPEC vs RANDOM MSE |
| --- | ---: | ---: |
| H1–48 | -0.4240% | +0.9852% |
| H49–96 | -0.7691% | +0.2848% |
| H97–144 | -0.8596% | +0.2765% |
| H145–192 | -0.7810% | +0.2750% |
| H193–288 | -0.6846% | +0.1799% |
| H289–336 | -0.6984% | +0.0858% |
| H337–512 | -0.6794% | +0.0190% |
| H513–720 | -0.7249% | -0.0114% |

[Strong Evidence] low-frequency geometry不是完全无效：它稳定优于random projection直到中长区间，但优势单调衰减，
最终消失；同时二者均不如不增加history bypass的A6。这支持“frequency semantics有弱signal，但当前injection没有
创造net forecast utility”。

## 7. Layer 2 — Matched mechanism attribution

SPEC-vs-RANDOM未过0.3% macro gate，不能建立正式frequency-specific mechanism claim。不过14/20 MSE cells、
4/5 datasets与3/4 horizons为正，说明结果也不能简化为“RANDOM完全解释SPEC”。更准确的状态是：

`weak_directional_specificity / insufficient_effect_size`。

RANDOM-vs-A6为`-0.9028%`，说明新增generic path不是无害容量扩充；SPEC只回收了其中约0.141个百分点，仍不足以
超过A6。参数数量差异不参与选择，结论来自function path与generalization。

## 8. Layer 3 — Internal mechanism health

remote frozen analyzer的11项health checks全部通过：protocol、finite、projectivity、paired initialization、projection
identity/orthogonality、trained weights、coefficient/prediction contribution与SPEC-RANDOM prediction diversity均健康。

| Dataset | SPEC/RANDOM contribution ratio | SPEC-RANDOM prediction NRMSE |
| --- | ---: | ---: |
| Weather | 4.44× | 0.0467 |
| ETTm1 | 2.65× | 0.0386 |
| ETTh1 | 1.86× | 0.0571 |
| ETTh2 | 1.78× | 0.0150 |
| ETTm2 | 2.12× | 0.0173 |

SPEC path不仅没有collapse，实际prediction contribution反而显著强于RANDOM。由于两个projection均orthonormal，
这一差异来自真实数据能量集中于low-frequency subspace及joint optimization，而不是参数量差异。[Hypothesis]
这种强旁路可能形成了validation-friendly shortcut，并在test distribution shift下过度介入。

## 9. Layer 4 — Failure attribution

remote frozen analyzer按预注册逻辑给出`compact_spectrum_transfer_failed`，该formal gate结论保留。但深层归因必须按
项目Diagnostic Failure Attribution Rule修正：

1. `optimization_or_numeric_pathology(validation_test_mismatch)`：validation transfer正、test负；
2. `intervention_point_wrong`：raw compact statistic作为additive coefficient bypass没有产生net utility；
3. `hypothesis_false`尚不成立：SPEC相对RANDOM存在弱而系统的directional specificity；
4. `capacity_control_explains`也不是完整描述：RANDOM解释新增function/capacity的负效应，却没有解释SPEC在14/20
   cells的相对优势。

因此结论为`exact_design_closed / direction_rejection_invalid`。D20只能否定$q=64$ low-frequency summary直接加到
A6 coefficients的tested implementation，不能否定所有history-spectrum、predictive-support或structured decoder。

## 10. Decision

- no confirmation seeds；no width/LR/gate sweep；
- `SC-D20-CST-v1`关闭为`diagnostic_only_failed_transfer_weak_specificity`；
- Contribution 1回Step2/4 source-informed redesign；
- 在新method前先执行D20-D1 within-model contribution direction/scale oracle，进一步分离over-injection与
  co-adaptation；
- Contribution 2继续停在Step2，不能叠加在失败的D20上。

详细可复算结果见本目录CSV/JSON；同步raw metadata位于相邻`d20_step9_remote_raw/`且不纳入paper table。
