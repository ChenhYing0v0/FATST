# ISCF-BSCA-MAIN-v1 H4N Formal Test 与 Weather 决策

## 1. Executive conclusion

H4N formal test 已完整结束并通过 artifact/protocol audit：40/40 frozen checkpoints、
160/160 standard-horizon rows、MSE/MAE、dense-prefix diagnostics 与 checkpoint
immutability 全部完整，无 retrain、mutation、partial artifact 或 ABORT。

按冻结的 full 12-baseline Weather target 与 0.1% near-tie selector，最终选择：

`Weather__h4n_seq608_p19_lr2e5`（`L=608, patch_num=19, lr=2e-5`）。

该 profile 改善了 H4M current profile 的 MAE，但没有提高 Weather lead-cell coverage，
且四项 H4N success gates 均失败。因此 Decision=
`H4N_complete_partial_improvement_gate_fail_no_automatic_H4O`。H4O、additional seeds、
architecture/objective redesign 与 selected-profile confirmation 均未自动授权。

## 2. Formal-test integrity

| Item | Result |
| --- | --- |
| candidate version | `ISCF-BSCA-MAIN-v1-weather-h4n-test-informed-20260805` |
| formal-test commit | `6184237994f1dea45b8af94575573e5afc2ed264` |
| checkpoint manifest SHA256 | `a0f152f9172acc193fe512001123b71aeae6d6d3ab1028c915074f24d54c1ed4` |
| test start / finish | 2026-08-06 14:15:48 / 14:22:25 +08:00 |
| checkpoints | 40/40 |
| standard cells | 160/160 |
| checkpoint hashes immutable | 40/40 |
| ABORT / partial target | none |
| post-test quota | 175G / 200G soft / 220G hard |

Canonical audit artifacts：

- `h4n_test_result/test_audit_completeness.json`；
- `h4n_test_result/test_audit_ledger.jsonl`；
- `h4n_test_result/all_trial_scorecard.csv`；
- `joint_objective_h4n_result_20260806/weather_all_profile_scorecard.csv`；
- `joint_objective_h4n_result_20260806/h4n_result_status.json`。

## 3. Selected Weather result

| Horizon | MSE | MAE | MSE vs H4M current | MAE vs H4M current |
| ---: | ---: | ---: | ---: | ---: |
| 96 | 0.141061 | 0.181594 | -0.080% | -1.046% |
| 192 | 0.182137 | 0.223136 | +0.118% | -0.600% |
| 336 | 0.232186 | 0.263780 | +0.206% | -0.364% |
| 720 | 0.304165 | 0.314772 | -0.013% | -0.563% |
| Mean | 0.214887 | 0.245821 | +0.063% | -0.608% |

相对本地复现的 TimeAlign Weather four-H mean，H4N selected profile 的 MSE 低
0.423%，MAE 高 0.448%。因此它是 competitive aggregate profile，但不能表述为 MSE/MAE
同时达到 Weather SOTA。

## 4. Frozen selector 与 gates

96 个 Weather profiles 中只有两个进入 0.1% near-tie band；primary-score minimum 本身即
`h4n_seq608_p19_lr2e5`，无需依赖有利 tie-break。其 full-table normalized scores 为：

- mean MSE target ratio=`0.997282`；
- mean MAE target ratio=`1.006631`；
- joint score=`1.001956`；
- exact leads=`3/4 MSE + 1/4 MAE = 4/8`。

| Frozen gate | Result | Pass |
| --- | ---: | --- |
| protocol-consistent joint-score improvement ≥0.3% | 0.205% | no |
| mean MSE ≤0.214752 | 0.214887 | no |
| mean MAE ≤0.244725 | 0.245821 | no |
| full-table leads ≥6/8 | 4/8 | no |

相对 H4M current selected profile `h4m_seq640_p20`，joint score 改善 0.297%，仍略低于
0.3%；相对所有 historical profiles 在同一 full-table target 下重算的最优
`h4m_seq512_p16_lr2e5`，改善为 0.205%。

## 5. Prelaunch comparator inconsistency audit

H4N config 中冻结的 `weather_historical_best_joint_score=1.0056094888` 来自旧的
5-baseline target；H4N primary objective 已切换到 12-baseline full-table target。若直接把
H4N score 与该旧标量相除，会得到 0.363% 并形式上通过 0.3% gate，但两侧 normalization
surface 不同，不能作合法比较。

为避免 test 后选择有利解释，本报告使用 conservative protocol-consistent recomputation：
在 H4N 冻结的同一 12-baseline target 上重算全部 56 historical Weather profiles，历史最优
score=`1.0040116170`，对应改善仅 0.205%，判定 gate fail。该 inconsistency 不影响
selected profile identity，也不影响其余三个独立 gates；即使采用旧标量，H4N 仍因
MSE、MAE 与 6/8 leads 三项失败而不能通过 overall gate。

## 6. Comparator-role separation

不能把不同 baseline surfaces 的 lead counts 混写：

1. H4N primary 12-baseline exact target：selected Weather=`4/8`；合并其他 H4M selected
   datasets 后为 MSE/MAE/combined=`14/28,14/28,28/56`。按论文三位小数显示规则为
   `15/28,14/28,29/56`，与 H4M Table-6-style draft 的 displayed best count 相同。
2. 旧 5-baseline 1% selector continuity audit：H4N trials不能替代
   `Weather__h4m_seq640_p20`；合法 selector 仍为 MSE/MAE/combined=
   `17/28,16/28,33/56`。因此 H4N 没有提高此前用户关注的 33/56 frontier。
3. 若把 H4N primary profile强行投影到旧 5-baseline exact target，会得到31/56；这只是
   target-role mismatch，不作为 H4N effectiveness gate 或 selector rollback依据。

## 7. Four-layer decision

1. `paper_facing_effectiveness`：`performance_partial_improvement_gate_fail`。MAE明显改善，
   aggregate MSE仍competitive，但 full-table leads与预注册均值gates失败。
2. `matched_mechanism_attribution`：H4N只做dataset-level HPO，不引入新mechanism；不能从
   本轮gain扩张ISCF/BSCA attribution claim。
3. `internal_mechanism_health`：40/40 numeric/artifact/checkpoint gates通过；无
   optimization/numeric pathology。Validation best与test best不同属预期，不改checkpoint。
4. `failure_attribution`：`search_space_performance_shortfall`，并伴随
   `prelaunch_historical_score_target_mismatch`。前者决定效果gate失败；后者要求未来HPO在
   launch前用同一target重算reference scalar，但不使本轮test artifacts无效。

Rollback=`Step 6 HPO/benchmark strategy decision`。Weather-only同架构继续扩张的边际收益
已经很小；下一步优先冻结 Main I final baseline role/table 与 Main II matched matrix。若仍要
H4O，必须另行授权并在test前修复target-consistent reference，不得自动沿用本轮结果调表。

Decision=`H4N_complete_partial_improvement_gate_fail_no_automatic_H4O`。
