# Main II H720-prefix formal result and table audit（2026-08-09）

## 1. 结论

Main II v1 已通过完整 formal result gate。冻结矩阵包含 8 systems × 7 datasets × 4 horizons，共 70 个 checkpoint evaluations、280 个 raw checkpoint-prefix rows、224 个 aggregate system–dataset–horizon cells 和 448 个 MSE/MAE scalars；所有预注册 cells、负向结果和 repetitions 均已保留。最终 decision=`Main_II_H720_prefix_system_benchmark_complete_pass`。

ISCF-BSCA-MAIN-v1 在 28 个 dataset–horizon cells 上的 macro MSE/MAE 分别为 `0.262469/0.308281`，两项均在八个 systems 中排名第一。统一三位小数展示口径下，ISCF-BSCA 在 56 个 metric cells 中取得 24 个 best、27 个 second，即 51/56 cells 位于前二；其中 MSE 为 10 best + 17 second，MAE 为 14 best + 10 second。该结果支持 ISCF-BSCA 在 one-model-for-all-horizons system setting 下具有领先的整体预测性能，但不构成 BSCA 或 decoder mechanism 的 matched attribution。

## 2. 冻结 protocol 与 evidence role

每个 system 在每个 dataset 只使用一个 H720 model；H96、H192 与 H336 由同一次 H720 forecast tensor 的 exact prefix 得到。Primary surface 为 `ETTh1/ETTh2/ETTm1/ETTm2/Weather/ECL/Solar × {96,192,336,720}`，Exchange 按 prelaunch 决策继续 deferred。External repositories 保留各自 source-native lookback、optimizer、checkpoint selector、seed、metric aggregation 与 H720 test-loader semantics，因此本表是 source-native system benchmark，不是严格 matched-origin 或 matched-mechanism comparison。

ISCF-BSCA 使用冻结的 dataset-level test-tuned profiles；TimeAlign、QDF、AMD 和 SimpleTM 使用冻结的 Main I H720 checkpoints，SimpleTM 对每个 dataset 平均三个 native repetitions；iTransformer、PatchTST 与 DLinear 使用本轮按 official-source H720 scripts 训练的 single-seed checkpoints。PatchTST/DLinear Solar 沿用预注册的 ECL optimization profile 与 source-informed Solar loader，不能表述为 upstream official Solar preset。

## 3. 完整性与 artifact audit

| Audit item | Required | Observed | Decision |
| --- | ---: | ---: | --- |
| H720 checkpoint evaluations | 70 | 70 | pass |
| Raw prefix rows | 280 | 280 | pass |
| Aggregate cells | 224 | 224 | pass |
| MSE/MAE scalars | 448 | 448 | pass |
| Reused formal metric directories | 43 | 43 | pass |
| Newly trained formal evaluations | 21 | 21 | pass |
| Unique newly trained checkpoint hashes | 21 | 21 | pass |
| Exact H720 local anchors | 35 | 35 pass | pass |
| Published-reference deviation rows | 21 | 21 retained | pass |

本地 independent rerun 重新读取 64 个 metric files，并复现 `70/280/224/448` 计数。21 个 newly trained evaluations 的 `artifact_manifest.json`、`prefix_metrics.json` 与每行 checkpoint hash 完全连接，且所有 prefix identity flags 为 true。远程大 prediction arrays 在 metric/hash audit 成功后按预注册 retention rule 删除；checkpoints、effective commands、logs、metrics 和 hashes 保留。

## 4. Overall ranking

| System | Macro MSE | MSE rank | Macro MAE | MAE rank |
| --- | ---: | ---: | ---: | ---: |
| ISCF-BSCA-MAIN-v1 | 0.262469 | 1 | 0.308281 | 1 |
| TimeAlign | 0.265935 | 2 | 0.310501 | 2 |
| PatchTST | 0.268637 | 3 | 0.321506 | 3 |
| AMD | 0.273581 | 4 | 0.324574 | 4 |
| QDF | 0.283815 | 5 | 0.336381 | 6 |
| SimpleTM | 0.287747 | 6 | 0.331377 | 5 |
| DLinear | 0.305559 | 7 | 0.357909 | 8 |
| iTransformer | 0.307825 | 8 | 0.342896 | 7 |

MSE 与 MAE 的系统次序不完全相同：SimpleTM 的 MAE 优于 QDF，iTransformer 的 MAE 优于 DLinear。相对最接近的 TimeAlign，ISCF-BSCA 的 macro MSE 与 MAE 分别降低 `1.303%` 与 `0.715%`。这一 aggregate conclusion 不表示 ISCF-BSCA 在每个 dataset 或每个 metric 上均为最优。

## 5. Dataset-mean 与 horizon-mean 排名

| Dataset | ISCF MSE | MSE rank | ISCF MAE | MAE rank | Best system if not ISCF |
| --- | ---: | ---: | ---: | ---: | --- |
| ETTh1 | 0.393520 | 2 | 0.421034 | 3 | AMD for both metrics |
| ETTh2 | 0.307332 | 2 | 0.365116 | 2 | PatchTST for both metrics |
| ETTm1 | 0.330699 | 1 | 0.363879 | 1 | — |
| ETTm2 | 0.248733 | 1 | 0.305693 | 1 | — |
| Weather | 0.214887 | 1 | 0.245821 | 1 | — |
| ECL | 0.151625 | 2 | 0.245635 | 2 | TimeAlign for both metrics |
| Solar | 0.190485 | 3 | 0.210792 | 1 | SimpleTM for MSE |

ISCF-BSCA 的 dataset-mean 优势集中在 ETTm1、ETTm2 和 Weather，两项指标均排名第一；Solar MAE 也排名第一。明确弱项为 ETTh1 MAE（rank 3）、Solar MSE（rank 3），以及 ETTh2/ECL 的双指标 rank 2。所有弱项均保留在主表中，未进行 dataset、horizon、metric 或 seed 选择性删除。

按 horizon 跨七 datasets 聚合时，ISCF-BSCA 的 MSE 在 H96/H192/H336/H720 均为 rank 1；MAE 在 H96/H192/H336 为 rank 1，在 H720 为 rank 2。H720 macro MAE=`0.356220`，TimeAlign=`0.356175`，二者只差约 `0.000045`，因此不能把 H720 MAE 表述为 ISCF 的明确领先。

## 6. Pairwise cell wins

以下为 raw-value pairwise wins/losses，单位为 28 个 dataset–horizon cells；不存在 exact ties。

| Baseline | MSE wins/losses | MAE wins/losses | Combined wins/losses |
| --- | ---: | ---: | ---: |
| TimeAlign | 20/8 | 17/11 | 37/19 |
| QDF | 28/0 | 28/0 | 56/0 |
| AMD | 25/3 | 24/4 | 49/7 |
| SimpleTM | 24/4 | 27/1 | 51/5 |
| iTransformer | 28/0 | 28/0 | 56/0 |
| PatchTST | 23/5 | 24/4 | 47/9 |
| DLinear | 28/0 | 28/0 | 56/0 |

## 7. H720 continuity audit

ISCF-BSCA、TimeAlign、QDF、AMD 与 SimpleTM 共 35 个 exact local anchors 全部通过 `max(1e-8, four float32 ULPs at anchor)` gate。最大 absolute MSE delta 为 QDF ETTm1 的 `2.980232e-08`，对应 tolerance=`1.192093e-07`；最大 absolute MAE delta 为 QDF ETTh2 的 `5.960464e-08`，对应 tolerance=`1.192093e-07`。因此 Main II H720 与冻结 Main I 的同-checkpoint continuity 成立，Main I 未被回写。

iTransformer、PatchTST 与 DLinear 的 Main I anchors 是 published three-run means，本轮 Main II 是 official-source single-seed reproduction；21 个 signed deviations 已完整保存在 `aggregate_audit/h720_main_i_continuity.csv`，只作 reproducibility context，不作为 exact-continuity failure，也不替换 Main I published values。

## 8. Recovery 与 failure attribution

执行中出现的异常均属于 artifact/metric/parser implementation 层，而非模型性能失败。首先，upstream evaluator 重复保留 prediction arrays 导致 storage projection 接近 220 GiB hard limit；在逐文件 hash 与 metric audit 后只删除可再生 arrays，并将 evaluator 改为 workspace-local ephemeral retention，未删除 checkpoint、log 或 metric。其次，TimeAlign、QDF 与 AMD 的 official metric reductions 分别是 NumPy float32 per-step cumulative mean、CPU torch float32 global mean、GPU float32 recursive batch mean；evaluator 已按各自 native contract 修正，同时保留 float64 global audit columns。最后，aggregate parser 首次将 Main I 展示用 `Avg.` row 转为整数而失败；修复只过滤非标准 horizon rows，未重新访问 test、checkpoint 或 prediction tensor。

按项目 failure taxonomy，上述问题归为已修复的 `optimization_or_numeric_pathology`/artifact collection implementation defects，不能据此拒绝任何 forecasting direction。完整聚合通过后不存在未解决的 numeric、hash 或 matrix-completeness blocker。

## 9. Four-layer evidence decision

1. `paper_facing_effectiveness`：pass。完整 224-cell official-test system surface 显示 ISCF-BSCA macro MSE/MAE 均排名第一，且 51/56 displayed metric cells 位于前二。
2. `matched_mechanism_attribution`：missing by design。External systems 的 source-native contracts 不 matched；Main II 不能证明优势由 BSCA、ISCF decoder 或任一单独组件造成。
3. `internal_mechanism_health`：not evaluated in Main II。该层继续由 exact five-dataset ablation、routing/gradient diagnostics 与 two-backbone decoder transfer 承担。
4. `failure_attribution`：Main II system benchmark 本身通过；现阶段不能把 system-level effectiveness 自动提升为 `passed_core_candidate`，也不能用本表替代后续 matched ablation/transfer evidence。

## 10. Canonical artifacts

- Aggregate cells：`aggregate_audit/main_ii_aggregate_cells.csv`
- Raw checkpoint-prefix rows：`aggregate_audit/raw_checkpoint_prefix_metrics.csv`
- H720 continuity：`aggregate_audit/h720_main_i_continuity.csv`
- Machine audit：`aggregate_audit/main_ii_result_audit.json`
- Paper table data：`table/table_data_long.csv`
- LaTeX table：`table/table_iscf_bsca_main_ii.tex`
- Table build summary：`table/table_build_summary.json`

当前 rollback：若后续发现 source/dataset hash 或 metric-contract 漂移，回到 Step 7 并重跑完整受影响 baseline block；性能结论变化时完整重建 Main II，不允许只修改有利 cells。当前不启动 Exchange、common-origin sensitivity 或 optional 3-seed extension。
