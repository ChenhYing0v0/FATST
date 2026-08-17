# Efficiency Accuracy--Memory--Storage Result and Table Audit

日期：2026-08-17

Decision：`complete_five_system_accuracy_advantage_resource_mixed`

## 1. 完整性

- canonical Main I accuracy：5 systems × 7 datasets × 4 horizons=`140/140` cells；
- peak-memory与checkpoint-storage service units：5 systems × 7 datasets=`35/35`；
- 既有独占RTX 3090 profiler复用ISCF-BSCA、TimeAlign、QDF的21 units；本轮新增AMD/SimpleTM 14/14 fresh-process units；
- table-role checkpoint objects：ISCF-BSCA 7个，四个horizon-specific families各28个，共`119/119`；新增AMD/SimpleTM 56/56 hashes逐文件校验；
- new training=`0`；formal test access=`0`；本轮synthetic input profiling不访问dataset loader或labels。

## 2. Canonical result

| System | Main I MSE | Main I MAE | Peak memory (MiB) | Checkpoints (MiB) |
| --- | ---: | ---: | ---: | ---: |
| **ISCF-BSCA** | **0.261** | **0.306** | 38.817 | <u>17.677</u> |
| TimeAlign | 0.274 | 0.314 | 109.102 | 95.433 |
| QDF | 0.288 | 0.331 | <u>31.939</u> | 20.381 |
| AMD | 0.282 | 0.328 | 238.035 | 221.150 |
| SimpleTM | 0.293 | 0.333 | **14.841** | **2.835** |

Accuracy是Main I七datasets × four horizons的macro mean。Peak memory与checkpoint storage先按每dataset构造完整four-horizon service，再对七个datasets作macro mean。上表为文字审计，正式LaTeX中的bold/underline按五个systems逐列计算。

## 3. 比较结果

- ISCF-BSCA在五系统中取得最佳macro MSE/MAE；相对TimeAlign、QDF、AMD、SimpleTM的MSE改善分别为`4.936%`、`9.320%`、`7.591%`、`10.941%`，MAE改善分别为`2.536%`、`7.639%`、`6.717%`、`8.032%`；
- 相对TimeAlign与AMD，ISCF-BSCA的peak memory分别降低`64.422%`与`83.693%`，checkpoint storage分别降低`81.477%`与`92.007%`；
- 相对QDF，ISCF-BSCA checkpoint storage降低`13.269%`，但peak memory高`21.535%`；
- SimpleTM是轻量级反例：其peak memory与checkpoint storage均低于ISCF-BSCA，分别只有`14.841 MiB`和`2.835 MiB`；
- 因此本表支持的是`best macro accuracy + one-checkpoint service consolidation`，以及相对TimeAlign/AMD的显著resource advantage；不支持“ISCF-BSCA在memory或storage上优于所有baseline”。

## 4. Measurement boundary

- hardware=`one exclusive NVIDIA GeForce RTX 3090`；precision=`FP32`；batch size=`1`；input=`synthetic standardized`；
- peak memory为fresh process中的`torch.cuda.max_memory_allocated()`，包含完整service的resident model weights、synthetic inputs与一次all-horizon inference activation peak；
- ISCF-BSCA resident一个unified checkpoint，执行一次$H=720$ forward并生成prefix views；
- TimeAlign/QDF/AMD/SimpleTM resident四个native fixed-H checkpoints，并顺序执行四次native forwards；
- checkpoint storage使用实际文件bytes，而非parameter count估算；
- SimpleTM Main I accuracy沿用冻结multi-run summary；资源统计使用预先冻结的repeat 0，一个repeat代表一次非ensemble deployment，不将三个统计重复同时常驻；
- 该结果描述inference service GPU allocation，不等同于training peak memory、CUDA reserved memory或production end-to-end host memory。

## 5. Exclusions

- iTransformer、PatchTST、DLinear目前只有Main II每dataset一个H720 checkpoint；将其作为four-model cost会违反本表协议；
- TVNet只有ETTh2具备author-corrected local reproduction evidence；
- 其他Main I published-context baselines缺少完整7×4 locally audited checkpoint set，均不进入cost ranking。

## 6. Canonical artifacts

- protocol：`configs/iscf_bsca_efficiency_accuracy_memory_storage_protocol.json`；
- additional unit evidence：`additional_units/*.json`；
- dataset results：`efficiency_dataset_results.csv`；
- macro results：`efficiency_system_macro_results.csv`；
- summary：`efficiency_result_summary.json`；
- LaTeX：`table/table_iscf_bsca_efficiency.tex`；
- standalone LaTeX：`table/table_iscf_bsca_efficiency_standalone.tex`；
- review PDF：`output/pdf/iscf_bsca_efficiency_accuracy_memory_storage_20260817.pdf`。

上一版accuracy--parameters--one-epoch表保留为historical compute-trade-off audit，不删除其负向training-time evidence。
